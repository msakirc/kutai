# Fatih Hoca Phase 2d — Unified Utilization Equation + Stateful Simulator

**Status**: Design kickoff (2026-04-19). Implementation deferred to a fresh session.
**Predecessor**: Phase 2c (`feat/fatih-hoca-phase2c`) shipped pools/urgency/gate machinery but empirically fails to balance. See "What Phase 2c leaves behind" below.

---

## 1. The frame Phase 2d must implement

Model selection is a **utilization balance** problem, not two separate mechanisms. "Don't waste Claude on d=3" and "use Groq before its quota resets in 20 minutes" are the same decision viewed from different ends. Splitting them into urgency-boost + capability-reservation layers creates contradictions (one boosts per_call while the other penalizes it) and fails to reach equilibrium.

**The equilibrium**: for each (model, task) pair, compute a single scalar reflecting the opportunity cost of allocating this model to this task, given:
1. How scarce the model's remaining capacity is (pool + reset horizon + queue depth)
2. How much of that capacity this task consumes vs. what a lighter model would spend
3. How many tasks at various difficulties are still coming

When it balances, Claude runs the hard tasks exactly when they arrive, Groq's free quota gets spent before reset, locals fill in the overflow, and nothing sits idle while nothing is over-served.

## 2. What Phase 2c leaves behind (merge as foundation)

Keep:
- `packages/fatih_hoca/src/fatih_hoca/pools.py` — pool classification is still correct
- `packages/fatih_hoca/src/fatih_hoca/grading.py` — `grading_perf_score` is orthogonal and useful
- `LocalModelState.idle_seconds` in nerd_herd — still the right signal
- `model_pick_log.pool` + `urgency` columns, `candidates_json` with `cap_score`/`pool`/`urgency` per candidate — telemetry lands unchanged
- Simulator fixture's `idle_seconds=300` default — fine starting point
- `packages/fatih_hoca/src/fatih_hoca/counterfactual.py` — CLI format survives; internal `_rescore` gets rewritten to call the new equation

Replace:
- `_apply_urgency_layer` in `ranking.py` — delete body, rewrite as `_apply_utilization_layer` with the unified equation
- `CAP_GATE_RATIO` — delete; gate-via-threshold approach is retired
- `URGENCY_MAX_BONUS` — keep as a tunable, but its role changes (max positive adjustment, not max "boost")
- Simulator's single-snapshot assumption — replaced by stateful model (§5)

## 3. The unified equation

For a candidate (model, task, snapshot, queue-state), compute:

```
fit_excess = (cap_score_100 - cap_needed_for_difficulty(d)) / 100
scarcity   = pool_scarcity(model, snapshot, queue_state)   # range [-1, +1]

# Positive scarcity boosts (use-it-or-lose-it)
# Negative scarcity penalizes proportional to over-qualification
# Neutral scarcity leaves composite alone
adjustment = 1.0 + K * scarcity * (1 - max(0, fit_excess))

composite *= adjustment
```

One multiplier. Sign falls out of the math. `K` is the max magnitude (0.25 is a reasonable start).

### 3.1 `pool_scarcity(model, snapshot, queue_state) -> float`

Returns a value in `[-1, +1]`:

| Pool + state | Scarcity | Interpretation |
|---|---|---|
| Local, idle seconds saturated | `+0.3` to `+0.5` | Paid capacity going unused; mild boost |
| Local, currently busy | `0` or small `-` | Don't pile on; neutral or soft penalty |
| Time_bucketed, reset imminent (<1h) with remaining > 50% | `+0.6` to `+1.0` | Burning; maximum boost |
| Time_bucketed, reset far (>4h) with remaining < 20% | `-0.3` to `-0.5` | Conserve; harder tasks ahead |
| Time_bucketed, balanced | `~0` | Normal |
| Per_call, budget utilization low + no queue pressure | `~0` | Normal cost math |
| Per_call, high utilization OR known hard tasks queued | `-0.6` to `-1.0` | Reserve aggressively |

This integrates with (but doesn't replace) `QuotaPlanner.expensive_threshold` — the threshold drives the per_call scarcity.

### 3.2 `cap_needed_for_difficulty(d) -> float`

**Open question — calibration TBD in Phase 2d.** Two candidate approaches:

**A. Declarative curve.** Hand-tuned linear/piecewise:
```
d=1-3: 30      # trivial; any model
d=4-5: 45
d=6-7: 60
d=8:   75
d=9-10: 88
```

**B. Empirically derived from `model_stats`.** For each difficulty bucket, find the lowest-cap model whose historical `success_rate` exceeds some threshold (say 0.8). That's the cap floor for that difficulty. Gives automatic re-calibration as the pipeline accumulates data.

Phase 2d should probably start with A (data is sparse today) and move to B when `model_stats` has enough samples per (model, difficulty) pair. Design so the curve is a pluggable function.

## 4. Queue-awareness

Scarcity signals must consider not just current state but expected demand. `QuotaPlanner.set_queue_profile` already tracks upcoming task profile (hard_tasks_count, needs_thinking_count, etc.) — wire this into `pool_scarcity`:

- If 5 d≥7 tasks are queued and Claude has 50% budget left, per_call scarcity goes sharply negative for low-difficulty picks — reserve for the hard work coming.
- If only easy tasks are queued and a quota resets in 30 min, scarcity stays positive — nothing to save it for.

This is the **"bin packing" frame** made numeric. Not full lookahead scheduling (expensive and brittle), just a scalar that reads the queue's shape.

## 5. Stateful simulator (new)

Current `simulate_i2p` scores all 182 tasks against one frozen snapshot. Phase 2d needs a simulator that **evolves state across the task sequence**:

- Each pick decrements the picked model's remaining counter
- Simulated wall-clock advances per task (derived from picked model's tps + estimated output tokens)
- Reset events fire at simulated thresholds (daily UTC-midnight for RPD, sliding windows for RPM)
- `idle_seconds` on local updates based on whether the loaded model was used this tick

**Virtual clock**, not real — deterministic and fast. ~1-2 seconds to replay 182 tasks.

Proposed module structure:
- `packages/fatih_hoca/src/fatih_hoca/sim/scenarios.py` — declarative scenario definitions (initial pool state + reset timings)
- `packages/fatih_hoca/src/fatih_hoca/sim/state.py` — mutable simulator state (per-pool counters + clock)
- `packages/fatih_hoca/src/fatih_hoca/sim/runner.py` — evolves state through the 182-task queue; wraps existing `simulate_i2p`
- `packages/fatih_hoca/src/fatih_hoca/sim/report.py` — metrics: Claude-quota-at-d8, local-idle-time, free-quota-wasted-at-reset

Scenarios to include as presets:
1. **baseline** — current default (locals + cloud, cold start)
2. **claude_constrained** — Claude limited to 30 req/day; 182 tasks include 18 d≥7
3. **groq_near_reset** — Groq with 30min to reset at sim-start, 85% remaining
4. **diverse_pool** — claude + groq + gemini-free + openrouter, each with realistic limits
5. **exhaustion_sequence** — progressive exhaustion of free tiers; does the system degrade gracefully?

## 6. Metrics for "balanced"

A policy is balanced when:

- **Hard-task satisfaction**: ≥90% of d≥7 tasks pick a candidate with `cap_score_100 ≥ cap_needed_for_difficulty(d)`
- **Easy-task waste**: <10% of d≤4 tasks pick a candidate with `fit_excess > 0.4` (i.e., grossly over-qualified)
- **Free-quota utilization**: time_bucketed pools run above 70% of their daily capacity in the diverse_pool scenario
- **Local idle**: local idle_seconds rarely exceeds 2× `LOCAL_IDLE_SATURATION_SECS` (not sitting idle for hours while cloud handles everything)

## 7. Scope guardrails

Out of scope for Phase 2d:
- Full scheduler (only a score-layer that considers queue shape)
- Cloud execution wiring (still separate stream; Phase 2d prepares selection for when wiring lands)
- Changing the existing weighted composite (capability/cost/availability/performance/speed weights) — the utilization adjustment is orthogonal
- Prepaid balance tracking infrastructure (still forward-looking)

In scope:
- Unified utilization equation (§3)
- Stateful simulator (§5)
- Queue-aware scarcity (§4)
- Scenario-based validation (§6)
- Retirement of Phase 2c's gate + urgency-only layer

## 8. Open design questions for the fresh session

1. **`cap_needed_for_difficulty` calibration** — hand curve vs. empirical. Start with hand curve; decide empirical timeline during Phase 2d brainstorm.
2. **`K` magnitude** — start at 0.25 (same as Phase 2c's max bonus) and tune via simulator scenarios.
3. **Scarcity formula per pool** — §3.1 gives ranges; finalize exact formulas during Phase 2d brainstorm.
4. **Queue profile integration** — reuse existing `QuotaPlanner.set_queue_profile` or extend Nerd Herd snapshot?
5. **Simulator determinism vs realism** — fixed random seed, or accept some run-to-run variance to expose edge cases?

## 9. First 10 minutes of the Phase 2d session

1. Read this doc + `docs/superpowers/specs/2026-04-18-fatih-hoca-phase2c-design.md` + the Phase 2c plan.
2. Confirm current branch state: `git -C C:/Users/sakir/Dropbox/Workspaces/kutay log --oneline origin/main..main` — expect Phase 2c commits merged to main.
3. Create a new worktree + branch: `git worktree add .worktrees/fatih-hoca-phase2d -b feat/fatih-hoca-phase2d main`.
4. Invoke `superpowers:brainstorming` — work through §8's open questions before writing any code.
5. Write plan; execute subagent-driven.

## 10. Constraints inherited from Phase 2c

- TDD, subagent-driven, commit per green task.
- Worktree required; shared venv at `../../.venv/Scripts/python.exe`; never `pip install -e`.
- Tests: `PYTHONPATH=packages/fatih_hoca/src` for package-local tests, `PYTHONPATH=.` for `tests/unit`, `PYTHONPATH=packages/fatih_hoca/src:.` for `tests/fatih_hoca`. Always `timeout N pytest`.
- The stateful simulator becomes the fast feedback loop (replaces the old snapshot-simulator's role).
