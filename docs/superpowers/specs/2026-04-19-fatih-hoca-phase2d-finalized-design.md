# Fatih Hoca Phase 2d — Finalized Design (Unified Utilization Equation)

**Status**: Approved design (2026-04-19). Supersedes the Phase 2d kickoff spec (`2026-04-19-fatih-hoca-phase2d-design.md`) with concrete decisions on open questions.
**Predecessor**: Phase 2c merged to main. Pools/grading/telemetry retained as foundation.

---

## 1. Frame

Model selection is a **utilization balance** problem. Every (model, task) pair gets one scalar expressing the opportunity cost of allocating that model to that task, combining pool scarcity and capability fit-excess. One multiplier, sign falls out of the math. No separate boost/penalty layers, no gate.

## 2. Core equation (replaces `_apply_urgency_layer`)

In `packages/fatih_hoca/src/fatih_hoca/ranking.py`:

```python
fit_excess = (cap_score_100 - cap_needed_for_difficulty(d)) / 100
scarcity   = pool_scarcity(model, snapshot, queue_state)   # clamped to [-1, +1]
composite *= 1 + K * scarcity * (1 - max(0, fit_excess))
```

- `K = 0.25` starting constant. Tuned via simulator scenarios.
- Positive scarcity (use-it-or-lose-it) boosts; negative (conserve) penalizes proportional to over-qualification.
- Under-qualified candidates (`fit_excess < 0`) clamp to `fit_excess = 0` — scarcity still applies at full strength.
- `CAP_GATE_RATIO` deleted. `_apply_urgency_layer` renamed/rewritten as `_apply_utilization_layer`.

## 3. `cap_needed_for_difficulty(d)` — `capability_curve.py`

Plain dict, pluggable:

```python
CAP_NEEDED_BY_DIFFICULTY = {
    1: 30, 2: 30, 3: 30,
    4: 45, 5: 45,
    6: 60, 7: 60,
    8: 75,
    9: 88, 10: 88,
}
```

Accessor `cap_needed_for_difficulty(d) -> float` returns the dict value, clamping unknown `d` into `[1, 10]` range. Graduation to empirical derivation from `model_stats` is deferred until sample counts warrant it.

## 4. `pool_scarcity(model, snapshot, queue_state)` — `scarcity.py`

Dispatches on `model.pool` (from Phase 2c `pools.py`). Returns float clamped to `[-1, +1]`.

| Pool | State signals | Formula shape |
|---|---|---|
| **local** | `idle_seconds`, `is_loaded`, `requests_processing` | Busy → small negative (-0.1). Idle saturation (≥`LOCAL_IDLE_SATURATION_SECS`) → positive (+0.3 to +0.5). Loaded-but-idle gets the larger boost. |
| **time_bucketed** | `remaining_pct`, `seconds_until_reset` | Reset imminent (<1h) and high remaining (>50%) → +0.6 to +1.0. Reset far (>4h) and low remaining (<20%) → -0.3 to -0.5. Balanced → ~0. |
| **per_call** | `expensive_threshold` from QuotaPlanner, `queue_state.hard_tasks_count`, `queue_state.total_pending` | Hard tasks queued and current task is easy → sharply negative. Low queue pressure → ~0. Never positive (per_call costs never encourage spending). |

Exact piecewise formulas finalized during implementation, tuned against scenarios.

`queue_state` is fetched via the existing `QuotaPlanner.set_queue_profile` / corresponding getter. No nerd_herd changes.

## 5. Simulator — test infrastructure under `packages/fatih_hoca/tests/sim/`

Scenarios are pytest tests that assert §7 metrics. The simulator is not shipped runtime code — it's a test harness. If a future caller needs it in package code, promote then.

Layout:
- `tests/sim/state.py` — `SimState` dataclass: per-pool counters (per_call spend, time_bucketed remaining + reset_at, local idle_seconds + loaded_model), virtual clock.
- `tests/sim/runner.py` — evolves `SimState` across a task sequence. Each pick:
  1. Calls `fatih_hoca.select()` with a snapshot derived from current `SimState`
  2. Decrements the picked model's pool counter
  3. Advances virtual clock by `estimated_output_tokens / picked_model.tps`
  4. Fires reset events when clock crosses thresholds
  5. Updates `idle_seconds` for locals (zero for the loaded model if used this tick, else +=delta)
- `tests/sim/report.py` — computes metrics across a run: hard-task satisfaction, easy-task waste, free-quota utilization, local idle distribution.
- `tests/sim/scenarios.py` — scenario factory: task sequences + initial `SimState`.
- `tests/test_scenarios.py` — 7 pytest cases, each loads a scenario, runs the runner, asserts §7 targets.

Fixed seed (`random.seed(42)`) for determinism. Variance scenarios can be added later if flakiness is never the goal.

## 6. Scenarios (7 total)

1. **baseline** — default cold-start: locals + cloud, standard i2p v3 task mix (182 tasks).
2. **claude_constrained** — Claude limited to 30 req/day; 182 tasks include 18 d≥7.
3. **groq_near_reset** — Groq with 30min to reset at sim-start, 85% remaining. Should burn before Claude.
4. **diverse_pool** — claude + groq + gemini-free + openrouter, each with realistic limits.
5. **exhaustion_sequence** — progressive exhaustion of free tiers; graceful degradation required.
6. **back_to_back_i2p** — 3 fresh i2p workflows queued simultaneously (~546 tasks). Tests queue-profile scaling: scarcity must reserve Claude for d≥7 across all three, Groq/free pools must burn before Claude on any d≤5 work.
7. **staggered_i2p** — one i2p ~50% through (partial per_call utilization, remaining queue skewed easy) when a second i2p starts fresh. Tests mid-flight queue-profile updates: new i2p's hard steps get served without starving the in-progress one.

## 7. Pass criteria (simulator metrics)

A policy is balanced when **all scenarios pass**:

- **Hard-task satisfaction**: ≥90% of d≥7 tasks pick a candidate with `cap_score_100 ≥ cap_needed_for_difficulty(d)`
- **Easy-task waste**: <10% of d≤4 tasks pick a candidate with `fit_excess > 0.4`
- **Free-quota utilization** (diverse_pool, back_to_back_i2p): time_bucketed pools run above 70% of their daily capacity
- **Local idle**: local `idle_seconds` rarely exceeds 2× `LOCAL_IDLE_SATURATION_SECS`
- **Exhaustion graceful** (exhaustion_sequence): no selection crash when a pool empties; fallback to remaining eligible candidates

## 8. What Phase 2c keeps (foundation, no churn)

- `pools.py` — pool classification unchanged
- `grading.py` — `grading_perf_score` unchanged
- `LocalModelState.idle_seconds` in nerd_herd — unchanged
- `model_pick_log.pool` + `urgency` columns, `candidates_json` with `cap_score`/`pool`/`urgency` — unchanged (urgency column repurposed to store the utilization adjustment for telemetry continuity)
- Counterfactual CLI (`counterfactual.py`) — surface unchanged; internal `_rescore` rewrites to invoke new equation

## 9. What gets deleted

- `_apply_urgency_layer` body (rewritten as `_apply_utilization_layer`)
- `CAP_GATE_RATIO` constant
- `URGENCY_MAX_BONUS` → renamed to `UTILIZATION_K` (the `K` in the equation), value `0.25`

## 10. Scope guardrails

**In scope:**
- Unified utilization equation in `ranking.py`
- `capability_curve.py`, `scarcity.py` (new runtime modules)
- Stateful simulator as test infrastructure under `tests/sim/`
- 7 scenario tests in `tests/test_scenarios.py`
- Queue-awareness wired through existing `QuotaPlanner.set_queue_profile`

**Out of scope:**
- Empirical `cap_needed_for_difficulty` derivation from `model_stats`
- Nerd Herd snapshot extensions
- Cloud execution wiring
- Changing existing weighted composite (capability/cost/availability/performance/speed weights)
- Prepaid balance tracking

## 11. Testing approach

TDD per task. Each runtime module (`capability_curve`, `scarcity`, `ranking._apply_utilization_layer`) gets a unit test suite written first. Simulator scenarios serve as integration tests — the §7 targets are the acceptance criteria. Implementation tasks commit per green test module.

## 12. Constraints inherited from Phase 2c

- Worktree: `.worktrees/fatih-hoca-phase2d`, branch `feat/fatih-hoca-phase2d`
- Shared venv at `../../.venv/Scripts/python.exe`; **never `pip install -e` from worktree paths** (corrupts editable install pointer)
- Test paths: `PYTHONPATH=packages/fatih_hoca/src` for package-local tests, `PYTHONPATH=.` for `tests/unit`, `PYTHONPATH=packages/fatih_hoca/src:.` for `tests/fatih_hoca`
- Always `timeout N pytest` — never unbounded
- Subagent-driven execution per kickoff memory

## 13. Implementation order (informs the plan)

1. `capability_curve.py` + tests
2. `scarcity.py` + tests (per-pool formulas, tuned iteratively)
3. `ranking._apply_utilization_layer` + tests (wire equation, delete gate)
4. `tests/sim/state.py` + `runner.py` + `report.py` (harness)
5. `tests/sim/scenarios.py` + `tests/test_scenarios.py` (7 scenarios)
6. Simulator-driven tuning: adjust `K`, scarcity piecewise values, curve entries until all scenarios pass
7. `counterfactual.py._rescore` rewrite to use new equation
8. `ranking.py` cleanup: delete gate constants, rename `URGENCY_MAX_BONUS` → `UTILIZATION_K`
9. Docs: CLAUDE.md Phase 2d note, architecture doc update
