# Fatih Hoca Phase 2d Simulator

Test infrastructure for validating the unified utilization equation against
realistic demand patterns. **Not shipped runtime code** — lives under
`tests/sim/` and exists solely to exercise `fatih_hoca.select()` with stateful
pool counters, a virtual clock, and a live `QueueProfile` per tick.

## The Equation Being Tested

For each (model, task, snapshot, queue_state) triple:

```
fit_excess = (cap_score_100 - cap_needed_for_difficulty(d)) / 100
scarcity   = pool_scarcity(model, snapshot, queue_state, task_difficulty)   # [-1, +1]

if scarcity > 0:
    fit_dampener = max(0, 1 - abs(fit_excess))   # symmetric (under- & over-qual)
else:
    fit_dampener = 1 - max(0, fit_excess)        # over-qual only
composite *= 1 + K * scarcity * fit_dampener
```

`K = UTILIZATION_K = 1.0`.

## Running

From the repo root (or any worktree):

```bash
# Canonical: pretty table of all 7 scenarios
python packages/fatih_hoca/tests/sim/run_scenarios.py

# Via pytest (parametrized assertions)
python -m pytest packages/fatih_hoca/tests/test_scenarios.py -v

# All sim-related tests including the simulator unit tests
python -m pytest packages/fatih_hoca/tests/sim/ packages/fatih_hoca/tests/test_scenarios.py
```

Use `C:\Users\sakir\Dropbox\Workspaces\kutay\.venv\Scripts\python.exe` on Windows.

## The 7 Scenarios

| scenario | setup | exercises |
|---|---|---|
| `baseline` | Local + Groq(1000) + Claude(1000), 24h resets | Default cold-start — capability respected |
| `claude_constrained` | Claude limit=60, ~47 hard tasks | Abundance → depletion transition as budget drains |
| `groq_near_reset` | Groq 30min to reset, 85% left | Imminent-reset "burn before expiry" |
| `diverse_pool` | groq+gemini+or with staggered resets (6h/3h/1.5h) | Free-quota utilization across mixed urgencies |
| `exhaustion_sequence` | Tight free budgets (40/40), full Claude | Graceful degradation as pools empty |
| `back_to_back_i2p` | 3 workflows concurrent (546 tasks) | Demand surge vs conservation |
| `staggered_i2p` | 91 tasks of one i2p, then 182 of another | Mid-flight queue-profile shifts |

## Pass Criteria (spec §7)

- **hard_task_satisfaction ≥ 90%**: fraction of d≥7 picks meeting `cap_needed`.
- **easy_task_waste < 10%**: fraction of d≤4 picks where a **per_call** (paid)
  model was picked with `fit_excess > 0.4`. Free-pool over-qualified picks
  don't count — they are intended "burn before reset" behavior.
- **free_quota_utilization > 70%** (diverse_pool only): average of
  `min(1.0, picks_per_pool / pool.limit)` across time_bucketed pools.
- **exhaustion_sequence_no_crashes**: runner completes all tasks without
  raising when pools empty.

## Metric Semantics

### `easy_task_waste` excludes free pools
Spec §7 originally defined waste as any d≤4 pick with `fit_excess > 0.4`.
This conflicted with the utilization equation's intent: Groq on d=3 with
imminent reset is exactly what we want. Waste is only the opportunity cost
of spending a **paid** budget on an easy task — free-pool over-qualification
is not waste because we get no refund for unused quota.

### `free_quota_utilization` counts picks, not final remaining
A pool that resets mid-run has its `remaining` counter refilled, erasing
pre-reset picks from a final-state reading. The metric counts picks per
pool (reset-proof) and divides by `limit`, capped at 1.0 per pool so a
single heavily-used pool doesn't mask a neglected one in the average.

## Why this exists — design reasoning

The utilization equation's job is to express a **single equilibrium** with
two ends: don't waste Claude on easy tasks (negative scarcity when budget
is healthy + there are hard tasks queued) AND don't waste free quota that
resets in 20 minutes (positive scarcity when reset is imminent). The
equation combines them into one scalar per (model, task) pair.

During tuning, a few structural issues surfaced:

1. **Stickiness at 1.4× crushed cloud.** A loaded local would win every task
   by base ranking alone because the 40% composite multiplier overpowered
   any utilization adjustment. Dialed to 1.10× (main) / 1.50× (overhead) —
   enough to break ties between close-cap locals (its actual purpose), not
   enough to override capability gaps.

2. **Stickiness on under-qualified locals was a capability override.** Even
   at 1.10×, a loaded local on a d=8 task would sometimes beat Claude
   because the multiplier compounded with cost/speed advantages. Added a
   qual_factor: stickiness fades linearly from full at `fit_excess=0` to
   zero at `fit_excess=-0.2`. The loaded model is a good default, not a
   right answer — when it's the wrong tool, stickiness disappears.

3. **Symmetric fit dampener on positive scarcity.** Originally the
   over-qualification dampener only applied to positive scarcity for
   over-qualified candidates (Claude on d=3 gets diminished boost). But an
   under-qualified candidate with positive scarcity (loaded local at idle
   on a hard task where it's under-qual) was getting the full "burn me"
   signal. Burning a wrong tool is itself wasteful. Made the dampener
   symmetric for positive scarcity: `1 - abs(fit_excess)`. Conservation
   signals (scarcity < 0) still apply at full strength regardless of fit.

4. **Per-call scarcity needed a positive arm.** Spec §3.1 said per_call
   was conservation-only — but then Claude has no positive signal when
   budget is flush. The base ranking had to carry all the capability
   preference, and it wasn't strong enough on hard tasks vs fast+free
   groq. Added an abundance arm: full +1 when `remaining_frac > 15% AND
   task_difficulty ≥ 7`, then the depletion arm takes over below 15%.

5. **Time-bucketed signal is continuous, not binary.** Original spec had
   "imminent (<1h)" vs "far (>4h)" with neutral in between. But a 24h
   reset with 100% remaining is still waste if you never use it. Switched
   to exponential decay with 24h characteristic time:
   `time_weight = exp(-reset_in / 86400)`. 1h → 0.96, 24h → 0.37, 72h → 0.05.

## Swap-storm verification

Stickiness's original purpose — preventing swap storms between close-cap
locals — still works. Verified with the real model registry (23 GGUFs,
7 general-purpose locals with cap range 35-80):

| start | swaps/200 | behavior |
|---|---|---|
| Qwen-9B (cap=68) | 0 | Stays loaded, wins all |
| Qwen-27B (cap=80) | 0 | Stays loaded, wins all |
| GigaChat (cap=35) | 1 (0.5%) | Handles 16 easy tasks as-loaded, swaps to gemma on first d=10 task, holds for 184 more |

The GigaChat case is the diagnostic moment: stickiness holds when the
loaded model is adequate for the task (qual_factor=1 at `fit_excess ≥ 0`),
but disappears when it's grossly wrong (qual_factor=0 at `fit_excess ≤ -0.2`).
One corrective swap, then equilibrium.

## File map

- `state.py` — `SimState`, `SimPoolCounter`, `SimLocalModel` (per-pool counters + virtual clock + idle tracking)
- `runner.py` — `run_simulation()` evolves state across a task sequence, returns `SimRun` with per-task picks
- `report.py` — `compute_metrics()` → `SimMetrics` (hard_sat, easy_waste, free_q, max_local_idle)
- `scenarios.py` — 7 `Scenario` factories + `_build_select_fn` closure wiring `selector.select_for_simulation`
- `run_scenarios.py` — runnable entry point (prints the table)
- `test_state.py`, `test_runner.py`, `test_report.py`, `test_scenarios_smoke.py` — unit tests

## Extending

To add a new scenario:

1. Define a factory in `scenarios.py` returning a `Scenario(name, tasks, initial_state, snapshot_factory, select_fn)`.
2. Reuse `_standard_i2p_task_mix()` for realistic difficulty distributions or build a custom task list.
3. Add to `SCENARIOS` in `run_scenarios.py` and `test_scenarios.py` for parametrized coverage.

To modify the equilibrium constants, tune in this order:
1. `UTILIZATION_K` in `pools.py` — overall magnitude of the utilization layer.
2. Per-pool scarcity parameters in `scarcity.py` — relative strength of each signal.
3. Stickiness in `ranking.py::_apply_utilization_layer`'s Group C — swap-storm damper strength.

Always verify all 7 scenarios + the real-registry swap-storm test after any tuning change.
