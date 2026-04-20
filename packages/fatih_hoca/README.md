# Fatih Hoca — Model Selection

The model-selection brain of KutAI. Given a task and current system state,
returns the best model for the job. Owns the catalog, scoring, pool
taxonomy, and utilization equilibrium.

## Layers

```
selector.select()                  Layer 1: eligibility filtering
    ↓
ranking.rank_candidates()          Layer 2: 5-dim weighted composite
    ↓ (for each candidate)
_apply_utilization_layer()         Layer 3: Phase 2d equilibrium adjustment
    ↓
returns ScoredModel list, best first
```

## Phase 2d: The Utilization Equilibrium (2026-04-20)

Selection is a utilization balance problem. Two sides of one medallion:

- "Don't waste Claude on a d=3 task" (conserve expensive budget for hard tasks)
- "Don't bother with local 9B on d=3 when Groq resets in 20 min" (burn
  free quota before it expires)

Both are expressed by one scalar in the equation:

```python
fit_excess = (cap_score_100 - cap_needed_for_difficulty(d)) / 100
scarcity   = pool_scarcity(model, snapshot, queue_state, task_difficulty)

if scarcity > 0:
    fit_dampener = max(0, 1 - abs(fit_excess))   # symmetric
else:
    fit_dampener = 1 - max(0, fit_excess)        # over-qual only

composite *= 1 + K * scarcity * fit_dampener     # K = 1.0
```

See `docs/architecture/fatih-hoca-phase2d-equilibrium.md` for the full
design record — scarcity semantics per pool, the dampener's asymmetry
rationale, and which structural bugs surfaced during tuning.

## Running the Simulator

The canonical way to see the equation work against realistic demand:

```bash
# Pretty table of all 7 scenarios
python packages/fatih_hoca/tests/sim/run_scenarios.py

# Stickiness / swap-storm verification with real GGUF registry
python packages/fatih_hoca/tests/sim/run_swap_storm_check.py

# pytest-parametrized scenario assertions
python -m pytest packages/fatih_hoca/tests/test_scenarios.py -v
```

`tests/sim/README.md` has extension guidance, metric semantics, and
swap-storm reasoning.

## Running All Tests

```bash
# fatih_hoca package tests
python -m pytest packages/fatih_hoca/

# fatih_hoca + nerd_herd (dependency-clean)
python -m pytest packages/fatih_hoca/ packages/nerd_herd/

# cross-package integration (requires PYTHONPATH)
PYTHONPATH=packages/fatih_hoca/src python -m pytest tests/fatih_hoca/
```

378 tests across the three buckets as of Phase 2d landing.

## Key Modules

| module | role |
|---|---|
| `capabilities.py` | `score_model_for_task` — 0-10 capability fit |
| `capability_curve.py` | `cap_needed_for_difficulty(d)` lookup (Phase 2d) |
| `counterfactual.py` | CLI replaying `model_pick_log` under candidate K values |
| `grading.py` | `grading_perf_score` from `model_stats` success rates (Phase 2c) |
| `pools.py` | `Pool` enum + `classify_pool` + `UTILIZATION_K` |
| `profiles.py` | Task profiles (coder, planner, analyst, etc.) |
| `ranking.py` | `rank_candidates` + `_apply_utilization_layer` |
| `registry.py` | Model catalog (YAML + GGUF scan) |
| `requirements.py` | `ModelRequirements`, `QueueProfile`, `QuotaPlanner` |
| `scarcity.py` | Phase 2d signed scarcity per pool |
| `selector.py` | Layer 1 eligibility + `select()` / `select_for_simulation()` |
| `simulate_i2p.py` | Dev tool: replay i2p v3 against the selector |
| `types.py` | `Failure` dataclass + shared types |

## Tuning Checklist

When adjusting equilibrium parameters:

1. Make the change (usually in `scarcity.py`, `pools.py::UTILIZATION_K`,
   or `ranking.py::_apply_utilization_layer`'s stickiness block).
2. Run `run_scenarios.py` — all 7 scenarios should keep hard_sat ≥ 90%,
   easy_waste < 10%, diverse_pool free_q > 70%.
3. Run `run_swap_storm_check.py` — swap rate should stay < 5% across
   the three starting configurations.
4. Run `pytest packages/fatih_hoca/` — no unit-test regressions.
5. Commit with a note explaining the tuning rationale.

Do NOT tune by eyeball on a single trace. The scenarios are the
equilibrium specification.
