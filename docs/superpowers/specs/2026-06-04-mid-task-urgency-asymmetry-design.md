# Design — Mid-task urgency asymmetry fix ("No model candidates available" too often)

**Date:** 2026-06-04
**Status:** Approved (Approach A), ready for implementation plan
**Related:** `docs/handoff/2026-06-03-model-selection-and-grader-divergence-handoff.md` (§7b correction), memory `project_no_models_available_gate_20260603`, `docs/architecture/fatih-hoca-phase2d-equilibrium.md`

## Problem

`ModelCallFailed: No model candidates available (category=availability)` fires too often (16× in one log rotation `kutai.jsonl.1`). It is noise + burns worker attempts + adds latency on tasks that were already running fine.

### Root cause (verified against logs + code)

The pool-pressure gate (`packages/fatih_hoca/src/fatih_hoca/selector.py:317-360`) returns `None` when every ranked candidate is below `threshold = max(-1.0, -0.5 - 0.5 * urgency)`. There are **two readers of that gate with opposite behavior on `None`:**

1. **Admission** (`packages/general_beckman/src/general_beckman/__init__.py:642-651`): runs `select()`; on `None` it `continue`s — the task is **not claimed, left pending, waits** for capacity. No error. This is correct and is exactly the desired behavior.

2. **Mid-task ReAct re-selection** (`packages/coulson/src/coulson/dispatch_helpers.py:pick_for_iter` → raise at `packages/coulson/src/coulson/react.py:600`; parallel site `packages/husam/src/husam/worker.py:188-266`): a task admitted *with* capacity runs iteration 1 on its admission pick, then a later iteration (or a failed-call re-select) queries the gate again. If the pool emptied meanwhile, it **raises `ModelCallFailed(availability)`** instead of waiting.

**The asymmetry is compounded by a self-inflicted strictness bug:** `dispatch_helpers.py:61` hardcodes the mid-task base `urgency = 0.5`, ignoring the task's actual admission urgency. `compute_urgency` (`general_beckman/admission.py:22`) = `priority/10 + age·0.05 + unblocks·0.05`, so a priority>5 or aged task is admitted at, e.g., urgency 0.8 (threshold −0.90) but **re-judged mid-task at 0.5 (threshold −0.75)** — stricter than at admission. An alive-band candidate (e.g. a rate-limited cloud model at scalar −0.83, which the gate's own comments classify as "pressured but **alive**", −1.0 being the only "dead" value) that was admittable becomes vetoed mid-flight → raise. A started task is treated as *less* deserving of a model than a fresh one — the inverse of the intent.

The `dispatch_helpers.py` docstring (lines 51-54) already quotes the correct design (user, 2026-05-03): *"mid task urgency of the task can be a little higher than pre-dispatch urgency to help react loops finish."* The code never implemented it.

### Out of scope (deliberately not changed)

- The **−1.0 hard veto** (`packages/nerd_herd/src/nerd_herd/signals/s9_perishability.py` `LOCAL_BUSY_PENALTY`): a busy local on a single-GPU `--parallel 1` host must stay out. The handoff §4 "local fallback" theory was wrong; locals correctly peg at −1.0 when one local call is in-flight.
- **Admission-wait-on-None**: already correct.
- The **gate formula** `threshold = −0.5 − 0.5·urgency`: unchanged.
- **Bounded wait-and-reselect** (proposal B) and **urgency escalation across Beckman retries** (proposal C-style): not in this change.

## Approach A — one source of truth for mid-task urgency

Thread the admission urgency to the mid-task selection and add a small finish-bias, so a started task finishes on the same alive band it was admitted under.

### Change 1 — Beckman stamps admission urgency

`packages/general_beckman/src/general_beckman/__init__.py`, where `task["preselected_pick"] = pick` is set (~line 692): also set

```python
task["_admission_urgency"] = urgency   # the compute_urgency(task) value from ~line 584
```

Rides the same in-memory task-dict path already used for `preselected_pick`/`_held_pick`. Recomputed and re-stamped fresh on every (re)admission → no staleness across retries.

### Change 2 — coulson mid-task urgency

`packages/coulson/src/coulson/dispatch_helpers.py:pick_for_iter`, replace the flat base:

```python
FINISH_BIAS = 0.1     # "a little higher than pre-dispatch" — user design 2026-05-03
FAILURE_BUMP = 0.1    # existing per-iter escalation on failures

base = task.get("_admission_urgency", 0.5)
urgency = min(1.0, base + FINISH_BIAS)
if failures:
    urgency = min(1.0, urgency + FAILURE_BUMP)
```

(Constants named/centralized; exact home decided in the plan.)

### Change 3 — husam alignment

`packages/husam/src/husam/worker.py:188`, converge on the same formula: base from `task.get("_admission_urgency")` (fallback to the current `urgency_in`, then `0.5`), add `FINISH_BIAS`, keep the failure bump. One formula, both raise-sites. (During implementation, confirm what `urgency_in` currently carries; reading the stamped value makes the source unambiguous regardless.)

## Behavior after the fix

- A priority>5 / aged / blocker-unblocking task that was admitted on the alive band continues to be served that band mid-flight → no spurious raise.
- The −1.0 hard veto and admission-wait are untouched; genuinely empty pools still raise (mid-task) / wait (admission) as before.

### Accepted limitation

A baseline priority-5 task: admission urgency 0.5 → mid 0.6 → threshold −0.80. A candidate at exactly −0.83 is still rejected (needs urgency ≥ 0.66, i.e. threshold ≤ −0.83). Such tasks wait under total contention — acceptable per user ("low-urgency tasks can wait"). `FINISH_BIAS` is the tuning knob if baseline tasks should later be more aggressive.

## Testing (TDD)

- **coulson** (`packages/coulson/tests/`): `pick_for_iter` computes `base + FINISH_BIAS` from `_admission_urgency`; `+FAILURE_BUMP` when `failures` present; clamp at 1.0; default 0.5 when unstamped. Regression: a high-`_admission_urgency` started task admits an alive-band candidate that flat-0.5 rejected (assert via a stubbed/fake selector or a constructed scored pool).
- **husam** (`packages/husam/tests/`): same formula honored at `worker.py` mid-task path.
- **Beckman** (`packages/general_beckman/tests/`): admission stamps `_admission_urgency` equal to `compute_urgency(task)`.
- Run suites separately (the `tests/` vs `packages/*/tests/` conftest collision), with a timeout and `-p no:warnings`, venv `.venv/Scripts/python.exe`.

### Mandatory Phase 2d validation (CLAUDE.md)

Urgency feeds the gate, so re-run and confirm no equilibrium regression:

```
.venv/Scripts/python.exe packages/fatih_hoca/tests/sim/run_scenarios.py
.venv/Scripts/python.exe packages/fatih_hoca/tests/sim/run_swap_storm_check.py
```

(These set their own urgency inputs at the selection layer; this change alters only the urgency *passed in* by the mid-task ReAct callers, not the gate formula — expect no scenario drift, but the run is non-negotiable.)

## Rollout

- Commit per conventional-commits; pushes to `main`. Not live until the next KutAI restart (user-managed via Telegram).
