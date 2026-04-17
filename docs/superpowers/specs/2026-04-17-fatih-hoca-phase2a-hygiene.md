# Fatih Hoca Phase 2a — Scoring Hygiene

**Date:** 2026-04-17
**Scope:** `packages/fatih_hoca/` only. No cross-package changes.
**Predecessor:** `docs/superpowers/plans/2026-04-17-fatih-hoca-selection-intelligence.md` (Phase 1, merged in `c79b770`).
**Successor (deferred):** scheduled benchmark refresh. Blocked on `refactor/orchestrator-phase1` merging `src/app/scheduled_jobs.py` to main; lands as a small follow-up once that infrastructure is available.

## Goal

Two cleanup items on Fatih Hoca that were deferred from Phase 1 because they needed the benchmark signal in place to evaluate correctly:

1. **Remove `asyncio.get_event_loop()` deprecation** in `selector.py` so pick telemetry stays forward-compatible with Python 3.12+.
2. **Audit and likely remove the specialty-bonus multiplier** in `ranking.py`, which now double-counts with blended benchmark+profile capability scores.

Both are hygiene: no new behavior, just removing noise that distorts selection or breaks on future Python upgrades.

## Non-goals

- Scheduled benchmark refresh (deferred, see Successor above)
- Weight auto-calibration (waits on `model_pick_log` data)
- Any change to eligibility filtering, swap budget, or failure adaptation
- Touching the scheduled_tasks table or orchestrator

## Item 1 — Deprecation fix

**Current code** (`selector.py:~290`):

```python
try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(_write())
    else:
        loop.run_until_complete(_write())
except RuntimeError:
    pass
```

`asyncio.get_event_loop()` emits `DeprecationWarning` in Python 3.10+ when no running loop exists and will raise in 3.12+.

**Fix:**

```python
try:
    asyncio.get_running_loop()
    asyncio.create_task(_write())
except RuntimeError:
    # No running loop — sync context. Skip persistence for this call.
    # Telemetry is best-effort; callers running outside an event loop
    # (CLI tools, sync tests) simply won't write. Production hot path
    # is always inside an asyncio loop.
    pass
```

**Behavioral change:** telemetry writes in sync context used to attempt `run_until_complete` on a (possibly defunct) loop; now they're silently skipped. This is deliberate — the fire-and-forget helper was never meant to block a sync caller, and `run_until_complete` on a background loop could stall. The only sync callers today are tests and smoke checks; production selection always runs under asyncio.

**Testing:**

- Existing `tests/fatih_hoca/test_pick_telemetry.py::test_select_persists_pick_to_db` already exercises the async-context path. Must still pass.
- New test: call `fatih_hoca.select()` from a pure-sync context (no event loop). Assert no crash, no `RuntimeWarning`, no DB row written. Confirms silent-skip.
- Run with `python -W error::DeprecationWarning -m pytest tests/fatih_hoca/ -q` to guarantee no deprecation leaks remain.

## Item 2 — Specialty-bonus audit

**Current code** (`ranking.py`, Layer 3):

```python
# Group B: Specialty alignment
if model.specialty == task_specialty:
    composite *= 1.15
    reasons.append(f"specialty={model.specialty}")
```

`task_specialty` is derived from the agent/task name (`coder` → `coding`, `researcher` → `research`, etc.). The bonus was added before benchmark data was wired in, to push specialized local models up the rankings when profile capabilities alone couldn't separate them.

**Why it's probably redundant now:**

- `TASK_PROFILES` already weights the relevant capability dimensions heavily. For a coder task, `code_generation` and `code_reasoning` profile weights are ~0.9–1.0 while unrelated dims are ~0.2.
- `blend_capability_scores()` (wired in Phase 1) pulls real AA benchmark signal into `ModelInfo.capabilities` for the dimensions the task profile weights most. A coding-specialized model with strong AA coder scores now has a high blended `code_generation` AND gets double-weighted by the task profile.
- The 1.15× multiplier stacks on top — a coding model on a coder task gets its already-high composite score multiplied again. Non-specialized coders (a general model with strong coding benchmarks) don't get the multiplier, even if their benchmarks say they should outrank the specialty model.

**Fix:** remove the 1.15× multiplier. Keep `model.specialty` as a field (used elsewhere for eligibility — e.g., don't route prose tasks to a dedicated coder). Add the specialty name to `reasons` for observability only, without numeric effect.

**Risk & rollback:**

- The Phase 1 E2E test (`test_e2e_benchmark_driven_ranking.py`) already asserts that benchmark signal promotes the right coder without relying on the specialty bonus (neither seeded model has `specialty="coding"`). So removing the multiplier should not flip that assertion.
- If any existing `packages/fatih_hoca/tests/test_ranking.py` test asserts that a specialty model wins over a non-specialty model of equal raw capability, the test was measuring the bonus itself. Update those tests to assert ties or rely on a realistic capability-delta instead.
- Rollback if broad regressions appear: reinstate as 1.05× (half-strength, documents the design intent) instead of full removal. This is a one-character config in the fix, not a new design.

**Testing:**

- Existing `test_ranking.py` suite must pass (or be updated per the above note).
- Phase 1 E2E must still pick `qwen3-32b` for the coder task.
- New test: two models with identical capabilities, one with `specialty="coding"` and one without. Before the fix they'd rank differently; after, they tie on composite (bonus gone).
- New test: a model with `specialty="coding"` but weak AA coder scores loses to a general model with strong AA coder scores on a coder task — proving benchmark signal now dominates.

## Observability

- `reasons` list: `specialty` entry retained but no longer emits a multiplier value. If future debugging wants to see whether a specialty matched, the name is still there.
- Pick log (`model_pick_log.candidates_json`) unchanged shape.

## Out-of-scope follow-ups tracked for Phase 2b+

- Scheduled benchmark refresh (depends on `src/app/scheduled_jobs.py` merging).
- Weight auto-calibration using `model_pick_log` outcomes (needs ~2 weeks of data).
- Task profile → specialty remapping (if it turns out some task profiles rely on specialty matching for hard filtering rather than ranking, document and keep, but that's a later audit).
- `asyncio.get_running_loop()` plus a proper dedicated telemetry queue — current fix keeps best-effort semantics; a real queue would give durability, but that's over-engineering for Phase 2a.
