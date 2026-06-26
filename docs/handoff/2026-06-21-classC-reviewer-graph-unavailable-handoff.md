# Handoff — Class C: reviewer-failure routing can never load the workflow graph

**Date:** 2026-06-21
**Status:** REAL defect, restart-independent, **root cause fully established** (no fix applied).
**Severity:** HIGH — silently defeats `e756355c` (autonomous reviewer→producer re-pend) for **every mission**, not just mission 87. A reviewer 'fail' that should re-pend the at-fault producer instead DLQs the reviewer.

## Symptom (live evidence)
Task **524380** `[1.13] research_quality_review`, mission 87:
- `status=failed`, `error_category=None`, `result` length 2544 = a **valid reviewer verdict** (`{"verdict":"fail","summary":"…competitive positioning file is malformed…charter missing Boundaries…interview script is a placeholder…","issues":[…]}`).
- `error = "reviewer rejected artifact but workflow graph unavailable"`.

So the reviewer did its job (produced a fail verdict). The failure is **downstream routing**.

## Root cause (traced end-to-end)
1. `apply.py::_apply_review_verdict` (≈4283): on `verdict_class == "fail"` it calls
   `wf = await _load_mission_workflow(mission_id)`. If `wf is None` it emits the exact error string and `_retry_or_dlq`s the reviewer (`apply.py:4296-4300`). **This is the only thing that produces that string.**
2. `_load_mission_workflow` (`apply.py:4224-4241`) resolves the workflow name **solely** from the checkpoint table:
   ```python
   checkpoint = await get_workflow_checkpoint(int(mission_id))
   if not checkpoint or not checkpoint.get("workflow_name"):
       return None
   wf = load_workflow(str(checkpoint["workflow_name"]))
   ```
3. **`workflow_checkpoints` has 0 rows — the entire table is empty** (verified: `SELECT count(*) FROM workflow_checkpoints` = 0). So `get_workflow_checkpoint(87)` returns None → `_load_mission_workflow` returns None → error.

### Why the table is empty (two compounding bugs in the writer)
Sole writer = `_check_phase_completion` (`src/workflows/engine/hooks.py:1780-1839`):
- **(a) Writes only on full-phase completion.** It upserts a checkpoint only when **every** task in a `workflow_phase` is in `{completed, skipped, cancelled}` (`all_done`). Mission 87's `phase_0` is NOT fully done (`[0.6a]` 524361 pending, `[0.6a.draft]` 524360 failed), so no checkpoint is ever written — even though the `[1.13]` reviewer in `phase_1` already ran and failed. Phases overlap; the reviewer can fire long before its phase (or phase_0) is `all_done`.
- **(b) Even when it eventually writes, `workflow_name` is seeded `""`.** Line 1815: `workflow_name = checkpoint["workflow_name"] if checkpoint else ""`. On the FIRST write there is no prior checkpoint, so it persists `workflow_name=""`. The loader's `if not checkpoint.get("workflow_name")` treats `""` as falsy → returns None anyway. **The real workflow name is never seeded at mission/workflow start.**

### The name IS available elsewhere
`missions.context.workflow_name = 'i2p_v3'` is reliably present (verified for mission 87). `_load_mission_workflow` simply never looks there.

## Recommended fix (design, not yet applied)
Primary, minimal, restart-independent:
- In `_load_mission_workflow`, **resolve the workflow name from `missions.context.workflow_name` as the primary (or fallback) source**, independent of the checkpoint table. `load_workflow('i2p_v3')` then succeeds and `route_review_failure` can re-pend the producer.

Secondary (fixes the writer so checkpoints are actually useful, and removes a latent footgun for anything else that reads them):
- Seed the checkpoint with the real `workflow_name` at workflow **start** (when the mission is expanded), instead of `""` on first phase completion. Find the expansion/seed site for i2p and write `upsert_workflow_checkpoint(mission_id, workflow_name=<name>, …)` there.
- Optionally make `_check_phase_completion` carry the name forward from mission context rather than from a possibly-absent prior checkpoint.

Prefer fixing **both** the loader (so it works now, for in-flight missions with no checkpoint) **and** the writer (so the table stops being dead). The loader fix alone fully resolves Class C.

## TDD plan
- RED: a `route_review_failure` / `_apply_review_verdict` test where the mission has **no `workflow_checkpoints` row** but `missions.context.workflow_name='i2p_v3'`; assert the producer is re-pended (not the reviewer DLQ'd). This currently fails with the exact error string.
- Existing fixtures `packages/general_beckman/tests/test_review_routing_io.py` and `…_e2e.py` both **manually `INSERT INTO workflow_checkpoints (mission_id, workflow_name)`** — i.e. the tests have been masking this by pre-seeding a row prod never writes. The new test must NOT pre-seed.
- GREEN: loader fallback to mission context.
- Regression: `get_task_by_workflow_step`-backed re-pend path (`review_routing.py::_repend_producer`) already works given a workflow dict.

## Files
- `packages/general_beckman/src/general_beckman/apply.py:4224` (`_load_mission_workflow`), `:4283` (`_apply_review_verdict` FAIL branch), `:4298` (error string).
- `packages/general_beckman/src/general_beckman/review_routing.py` (the routing it's starved of).
- `src/workflows/engine/hooks.py:1780-1839` (`_check_phase_completion` — the broken writer).
- `packages/db/src/dabidabi/__init__.py:6663` (`upsert_workflow_checkpoint`), `:6691` (`get_workflow_checkpoint`).
- Tests masking the bug: `packages/general_beckman/tests/test_review_routing_io.py:178`, `…/test_review_routing_e2e.py:78`.

## Verify method
Read-only DB:
```python
import sqlite3
c=sqlite3.connect("file:C:/Users/sakir/ai/kutai/kutai.db?mode=ro",uri=True)
c.execute("SELECT count(*) FROM workflow_checkpoints").fetchone()   # currently (0,)
```
After fix + restart + a reviewer 'fail': the producer step row should flip to `pending` with `error_category='quality'` and `_schema_error` in its context, NOT the reviewer DLQ'd with "workflow graph unavailable".
