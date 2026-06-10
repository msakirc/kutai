# Handoff — SP5 remaining: delete `await_inline` (the blocked finale)

**Date:** 2026-06-10
**For:** the session that finishes SP5 by deleting the `await_inline` blocking primitive.
**From:** the SP5 deletion-sweep session that retired `LLMDispatcher.request()`.
**Predecessor handoff:** `docs/handoff/2026-06-09-sp5-deletion-sweep-handoff.md` (Steps 1–3 are now DONE; its Step 4 is this doc).
**Read first:** memory `project_shopping_cps_migration_20260609`, `project_cps_sp4b_plan1_plan3_merged_20260608`.

---

## What just shipped (this session) — MERGED to main + PUSHED

`LLMDispatcher.request()` is **deleted**. Merge commit **`786b0ad9`**, pushed to `origin/main`. Live KutAI still runs old code until a founder `/restart` (restart-gated).

Deleted (all re-grepped to zero live callers first):
- `LLMDispatcher.request()` + `_request_kwargs_to_spec` + `_task_result_to_request_response` (`src/core/llm_dispatcher.py`, −~190 LOC). Only call surface left is `execute()`.
- `coulson.reflection.self_reflect` (the function only — kept the re-exported `REFLECTION_BLOCKS`/`STACK_BLOCKS`/`LAYER_BLOCKS`/`build_reflection_prompt`/`build_reflect_messages`, which the Z2/Z3 tests + the posthook child import).
- `coulson.single_shot` + the `execution_pattern=="single_shot"` branch in `coulson.execute` + the `src/runtime/single_shot.py` shim. No live profile sets `single_shot`.
- `src/workflows/engine/constrained_emit.py::maybe_apply` (+ its test). Dead — constrained-emit is a CPS posthook child.

Test fallout fixed: rewrote the 5 `coulson._single_shot_run` patches (z6 detect-and-bail); deleted `tests/migration/test_dispatcher_alias_compat.py` (validated the deleted alias); deleted 2 obsolete real-LLM tests in `test_e2e_llm_pipeline.py`; retargeted 3 "no-LLM" guards (`test_ab_harness`, `test_hypothesis_verdict`, `test_growth_backlog`) from `LLMDispatcher.request` → `LLMDispatcher.execute`.

Verified green: 109 packages + 105 reflection-consumer + 11 guards + 9 z6, plus merged-main re-run (9 z6 + import smoke).

---

## What remains — delete `await_inline` (the real SP5 finale)

`await_inline=True` is the blocking inline-waiter on `general_beckman.enqueue`. After this session, it had **exactly two live callers** (verified — the `request()` shim that was the third is gone). **Update 2026-06-11:** the image-gen Plan 3 swap mechanic added two more (carve-outs 3+4 below). All four are deliberate, documented deferrals. Deleting `await_inline` requires migrating ALL FOUR off it first.

### Carve-out 1 — `src/core/task_classifier.py:284` (`_enqueue_inline_classifier`)
- `tr = await general_beckman.enqueue(spec, parent_id=None, await_inline=True)`.
- Comment (lines 280–283): *"the one edge-group await_inline site SP2 keeps, because classify_task's caller (add_task) consumes the returned TaskClassification synchronously. CPS-migrating this requires redesigning task admission — see SP2 spec §Site 2 special case."*
- **This is the hard blocker.** `add_task` calls the classifier and uses its result inline to set agent_type/difficulty before the row is even enqueued. Making it async (CPS) means task admission can no longer classify-then-enqueue in one synchronous step — admission itself becomes two-phase. That is a task-admission redesign, not a sweep. Scope it as its own spec.
- NOTE: this is distinct from the telegram message classifier — `telegram_bot._classify_user_message` (line 7889) was ALREADY CPS-migrated by SP2 (returns `None`, enqueues with `on_complete="telegram.message_route_resume"`). It does NOT use `await_inline`. Only `task_classifier`'s `add_task`-path classifier still does.

### Carve-out 2 — `src/app/jobs/investor_bullets.py:211` (`_call_llm_anomaly_hypothesis`)
- `await_inline=True` inside the ONESHOT-lane enqueue.
- Comment (lines 204–210): SP5-DEFERRED — *"investor_bullets' anomaly-hypothesis path is unreachable in production today (missions.product_id is NULL … fetchers return {} and this code path never fires). CPS-migrating it costs either ~80 LOC + a pending-table schema or a kickoff/finalize split — neither justified while the upstream producer is missing."*
- **Dormant.** Cheapest resolution: when you do tackle await_inline, either (a) CPS-migrate it (kickoff enqueue + `on_complete` that resumes `run_investor_bullets` from a pending row), or (b) since it never fires in prod, drop the LLM-hypothesis call entirely (render bullets with `_needs founder explanation_` placeholders, which the render layer already supports) and delete `_call_llm_anomaly_hypothesis`. Confirm with founder which.

### Carve-outs 3 + 4 — image-gen Plan 3 swap mechanic (added 2026-06-11)
- `packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py:274` — the prompt_writer enqueue (`_enqueue_prompt_writer`) blocks on `await_inline=True` for the single diffusion-prompt task.
- `packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py:345` — the per-image enqueue (`_generate_one_image`) blocks on `await_inline=True` once per placeholder inside the swap fanout.
- Future CPS migration = kickoff + `on_complete` continuation resuming the fanout; today both sites graceful-degrade on timeout (placeholders keep their placehold.co URLs).

### Deletion order for the await_inline finale
1. Re-grep: `rg -n "await_inline\s*=\s*True" src packages --glob '!*/tests/*'` — confirm still exactly the four above (no new caller crept in).
2. Resolve carve-out 2 first (it's small / dormant).
3. Resolve carve-outs 3+4 (the swap-mechanic CPS continuation described above).
4. Resolve carve-out 1 (the task-admission redesign — its own spec + plan).
5. Only when all four are off `await_inline`: delete the `await_inline` parameter from `general_beckman.enqueue` (+ the inline-waiter future machinery). Re-grep tests for `await_inline=True` and update the deadlock guards (`packages/general_beckman/tests/test_no_inline_deadlock.py` asserts NObody passes it — those become stronger / trivially-true).

---

## Residual (NOT blocking, left deliberately)

- **Pre-existing stale tests still referencing the deleted `LLMDispatcher.request`** — left untouched because they were ALREADY red on main for unrelated contract changes (fixing the `request` ref would not make them pass; they need full rewrites that belong to other cleanups):
  - `tests/test_mission_workflow_integration.py::TestLLMClassification` (3 tests) — assert a return value from `_classify_user_message`, which SP2's CPS migration changed to return `None`. Rewrite for the async/CPS classifier contract, or delete.
  - `tests/integration/test_agent_basic.py` (the ReAct-iteration test ~line 300) — patches `src.agents.base.execute_tool`, removed by Runtime Phase A (`base.py` is now a 4-line delegator to `runtime.execute`). Delete or rewrite against `coulson.execute`.
- **`src/tools/vision.py`** calls `husam.run()` directly (ruling-#1 non-compliant). Does NOT touch `request()`/`await_inline` — separate compliance fix, still deferred.
- **Orphaned worktree dir** `.claude/worktrees/sp5-deletion-sweep` — git-pruned (deregistered) but the directory survived a Windows file-lock on removal. Safe to `rm -rf` manually; branch already deleted.

## Gotchas / discipline (carried forward)
- **Worktree + editable-package trap:** `pip install -e` packages (coulson/general_beckman/…) resolve to the **main** checkout, not your worktree. Running pytest in a worktree tests the OLD package code. Fix: `PYTHONPATH=".;$(printf '%s;' packages/*/src)" .venv/Scripts/python -m pytest …` to force worktree sources ahead of the editable installs. (After merge-to-main, the main checkout's editable packages reflect the merge, so this hack is only needed pre-merge.)
- **conftest collision:** never put `tests/` and `packages/*/tests/` in the SAME pytest invocation — pytest registers two `tests.conftest` and aborts. Separate invocations.
- `timeout` always; never concurrent pytest (SQLite WAL lock crash-loops live KutAI).
- **Re-grep every symbol before deleting** (`feedback_audit_call_sites`); the predecessor handoff's line numbers + its claim that `single_shot`/`constrained_emit` lived in `packages/coulson` were stale (the live constrained_emit was `src/workflows/engine/`).
