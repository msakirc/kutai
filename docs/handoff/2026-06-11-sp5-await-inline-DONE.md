# Handoff — SP5 CLOSED: `await_inline` deleted

**Date:** 2026-06-11
**From:** the session that finished SP5.
**Predecessor:** `docs/handoff/2026-06-10-sp5-await-inline-remaining-handoff.md` (its plan premise was partly wrong — see below).
**Plan executed:** `docs/superpowers/plans/2026-06-11-sp5-await-inline-finale.md`.

---

## Outcome — MERGED to main (NOT pushed, restart-gated)

`general_beckman.enqueue`'s blocking `await_inline=True` primitive is **deleted**, along with all its inline-waiter machinery (`resolve_inline`, `_inline_waiters`, `INLINE_TIMEOUT`, the `TaskResult` dataclass, the terminal-hook resolve block, the mutual-exclusion `ValueError`). `enqueue` is now fire-and-continue → `int | None`. SP5 is closed.

Merge commit **`ee538bb2`** on `main` (6 feature commits `0d4fec83`..`25f68c19`). **Not pushed** — live KutAI runs old code until a founder `/restart`.

## The handoff premise was WRONG (corrected)

The predecessor handoff scoped **carve-out 1 (`task_classifier`)** as a "task-admission redesign" — its own spec — because a code comment claimed `add_task` consumes `classify_task`'s result synchronously. **It does not.** Verified by grep:
- `classify_task` / `_classify_with_llm` / `_enqueue_inline_classifier` had **zero live callers** (only tests + `_extract_json`, which telegram_bot imports). `add_task` (db.py) is dedup+insert and takes `agent_type` as a param.
- Live classification is `telegram_bot._classify_user_message` (already CPS) and `/task` admits typed tasks directly.

So carve-out 1 was **dead-but-kept** code, not a redesign. **Founder ruling (2026-06-11): CPS-migrate BOTH carve-outs** (keep the intelligence, don't delete) — accepting that the migrated classifier is "a CPS shell nobody invokes."

## What changed

**Carve-out 1 — `src/core/task_classifier.py`:**
- `classify_task(title, desc, *, on_complete="task_classifier.classify.resume", cont_state=None) -> int | None` — CPS kickoff (mirrors `_classify_user_message`). No sync return.
- Extracted the field-mapping intelligence into pure `parse_classification(result, *, title, description) -> TaskClassification` (sync, no LLM, unit-testable).
- `_classify_resume` continuation rebuilds the classification; `_on_classified` is the default (logging) consumer. `register_continuations()` at import.
- Deleted `_classify_with_llm` + `_enqueue_inline_classifier`. Kept `_extract_json`, `TaskClassification`, keyword fallback.

**Carve-out 2 — `src/app/jobs/investor_bullets.py`:**
- `run_investor_bullets` → kickoff: detect anomalies, then finalize immediately (no anomalies) or start a **sequential CPS chain** (`_enqueue_hypothesis_child` → `_hypothesis_resume`/`_resume_err` → `_advance_chain` → `_finalize_bullets`). State threaded via `cont_state` (≤5 capped anomalies), **no new table**.
- Deleted `_call_llm_anomaly_hypothesis`. `register_continuations()` at import.

**Substrate — `packages/general_beckman/`:**
- `_HANDLER_MODULES` += `src.core.task_classifier`, `src.app.jobs.investor_bullets` (restart-recovery).
- `__init__.py`: deleted the await_inline param + machinery; `enqueue -> int | None`. README EN+TR updated.
- `mr_roboto` conftest hang-guard (shrank `INLINE_TIMEOUT`) removed as obsolete; `alert_triage` dead `TaskResult` else-branch removed.
- Deadlock guards strengthened: `test_enqueue_has_no_await_inline_param` + `test_no_inline_waiter_machinery` (passing await_inline is now a `TypeError`).

## Verification

Green (worktree, PYTHONPATH-forced to worktree sources): task_classifier 14 + investor_bullets 35 + handler-modules 2 + deadlock guards 11; migrated classifier/e2e/shopping files collect clean.

Full `general_beckman` suite: **273 passed**, 2 failed — `test_admission_cache.py::{test_cache_skips_redundant_scan_when_state_unchanged, test_cache_invalidates_when_in_flight_changes}` with `no such table: tasks`. **Proven to be a worktree+PYTHONPATH-hack harness artifact, NOT a regression:**
- main (no hack): 3 passed. main + hack: 3 passed.
- worktree + hack with my code / HEAD `__init__` / HEAD `continuations`: all 2 failed.
- i.e. reverting my beckman changes in the worktree still failed → the dual sys.path (hack-prepended worktree `packages/*/src` + editable `.pth` main `packages/*/src`) splits the `nerd_herd.refresh_snapshot` mock for the only two tests that rely on the snapshot-reject path without a DB fixture.
- **Authoritative post-merge run on main (no hack) is the real check** — see the run referenced in the closing message.

## Test migrations (classify_task contract changed → int)

- `parse_classification` unit tests (`tests/core/test_parse_classification.py`) + CPS kickoff/resume tests (`test_classify_cps.py`).
- Mocked return-asserting tests → `parse_classification`; agent-execution e2e → deterministic keyword/pinned agent_type; pure-LLM-classification-quality tests on the now-uninvoked shell **deleted** with documented pointers (the prompt is unchanged; pick-rules covered by `test_task_classifier_picks.py`).
- investor_bullets: old `await_inline`/`TaskResult` tests → CPS resume/chain tests (`tests/z7/test_investor_bullets_cps.py` + rewrites in `test_a9_investor_bullets.py`).

## Residual (NOT blocking)

- `src/tools/vision.py` calls `husam.run()` directly (ruling-#1 non-compliant) — separate compliance fix, untouched.
- The pre-existing stale tests the predecessor handoff flagged (`test_mission_workflow_integration.py::TestLLMClassification` was actually rewritten here; `tests/integration/test_agent_basic.py` ReAct-iteration patch) — `test_agent_basic` left as-is (Runtime Phase A debt, unrelated).
- `src/core/llm_dispatcher.py:96` docstring mentions the old `await_inline` bridge historically — accurate, left as-is.

## NEXT

1. **Founder `/restart`** to make the deletion live (restart-gated).
2. Decide push vs local-only (this session left it merged-but-unpushed, matching the restart-gated convention; the SP5 request() session chose to push).
