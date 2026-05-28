# CPS SP2 Shipped — Handoff

**Date:** 2026-05-29
**Branch:** merged to `main` (commit follows merge of `feat/cps-sp2-edge-group`)
**Author:** SP2 session (subagent-driven, two review passes + fix-up pass)
**Parent specs / plans:**
- `docs/superpowers/specs/2026-05-27-cps-migration-design.md` (rev2 umbrella)
- `docs/superpowers/specs/2026-05-27-cps-migration-call-site-inventory.md`
- `docs/superpowers/specs/2026-05-27-cps-sp2-edge-group-design.md` (SP2 spec)
- `docs/superpowers/plans/2026-05-27-cps-sp2-edge-group.md` (SP2 plan)
**Companion handoff:** `docs/handoff/2026-05-28-sp3-kickoff.md` (parallel session, SP3 scope)

---

## What shipped (13 commits + merge)

SP2 migrated **4 of 6** edge-group `await_inline=True` sites onto the SP1 durable continuation substrate. The substrate proper (`continuations.py` fire logic, `__init__.py`, `db.py`) was **not touched** by SP2 beyond extending `_HANDLER_MODULES`. `await_inline` itself stays intact through SP5.

### Migrated (4 sites)

| Site | File | Resume / on_error |
|------|------|-------------------|
| #1 telegram | `src/app/telegram_bot.py` (`_handle_casual`, `_classify_user_message`, `cmd_mission`) | `telegram.casual_reply_resume`/`_err`, `telegram.message_route_resume`/`_err` |
| #3 interview | `src/app/interview.py` (`summarize_interview`) | `interview.summary_persist_resume`/`_err` |
| #4 meetings | `src/app/meetings.py` (`_call_llm_meeting_brief`) | `meetings.brief_persist_resume`/`_err` |
| #5 faq_regen | `src/app/jobs/faq_regen.py` (`_llm_cluster_draft`) | `faq_regen.draft_persist_resume`/`_err` |

Each migrated module exposes `register_continuations()` at import time and is listed in `_HANDLER_MODULES` (now a mutable `list[str]` per parallel SP1.1 hardening; SP2 take-both kept that shape and appended the 4 SP2 entries).

### Deferred to SP5+ (2 sites — explicit carve-outs)

| Site | File:line | Why deferred |
|------|-----------|--------------|
| #2 task_classifier | `src/core/task_classifier.py:287` | `classify_task` is consumed synchronously by `add_task` → CPS-ing it inverts the task-admission contract. Out of SP2 scope. |
| #6 investor_bullets | `src/app/jobs/investor_bullets.py:211` | Path is **dead in prod** (per `2026-05-17-z7-unwired-features.md`: `missions.product_id` is NULL → fetchers `{}` → hypothesis unreachable). CPS cost not justified until upstream `metric_emit` producer ships. |

Both have `# SP5-DEFERRED:` comments directly above the `await_inline=True` line documenting the reason. SP5 must carve them out of the deletion guard.

### Latent bug fixed (incidental)

`src/app/interview.py:265` (pre-SP2) read `getattr(task_result, "output", None)`. `TaskResult` has only `status`/`result`/`error` — `.output` doesn't exist. Summary, quotes, insights, action_items were **always empty** in production. The CPS migration's resume reads `result["result"]["content"]` (the documented shape) and is tested by `test_summary_resume_writes_db_from_result_content` driving the parsed payload all the way into the DB row.

### Three post-review fix-ups (round 2)

The first review pass caught three behaviors the original SP2 commits dropped from the LLM-classifier path. The second review pass cleared the fixes.

1. **followup / clarification_response parent-task linkage** — `_route_classified_message` now has explicit branches calling `_find_followup_parent(chat_id, text)` / `find_followup_context(chat_id, text)` and threading `parent_task_id` + `recent_conversation` into the new task. Tests: `test_followup_route_sets_parent_task_id`, `test_clarification_response_route_sets_parent_task_id`.
2. **Z0 cost-ceiling prompt for plain `/mission`** — `_pending_mission` resume's no-workflow branch now: `add_mission` → `ensure_mission_topic` (try/except) → `plan_mission` (try/except) → `user_last_task_id[chat_id] = None` → arms `tg._pending_action[chat_id] = {"kind": "z0_ceiling", "mission_id"}` → sends ceiling prompt via `_send_telegram_via_resume`. ARM-before-PROMPT ordering is **race-safer than pre-SP2** (which sent prompt first). Test: `test_mission_plain_sets_z0_ceiling_pending_action`.
3. **Rich `status_query` / `progress_inquiry` reply** — extracted `_build_status_query_response(text) -> list[(text, parse_mode)] | None` shared by `_handle_status_query` (live `Update`) and `_handle_status_query_chat(chat_id, text)` (CPS path). Zero duplicated DB query logic. Test: `test_status_query_resume_invokes_status_handler`.

### Test count

Post-merge gate (`tests/beckman/test_continuations*.py` + `tests/app/test_cps_sp2_*.py`): **63/63 pass in 37s.**

---

## Interaction with parallel SP1.1 hardening

Mid-session, a parallel branch shipped 4 SP1.1 hardening commits to `main` (all useful, all preserved through the merge):

- `5b868b4d` fix(beckman): fire continuation on real DB terminal + agent-result shape
- `bc903040` fix(run): save periodic reconcile task handle so asyncio doesn't GC it
- `a3738f11` fix(beckman): handler-import warning + pre-check before CAS claim
- `1c2b8f44` fix(beckman): legacy-shim probe + raw envelope reconcile test + telemetry

Merge conflict point: `_HANDLER_MODULES` (parallel session converted tuple → `list[str]` and added `register_continuations_module(name)` dynamic-add helper; SP2 extended the tuple with 4 strings). Resolved take-both: kept the list-shape + dynamic helper, appended SP2 entries. Substrate invariants documented in the SP3 kickoff still hold.

**One implication:** the SP3 author's substrate invariant "Handler-presence pre-check is on by default" (from SP1.1) is now load-bearing for SP2 too — if a future migration forgets `_HANDLER_MODULES` registration, `fire_for_task` won't claim, the row stays pending, and reconcile picks it up. SP2 verified all 4 new modules are registered (acceptance grep + integration smoke).

---

## Open items / follow-ups

### Carried forward to SP3+ (not blocking this merge)

- **SP3 sites** (per `docs/handoff/2026-05-28-sp3-kickoff.md`): `grading`, `code_review`, `hooks._llm_summarize`, `dispatcher.request` shim. This is the migration that actually kills the DLQ deadlock — SP2 only validated the substrate against edge traffic.
- **SP4 sites** (tools + mechanicals): `vision.py`, all `mr_roboto/{crisis_draft_holding,demo_storyboard,incident_draft_update,press_kit_assemble,reviews_{classify,draft_reply}}.py`, `yalayut/discovery/synthesize.py`, both `posthook_handlers/`.
- **SP5**: delete `await_inline` / `resolve_inline` / `_inline_waiters` / `INLINE_TIMEOUT`. **Must explicitly carve out site #2 (task_classifier) AND site #6 (investor_bullets)** — both have `# SP5-DEFERRED:` comments documenting why.

### Low-priority polish in shipped code

- **`_handle_status_query_chat` "no matches" fallback** (`src/app/telegram_bot.py:~8341`): CPS path sends a brief `"📊 No matching tasks found. Use /tasks or /missions to see status."` instead of dumping recent progress notes (which pre-SP2 did via `cmd_progress(update, context)`). If parity is desired, extract a `_format_recent_progress_notes()` helper from `cmd_progress` and call it from `_handle_status_query_chat`.
- **`_pending_action` snapshot-and-compare** (spec §B2): SP2 spec described capturing a snapshot at enqueue time and comparing against current state before mutating. SP2 ships the simpler version (unconditional write of `user_last_task_id[chat_id]` + `_pending_action`) — matches today's "idempotence by accident". If a user actually hits the race in production (sends message B mutating `_pending_action` before message A's resume fires), pick this up.

### Site #6 prerequisite (independent of CPS work)

`investor_bullets` anomaly hypothesis can't migrate until the upstream `metric_emit` producer ships AND `missions.product_id` is wired. Per `2026-05-17-z7-unwired-features.md`: "**Decisions needed: who/what sets `missions.product_id`; who emits the metric `growth_events`. Until then A9 produces empty founder cards.**" Same blocker still open.

### Pre-existing test failures (NOT SP2)

Reviewer flagged these for separate triage. Verified failing on `main` pre-SP2:

- `tests/z7/test_a8_faq_flywheel.py::test_faq_regen_approve_appends_to_faq_file` — `sqlite3.OperationalError: no such table: missions`. Fixture hygiene bug (uses `tmp_path, monkeypatch` but never wires them to the DB). Unrelated to faq_regen CPS migration.
- `tests/core/test_dispatcher_in_flight.py::test_dispatcher_calls_begin_end_for_cloud`
- `tests/core/test_dispatcher_in_flight.py::test_dispatcher_ends_call_even_on_exception`
- `tests/core/test_dispatcher_records_swap.py::test_dispatcher_records_swap_after_swap`

The three `tests/core/test_dispatcher_*` failures may be related to ongoing dispatcher work (parallel session's SP1.1 touched it). Worth a 5-min triage in the next session — they were failing on main before SP2 work, but the root cause might already be fixed by a later SP1.1 commit.

---

## Substrate purity guarantee

Verified by the second reviewer pre-merge:

- `git diff <pre-SP2>..HEAD -- packages/general_beckman/src/general_beckman/__init__.py` → SP2 contributes **zero lines** (parallel SP1.1 owns all changes there).
- `git diff <pre-SP2>..HEAD -- src/infra/db.py` → SP2 contributes **zero lines**.
- `git diff <pre-SP2>..HEAD -- packages/general_beckman/src/general_beckman/continuations.py` → SP2 contributes only the 4 new entries in `_HANDLER_MODULES` (parallel SP1.1 contributes the list-shape + `register_continuations_module` + fire-logic + reconcile work).

The substrate's contract surface (the SP3 kickoff's "Substrate invariants SP3 MUST honor" section) is therefore the **single source of truth** for SP3 design — SP2 didn't move any of it.

---

## Commands cheatsheet for the next session

```bash
# Verify SP2 + substrate green
timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_continuations.py tests/beckman/test_continuations_durable.py tests/app/test_cps_sp2_*.py -q

# Inspect registered handlers at runtime
.venv/Scripts/python -c "from general_beckman.continuations import register_startup_handlers, _HANDLERS; register_startup_handlers(); print(sorted(_HANDLERS.keys()))"

# Find remaining await_inline=True sites (SP3 + SP4 targets, plus SP5-deferred carve-outs)
rtk grep "await_inline=True" --type py src packages

# Check no SP2 module forgot _HANDLER_MODULES
.venv/Scripts/python -c "from general_beckman.continuations import _HANDLER_MODULES; print(_HANDLER_MODULES)"
```

---

## TL;DR for the next session

- **SP1+SP1.1+SP2 are shipped on `main`.** Substrate is hardened + production-validated against edge traffic.
- **SP3 is the next sub-project** — see `docs/handoff/2026-05-28-sp3-kickoff.md`. It's the migration that actually closes the DLQ deadlock (sites #7, #8, #9, #10 in the inventory).
- **No SP2 work remains to do.** The 2 SP5-deferred sites (#2, #6) are documented; the 3 round-2 fix commits are tested; the 1 low-priority polish (`_format_recent_progress_notes`) is optional.
- **Parallel-session-merge pattern worked** here (no double work, no lost commits) but only because the conflict was a small enum-style file. Bigger overlap would benefit from coordination before parallel SPs.
