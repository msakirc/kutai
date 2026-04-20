# Task 13 Bug-Fix Handoff — 2026-04-20

Session-end handoff for whoever picks this up next. Task 13 (Beckman simplification + orchestrator main-loop rewrite) merged to main as `cac193c` earlier this session. A follow-up deep code review found **five critical runtime bugs** in the merged code; all five are fixed in the bug-fix commit that follows this document. This handoff explains what was found, what was fixed, what's still outstanding, and what the next session should do.

## TL;DR

- Task 13 is merged and the core architecture is correct.
- Five bugs that would break production at first run (ModuleNotFoundError-style, silent "unknown action" failures for every mechanical task spawned by Beckman) are **fixed** in the post-merge bug-fix commit.
- 33/33 targeted beckman tests green. No push yet — user is the gatekeeper.
- A small set of non-blocker follow-ups remains (listed below).
- **Before restarting KutAI**, `pip install -r requirements.txt` (Task 5 added `-e ./packages/workflow_engine`).

## Current state

- Branch: `main` (Task 13 merged at `cac193c`, bug-fix commit after that)
- Worktree: `C:/Users/sakir/Dropbox/Workspaces/kutay` (the task13 worktree was cleaned up)
- Production DB was leak-cleaned earlier in the session; no garbage rows remain
- KutAI is stopped (user confirmed). Yaşar Usta processes from earlier are stale
- No llama-server running

## What the bug-fix commit addresses

### BUG 1-4: malformed mechanical-task `context` shape (7 call sites)

**Symptom:** Every mechanical task Beckman spawned (`MissionAdvance → workflow_advance`, `RequestClarification → salako.clarify`, DLQ notify, cron-fired todo_reminder / daily_digest / api_discovery, nerd_herd health alert, sweep escalation) would hit dispatch, salako.run would return `Action(status="failed", error="unknown mechanical action: None")`, and the task row would land in `failed` — then retry-loop through the queue until DLQ. Missions would fail to progress. Clarifications would never reach Telegram. DLQ alerts would never fire.

**Root cause:** The workflow engine's canonical shape (verified by `tests/workflows/test_mechanical_step_materializes_with_executor_tag.py`) is:

```python
context = {
    "executor": "mechanical",
    "payload": {"action": "<name>", **kwargs},
}
```

The orchestrator's `_dispatch` does `t["payload"] = ctx["payload"]` and salako routes on `payload["action"]`. Task 2/5/7 subagents stored the payload fields flat at the top of `context` (no nested `payload` dict), and sometimes used `executor=<name>` instead of `action=<name>`. Both mistakes combined to make every spawned mechanical task unrouteable.

**Fix (1 commit):**

- New helper `apply._mechanical_context(action, **payload_fields)` produces the canonical shape. Public from apply.py for reuse.
- `apply._apply_clarify` — fixed to use the helper.
- `apply._apply_mission_advance` — fixed.
- `apply._dlq_write` notify — fixed.
- `cron._insert_scheduled_task` — fixed (used for todo_reminder / daily_digest / api_discovery).
- `cron._nerd_herd_health_alert` — fixed.
- `sweep._notify` — fixed (5 call sites go through this helper, all fixed in one edit).

### BUG 5: `plan_mission` NameError at runtime

`src/core/orchestrator.py:160,162` calls `get_mission_workspace_relative(mission_id)` but the import was removed during Task 7's "clean up unused imports" pass. Any new mission would raise `NameError: name 'get_mission_workspace_relative' is not defined`.

**Fix:** restore the import. Also removed two genuinely dead imports (`set_context`, `MAX_CONTEXT_CHAIN_LENGTH`) that moved to `context_injection.py` during the Task 8 trim.

### Regression guard

New test file `tests/test_mechanical_context_shape.py`:

1. Asserts `_mechanical_context` produces the canonical shape.
2. Simulates the orchestrator's dispatch-unpack logic and verifies `task["payload"]` has the action key.
3. **Negative test** that documents the old broken shape and why it fails (so nobody re-introduces it).

Updated `tests/test_beckman_apply.py` and `tests/test_beckman_on_task_finished.py` assertions to check the new canonical shape (nested `ctx["payload"]["action"]`) instead of the flat shape they accidentally codified.

## Test status

After bug-fix: **33/33 targeted tests green**, including:

```
tests/test_beckman_cron_seed.py            2 tests
tests/test_beckman_rewrite.py              6 tests
tests/test_beckman_retry.py                5 tests
tests/test_beckman_apply.py                4 tests  (2 updated)
tests/test_beckman_next_task.py            3 tests
tests/test_beckman_on_task_finished.py     4 tests  (2 updated)
tests/test_salako_workflow_advance.py      2 tests
tests/test_workflow_engine_advance.py      4 tests
tests/test_mechanical_context_shape.py     3 tests  (NEW)
```

Mandatory pytest invocation (Windows PYTHONPATH separator is `;`):

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
export PATH="$PATH:/c/Users/sakir/ai/util"
DB_PATH="$PWD/worktree_test.db" \
  timeout 60 .venv/Scripts/python.exe -m pytest \
    tests/test_beckman_cron_seed.py tests/test_beckman_rewrite.py \
    tests/test_beckman_retry.py tests/test_beckman_apply.py \
    tests/test_beckman_next_task.py tests/test_beckman_on_task_finished.py \
    tests/test_salako_workflow_advance.py tests/test_workflow_engine_advance.py \
    tests/test_mechanical_context_shape.py -v
rm -f worktree_test.db
```

**Do not run the full `pytest tests/`** — test collection imports `src.app.*` modules that eagerly instantiate the local_model_manager singleton and spawn llama-server. Kutay is stopped; don't accidentally spawn llama-servers during review.

## Smoke test before user restarts KutAI

1. `pip install -r requirements.txt` (picks up the new `workflow_engine` package; subagent's editable install during Task 5 was already fixed up to point at production during cleanup).
2. Manual smoke per the spec success criteria:
   - `/task echo hello` — task spawns, dispatches, completes, Telegram reply
   - `/shop coffee beans` — shopping mission runs through `MissionAdvance` spawns `workflow_advance` tasks; scraping runs
   - A mission phase that emits a clarification — should produce a `salako.clarify` task that sends the Telegram question
   - DLQ retry of an already-failed task — should generate a DLQ notify message in Telegram
   - Wait ~5 minutes after startup and check `scheduled_tasks` table: `beckman_sweep` and `nerd_herd_health_alert` markers should have `next_run` advanced once

## Remaining follow-ups (not blockers)

Carried over from the Task 13 final review, mostly unchanged:

1. **Salako executors for cron markers** — `todo_reminder`, `daily_digest`, `api_discovery`. The cron rows now correctly dispatch with the right shape, but the executor implementations don't exist in `packages/salako/`. They'll fall through to salako's "unknown action" branch until built. Each deserves a proper brainstorm (what's in the digest? how does LLM-driven todo reminder work?). `_parse_todo_suggestions` parser is recoverable from `86dea8c^:packages/general_beckman/.../scheduled_jobs.py`.

2. **Advance_recipe extraction** — `packages/workflow_engine/src/workflow_engine/advance.py` catches `ImportError` on `src.workflows.engine.recipe.advance_recipe`. The Task 6 subagent read `_handle_complete` end-to-end and found zero phase-progression logic there — `post_execute_workflow_step` in `src/workflows/engine/hooks.py` handles phase completion via `_check_phase_completion`. So missions should progress correctly without `advance_recipe`. **Verify this empirically during smoke-test**: start a mission, watch phase transitions happen. If phases never advance to N+1, then extraction is needed.

3. **Orchestrator `plan_mission` line 160-162** — the fix restored the import, but the function also relies on `create_mission_branch(..., path=get_mission_workspace_relative(mission_id))`. Confirm `create_mission_branch`'s signature still accepts `path=`; grep if you're unsure. (I did not verify this during the fix — only the import.)

4. **Test `test_silent_task_clarify_becomes_failure`** — passes, but the check on `worker_attempts` assumes the `add_task` path left `worker_attempts` unset. Verify this is still true with current `add_task` if touching this area.

5. **Seed race** — `cron_seed` now has an `asyncio.Lock`. No change needed.

6. **`_parse_todo_suggestions` test** (removed from `tests/test_todo.py`) — resurrect from git history if/when the `todo_reminder` salako executor is built.

## Key files for the next session

| File | Why |
|---|---|
| `docs/superpowers/specs/2026-04-19-beckman-simplification-design.md` | The spec that governs the architecture |
| `docs/superpowers/plans/2026-04-19-beckman-task13.md` | The task-by-task plan Task 13 executed |
| `docs/superpowers/plans/2026-04-19-phase2b-task13-handoff.md` | The original Phase 2b → Task 13 handoff |
| `packages/general_beckman/README.md` | Post-Task-13 package README (bilingual, follows dallama/nerd_herd convention) |
| `packages/general_beckman/src/general_beckman/apply.py` | `_mechanical_context` lives here; used by cron + sweep |
| `tests/test_mechanical_context_shape.py` | Regression guard; read first if a mechanical task misbehaves |
| Memory: `project_task13_shipped_20260420.md` | Overall Task 13 retrospective |

## What NOT to do in the next session

- **Don't run the full pytest suite.** It spawns llama-server via `src.app.*` imports. Use only the targeted list above unless you've explicitly stubbed out or killed the manager singleton.
- **Don't `pip install` in a worktree.** Shared venv with production; rewrites the editable install pointer.
- **Don't monkeypatch DB_PATH via setenv** in new tests — `DB_PATH` is read at `src.infra.db` import time. Use `monkeypatch.setattr(_db_mod, "DB_PATH", ...)` + close `_db_connection` + reset `cron_seed._seeded` and `paused_patterns._patterns`.
- **Don't touch production DB** from tests. Even with proper monkeypatching, use `DB_PATH=<absolute test path>` env var as belt-and-suspenders.

## Session handoff summary

This session: Task 13 executed via subagents → merged → deep review → five bugs caught → all fixed → regression guard added → 33/33 tests green → this handoff written.

Next session: verify the bug-fix commit with the user, smoke-test KutAI, and optionally pick up one of the salako reminder executor builds (each is its own brainstorm + spec + plan cycle).
