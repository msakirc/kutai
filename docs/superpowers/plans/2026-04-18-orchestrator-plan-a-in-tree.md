# Plan A — Finish In-Tree Untangle of `process_task`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-04-18
**Base branch:** `main` (Phase 1 landed at commit 95404d6)
**Predecessor plan:** `docs/superpowers/plans/2026-04-17-orchestrator-phase1-in-tree.md`
**Handoff doc:** `docs/superpowers/plans/2026-04-17-phase2-handoff.md`

**Goal:** Unblock the two hard parts deferred from Phase 1 (D1 — `result_router` wire-up; D2 — `process_task` split) so `process_task` drops from 945 lines to under 50. This plan stays in-tree; no package extraction.

**Scope:**
1. Extract `_handle_exhausted` and `_handle_failed` as verbatim moves from inline code in `process_task`.
2. Extend `src/core/result_router.py` Action types to carry the raw agent-result dict so existing `_handle_*(task, result)` signatures are preserved (Option 1).
3. Port per-branch guard code (workflow post-hooks, quality retries, bonus attempts, schema validation, DLQ cascade) into typed helpers in a new `src/core/result_guards.py`.
4. Wire `route_result` + guards + handlers into `process_task` via two new slices: `_dispatch` (agent/pipeline invocation + timeout/partial-result recovery) and `_record` (route → run guards → call handler).
5. Shrink `process_task` to < 50 lines.

**Option choice — Option 1 committed.** Rationale: (a) existing `_handle_complete/_subtasks/_clarification/_review(task, result)` signatures stay identical, so the verbatim diff shrinks from ~200 lines of signature-threading churn to zero; (b) guard code is the complex part — it gets a clean home in `result_guards.py` regardless of handler shape — so Option 2's extra indirection buys nothing in Phase 1; (c) Phase 2b will invert everything to Decision emissions anyway, making Option 2's scaffolding throwaway.

**Invariants to preserve throughout:**
- Every existing test continues to pass. Run `timeout 300 python -m pytest tests/` at the end of each task that touches `process_task`.
- No behavioral change visible from Telegram (task #/emoji, retry cadence, DLQ messages).
- `/restart` still works.

---

## File Structure

Files created:
- `src/core/result_guards.py` — typed helpers for pre-handler guard code
- `tests/test_result_guards.py` — unit tests for the guards

Files modified:
- `src/core/orchestrator.py` — extract `_handle_exhausted`, `_handle_failed`, `_dispatch`, `_record`; shrink `process_task`
- `src/core/result_router.py` — extend Action types to carry raw result dict
- `tests/test_result_router.py` — new assertions for the raw-dict field

---

## Task 1: Extract `_handle_exhausted` (Verbatim Move)

**Files:** Modify `src/core/orchestrator.py`

**Goal:** Turn the ~90-line inline `elif status == "exhausted":` block into an `async def _handle_exhausted(self, task, result)` method. No logic changes — pure move.

- [ ] **Step 1: Identify boundaries.** `grep -n 'elif status == "exhausted"\|elif status == "failed"' src/core/orchestrator.py`. Read the exhausted block.

- [ ] **Step 2: Write a behavioral regression test FIRST** in `tests/test_orchestrator_routing.py` covering budget/guards/tool_failures reasons. Expected FAIL with AttributeError.

- [ ] **Step 3: Move the block verbatim.** Cut the full `elif status == "exhausted":` body including inner branches. Paste as a method after `_handle_review`. Replace the old inline block with `await self._handle_exhausted(task, result)`.

- [ ] **Step 4: Run regression + full routing suite.** `timeout 120 python -m pytest tests/test_orchestrator_routing.py tests/test_lifecycle_fixes.py tests/test_exhaustion.py -v`

- [ ] **Step 5: Commit.** `refactor(core): extract _handle_exhausted from process_task`

---

## Task 2: Extract `_handle_failed` (Verbatim Move)

**Files:** Modify `src/core/orchestrator.py`

**Goal:** Same pattern as Task 1 for the ~55-line inline `elif status == "failed":` block.

- [ ] **Step 1: Write regression test** for terminal→DLQ and retryable→pending cases. Expect FAIL.

- [ ] **Step 2: Move the block verbatim.**

- [ ] **Step 3: Run tests.** `timeout 120 python -m pytest tests/test_orchestrator_routing.py tests/test_lifecycle_fixes.py tests/test_retry.py -v`

- [ ] **Step 4: Commit.** `refactor(core): extract _handle_failed from process_task`

---

## Task 3: Extend `result_router` Action Types with Raw Dict (Option 1)

**Files:** Modify `src/core/result_router.py`, `tests/test_result_router.py`

**Goal:** Add a `raw: dict` field on every Action so `_record` can call `_handle_*(task, action.raw)` without reconstruction.

- [ ] **Step 1: Update tests first.** For each Action type, add `test_*_action_carries_raw_result` verifying `actions[0].raw == agent_result`.

- [ ] **Step 2: Add `raw: dict = field(default_factory=dict)` to every Action dataclass** (`Complete`, `SpawnSubtasks`, `RequestClarification`, `RequestReview`, `Exhausted`, `Failed`). In `route_result`, pass `raw=agent_result or {}` into each constructor.

- [ ] **Step 3: Run router tests.** All PASS.

- [ ] **Step 4: Commit.** `refactor(result_router): add raw dict to every Action type`

---

## Task 4: Create `src/core/result_guards.py`

**Files:** Create `src/core/result_guards.py`, `tests/test_result_guards.py`

**Goal:** Move the 30–80 lines of guard code that currently sit between "result received" and `_handle_*` into typed async helpers. Each guard returns `None` (fall through) or a terminal `GuardHandled(reason)` meaning "process_task should return now".

**Identified guards:**

| Guard function | Trigger |
|---|---|
| `guard_workflow_step_post_hook` | `status == "completed"` AND `is_workflow_step(task_ctx)` |
| `guard_workflow_clarification_validation` | post-hook flipped status to `needs_clarification` |
| `guard_quality_retry_after_post_hook` | post-hook flipped status to `failed` |
| `guard_pipeline_artifacts` | pipeline + workflow step |
| `guard_subtasks_blocked_for_workflow` | `needs_subtasks` + workflow step |
| `guard_clarification_suppression` | `needs_clarification` + (silent OR `may_need_clarification=False` OR existing history) |
| `guard_ungraded_post_hook` | `status == "ungraded"` |

- [ ] **Step 1: Sketch the API** with `@dataclass(frozen=True) class GuardHandled: reason: str` and `GuardOutcome = GuardHandled | None`.

- [ ] **Step 2: Write failing tests for each guard** — happy (None), terminal (GuardHandled + DB side effect), retry path.

- [ ] **Step 3: Implement each guard by verbatim-porting** the corresponding inline block. Keep `self` as first arg; guards use `self.telegram.*`, `self._validate_clarification`, etc.

- [ ] **Step 4: Run the guards suite + existing shadowed suites.** All PASS (inline code still in place).

- [ ] **Step 5: Commit.** `feat(core): add result_guards module for pre-handler guard code`

---

## Task 5: Wire `route_result` + Guards + Handlers into `process_task`

**Files:** Modify `src/core/orchestrator.py`

**Goal:** Replace the long `if/elif` status chain in `process_task` with `route_result → guards → _dispatch_action`.

- [ ] **Step 1: Add `_dispatch_action(action, task)`** — isinstance-dispatches to `_handle_*(task, action.raw)`.

- [ ] **Step 2: Add `_run_guards_for(action, task, task_ctx, result) -> bool`** — runs the guards that apply to this action type; returns True if caller must return.

- [ ] **Step 3: Replace the if/elif chain.** Keep `status == "ungraded"` and `status == "pending"` inline for Phase 1 (not router Actions). For routed statuses: `actions = route_result(task, result); for action in actions: if await self._run_guards_for(...): return; await self._dispatch_action(action, task)`

- [ ] **Step 4: Run full suite.** `timeout 300 python -m pytest tests/ -v`. Diff against verbatim block if anything fails.

- [ ] **Step 5: Manual Telegram smoke.** `/task`, `/shop`, workflow, clarification, human-gate.

- [ ] **Step 6: Commit.** `refactor(core): wire result_router + result_guards into process_task`

---

## Task 6: Extract `_dispatch` (Agent/Pipeline Invocation + Timeout)

**Files:** Modify `src/core/orchestrator.py`

**Goal:** Pull the ~260 lines covering "agent selection → timeout wrapper → partial-result recovery → retry/DLQ on timeout" into `async def _dispatch(self, task, agent_type, timeout_seconds) -> dict | None`. Returns `None` if timeout handler fully consumed the task.

- [ ] **Step 1: Define the seam.**

- [ ] **Step 2: Move the block.** Replace `return` with `return None` in timeout path; `return result` in success path.

- [ ] **Step 3: Wire.** `result = await self._dispatch(task, agent_type, timeout_seconds); if result is None: return`

- [ ] **Step 4: Test.** Full suite + focus on `test_timeout_recovery.py`, `test_lifecycle_fixes.py`.

- [ ] **Step 5: Commit.** `refactor(core): extract _dispatch from process_task`

---

## Task 7: Extract `_record` (Route → Guards → Handler)

**Files:** Modify `src/core/orchestrator.py`

**Goal:** The glue code from Task 5 moves into `async def _record(self, task, task_ctx, result) -> None`. `process_task` just calls `_prepare` → `_dispatch` → `_record` plus two outer `except` handlers.

Target `process_task` shape:

```python
async def process_task(self, task: dict):
    task_id = task["id"]
    title = task["title"]
    agent_type = task.get("agent_type", "executor")
    logger.info("task received", task_id=task_id, title=title, agent_type=agent_type)

    task_ctx = {}
    result = None
    try:
        prepared = await self._prepare(task)
        if prepared is None:
            return
        task, agent_type, timeout_seconds = prepared
        task_ctx = parse_context(task)

        result = await self._dispatch(task, agent_type, timeout_seconds)
        if result is None:
            return

        await self._record(task, task_ctx, result)

    except ModelCallFailed as mcf:
        await self._handle_availability_failure(task, task_ctx, mcf)
    except Exception as e:
        await self._handle_unexpected_failure(task, task_ctx, result, e)
```

- [ ] **Step 1: Define the seam** including `ungraded`/`pending` inline branches and `release_task_locks` cleanup.

- [ ] **Step 2: Move the code; shrink `process_task`.** Extract outer exception handlers into `_handle_availability_failure` and `_handle_unexpected_failure` (verbatim moves).

- [ ] **Step 3: Verify line count.** `awk '/async def process_task/,/^    async def |^    def /' src/core/orchestrator.py | wc -l` → < 50.

- [ ] **Step 4: Run full suite + smoke test.**

- [ ] **Step 5: Commit.** `refactor(core): split process_task into _prepare/_dispatch/_record`

---

## Task 8: Docs + Line Count + Final Commit

- [ ] **Step 1:** Full verbose suite.
- [ ] **Step 2:** `wc -l src/core/orchestrator.py src/core/result_router.py src/core/result_guards.py`. Orchestrator target: < 2,300. `process_task` < 50.
- [ ] **Step 3:** Update `docs/architecture-modularization.md` with Plan A outcome.
- [ ] **Step 4:** Append memory note to `MEMORY.md`.
- [ ] **Step 5:** Final commit.

---

## Plan A Definition of Done

- [ ] `process_task` < 50 lines
- [ ] `src/core/result_guards.py` exists with tests for every guard
- [ ] Every Action type in `result_router.py` carries `raw: dict`
- [ ] `_handle_exhausted` and `_handle_failed` are methods, not inline code
- [ ] All existing behavioral tests pass
- [ ] Manual Telegram smoke matches pre-refactor behavior
- [ ] Architecture doc updated; memory note saved

## Out of Scope

- Moving `_handle_*` methods out of `Orchestrator` (Phase 2b)
- Extracting `result_guards` to a package (Phase 2b)
- Replacing `ungraded` inline handling with a router Action type (defer)
- Inverting `self.telegram.*` coupling inside guards (Phase 2b)
