# Phase 2b — General Beckman (Task Master Package) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-04-19
**Base branch:** `main`
**Design spec:** `docs/superpowers/specs/2026-04-18-phase2b-general-beckman-design.md` (source of truth)
**Predecessor plans:** Plan A (`2026-04-18-orchestrator-plan-a-in-tree.md`) + Plan B (`2026-04-18-phase2a-salako-package.md`) — both merged.

**Goal:** Extract task-queue logic from `src/core/orchestrator.py` (currently 2,569 lines) into a new package `packages/general_beckman/`. End state: `orchestrator.py` ≤ 250 lines (target 200, hard cap 300). Named after General Beckman from *Chuck* — the NSA commander who hands out missions. This package answers: **"what should we do next, and how many of it?"**

**Architecture (end state):**

```
Orchestrator.main_loop
  while running:
      while (task := await beckman.next_task()):
          asyncio.create_task(_dispatch(task))
      await beckman.tick()
      await asyncio.sleep(3)

_dispatch(task):
  if task["agent_type"] == "mechanical": result = await salako.run(task)
  else:                                  result = await llm_dispatcher.request(task)
  await beckman.on_task_finished(task["id"], result)
```

**Tech stack:** Python 3.10 async, setuptools editable install (matches `fatih_hoca` / `salako` / `nerd_herd`), existing pytest infra with `timeout` prefix.

**Invariants to preserve:**

- `timeout 120 pytest tests/` — baseline is 253 pre-existing failures (documented). Any new failure touching orchestrator / beckman / `result_*` / `watchdog` / `scheduled_jobs` must be investigated before merging.
- No behavioral change visible from Telegram (task #/emoji formatting, retry cadence, DLQ messages).
- `/restart` still works.
- Existing test suites pass through shims without edits.

**Never** run `pytest` without a `timeout` prefix. Targeted: `timeout 30 pytest tests/foo`. Full suite: `timeout 120 pytest tests/`. Integration only: `timeout 60 pytest tests/integration/`.

---

## File Structure

**Created:**

- `packages/general_beckman/pyproject.toml`
- `packages/general_beckman/README.md`
- `packages/general_beckman/src/general_beckman/__init__.py` — public API: `next_task`, `on_task_finished`, `tick`
- `packages/general_beckman/src/general_beckman/types.py` — `Task`, `AgentResult` aliases, `Lane` enum
- `packages/general_beckman/src/general_beckman/queue.py` — eligibility filters + task selection
- `packages/general_beckman/src/general_beckman/lookahead.py` — quota look-ahead (reinstated)
- `packages/general_beckman/src/general_beckman/lifecycle.py` — `on_task_finished` + all `_handle_*` methods
- `packages/general_beckman/src/general_beckman/result_router.py` — moved from `src/core/`
- `packages/general_beckman/src/general_beckman/result_guards.py` — moved from `src/core/`
- `packages/general_beckman/src/general_beckman/task_context.py` — moved from `src/core/`
- `packages/general_beckman/src/general_beckman/watchdog.py` — moved from `src/core/`
- `packages/general_beckman/src/general_beckman/scheduled_jobs.py` — moved from `src/app/`
- `packages/general_beckman/tests/test_init.py`
- `packages/general_beckman/tests/test_queue.py`
- `packages/general_beckman/tests/test_lookahead.py`
- `packages/general_beckman/tests/test_lifecycle.py`
- `packages/salako/src/salako/clarify.py`
- `packages/salako/src/salako/notify_user.py`
- `packages/salako/tests/test_clarify.py`
- `packages/salako/tests/test_notify_user.py`

**Modified (become shims):**

- `src/core/task_context.py` — re-export from `general_beckman.task_context`
- `src/core/result_router.py` — re-export
- `src/core/result_guards.py` — re-export
- `src/core/watchdog.py` — re-export
- `src/app/scheduled_jobs.py` — re-export
- `packages/salako/src/salako/__init__.py` — register new executors in `run()`
- `src/core/orchestrator.py` — massive shrink: delete `_handle_*`, delete gate call, main loop rewired

**Deleted:**

- `src/security/risk_assessor.py`
- `src/core/task_gates.py` (not shimmed — content is dead)
- `tests/test_human_gates.py`
- `tests/test_resilience_approvals.py`

---

## Task 1: Add `clarify` Mechanical Executor to Salako

**Goal:** Add the `clarify` executor so beckman's `_handle_clarification` can emit it as a regular mechanical task instead of calling Telegram directly.

**Files:**

- Create `packages/salako/src/salako/clarify.py`
- Create `packages/salako/tests/test_clarify.py`
- Modify `packages/salako/src/salako/__init__.py` (lines 1–40) — register `"clarify"` action in `run()`

**Steps:**

- [ ] **Step 1: Write failing tests first.** Create `packages/salako/tests/test_clarify.py`:
  ```python
  import pytest
  from unittest.mock import AsyncMock, patch

  @pytest.mark.asyncio
  async def test_clarify_sends_via_telegram_and_marks_pending():
      task = {
          "id": 42,
          "title": "Book a flight",
          "payload": {
              "action": "clarify",
              "question": "Which city?",
              "chat_id": 111,
          },
      }
      fake_tg = AsyncMock()
      with patch("salako.clarify.get_telegram", return_value=fake_tg), \
           patch("salako.clarify.update_task", new=AsyncMock()) as ut:
          from salako import run
          action = await run(task)
      assert action.status == "completed"
      fake_tg.request_clarification.assert_awaited_once_with(42, "Book a flight", "Which city?")
      ut.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_clarify_missing_question_fails():
      from salako import run
      action = await run({"id": 1, "payload": {"action": "clarify"}})
      assert action.status == "failed"
      assert "question" in (action.error or "")
  ```

- [ ] **Step 2: Implement** `packages/salako/src/salako/clarify.py`:
  ```python
  """Mechanical clarify executor: sends clarification prompt via Telegram."""
  from __future__ import annotations
  from src.infra.db import update_task
  from src.app.telegram_bot import get_telegram

  async def clarify(task: dict) -> dict:
      payload = task.get("payload") or {}
      question = payload.get("question")
      if not question:
          raise ValueError("clarify payload requires 'question'")
      tg = get_telegram()
      await tg.request_clarification(task["id"], task.get("title", ""), question)
      await update_task(task["id"], status="waiting_human")
      return {"sent": True, "question": question}
  ```
  If `get_telegram()` does not exist in `telegram_bot.py`, add a module-level singleton accessor. Verify via grep first: `grep -n 'def get_telegram\|_telegram_singleton' src/app/telegram_bot.py`. If missing, add:
  ```python
  _TG_INSTANCE: "TelegramInterface | None" = None
  def get_telegram() -> "TelegramInterface":
      if _TG_INSTANCE is None:
          raise RuntimeError("telegram not initialized")
      return _TG_INSTANCE
  def set_telegram(instance): global _TG_INSTANCE; _TG_INSTANCE = instance
  ```
  And call `set_telegram(self)` from `TelegramInterface.__init__`.

- [ ] **Step 3: Wire into `salako.run()`.** Edit `packages/salako/src/salako/__init__.py`, add between `git_commit` branch and the fallback:
  ```python
  if action == "clarify":
      from salako.clarify import clarify
      try:
          res = await clarify(task)
          return Action(status="completed", result=res)
      except Exception as e:
          return Action(status="failed", error=str(e))
  ```

- [ ] **Step 4: Verify.** `timeout 30 pytest packages/salako/tests/test_clarify.py -v` → all green.

- [ ] **Step 5: Commit.**
  ```
  git add packages/salako/src/salako/clarify.py packages/salako/src/salako/__init__.py packages/salako/tests/test_clarify.py src/app/telegram_bot.py
  git commit -m "feat(salako): add clarify mechanical executor"
  ```

---

## Task 2: Add `notify_user` Mechanical Executor to Salako

**Goal:** Plain-status Telegram send for meaningful notifications (mission complete, DLQ alerts, rejections). Parallels `clarify`.

**Files:**

- Create `packages/salako/src/salako/notify_user.py`
- Create `packages/salako/tests/test_notify_user.py`
- Modify `packages/salako/src/salako/__init__.py`

**Steps:**

- [ ] **Step 1: Write failing tests.** `packages/salako/tests/test_notify_user.py`:
  ```python
  import pytest
  from unittest.mock import AsyncMock, patch

  @pytest.mark.asyncio
  async def test_notify_user_sends_message():
      task = {"id": 7, "payload": {"action": "notify_user", "chat_id": 222, "text": "Mission done"}}
      fake_tg = AsyncMock()
      with patch("salako.notify_user.get_telegram", return_value=fake_tg):
          from salako import run
          action = await run(task)
      assert action.status == "completed"
      fake_tg.send_message.assert_awaited_once_with(222, "Mission done")

  @pytest.mark.asyncio
  async def test_notify_user_missing_text_fails():
      from salako import run
      action = await run({"id": 1, "payload": {"action": "notify_user", "chat_id": 1}})
      assert action.status == "failed"
  ```

- [ ] **Step 2: Implement** `packages/salako/src/salako/notify_user.py`:
  ```python
  """Mechanical notify_user executor: plain status Telegram send."""
  from __future__ import annotations
  from src.app.telegram_bot import get_telegram

  async def notify_user(task: dict) -> dict:
      payload = task.get("payload") or {}
      chat_id = payload.get("chat_id")
      text = payload.get("text")
      if not text or chat_id is None:
          raise ValueError("notify_user payload requires 'chat_id' and 'text'")
      tg = get_telegram()
      await tg.send_message(chat_id, text)
      return {"sent": True}
  ```

- [ ] **Step 3: Wire into `salako.run()`.** Add a third branch in `__init__.py`:
  ```python
  if action == "notify_user":
      from salako.notify_user import notify_user
      try:
          res = await notify_user(task)
          return Action(status="completed", result=res)
      except Exception as e:
          return Action(status="failed", error=str(e))
  ```

- [ ] **Step 4: Verify.** `timeout 30 pytest packages/salako/tests/test_notify_user.py -v`.

- [ ] **Step 5: Commit.**
  ```
  git add packages/salako/src/salako/notify_user.py packages/salako/src/salako/__init__.py packages/salako/tests/test_notify_user.py
  git commit -m "feat(salako): add notify_user mechanical executor"
  ```

---

## Task 3: Dead-Code Deletions

**Goal:** Remove `risk_assessor`, `task_gates`, `human_gate` plumbing, and associated tests. These gates never fire in practice and the orchestrator gate call-site at line ~619 is removed.

**Files:**

- Delete `src/security/risk_assessor.py`
- Delete `src/core/task_gates.py`
- Delete `tests/test_human_gates.py`
- Delete `tests/test_resilience_approvals.py`
- Modify `src/core/orchestrator.py` — remove imports and gate call (lines 25, 27, 615–624)
- Modify `src/app/telegram_bot.py` — delete `request_approval` method (line ~5698)
- Grep-remove `human_gate` context reads (leave workflow JSON schema alone)

**Steps:**

- [ ] **Step 1: Grep-audit.**
  ```
  grep -rn "from src.security.risk_assessor\|import risk_assessor\|request_approval\|approval_fn\|human_gate\|run_gates\|task_gates" src/ tests/ packages/ | tee /tmp/dead_code_refs.txt
  ```
  Expected: references in `orchestrator.py` (gate call), `telegram_bot.py` (`request_approval`), `tests/test_human_gates.py`, `tests/test_resilience_approvals.py`. Any other reference must be investigated before deleting.

- [ ] **Step 2: Remove gate call from orchestrator.** Edit `src/core/orchestrator.py`:
  - Delete line 25: `from .task_gates import run_gates`
  - Delete line 27: `from .decisions import Cancel as GateCancel`
  - Delete lines 615–624 (the `run_gates` call + `GateCancel` branch). Replace with a blank line (this block sits between "context enrichment" and "Phase 6: Snapshot workspace").

- [ ] **Step 3: Remove `request_approval` method.** Edit `src/app/telegram_bot.py` at line ~5698: delete the whole `async def request_approval(...)` method body. Grep afterwards: `grep -n 'request_approval' src/` must return zero.

- [ ] **Step 4: Delete files.**
  ```
  rm src/security/risk_assessor.py
  rm src/core/task_gates.py
  rm tests/test_human_gates.py
  rm tests/test_resilience_approvals.py
  ```

- [ ] **Step 5: Remove `human_gate` context reads.** `grep -n 'human_gate' src/ packages/` — strip any `task_ctx.get("human_gate")` reads remaining (leave workflow JSON step definitions alone — the field may still live in `src/workflows/i2p/*.json`; unread fields are harmless).

- [ ] **Step 6: Smoke-import.**
  ```
  python -c "from src.core import orchestrator; print('ok')"
  python -c "from src.app import telegram_bot; print('ok')"
  ```

- [ ] **Step 7: Run targeted tests.**
  ```
  timeout 60 pytest tests/test_orchestrator_routing.py tests/test_lifecycle_fixes.py -v
  ```

- [ ] **Step 8: Commit.**
  ```
  git add -A
  git commit -m "chore: delete dead gate/approval plumbing (risk_assessor, task_gates, human_gate tests)"
  ```

---

## Task 4: Scaffold `packages/general_beckman/`

**Goal:** Empty installable package with public-API stub and smoke test. Mirrors `salako` / `fatih_hoca` layout.

**Files:**

- Create `packages/general_beckman/pyproject.toml`
- Create `packages/general_beckman/README.md`
- Create `packages/general_beckman/src/general_beckman/__init__.py`
- Create `packages/general_beckman/src/general_beckman/types.py`
- Create `packages/general_beckman/tests/__init__.py`
- Create `packages/general_beckman/tests/test_init.py`
- Modify root install script / editable-install list (see Task 14)

**Steps:**

- [ ] **Step 1: `pyproject.toml`:**
  ```toml
  [build-system]
  requires = ["setuptools>=42"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "general_beckman"
  version = "0.1.0"
  description = "Task master — task queue, lifecycle, look-ahead against cloud quota"
  requires-python = ">=3.10"
  dependencies = ["nerd_herd", "salako"]

  [tool.setuptools.packages.find]
  where = ["src"]
  ```

- [ ] **Step 2: `src/general_beckman/types.py`:**
  ```python
  """Core types: Task, AgentResult, Lane."""
  from __future__ import annotations
  from enum import Enum
  from typing import Any

  Task = dict[str, Any]
  AgentResult = dict[str, Any]

  class Lane(str, Enum):
      LOCAL_LLM = "local_llm"
      CLOUD_LLM = "cloud_llm"
      MECHANICAL = "mechanical"
  ```

- [ ] **Step 3: `src/general_beckman/__init__.py` (stub):**
  ```python
  """General Beckman — the task master."""
  from __future__ import annotations
  from general_beckman.types import Task, AgentResult, Lane

  __all__ = ["next_task", "on_task_finished", "tick", "Task", "AgentResult", "Lane"]

  async def next_task() -> Task | None:
      """Stub — filled in by Task 10."""
      return None

  async def on_task_finished(task_id: int, result: AgentResult) -> None:
      """Stub — filled in by Task 9."""
      return None

  async def tick() -> None:
      """Stub — filled in by Task 11."""
      return None
  ```

- [ ] **Step 4: `tests/test_init.py`:**
  ```python
  import pytest

  @pytest.mark.asyncio
  async def test_public_api_importable():
      import general_beckman
      assert hasattr(general_beckman, "next_task")
      assert hasattr(general_beckman, "on_task_finished")
      assert hasattr(general_beckman, "tick")

  @pytest.mark.asyncio
  async def test_next_task_stub_returns_none():
      import general_beckman
      assert await general_beckman.next_task() is None
  ```

- [ ] **Step 5: `README.md`** — 1-paragraph summary + public API + test command. Mirrors `packages/salako/README.md`.

- [ ] **Step 6: Install + verify.**
  ```
  pip install -e packages/general_beckman
  python -c "import general_beckman; print(general_beckman.__all__)"
  timeout 30 pytest packages/general_beckman/tests/test_init.py -v
  ```

- [ ] **Step 7: Commit.**
  ```
  git add packages/general_beckman/
  git commit -m "feat(general_beckman): scaffold package with public API stub"
  ```

---

## Task 5: Move `task_context` into Beckman with Shim

**Goal:** Relocate 31-line `src/core/task_context.py` into the package; leave a re-export shim.

**Files:**

- Create `packages/general_beckman/src/general_beckman/task_context.py` (verbatim from `src/core/task_context.py`)
- Modify `src/core/task_context.py` → shim
- Existing test: `tests/test_task_context.py` (53 lines) — must pass unchanged via shim

**Steps:**

- [ ] **Step 1: Copy file verbatim.** Contents of `src/core/task_context.py` → `packages/general_beckman/src/general_beckman/task_context.py`.

- [ ] **Step 2: Rewrite shim.** Replace `src/core/task_context.py` with:
  ```python
  """Backward-compat shim. Real module lives in general_beckman.task_context."""
  from general_beckman.task_context import parse_context, set_context  # noqa: F401
  __all__ = ["parse_context", "set_context"]
  ```

- [ ] **Step 3: Verify.**
  ```
  timeout 30 pytest tests/test_task_context.py -v
  python -c "from src.core.task_context import parse_context, set_context; print('ok')"
  python -c "from general_beckman.task_context import parse_context; print('ok')"
  ```

- [ ] **Step 4: Commit.**
  ```
  git add packages/general_beckman/src/general_beckman/task_context.py src/core/task_context.py
  git commit -m "refactor(general_beckman): move task_context from src/core (shim preserved)"
  ```

---

## Task 6: Move `result_router` + `result_guards` into Beckman with Shims

**Goal:** Relocate the 109-line router and 353-line guards modules. Both have dedicated test files (`test_result_router.py`, `test_result_guards.py`) that must pass unchanged.

**Files:**

- Create `packages/general_beckman/src/general_beckman/result_router.py` (verbatim)
- Create `packages/general_beckman/src/general_beckman/result_guards.py` (verbatim)
- Modify `src/core/result_router.py` → shim
- Modify `src/core/result_guards.py` → shim

**Steps:**

- [ ] **Step 1: Copy files verbatim.** Both copied into `packages/general_beckman/src/general_beckman/`.

- [ ] **Step 2: Check internal imports.** `grep -n '^from\|^import' packages/general_beckman/src/general_beckman/result_router.py packages/general_beckman/src/general_beckman/result_guards.py`. Any `from .task_context import` must be updated to `from general_beckman.task_context import`. Any `from src.core.decisions import` stays (decisions module remains in `src/core/`).

- [ ] **Step 3: Rewrite shims.**
  ```python
  # src/core/result_router.py
  """Backward-compat shim."""
  from general_beckman.result_router import *  # noqa: F401, F403
  from general_beckman.result_router import (  # noqa: F401
      route_result, Complete, SpawnSubtasks, RequestClarification,
      RequestReview, Exhausted, Failed,
  )
  ```
  ```python
  # src/core/result_guards.py
  """Backward-compat shim."""
  from general_beckman.result_guards import *  # noqa: F401, F403
  ```

- [ ] **Step 4: Verify.**
  ```
  timeout 60 pytest tests/test_result_router.py tests/test_result_guards.py -v
  python -c "from src.core.result_router import route_result, Complete; print('ok')"
  python -c "from general_beckman.result_router import route_result; print('ok')"
  ```

- [ ] **Step 5: Commit.**
  ```
  git add packages/general_beckman/src/general_beckman/result_router.py packages/general_beckman/src/general_beckman/result_guards.py src/core/result_router.py src/core/result_guards.py
  git commit -m "refactor(general_beckman): move result_router + result_guards (shims preserved)"
  ```

---

## Task 7: Move `watchdog` into Beckman with Shim

**Goal:** Relocate the 524-line `src/core/watchdog.py`. Test: `tests/test_stuck_tasks.py` (189 lines).

**Files:**

- Create `packages/general_beckman/src/general_beckman/watchdog.py` (verbatim)
- Modify `src/core/watchdog.py` → shim

**Steps:**

- [ ] **Step 1: Copy verbatim.**

- [ ] **Step 2: Fix internal imports.** The watchdog imports `from .task_context import parse_context` and `from .router import get_kdv` (lines 19–20). Rewrite in the package copy to:
  ```python
  from general_beckman.task_context import parse_context
  from src.core.router import get_kdv
  ```

- [ ] **Step 3: Shim.**
  ```python
  # src/core/watchdog.py
  """Backward-compat shim."""
  from general_beckman.watchdog import *  # noqa: F401, F403
  from general_beckman.watchdog import check_stuck_tasks, check_resources  # noqa: F401
  ```

- [ ] **Step 4: Verify.**
  ```
  timeout 60 pytest tests/test_stuck_tasks.py tests/unit/test_idle_watchdog_race.py -v
  python -c "from src.core.watchdog import check_stuck_tasks; print('ok')"
  ```

- [ ] **Step 5: Commit.**
  ```
  git add packages/general_beckman/src/general_beckman/watchdog.py src/core/watchdog.py
  git commit -m "refactor(general_beckman): move watchdog from src/core (shim preserved)"
  ```

---

## Task 8: Move `scheduled_jobs` into Beckman with Shim

**Goal:** Relocate the 527-line `src/app/scheduled_jobs.py`. Test: `tests/test_scheduled_jobs.py` (14 lines).

**Files:**

- Create `packages/general_beckman/src/general_beckman/scheduled_jobs.py` (verbatim)
- Modify `src/app/scheduled_jobs.py` → shim

**Steps:**

- [ ] **Step 1: Copy verbatim.** Imports in this module all use absolute `src.infra.*` paths — no rewriting needed.

- [ ] **Step 2: Shim.**
  ```python
  # src/app/scheduled_jobs.py
  """Backward-compat shim."""
  from general_beckman.scheduled_jobs import *  # noqa: F401, F403
  ```

- [ ] **Step 3: Verify.**
  ```
  timeout 30 pytest tests/test_scheduled_jobs.py -v
  python -c "from src.app.scheduled_jobs import tick_todos_reminder; print('ok')" || python -c "from general_beckman.scheduled_jobs import *; print('ok')"
  ```
  (The second command is a fallback if the specific function name differs — intent is to confirm package import succeeds.)

- [ ] **Step 4: Commit.**
  ```
  git add packages/general_beckman/src/general_beckman/scheduled_jobs.py src/app/scheduled_jobs.py
  git commit -m "refactor(general_beckman): move scheduled_jobs from src/app (shim preserved)"
  ```

---

## Task 9: Move `_handle_*` Methods into `lifecycle.py`

**Goal:** Relocate the 8 lifecycle handlers from `src/core/orchestrator.py` into `packages/general_beckman/src/general_beckman/lifecycle.py`. Convert `_handle_clarification` to emit a salako `clarify` task instead of calling Telegram directly. Build `on_task_finished(task_id, result)` as the drain entry point.

**Current handler locations in orchestrator.py (exact line numbers):**

| Handler | Line |
|---|---|
| `_handle_availability_failure` | 1003 |
| `_handle_unexpected_failure` | 1058 |
| `_handle_complete` | 1150 |
| `_handle_subtasks` | 1344 |
| `_handle_clarification` | 1547 |
| `_handle_review` | 1554 |
| `_handle_exhausted` | 1576 |
| `_handle_failed` | 1672 |

**Files:**

- Create `packages/general_beckman/src/general_beckman/lifecycle.py`
- Create `packages/general_beckman/tests/test_lifecycle.py`
- Modify `src/core/orchestrator.py` — delete handler bodies; keep thin wrappers for now (call `await beckman.on_task_finished(...)`)

**Steps:**

- [ ] **Step 1: Write behavioral tests FIRST** in `packages/general_beckman/tests/test_lifecycle.py`. Cover each handler's primary path. Example:
  ```python
  import pytest
  from unittest.mock import AsyncMock, patch

  @pytest.mark.asyncio
  async def test_handle_complete_marks_task_completed():
      from general_beckman.lifecycle import handle_complete
      task = {"id": 1, "title": "t"}
      result = {"status": "completed", "result": "ok"}
      with patch("general_beckman.lifecycle.update_task", new=AsyncMock()) as ut:
          await handle_complete(task, result)
      ut.assert_awaited()

  @pytest.mark.asyncio
  async def test_handle_clarification_emits_salako_task():
      """No direct telegram call — emits mechanical task with action='clarify'."""
      from general_beckman.lifecycle import handle_clarification
      task = {"id": 5, "title": "plan trip", "mission_id": 2, "chat_id": 99}
      result = {"clarification": "Which dates?"}
      with patch("general_beckman.lifecycle.add_task", new=AsyncMock()) as at, \
           patch("general_beckman.lifecycle.update_task", new=AsyncMock()):
          await handle_clarification(task, result)
      at.assert_awaited_once()
      kwargs = at.await_args.kwargs
      assert kwargs["agent_type"] == "mechanical"
      assert kwargs["payload"]["action"] == "clarify"
      assert kwargs["payload"]["question"] == "Which dates?"
  ```
  Add tests for exhausted (budget/guards/tool_failures), failed (retryable + terminal), subtasks happy path, review, availability_failure, unexpected_failure.

- [ ] **Step 2: Create `lifecycle.py`.** Copy each handler verbatim, rename `_handle_*` → `handle_*` (free functions, not methods). Drop the `self` parameter; where `self.telegram.*` was called in `_handle_complete`, `_handle_exhausted`, `_handle_failed`, etc., emit a mechanical `notify_user` task via `add_task` OR keep direct `get_telegram().send_message(...)` for now — choose per the spec §12 guidance ("progress chatter is ephemeral, not Plan C"). Rewrite `_handle_clarification` to:
  ```python
  async def handle_clarification(task: dict, result: dict) -> None:
      task_id = task["id"]
      question = result.get("clarification", "Need more information")
      await update_task(task_id, status="waiting_human")
      await add_task(
          title=f"Clarify: {task.get('title','')[:40]}",
          description=question,
          mission_id=task.get("mission_id"),
          parent_task_id=task_id,
          agent_type="mechanical",
          executor="mechanical",
          payload={
              "action": "clarify",
              "question": question,
              "chat_id": task.get("chat_id"),
          },
          depends_on=[],
      )
      logger.info(f"[Task #{task_id}] Emitted clarify mechanical task")
  ```

- [ ] **Step 3: Build `on_task_finished`.** At the bottom of `lifecycle.py`:
  ```python
  from general_beckman.result_router import (
      route_result, Complete, SpawnSubtasks, RequestClarification,
      RequestReview, Exhausted, Failed,
  )
  from general_beckman.result_guards import run_guards_for

  async def on_task_finished(task_id: int, result: dict) -> None:
      task = await get_task(task_id)
      if task is None:
          logger.warning("on_task_finished called for missing task", task_id=task_id)
          return
      task_ctx = parse_context(task)
      actions = route_result(task, result)
      for action in actions:
          if await run_guards_for(action, task, task_ctx, result):
              return
          await _dispatch_action(action, task)

  async def _dispatch_action(action, task):
      if isinstance(action, Complete):            await handle_complete(task, action.raw)
      elif isinstance(action, SpawnSubtasks):     await handle_subtasks(task, action.raw)
      elif isinstance(action, RequestClarification): await handle_clarification(task, action.raw)
      elif isinstance(action, RequestReview):     await handle_review(task, action.raw)
      elif isinstance(action, Exhausted):         await handle_exhausted(task, action.raw)
      elif isinstance(action, Failed):            await handle_failed(task, action.raw)
  ```
  Note: if `run_guards_for` does not exist, port the isinstance-dispatch guard logic from the current `_record`/`process_task` path into `result_guards.py` as part of this task. Grep first: `grep -n 'def run_guards_for\|def _run_guards' src/core/result_guards.py packages/general_beckman/src/general_beckman/result_guards.py`.

- [ ] **Step 4: Export from `general_beckman/__init__.py`.**
  ```python
  from general_beckman.lifecycle import on_task_finished
  ```

- [ ] **Step 5: Remove handlers from orchestrator.** Delete lines 1003, 1058, 1150, 1344, 1547, 1554, 1576, 1672 (each `_handle_*` method body). Any remaining `await self._handle_*` call-site becomes `await beckman.on_task_finished(task_id, result)` — but most call-sites vanish once Task 12 rewires the main loop. For Task 9 intermediate state, leave a thin stub:
  ```python
  async def _handle_complete(self, task, result):
      from general_beckman import on_task_finished
      await on_task_finished(task["id"], result)
  ```
  applied uniformly to all 8 methods. These stubs die in Task 12.

- [ ] **Step 6: Verify.**
  ```
  timeout 60 pytest packages/general_beckman/tests/test_lifecycle.py tests/test_lifecycle_fixes.py tests/test_orchestrator_routing.py tests/test_exhaustion.py tests/test_retry.py -v
  wc -l src/core/orchestrator.py
  ```

- [ ] **Step 7: Commit.**
  ```
  git add packages/general_beckman/ src/core/orchestrator.py
  git commit -m "refactor(general_beckman): move _handle_* lifecycle methods; add on_task_finished drain"
  ```

---

## Task 10: Build `beckman.next_task()` — Eligibility + Priority + Look-Ahead

**Goal:** Implement the eligibility-filter + priority + look-ahead pipeline that selects one task to release per call. "How many in flight" is answered by reading the capacity snapshot directly — llama-server exposes 1 slot, kdv rate limits, and beckman's quota look-ahead. No artificial cap exists. `next_task()` returns `None` when the authoritative capacity math says stop.

**Files:**

- Create `packages/general_beckman/src/general_beckman/queue.py`
- Create `packages/general_beckman/tests/test_queue.py`
- Modify `packages/general_beckman/src/general_beckman/__init__.py` — replace `next_task` stub

**Steps:**

- [ ] **Step 1: Survey current task-fetch path.** `grep -n 'get_ready_tasks\|claim_task' src/core/orchestrator.py` to locate existing eligibility logic. Read the block that selects the next task (~main loop) to understand priority ordering (TASK_PRIORITY constant from `src/app/config.py`).

- [ ] **Step 2: Write failing tests for `queue.py`** in `packages/general_beckman/tests/test_queue.py`. Cover:
  - Eligibility filter skips `paused`, `waiting_human`, `cancelled` rows.
  - Priority order: workflow steps > direct user tasks > background missions.
  - Returns `None` when no eligible task.
  - Lanes at capacity saturation (as reported by snapshot) are excluded.
  - Mechanical tasks are never held back by LLM capacity constraints.

- [ ] **Step 3: Implement `queue.py`.**
  ```python
  """Task queue: eligibility filter + priority."""
  from __future__ import annotations
  from src.infra.db import get_ready_tasks, claim_task

  async def pick_ready_task(saturated_lanes: set[str]) -> dict | None:
      """Return one ready task eligible for dispatch, or None.

      saturated_lanes contains lanes where the capacity snapshot says no room.
      Tasks bound to those lanes are skipped.
      """
      rows = await get_ready_tasks()
      for row in rows:
          lane = _classify_lane(row)
          if lane in saturated_lanes:
              continue
          claimed = await claim_task(row["id"])
          if claimed:
              return row
      return None

  def _classify_lane(task: dict) -> str:
      if task.get("agent_type") == "mechanical":
          return "mechanical"
      # Cloud vs local classification — read from task_ctx or default local
      # (Fatih Hoca ultimately chooses the model; this is just lane pre-gating.)
      return "local_llm"
  ```

- [ ] **Step 4: Wire into `__init__.py`.**
  ```python
  from nerd_herd import snapshot as _snapshot
  from general_beckman.queue import pick_ready_task, _classify_lane

  _IN_FLIGHT: dict[str, int] = {"local_llm": 0, "cloud_llm": 0, "mechanical": 0}

  async def next_task() -> Task | None:
      snap = _snapshot()
      # Determine lanes at capacity saturation from the snapshot:
      # local_llm is saturated when llama-server has no free slot (VRAM headroom < 500 MB)
      # cloud_llm saturation is handled by look-ahead in Task 11
      saturated: set[str] = set()
      if snap.local is not None and snap.vram_available_mb < 500:
          saturated.add("local_llm")
      if len(saturated) == 3:
          return None
      task = await pick_ready_task(saturated)
      if task is not None:
          _IN_FLIGHT[_classify_lane(task)] += 1
      return task

  def _release_slot(task: Task) -> None:
      lane = _classify_lane(task)
      _IN_FLIGHT[lane] = max(0, _IN_FLIGHT[lane] - 1)
  ```
  `_release_slot` is called from `on_task_finished` (add to lifecycle.py).

- [ ] **Step 5: Verify.**
  ```
  timeout 30 pytest packages/general_beckman/tests/test_queue.py -v
  ```

- [ ] **Step 6: Commit.**
  ```
  git add packages/general_beckman/
  git commit -m "feat(general_beckman): implement next_task with eligibility + look-ahead"
  ```

---

## Task 11: Build `beckman.lookahead` (Quota Look-Ahead)

**Goal:** Reinstate the quota look-ahead lost during `quota_planner` extraction. Beckman holds back cloud-heavy tasks when projected demand exceeds quota headroom.

**Reference:** `packages/fatih_hoca/src/fatih_hoca/requirements.py:385` (`get_quota_planner`). Beckman consumes the planner's signals via the nerd_herd snapshot.

**Files:**

- Create `packages/general_beckman/src/general_beckman/lookahead.py`
- Create `packages/general_beckman/tests/test_lookahead.py`
- Modify `packages/general_beckman/src/general_beckman/__init__.py` — call lookahead inside `next_task()`

**Steps:**

- [ ] **Step 1: Write failing tests first.**
  ```python
  import pytest
  from unittest.mock import patch

  @pytest.mark.asyncio
  async def test_lookahead_holds_back_cloud_when_quota_low():
      from general_beckman.lookahead import should_hold_back
      from nerd_herd.types import SystemSnapshot, CloudProviderState
      snap = SystemSnapshot(vram_available_mb=0, local=None, cloud={
          "anthropic": CloudProviderState(provider="anthropic",
              requests_remaining=2, tokens_remaining=1000, reset_at=0),
      })
      # 5 cloud-heavy tasks queued; 2 requests left → hold back
      assert should_hold_back(candidate_task={"agent_type": "researcher"}, snapshot=snap,
                              cloud_queue_depth=5) is True

  @pytest.mark.asyncio
  async def test_lookahead_releases_mechanical_regardless():
      from general_beckman.lookahead import should_hold_back
      from nerd_herd.types import SystemSnapshot
      snap = SystemSnapshot(vram_available_mb=0, local=None, cloud={})
      assert should_hold_back(candidate_task={"agent_type": "mechanical"},
                              snapshot=snap, cloud_queue_depth=100) is False
  ```

- [ ] **Step 2: Implement `lookahead.py`.**
  ```python
  """Queue look-ahead: hold back tasks when quota headroom is insufficient."""
  from __future__ import annotations
  from nerd_herd.types import SystemSnapshot

  _CLOUD_AGENT_TYPES = {"researcher", "planner", "architect"}
  _HEADROOM_FACTOR = 1.5  # keep requests_remaining > queue_depth * factor

  def should_hold_back(candidate_task: dict, snapshot: SystemSnapshot,
                       cloud_queue_depth: int) -> bool:
      agent = candidate_task.get("agent_type", "")
      if agent == "mechanical":
          return False
      if agent not in _CLOUD_AGENT_TYPES:
          return False
      total_remaining = sum(s.requests_remaining for s in snapshot.cloud.values())
      if total_remaining == 0:
          return False  # no cloud providers registered → no quota signal
      required = max(1, int(cloud_queue_depth * _HEADROOM_FACTOR))
      return total_remaining < required
  ```

- [ ] **Step 3: Wire into `next_task()`.** Extend the `next_task()` implementation from Task 10 in `__init__.py` to add the look-ahead check after claiming a task:
  ```python
  from general_beckman.lookahead import should_hold_back

  async def next_task() -> Task | None:
      snap = _snapshot()
      # Lanes at capacity saturation per snapshot
      saturated: set[str] = set()
      if snap.local is not None and snap.vram_available_mb < 500:
          saturated.add("local_llm")
      if len(saturated) == 3:
          return None
      task = await pick_ready_task(saturated)
      if task is None:
          return None
      queue_depth = await _count_pending_cloud_tasks()
      if should_hold_back(task, snap, queue_depth):
          # Return task to pool — caller retries on next tick
          await _unclaim(task)
          return None
      _IN_FLIGHT[_classify_lane(task)] += 1
      return task
  ```
  Implement `_count_pending_cloud_tasks` and `_unclaim` as small DB helpers in `queue.py`.

- [ ] **Step 4: Verify.**
  ```
  timeout 30 pytest packages/general_beckman/tests/test_lookahead.py -v
  timeout 30 pytest packages/general_beckman/tests/ -v
  ```

- [ ] **Step 5: Commit.**
  ```
  git add packages/general_beckman/
  git commit -m "feat(general_beckman): reinstate queue look-ahead against cloud quota"
  ```

---

## Task 12: Build `beckman.tick()` (Watchdog + Scheduled Jobs)

**Goal:** Collapse watchdog + scheduled-jobs loops into a single periodic call invoked from the orchestrator main loop. Preserve each module's internal last-run cadence so tick's 3s rhythm does not over-fire them.

**Files:**

- Modify `packages/general_beckman/src/general_beckman/__init__.py` — implement `tick()`
- Modify `packages/general_beckman/src/general_beckman/watchdog.py` — add `_last_run` state (if not already present)
- Modify `packages/general_beckman/src/general_beckman/scheduled_jobs.py` — ensure tick_* functions are idempotent at 3s

**Steps:**

- [ ] **Step 1: Audit existing cadences.** `grep -n '_last_run\|last_check\|datetime.now' packages/general_beckman/src/general_beckman/watchdog.py packages/general_beckman/src/general_beckman/scheduled_jobs.py`. Confirm each tick function either (a) records its own last-run timestamp, or (b) is cheap enough to call every 3s.

- [ ] **Step 2: Implement `tick()`.**
  ```python
  # in general_beckman/__init__.py
  from general_beckman.watchdog import check_stuck_tasks
  from general_beckman.scheduled_jobs import (
      tick_todos_reminder, tick_price_watch, tick_api_discovery, tick_daily_digest,
  )

  async def tick() -> None:
      """Periodic maintenance. Called every 3s by orchestrator main loop."""
      import asyncio
      from src.infra.logging_config import get_logger
      log = get_logger("general_beckman.tick")
      for fn in (check_stuck_tasks, tick_todos_reminder, tick_price_watch,
                 tick_api_discovery, tick_daily_digest):
          try:
              await fn()
          except Exception as e:
              log.warning("tick subroutine failed", fn=fn.__name__, error=str(e))
  ```
  Grep for actual function names in `scheduled_jobs.py` — the names above are placeholders verified against what you find: `grep -n '^async def tick_\|^def tick_' packages/general_beckman/src/general_beckman/scheduled_jobs.py`. Use the real names.

- [ ] **Step 3: If any scheduled-job function lacks internal cadence**, wrap it in the tick function with a module-level `_LAST_RUN` dict keyed by function name:
  ```python
  _LAST_RUN: dict[str, float] = {}
  _CADENCES: dict[str, float] = {
      "tick_todos_reminder": 7200,   # 2h
      "tick_price_watch":    86400,  # 24h
      "tick_daily_digest":   3600,   # 1h
      "tick_api_discovery":  1800,   # 30m
      "check_stuck_tasks":   60,     # 1m
  }
  # Inside tick(): skip if time.time() - _LAST_RUN[name] < _CADENCES[name]
  ```

- [ ] **Step 4: Write a tick test.**
  ```python
  # packages/general_beckman/tests/test_tick.py
  import pytest
  from unittest.mock import patch, AsyncMock

  @pytest.mark.asyncio
  async def test_tick_invokes_all_subroutines():
      from general_beckman import tick
      with patch("general_beckman.watchdog.check_stuck_tasks", new=AsyncMock()) as w, \
           patch("general_beckman.scheduled_jobs.tick_todos_reminder", new=AsyncMock()) as t:
          await tick()
      # at least the watchdog fires; scheduled jobs may skip based on cadence
      w.assert_awaited()

  @pytest.mark.asyncio
  async def test_tick_swallows_subroutine_exceptions():
      from general_beckman import tick
      async def boom(): raise RuntimeError("x")
      with patch("general_beckman.watchdog.check_stuck_tasks", side_effect=boom):
          await tick()  # must not raise
  ```

- [ ] **Step 5: Verify.** `timeout 30 pytest packages/general_beckman/tests/test_tick.py -v`.

- [ ] **Step 6: Commit.**
  ```
  git add packages/general_beckman/
  git commit -m "feat(general_beckman): implement tick() — watchdog + scheduled jobs on 3s cadence"
  ```

---

## Task 13: Rewire Orchestrator Main Loop

**Goal:** Replace the ~2,000-line fat orchestrator with the spec §5 sketch. `process_task` collapses. All `_handle_*` thin-stubs die. `run_gates` call-site already removed (Task 3). Target: `orchestrator.py` ≤ 250 lines.

**Files:**

- Modify `src/core/orchestrator.py` — final shrink

**Steps:**

- [ ] **Step 1: Back up the current shape for diffing.** `cp src/core/orchestrator.py /tmp/orchestrator.pre-rewire.py`.

- [ ] **Step 2: Locate the main loop.** `grep -n 'async def main_loop\|async def run\|async def watchdog' src/core/orchestrator.py`.

- [ ] **Step 3: Rewrite the main loop + `_dispatch`.** Replace the main loop body with:
  ```python
  import asyncio
  import general_beckman as beckman
  import salako
  from src.core.llm_dispatcher import LLMDispatcher

  class Orchestrator:
      def __init__(self, telegram):
          self.telegram = telegram
          self.llm = LLMDispatcher()
          self.running = True

      async def main_loop(self):
          while self.running:
              # Drain beckman to saturation
              while (task := await beckman.next_task()):
                  asyncio.create_task(self._dispatch(task))
              await beckman.tick()
              await asyncio.sleep(3)

      async def _dispatch(self, task: dict):
          task_id = task["id"]
          try:
              if task.get("agent_type") == "mechanical":
                  result = await salako.run(task)
                  result_dict = {"status": result.status, **result.result, "error": result.error}
              else:
                  result_dict = await self.llm.request(task)
          except Exception as e:
              logger.exception("dispatch failed", task_id=task_id)
              result_dict = {"status": "failed", "error": str(e)}
          await beckman.on_task_finished(task_id, result_dict)
  ```

- [ ] **Step 4: Delete the remnant `_handle_*` thin-stubs** that Task 9 left behind. Delete `process_task` entirely (its logic is now `_dispatch` + `beckman.on_task_finished`). Delete `_prepare`, `_dispatch_action`, `_record` if they were extracted in Plan A — their logic belongs to beckman now. Grep to confirm: `grep -n 'async def _prepare\|async def _record\|async def _handle_\|async def _dispatch_action' src/core/orchestrator.py` → all should disappear except the new `_dispatch`.

- [ ] **Step 5: Delete orphaned imports.** Remove any import whose symbol is no longer referenced. Run `python -c "import ast; tree = ast.parse(open('src/core/orchestrator.py').read()); print('ok')"` and then `grep -n '^from\|^import' src/core/orchestrator.py` — prune unused ones.

- [ ] **Step 6: Verify line count.**
  ```
  wc -l src/core/orchestrator.py
  ```
  **Must be ≤ 300. Target 250. Stretch 200.** If over 300, audit what else can move to beckman (likely leftover helpers around `_prepare`, task enrichment, etc.).

- [ ] **Step 7: Run full suite.**
  ```
  timeout 120 pytest tests/ 2>&1 | tee /tmp/pytest_phase2b.log
  ```
  Compare failure set against baseline (253 pre-existing). Any **new** failure touching `orchestrator`, `beckman`, `result_*`, `watchdog`, `scheduled_jobs`, or `lifecycle` must be investigated and fixed before commit.

- [ ] **Step 8: Commit.**
  ```
  git add src/core/orchestrator.py
  git commit -m "refactor(core): rewire orchestrator main loop through beckman (≤ 300 lines)"
  ```

---

## Task 14: Install Wiring + Docs + Final Verification

**Goal:** Add `-e packages/general_beckman` to install scripts, update architecture docs, capture manual smoke evidence, verify final line count.

**Files:**

- Modify root install script (locate via `grep -rn 'pip install -e packages/salako\|pip install -e packages/fatih_hoca' .` — typically `install.ps1`, `setup.sh`, or `requirements-dev.txt`)
- Modify `CLAUDE.md` package-boundaries section
- Modify `docs/architecture-modularization.md` — append Phase 2b section
- Modify `MEMORY.md` — append Phase 2b project note

**Steps:**

- [ ] **Step 1: Locate install site.** `grep -rn 'pip install -e packages' . --include=*.ps1 --include=*.sh --include=*.txt --include=*.toml`. Add `pip install -e packages/general_beckman` next to existing salako / fatih_hoca lines.

- [ ] **Step 2: Update `CLAUDE.md`.** In the Architecture section (top of file), add a line after the Salako bullet:
  ```
  - **Task master**: `packages/general_beckman/` (General Beckman) — task queue, lifecycle, look-ahead against cloud quota. Public API: `next_task()`, `on_task_finished()`, `tick()`.
  ```
  In the Key Files table, add:
  ```
  | `packages/general_beckman/` | **General Beckman** — task master: queue selection, lifecycle handlers, quota look-ahead |
  ```

- [ ] **Step 3: Update `docs/architecture-modularization.md`.** Append a "Phase 2b — General Beckman" section summarizing: (a) what moved, (b) shim list, (c) deleted dead code, (d) new public API.

- [ ] **Step 4: Append MEMORY note.** Add to `MEMORY.md` top level:
  ```
  - [Phase 2b General Beckman](project_phase2b_general_beckman_20260419.md) — task-queue extraction: `packages/general_beckman/` with next_task/on_task_finished/tick; risk_assessor + task_gates deleted; orchestrator.py 2569→≤300 lines; capacity bound naturally by snapshot (llama-server slot, kdv rate limits, quota look-ahead)
  ```

- [ ] **Step 5: Final line-count verification.**
  ```
  wc -l src/core/orchestrator.py
  wc -l packages/general_beckman/src/general_beckman/*.py
  ```

- [ ] **Step 6: Full suite.**
  ```
  timeout 120 pytest tests/ 2>&1 | tail -50
  ```
  Compare to pre-Phase-2b baseline (253 failures). No new failures touching beckman / orchestrator / result_* / watchdog / scheduled_jobs / lifecycle.

- [ ] **Step 7: Package tests green in isolation.**
  ```
  timeout 60 pytest packages/general_beckman/tests/ -v
  timeout 30 pytest packages/salako/tests/ -v
  ```

- [ ] **Step 8: Manual smoke (executor runs this — do not skip).** Start KutAI via `/restart` (Telegram) or `python kutai_wrapper.py`. Verify:
  - A simple `/task Write hello world` dispatches and completes.
  - `/shop coffee machine` triggers clarification round-trip (beckman emits salako clarify task, user reply routes back to original task).
  - Watchdog fires at least once within 90 seconds (check `logs/orchestrator.jsonl` for `check_stuck_tasks` entries).
  - Todo reminder tick fires within its cadence (or is confirmed to be throttled by `_LAST_RUN` state).

- [ ] **Step 9: Final commit.**
  ```
  git add CLAUDE.md docs/architecture-modularization.md MEMORY.md install.ps1 setup.sh requirements-dev.txt
  git commit -m "docs: document Phase 2b general_beckman extraction + wire install"
  ```

---

## Plan Definition of Done

- [ ] `packages/general_beckman/` exists, installable, all package tests pass.
- [ ] `orchestrator.py` ≤ 300 lines (target 250, stretch 200).
- [ ] `src/security/risk_assessor.py` and `src/core/task_gates.py` deleted.
- [ ] No `approval_fn` / `request_approval` / `run_gates` references anywhere.
- [ ] Shims in place: `src/core/task_context.py`, `src/core/result_router.py`, `src/core/result_guards.py`, `src/core/watchdog.py`, `src/app/scheduled_jobs.py`.
- [ ] Salako has `clarify` + `notify_user` executors, both tested.
- [ ] Beckman public API `next_task / on_task_finished / tick` implemented + tested.
- [ ] Full suite: no new failures touching orchestrator / beckman / result_* / watchdog / scheduled_jobs / lifecycle.
- [ ] Manual smoke passes: dispatch + clarification + scheduled jobs.
- [ ] Architecture doc + CLAUDE.md + MEMORY.md updated.

## Out of Scope (Follow-Ups)

From spec §12 and §14:

- **kdv state persistence.** Cloud rate-limit state remains in-memory only; lost on restart. Separate fix.
- **Full Telegram module extraction.** Outbound flows through salako `clarify` / `notify_user`, but inbound reply routing and ephemeral progress chatter still live in `src/app/telegram_bot.py`.
- **Progress chatter standardization.** Iteration counters and scraping progress still call `self.telegram.send_message` directly. Worth tidying later, not now.

## Key Risks

- **Hidden couplings in the 2,569-line orchestrator.** Mitigation: each task commits separately with verification. Revert one task if it regresses.
- **Watchdog + scheduled-jobs timing.** Collapsing into shared `tick()` may shift timing. Mitigation: preserve internal cadences via `_LAST_RUN` + `_CADENCES` map (Task 12 Step 3).
- **Look-ahead reinstatement.** Original logic was lost; reimplementing without tests is risk. Mitigation: tests-first in Task 11 against synthetic snapshots.
- **Shim drift.** Existing suites must pass unchanged — any shim that doesn't re-export a public name will break a test. Mitigation: Tasks 5–8 each include a targeted test run before committing.
