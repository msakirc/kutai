# Unified Task Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the 5 separate retry/failure mechanisms into 2 failure types, add an `ungraded` quality gate state, kill in-memory queues, and simplify the task state machine from 12+ states to 9.

**Architecture:** Two-phase task execution (worker → grading) on a single DB row. Quality and availability as the only two failure types. `next_retry_at` replaces sleeping/paused states. Grade queue replaced by DB queries on `ungraded` tasks. All transitions enforced via `transition_task()`.

**Tech Stack:** Python 3.10, aiosqlite, asyncio, pytest

**Spec:** `docs/superpowers/specs/2026-04-05-unified-task-lifecycle-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/core/state_machine.py` | Modify | Updated enum, transitions, enforced usage |
| `src/core/retry.py` | Create | `compute_retry_timing()`, model exclusion helpers, `RetryDecision` |
| `src/infra/db.py` | Modify | Schema migration, `get_ready_tasks` update, `accelerate_retries`, remove sleeping queue |
| `src/core/grading.py` | Create | `grade_task()`, `apply_grade_result()`, `drain_ungraded_tasks()`, structured prompt/parsing |
| `src/core/orchestrator.py` | Modify | Unified retry calls, simplified watchdog, idle grading, `ensure_gpu_utilized` fix |
| `src/core/llm_dispatcher.py` | Modify | Remove `GradeQueue`/`PendingGrade`, update `on_model_swap`, update `ensure_gpu_utilized` |
| `src/core/router.py` | Modify | Read `excluded_models` from context, structured grading prompt |
| `src/agents/base.py` | Modify | Worker completion → `ungraded` flow |
| `src/infra/dead_letter.py` | Modify | Phase-aware DLQ retry |
| `src/app/telegram_bot.py` | Modify | Status displays, button handlers, DLQ retry UI |
| `src/workflows/engine/hooks.py` | Modify | Remove `_schema_retry_count`, use shared `attempts` |

---

## Task 1: State Machine Update

**Files:**
- Modify: `src/core/state_machine.py`
- Create: `tests/test_state_machine.py`

- [ ] **Step 1: Write tests for new state machine**

```python
# tests/test_state_machine.py
import pytest
from src.core.state_machine import TaskState, TRANSITIONS, validate_transition, InvalidTransition


class TestTaskStates:
    def test_all_states_exist(self):
        expected = {
            "pending", "processing", "ungraded", "completed", "failed",
            "waiting_subtasks", "waiting_human", "cancelled", "skipped",
        }
        assert {s.value for s in TaskState} == expected

    def test_old_states_removed(self):
        values = {s.value for s in TaskState}
        for old in ("paused", "sleeping", "needs_clarification",
                     "needs_review", "needs_subtasks", "rejected", "done"):
            assert old not in values


class TestTransitions:
    # Terminal states allow nothing
    @pytest.mark.parametrize("state", ["completed", "cancelled", "skipped"])
    def test_terminal_states_have_no_transitions(self, state):
        assert TRANSITIONS[state] == set()

    # Key new transitions
    def test_processing_to_ungraded(self):
        assert validate_transition("processing", "ungraded")

    def test_ungraded_to_completed(self):
        assert validate_transition("ungraded", "completed")

    def test_ungraded_to_pending(self):
        assert validate_transition("ungraded", "pending")

    def test_ungraded_to_failed(self):
        assert validate_transition("ungraded", "failed")

    def test_failed_to_ungraded(self):
        assert validate_transition("failed", "ungraded")

    def test_failed_to_pending(self):
        assert validate_transition("failed", "pending")

    # Invalid transitions
    def test_completed_to_anything_invalid(self):
        assert not validate_transition("completed", "pending")
        assert not validate_transition("completed", "processing")

    def test_pending_to_completed_invalid(self):
        assert not validate_transition("pending", "completed")

    def test_ungraded_to_processing_invalid(self):
        assert not validate_transition("ungraded", "processing")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_state_machine.py -v`
Expected: Failures — old states still in enum, new states missing.

- [ ] **Step 3: Update state_machine.py**

```python
# src/core/state_machine.py
"""
Task state machine — validates all status transitions.

States:
  pending → processing → ungraded / completed / failed /
                          waiting_human / waiting_subtasks
  pending → cancelled / skipped
  ungraded → completed / pending / failed
  failed → pending / ungraded  (DLQ retry)
  waiting_human → pending / cancelled
  waiting_subtasks → completed / failed / cancelled
"""

from enum import Enum
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("core.state_machine")


class TaskState(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    UNGRADED = "ungraded"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_SUBTASKS = "waiting_subtasks"
    WAITING_HUMAN = "waiting_human"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ErrorCategory(str, Enum):
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    INVALID_OUTPUT = "invalid_output"
    DEPENDENCY_FAILED = "dependency_failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


TRANSITIONS: dict[str, set[str]] = {
    TaskState.PENDING: {
        TaskState.PROCESSING,
        TaskState.CANCELLED,
        TaskState.SKIPPED,
    },
    TaskState.PROCESSING: {
        TaskState.UNGRADED,
        TaskState.COMPLETED,
        TaskState.PENDING,       # retry on failure
        TaskState.FAILED,
        TaskState.WAITING_SUBTASKS,
        TaskState.WAITING_HUMAN,
        TaskState.CANCELLED,
    },
    TaskState.UNGRADED: {
        TaskState.COMPLETED,     # grade pass or waive
        TaskState.PENDING,       # grade FAIL → worker retry
        TaskState.FAILED,        # availability DLQ from grading phase
    },
    TaskState.COMPLETED: set(),
    TaskState.FAILED: {
        TaskState.PENDING,       # DLQ retry (worker-phase)
        TaskState.UNGRADED,      # DLQ retry (grading-phase)
    },
    TaskState.WAITING_SUBTASKS: {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.WAITING_HUMAN: {
        TaskState.PENDING,
        TaskState.CANCELLED,
    },
    TaskState.CANCELLED: set(),
    TaskState.SKIPPED: set(),
}


class InvalidTransition(Exception):
    def __init__(self, task_id: int, from_state: str, to_state: str):
        self.task_id = task_id
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid transition for task #{task_id}: "
            f"'{from_state}' → '{to_state}'"
        )


def validate_transition(from_state: str, to_state: str) -> bool:
    allowed = TRANSITIONS.get(from_state, set())
    return to_state in allowed


async def transition_task(
    task_id: int,
    to_state: str,
    error: Optional[str] = None,
    **extra_fields,
) -> None:
    """Transition a task to a new state with validation.

    Loads current state from DB, validates the transition is legal,
    then updates. Raises InvalidTransition if the move is not allowed.
    """
    from ..infra.db import get_task, update_task

    task = await get_task(task_id)
    if not task:
        raise ValueError(f"Task #{task_id} not found")

    current_state = task.get("status", "pending")

    if not validate_transition(current_state, to_state):
        raise InvalidTransition(task_id, current_state, to_state)

    update_kwargs = {"status": to_state, **extra_fields}
    if error is not None:
        update_kwargs["error"] = error

    logger.info(
        "state transition",
        task_id=task_id,
        from_state=current_state,
        to_state=to_state,
    )

    await update_task(task_id, **update_kwargs)


def classify_error(exception: Exception) -> str:
    import asyncio

    exc_msg = str(exception).lower()

    if isinstance(exception, (asyncio.TimeoutError,)):
        return ErrorCategory.TIMEOUT
    if "timeout" in exc_msg:
        return ErrorCategory.TIMEOUT
    if "budget" in exc_msg or "cost" in exc_msg:
        return ErrorCategory.BUDGET_EXCEEDED
    if isinstance(exception, (asyncio.CancelledError,)):
        return ErrorCategory.CANCELLED

    model_indicators = [
        "rate_limit", "ratelimit", "429", "api_error",
        "authentication", "401", "403", "api_key",
        "litellm", "openai", "anthropic", "groq",
        "model", "completion",
    ]
    if any(ind in exc_msg for ind in model_indicators):
        return ErrorCategory.MODEL_ERROR

    tool_indicators = [
        "tool error", "command failed", "file not found",
        "permission denied", "not found", "no such file",
        "subprocess", "docker",
    ]
    if any(ind in exc_msg for ind in tool_indicators):
        return ErrorCategory.TOOL_ERROR

    if "parse" in exc_msg or "json" in exc_msg or "invalid" in exc_msg:
        return ErrorCategory.INVALID_OUTPUT

    return ErrorCategory.UNKNOWN
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_state_machine.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/state_machine.py tests/test_state_machine.py
git commit -m "refactor(state_machine): update to 9 states, add ungraded/waiting_human/skipped"
```

---

## Task 2: Retry Logic Module

**Files:**
- Create: `src/core/retry.py`
- Create: `tests/test_retry.py`

- [ ] **Step 1: Write tests for retry logic**

```python
# tests/test_retry.py
import pytest
from src.core.retry import (
    compute_retry_timing, RetryDecision,
    update_exclusions_on_failure, get_model_constraints,
)


class TestComputeRetryTiming:
    # Quality failures
    def test_quality_immediate_first_attempts(self):
        for i in range(3):
            d = compute_retry_timing("quality", attempts=i, max_attempts=6)
            assert d.action == "immediate"

    def test_quality_delayed_after_3(self):
        d = compute_retry_timing("quality", attempts=3, max_attempts=6)
        assert d.action == "delayed"
        assert d.delay_seconds == 600

    def test_quality_terminal_at_max(self):
        d = compute_retry_timing("quality", attempts=6, max_attempts=6)
        assert d.action == "terminal"

    # Availability failures
    def test_availability_first_failure_60s(self):
        d = compute_retry_timing("availability", last_avail_delay=0)
        assert d.action == "delayed"
        assert d.delay_seconds == 60

    def test_availability_doubles(self):
        d = compute_retry_timing("availability", last_avail_delay=60)
        assert d.delay_seconds == 120

    def test_availability_caps_at_7200(self):
        d = compute_retry_timing("availability", last_avail_delay=5000)
        assert d.delay_seconds == 7200

    def test_availability_terminal_after_cap(self):
        d = compute_retry_timing("availability", last_avail_delay=7200)
        assert d.action == "terminal"


class TestModelExclusions:
    def test_update_exclusions_adds_model(self):
        ctx = {}
        update_exclusions_on_failure(ctx, "model_a", 1)
        assert ctx["failed_models"] == ["model_a"]

    def test_update_exclusions_no_duplicates(self):
        ctx = {"failed_models": ["model_a"]}
        update_exclusions_on_failure(ctx, "model_a", 2)
        assert ctx["failed_models"] == ["model_a"]

    def test_constraints_no_exclusion_before_3(self):
        ctx = {"failed_models": ["model_a", "model_b"]}
        excluded, bump = get_model_constraints(ctx, attempts=2)
        assert excluded == []
        assert bump == 0

    def test_constraints_exclude_at_3(self):
        ctx = {"failed_models": ["model_a"]}
        excluded, bump = get_model_constraints(ctx, attempts=3)
        assert excluded == ["model_a"]
        assert bump == 0

    def test_constraints_difficulty_bump_at_4(self):
        ctx = {"failed_models": ["model_a"]}
        excluded, bump = get_model_constraints(ctx, attempts=4)
        assert excluded == ["model_a"]
        assert bump == 2

    def test_constraints_difficulty_bump_scales(self):
        ctx = {"failed_models": []}
        _, bump = get_model_constraints(ctx, attempts=5)
        assert bump == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retry.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement retry module**

```python
# src/core/retry.py
"""
Unified retry logic for all task failure types.

Two failure types:
  quality     — output bad/missing. Immediate retry, then delay. Model escalation.
  availability — couldn't execute. Signal-aware backoff. No model change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetryDecision:
    action: str  # "immediate", "delayed", "terminal"
    delay_seconds: int = 0

    @staticmethod
    def immediate() -> RetryDecision:
        return RetryDecision(action="immediate", delay_seconds=0)

    @staticmethod
    def delayed(seconds: int) -> RetryDecision:
        return RetryDecision(action="delayed", delay_seconds=seconds)

    @staticmethod
    def terminal() -> RetryDecision:
        return RetryDecision(action="terminal", delay_seconds=0)


def compute_retry_timing(
    failure_type: str,
    attempts: int = 0,
    max_attempts: int = 6,
    last_avail_delay: int = 0,
) -> RetryDecision:
    """Compute retry timing for a failed task.

    Args:
        failure_type: "quality" or "availability"
        attempts: quality failure count (worker or grading phase)
        max_attempts: quality hard cap
        last_avail_delay: from context, seconds (availability backoff)

    Returns:
        RetryDecision with action and optional delay.
    """
    if failure_type == "quality":
        if attempts >= max_attempts:
            return RetryDecision.terminal()
        if attempts < 3:
            return RetryDecision.immediate()
        return RetryDecision.delayed(600)

    elif failure_type == "availability":
        if last_avail_delay >= 7200:
            return RetryDecision.terminal()
        new_delay = max(60, min(last_avail_delay * 2, 7200))
        return RetryDecision.delayed(new_delay)

    raise ValueError(f"Unknown failure_type: {failure_type}")


def update_exclusions_on_failure(
    task_context: dict,
    failed_model: str,
    attempts: int,
) -> None:
    """Track a model that produced bad output."""
    failed = task_context.setdefault("failed_models", [])
    if failed_model and failed_model not in failed:
        failed.append(failed_model)


def get_model_constraints(
    task_context: dict,
    attempts: int,
) -> tuple[list[str], int]:
    """Get model exclusions and difficulty bump for next attempt.

    Returns:
        (excluded_models, difficulty_bump)
    """
    failed = task_context.get("failed_models", [])
    excluded = list(failed) if attempts >= 3 else []
    difficulty_bump = max(0, (attempts - 3) * 2) if attempts >= 4 else 0
    return excluded, difficulty_bump
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retry.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/retry.py tests/test_retry.py
git commit -m "feat(retry): unified retry logic with quality/availability failure types"
```

---

## Task 3: Database Schema Migration

**Files:**
- Modify: `src/infra/db.py`
- Modify: `tests/integration/test_task_lifecycle.py`

- [ ] **Step 1: Write migration test**

```python
# Add to tests/integration/test_task_lifecycle.py
import pytest
import asyncio

@pytest.mark.integration
def test_new_task_columns_exist(temp_db):
    """After init_db, new retry columns should exist on tasks table."""
    async def _check():
        from src.infra.db import get_db
        db = await get_db()
        cursor = await db.execute("PRAGMA table_info(tasks)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "attempts" in columns
        assert "max_attempts" in columns
        assert "grade_attempts" in columns
        assert "max_grade_attempts" in columns
        assert "next_retry_at" in columns
        assert "retry_reason" in columns
        assert "failed_in_phase" in columns

    asyncio.get_event_loop().run_until_complete(_check())


@pytest.mark.integration
def test_sleeping_tasks_migrated_to_pending(temp_db):
    """Sleeping tasks should become pending with next_retry_at after migration."""
    async def _check():
        from src.infra.db import get_db
        db = await get_db()
        # Insert a sleeping task with old-style sleep_state
        await db.execute(
            """INSERT INTO tasks (title, status, sleep_state)
               VALUES ('test', 'sleeping', '{"next_timer_wake":"2026-04-05 12:00:00"}')"""
        )
        await db.commit()
        # Run migration
        from src.infra.db import _migrate_task_lifecycle
        await _migrate_task_lifecycle(db)
        # Check result
        cursor = await db.execute("SELECT status, next_retry_at FROM tasks WHERE title='test'")
        row = await cursor.fetchone()
        assert row[0] == "pending"
        assert row[1] == "2026-04-05 12:00:00"

    asyncio.get_event_loop().run_until_complete(_check())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_task_lifecycle.py::test_new_task_columns_exist -v`
Expected: FAIL — columns don't exist yet.

- [ ] **Step 3: Add migration to db.py**

In `init_db()`, after the existing migration block (around line 580), add:

```python
    # Migration: Unified Task Lifecycle — new retry columns
    if "attempts" not in columns:
        for col_sql in [
            "ALTER TABLE tasks ADD COLUMN attempts INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN max_attempts INTEGER DEFAULT 6",
            "ALTER TABLE tasks ADD COLUMN grade_attempts INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN max_grade_attempts INTEGER DEFAULT 3",
            "ALTER TABLE tasks ADD COLUMN next_retry_at TIMESTAMP",
            "ALTER TABLE tasks ADD COLUMN retry_reason TEXT",
            "ALTER TABLE tasks ADD COLUMN failed_in_phase TEXT",
        ]:
            try:
                await db.execute(col_sql)
                await db.commit()
            except Exception:
                pass  # column may already exist
        logger.info("Added unified task lifecycle columns")

        # Run data migration
        await _migrate_task_lifecycle(db)
```

Add the migration function:

```python
async def _migrate_task_lifecycle(db) -> None:
    """One-time migration: sleeping/paused/rejected → unified model."""
    try:
        # Backfill attempts from retry_count
        await db.execute(
            "UPDATE tasks SET attempts = COALESCE(retry_count, 0) "
            "WHERE attempts = 0 AND COALESCE(retry_count, 0) > 0"
        )
        # Backfill max_attempts from max_retries (add 3 for grading headroom)
        await db.execute(
            "UPDATE tasks SET max_attempts = COALESCE(max_retries, 3) + 3 "
            "WHERE max_attempts = 6 AND max_retries IS NOT NULL AND max_retries != 3"
        )
        # Convert sleeping → pending with next_retry_at
        await db.execute(
            """UPDATE tasks SET status = 'pending',
               next_retry_at = json_extract(sleep_state, '$.next_timer_wake')
               WHERE status = 'sleeping'"""
        )
        # Convert paused → pending with 10-min delay
        await db.execute(
            """UPDATE tasks SET status = 'pending',
               next_retry_at = datetime('now', '+10 minutes')
               WHERE status = 'paused'"""
        )
        # Rename needs_clarification → waiting_human
        await db.execute(
            "UPDATE tasks SET status = 'waiting_human' "
            "WHERE status = 'needs_clarification'"
        )
        # Fix rejected → cancelled
        await db.execute(
            "UPDATE tasks SET status = 'cancelled' WHERE status = 'rejected'"
        )
        await db.commit()
        logger.info("Migrated task lifecycle data")
    except Exception as e:
        logger.warning(f"Task lifecycle data migration error: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_task_lifecycle.py -v -k "new_task_columns or sleeping_tasks_migrated"`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infra/db.py tests/integration/test_task_lifecycle.py
git commit -m "feat(db): add unified lifecycle columns and data migration"
```

---

## Task 4: Update `get_ready_tasks` and Add `accelerate_retries`

**Files:**
- Modify: `src/infra/db.py`
- Create: `tests/test_ready_tasks.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_ready_tasks.py
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock


class TestGetReadyTasksNextRetryAt:
    """Tasks with future next_retry_at should NOT be returned."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @pytest.mark.integration
    def test_pending_with_future_retry_excluded(self, temp_db):
        async def _test():
            from src.infra.db import get_db, add_task, get_ready_tasks
            db = await get_db()
            # Task with future next_retry_at
            await add_task("future", "desc")
            await db.execute(
                "UPDATE tasks SET next_retry_at = datetime('now', '+1 hour') WHERE title = 'future'"
            )
            await db.commit()
            # Task without next_retry_at (immediately ready)
            await add_task("ready", "desc")

            tasks = await get_ready_tasks()
            titles = [t["title"] for t in tasks]
            assert "ready" in titles
            assert "future" not in titles

        self._run(_test())

    @pytest.mark.integration
    def test_pending_with_past_retry_included(self, temp_db):
        async def _test():
            from src.infra.db import get_db, add_task, get_ready_tasks
            db = await get_db()
            await add_task("past", "desc")
            await db.execute(
                "UPDATE tasks SET next_retry_at = datetime('now', '-1 hour') WHERE title = 'past'"
            )
            await db.commit()

            tasks = await get_ready_tasks()
            titles = [t["title"] for t in tasks]
            assert "past" in titles

        self._run(_test())


class TestUngradedNotResolved:
    """Ungraded tasks should NOT count as resolved for dependency checks."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @pytest.mark.integration
    def test_dependent_blocked_by_ungraded(self, temp_db):
        async def _test():
            from src.infra.db import get_db, add_task, get_ready_tasks, update_task
            import json
            # Task A: ungraded (worker done, awaiting grade)
            task_a_id = await add_task("task_a", "desc")
            await update_task(task_a_id, status="ungraded")
            # Task B: depends on A
            task_b_id = await add_task("task_b", "desc", depends_on=[task_a_id])

            tasks = await get_ready_tasks()
            task_ids = [t["id"] for t in tasks]
            assert task_b_id not in task_ids  # blocked by ungraded A

        self._run(_test())

    @pytest.mark.integration
    def test_dependent_unblocked_when_completed(self, temp_db):
        async def _test():
            from src.infra.db import add_task, get_ready_tasks, update_task
            task_a_id = await add_task("task_a", "desc")
            await update_task(task_a_id, status="completed")
            task_b_id = await add_task("task_b", "desc", depends_on=[task_a_id])

            tasks = await get_ready_tasks()
            task_ids = [t["id"] for t in tasks]
            assert task_b_id in task_ids

        self._run(_test())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ready_tasks.py -v`
Expected: FAIL — `next_retry_at` not filtered, `ungraded` counted as blocking.

- [ ] **Step 3: Update `get_ready_tasks` in db.py**

In `get_ready_tasks()` (line 779), change the query from:

```python
    cursor = await db.execute(
        """SELECT * FROM tasks
           WHERE status = 'pending'
           ORDER BY priority DESC, created_at ASC"""
    )
```

to:

```python
    cursor = await db.execute(
        """SELECT * FROM tasks
           WHERE status = 'pending'
           AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))
           ORDER BY priority DESC, created_at ASC"""
    )
```

The dependency check (line 830) already only counts `status = 'completed'` and `status = 'skipped'`. `ungraded` is neither, so it correctly blocks dependents. No change needed there.

- [ ] **Step 4: Add `accelerate_retries` to db.py**

Replace `wake_sleeping_tasks` (lines 942-981) with:

```python
async def accelerate_retries(reason: str) -> int:
    """Pull next_retry_at to now for tasks waiting on availability.

    Called from: model_swap, gpu_available, rate_limit_reset,
    quota_restored, circuit_breaker_reset.

    Resets last_avail_delay in context so backoff starts fresh.
    Covers both phases: pending (worker) and ungraded (grading).

    Returns number of tasks accelerated.
    """
    import json as _json

    db = await get_db()
    cursor = await db.execute(
        """SELECT id, context FROM tasks
           WHERE status IN ('pending', 'ungraded')
           AND next_retry_at > datetime('now')
           AND retry_reason = 'availability'"""
    )
    rows = [dict(r) for r in await cursor.fetchall()]

    for row in rows:
        try:
            ctx = _json.loads(row.get("context") or "{}")
        except (ValueError, TypeError):
            ctx = {}
        ctx["last_avail_delay"] = 0
        await db.execute(
            """UPDATE tasks SET next_retry_at = datetime('now'),
               context = ? WHERE id = ?""",
            (_json.dumps(ctx), row["id"]),
        )

    if rows:
        await db.commit()
        logger.info(f"Accelerated {len(rows)} task(s) | reason={reason}")
    return len(rows)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ready_tasks.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/infra/db.py tests/test_ready_tasks.py
git commit -m "feat(db): filter next_retry_at in get_ready_tasks, add accelerate_retries"
```

---

## Task 5: Grading Module

**Files:**
- Create: `src/core/grading.py`
- Create: `tests/test_grading.py`
- Modify: `src/core/router.py` (grading prompt)

- [ ] **Step 1: Write tests for grade parsing**

```python
# tests/test_grading.py
import pytest
from src.core.grading import parse_grade_response, GradeResult


class TestParseGradeResponse:
    def test_all_yes(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: YES"
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.relevant is True
        assert result.complete is True

    def test_verdict_no(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: NO"
        result = parse_grade_response(raw)
        assert result.passed is False

    def test_verdict_fail_keyword(self):
        raw = "RELEVANT: YES\nCOMPLETE: NO\nVERDICT: FAIL"
        result = parse_grade_response(raw)
        assert result.passed is False
        assert result.complete is False

    def test_verdict_pass_keyword(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS"
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_derive_from_relevant_complete_when_no_verdict(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES"
        result = parse_grade_response(raw)
        assert result.passed is True  # derived: both YES

    def test_derive_fail_when_relevant_no(self):
        raw = "RELEVANT: NO\nCOMPLETE: YES"
        result = parse_grade_response(raw)
        assert result.passed is False

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="grader incapable"):
            parse_grade_response("Here is my analysis of the task...")

    def test_case_insensitive(self):
        raw = "relevant: yes\ncomplete: Yes\nverdict: YES"
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_with_reasoning_noise(self):
        raw = "The response looks good.\nRELEVANT: YES\nI think it is complete.\nCOMPLETE: YES\nVERDICT: YES"
        result = parse_grade_response(raw)
        assert result.passed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grading.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement grading module**

```python
# src/core/grading.py
"""
Task grading — structured binary evaluation.

Replaces the old 1-5 numeric grading with a structured YES/NO prompt.
All grading calls go through the LLM dispatcher as OVERHEAD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("core.grading")

GRADING_PROMPT = """Evaluate this task response.

Task: {title}
Description: {description}
Response: {response}

Answer each with YES or NO only:
RELEVANT: Does the response address the task?
COMPLETE: Does it contain a concrete deliverable, not just a plan or description?
VERDICT: Should this response be accepted?"""


@dataclass
class GradeResult:
    passed: bool
    relevant: Optional[bool] = None
    complete: Optional[bool] = None
    raw: str = ""
    score: float = 0.0  # 4.0 for PASS, 2.0 for FAIL (analytics compat)

    def __post_init__(self):
        self.score = 4.0 if self.passed else 2.0


def _parse_yes_no(text: str, key: str) -> Optional[bool]:
    """Extract a YES/NO value for a given key from grader output."""
    pattern = rf"{key}\s*:\s*(YES|NO|PASS|FAIL)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    val = match.group(1).upper()
    return val in ("YES", "PASS")


def parse_grade_response(raw: str) -> GradeResult:
    """Parse structured grader output into a GradeResult.

    Parse priority:
      1. VERDICT → use directly
      2. If no VERDICT but RELEVANT+COMPLETE → derive (both YES = PASS)
      3. If nothing parses → raise ValueError (grader incapable)
    """
    relevant = _parse_yes_no(raw, "RELEVANT")
    complete = _parse_yes_no(raw, "COMPLETE")
    verdict = _parse_yes_no(raw, "VERDICT")

    if verdict is not None:
        return GradeResult(
            passed=verdict,
            relevant=relevant,
            complete=complete,
            raw=raw,
        )

    if relevant is not None and complete is not None:
        return GradeResult(
            passed=(relevant and complete),
            relevant=relevant,
            complete=complete,
            raw=raw,
        )

    raise ValueError(f"grader incapable: could not parse VERDICT, RELEVANT, or COMPLETE from output: {raw[:150]}")


async def grade_task(
    task: dict,
    grader_model: str,
) -> GradeResult:
    """Grade a task's output using a specific model via dispatcher OVERHEAD.

    Args:
        task: Task dict with title, description, result, context
        grader_model: litellm_name of the model to use for grading

    Returns:
        GradeResult

    Raises:
        ValueError: grader parse failure (QualityError equivalent)
        RuntimeError: grading call failed (AvailabilityError equivalent)
    """
    import json
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    from src.core.router import ModelRequirements

    ctx = task.get("context", "{}")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}

    generating_model = ctx.get("generating_model", "")
    grade_excluded = ctx.get("grade_excluded_models", [])

    # Build exclusion list: generating model + previously failed graders
    all_excluded = list(set([generating_model] + grade_excluded))

    reqs = ModelRequirements(
        task="reviewer",
        difficulty=3,
        priority=1,
        estimated_input_tokens=800,
        estimated_output_tokens=100,
        prefer_speed=True,
        exclude_models=all_excluded,
        model_override=grader_model,
    )

    result_text = task.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        # Trivial output — auto-pass (nothing meaningful to grade)
        return GradeResult(passed=True, raw="auto-pass: trivial output")

    dispatcher = get_dispatcher()
    response = await dispatcher.request(
        CallCategory.OVERHEAD,
        reqs,
        messages=[{
            "role": "user",
            "content": GRADING_PROMPT.format(
                title=task.get("title", "")[:100],
                description=task.get("description", "")[:200],
                response=str(result_text)[:2000],
            ),
        }],
    )

    raw_content = response.get("content", "")
    if isinstance(raw_content, list):
        raw_content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        )

    return parse_grade_response(str(raw_content))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grading.py -v`
Expected: All PASS (only parse_grade_response is tested; grade_task needs integration tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/grading.py tests/test_grading.py
git commit -m "feat(grading): structured binary grading with YES/NO parsing"
```

---

## Task 5b: Router — Read Excluded Models from Context

**Files:**
- Modify: `src/core/router.py`

- [ ] **Step 1: Update `select_model` or `call_model` to read exclusions from task context**

In `src/core/router.py`, find where `ModelRequirements.exclude_models` is used during model selection. The orchestrator already passes `reqs` to `call_model`. Update the orchestrator's retry path to populate `reqs.exclude_models` from context:

In `src/core/orchestrator.py`, in the task processing section where `ModelRequirements` are built (or where AGENT_REQUIREMENTS are applied), add after building `reqs`:

```python
# Apply model exclusions from retry context
import json as _json
_task_ctx = task.get("context", "{}")
if isinstance(_task_ctx, str):
    try:
        _task_ctx = _json.loads(_task_ctx)
    except (ValueError, TypeError):
        _task_ctx = {}

_excluded = _task_ctx.get("failed_models", [])
_attempts = task.get("attempts", 0)
if _attempts >= 3 and _excluded:
    existing = list(reqs.exclude_models) if reqs.exclude_models else []
    reqs.exclude_models = list(set(existing + _excluded))

# Apply difficulty bump
if _attempts >= 4:
    reqs.difficulty = min(10, reqs.difficulty + (_attempts - 3) * 2)
```

This ensures the router sees the exclusion list when selecting models for retried tasks.

- [ ] **Step 2: Handle excluded models exhausting the pool → availability**

When `ModelCallFailed` is raised because all models are excluded, treat it as availability (not quality). In the `ModelCallFailed` handler in `process_task` (around line 1838), the existing code already enters the availability path. The key is: when `compute_retry_timing` is called, use `"availability"` as the failure type and read `last_avail_delay` from context. This is already the behavior since `ModelCallFailed` maps to availability. No additional code needed — the existing `ModelCallFailed` catch block will naturally handle this case once exclusions are wired into `reqs.exclude_models`. The router tries all non-excluded models, they're all excluded, raises `ModelCallFailed`, and the availability backoff kicks in.

- [ ] **Step 3: Verify with import check**

Run: `python -c "from src.core.router import call_model; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/core/orchestrator.py src/core/router.py
git commit -m "feat(router): apply excluded_models and difficulty bump from retry context"
```

---

## Task 6: `apply_grade_result` and `drain_ungraded_tasks`

**Files:**
- Modify: `src/core/grading.py`
- Create: `tests/test_grade_drain.py`

- [ ] **Step 1: Write tests for apply_grade_result**

```python
# tests/test_grade_drain.py
import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock
from src.core.grading import GradeResult


class TestApplyGradeResult:
    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @pytest.mark.integration
    def test_pass_transitions_to_completed(self, temp_db):
        async def _test():
            from src.infra.db import add_task, update_task, get_task
            from src.core.grading import apply_grade_result

            tid = await add_task("graded", "desc")
            await update_task(tid, status="ungraded",
                              context=json.dumps({"generating_model": "model_a"}))

            verdict = GradeResult(passed=True)
            await apply_grade_result(tid, verdict)

            task = await get_task(tid)
            assert task["status"] == "completed"
            assert task["quality_score"] == 4.0

        self._run(_test())

    @pytest.mark.integration
    def test_fail_transitions_to_pending(self, temp_db):
        async def _test():
            from src.infra.db import add_task, update_task, get_task
            from src.core.grading import apply_grade_result

            tid = await add_task("graded", "desc")
            await update_task(tid, status="ungraded",
                              context=json.dumps({"generating_model": "model_a"}))

            verdict = GradeResult(passed=False)
            await apply_grade_result(tid, verdict)

            task = await get_task(tid)
            assert task["status"] == "pending"
            assert task["attempts"] == 1
            assert task["grade_attempts"] == 0  # reset for next worker run
            ctx = json.loads(task["context"])
            assert "model_a" in ctx.get("failed_models", [])

        self._run(_test())

    @pytest.mark.integration
    def test_fail_at_max_attempts_goes_to_failed(self, temp_db):
        async def _test():
            from src.infra.db import add_task, update_task, get_task
            from src.core.grading import apply_grade_result

            tid = await add_task("graded", "desc")
            await update_task(tid, status="ungraded", attempts=5, max_attempts=6,
                              context=json.dumps({"generating_model": "model_a"}))

            verdict = GradeResult(passed=False)
            await apply_grade_result(tid, verdict)

            task = await get_task(tid)
            assert task["status"] == "failed"
            assert task["failed_in_phase"] == "worker"

        self._run(_test())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grade_drain.py -v`
Expected: FAIL — `apply_grade_result` doesn't exist.

- [ ] **Step 3: Implement `apply_grade_result`**

Add to `src/core/grading.py`:

```python
async def apply_grade_result(task_id: int, verdict: GradeResult) -> None:
    """Apply grade outcome to a task. Handles PASS and FAIL.

    PASS: transition to completed, trigger skill extraction.
    FAIL: increment attempts, add model to exclusions, retry or DLQ.
    """
    import json
    from datetime import datetime
    from src.infra.db import get_task, update_task
    from src.core.state_machine import transition_task
    from src.core.retry import (
        compute_retry_timing, update_exclusions_on_failure,
        get_model_constraints,
    )

    task = await get_task(task_id)
    if not task:
        logger.warning(f"apply_grade_result: task #{task_id} not found")
        return

    ctx = task.get("context", "{}")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}

    if verdict.passed:
        await transition_task(
            task_id, "completed",
            quality_score=verdict.score,
            completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Record model quality feedback
        try:
            from src.infra.db import record_model_call
            await record_model_call(
                model=ctx.get("generating_model", ""),
                agent_type=task.get("agent_type", "executor"),
                success=True,
                grade=verdict.score,
            )
        except Exception:
            pass

        # Skill extraction for deferred grades
        iterations = task.get("iterations", 1) or 1
        tools_used = ctx.get("tools_used_names", [])
        if iterations >= 2 and tools_used and verdict.score >= 4.0:
            try:
                from src.memory.skills import add_skill
                agent_type = task.get("agent_type", "executor")
                title = task.get("title", "")
                await add_skill(
                    name=f"auto:{agent_type}:{title[:40]}",
                    description=f"Task: {title[:100]}. Agent: {agent_type}.",
                    strategy_summary=f"Used {', '.join(sorted(tools_used)[:5])} in {iterations} iterations",
                    tools_used=sorted(tools_used),
                    avg_iterations=iterations,
                    source_grade="great",
                    source_task_id=task_id,
                )
            except Exception as e:
                logger.debug(f"deferred skill extraction failed: {e}")

        logger.info(f"grade PASS | task_id={task_id} score={verdict.score}")
    else:
        # VERDICT=FAIL — worker quality failure
        generating_model = ctx.get("generating_model", "")
        attempts = (task.get("attempts") or 0) + 1
        max_attempts = task.get("max_attempts") or 6

        update_exclusions_on_failure(ctx, generating_model, attempts)
        decision = compute_retry_timing("quality", attempts=attempts, max_attempts=max_attempts)

        if decision.action == "terminal":
            ctx_str = json.dumps(ctx)
            await transition_task(
                task_id, "failed",
                failed_in_phase="worker",
                attempts=attempts,
                context=ctx_str,
            )
            try:
                from src.infra.dead_letter import quarantine_task
                await quarantine_task(
                    task_id=task_id,
                    mission_id=task.get("mission_id"),
                    error=f"Quality gate failed after {attempts} attempts",
                    error_category="quality",
                    original_agent=task.get("agent_type", "executor"),
                    retry_count=attempts,
                )
            except Exception as e:
                logger.warning(f"DLQ quarantine failed: {e}")
            logger.warning(f"grade FAIL terminal | task_id={task_id} attempts={attempts}")
        else:
            next_retry = None
            if decision.action == "delayed":
                from datetime import timedelta
                next_retry = (datetime.now() + timedelta(seconds=decision.delay_seconds)).strftime("%Y-%m-%d %H:%M:%S")

            await transition_task(
                task_id, "pending",
                attempts=attempts,
                grade_attempts=0,
                next_retry_at=next_retry,
                retry_reason="quality",
                context=json.dumps(ctx),
            )
            logger.info(f"grade FAIL retry | task_id={task_id} attempts={attempts} delay={decision.delay_seconds}")
```

- [ ] **Step 4: Implement `drain_ungraded_tasks`**

Add to `src/core/grading.py`:

```python
async def drain_ungraded_tasks(new_model: str) -> int:
    """Grade all ungraded tasks that the new model can grade.

    Called from on_model_swap(). The new model can grade any task
    NOT generated by itself.

    Returns number of tasks graded.
    """
    import json
    from datetime import datetime, timedelta
    from src.infra.db import get_db, update_task
    from src.core.retry import compute_retry_timing

    db = await get_db()
    cursor = await db.execute(
        """SELECT * FROM tasks
           WHERE status = 'ungraded'
           AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))"""
    )
    tasks = [dict(row) for row in await cursor.fetchall()]

    if not tasks:
        return 0

    graded = 0
    for task in tasks:
        ctx_str = task.get("context") or "{}"
        try:
            ctx = json.loads(ctx_str)
        except (json.JSONDecodeError, TypeError):
            ctx = {}

        generating_model = ctx.get("generating_model", "")
        if generating_model == new_model:
            continue  # can't self-grade

        # Check grade_excluded_models
        if new_model in ctx.get("grade_excluded_models", []):
            continue  # this grader already failed for this task

        task_id = task["id"]

        try:
            verdict = await grade_task(task, new_model)
            await apply_grade_result(task_id, verdict)
            graded += 1
        except ValueError:
            # QualityError — grader parse failure
            g_attempts = (task.get("grade_attempts") or 0) + 1
            max_g = task.get("max_grade_attempts") or 3
            ctx.setdefault("grade_excluded_models", []).append(new_model)

            if g_attempts >= max_g:
                # Waive grading — promote with NULL score
                from src.core.state_machine import transition_task
                await transition_task(
                    task_id, "completed",
                    quality_score=None,
                    grade_attempts=g_attempts,
                    completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    context=json.dumps(ctx),
                )
                logger.warning(f"grading waived (parse failures) | task_id={task_id} grade_attempts={g_attempts}")
                graded += 1
            else:
                await update_task(
                    task_id,
                    grade_attempts=g_attempts,
                    context=json.dumps(ctx),
                )
                logger.info(f"grader parse fail, will retry | task_id={task_id} grade_attempts={g_attempts}")
        except (RuntimeError, Exception) as e:
            # Availability error — backoff, stay ungraded
            last_delay = ctx.get("last_avail_delay", 0)
            decision = compute_retry_timing("availability", last_avail_delay=last_delay)

            if decision.action == "terminal":
                from src.core.state_machine import transition_task
                from src.infra.dead_letter import quarantine_task
                await transition_task(
                    task_id, "failed",
                    failed_in_phase="grading",
                )
                await quarantine_task(
                    task_id=task_id,
                    mission_id=task.get("mission_id"),
                    error=f"Grading availability exhausted: {e}",
                    error_category="availability",
                )
                logger.warning(f"grading availability DLQ | task_id={task_id}")
            else:
                ctx["last_avail_delay"] = decision.delay_seconds
                next_retry = (datetime.now() + timedelta(seconds=decision.delay_seconds)).strftime("%Y-%m-%d %H:%M:%S")
                await update_task(
                    task_id,
                    next_retry_at=next_retry,
                    retry_reason="availability",
                    context=json.dumps(ctx),
                )
                logger.info(f"grading availability backoff | task_id={task_id} delay={decision.delay_seconds}")

    if graded:
        logger.info(f"drain_ungraded | graded={graded} total_checked={len(tasks)}")
    return graded
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_grade_drain.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/grading.py tests/test_grade_drain.py
git commit -m "feat(grading): apply_grade_result and drain_ungraded_tasks"
```

---

## Task 7: LLM Dispatcher Cleanup

**Files:**
- Modify: `src/core/llm_dispatcher.py`
- Modify: `tests/test_llm_dispatcher.py`

- [ ] **Step 1: Remove GradeQueue and PendingGrade**

In `src/core/llm_dispatcher.py`:

1. Delete the `PendingGrade` dataclass (lines 95-108)
2. Delete the `GradeQueue` class (lines 111-233)
3. From `LLMDispatcher.__init__`, remove `self.grade_queue = GradeQueue(max_pending=20)`
4. Delete `request_grade` method (lines 610-678)
5. Delete `drain_grades_if_idle` method (lines 706-714)
6. Delete `drain_grades_if_full` method (lines 716-720)
7. Update `on_model_swap` to call new grading and accelerate_retries:

```python
    async def on_model_swap(self, old_model: str | None, new_model: str | None):
        """Called when a model swap occurs. Grades ungraded tasks and
        wakes availability-delayed tasks.
        """
        # 1. Wake availability-delayed tasks
        try:
            from src.infra.db import accelerate_retries
            woken = await accelerate_retries("model_swap")
            if woken:
                logger.info(f"accelerated {woken} task(s) after swap")
        except Exception as e:
            logger.debug(f"accelerate_retries failed: {e}")

        # 2. Grade ungraded tasks the new model can handle
        if new_model:
            try:
                from src.core.grading import drain_ungraded_tasks
                graded = await drain_ungraded_tasks(new_model)
                if graded:
                    logger.info(f"graded {graded} task(s) after swap to {new_model}")
            except Exception as e:
                logger.debug(f"drain_ungraded_tasks failed: {e}")
```

8. Update `get_stats` to remove grade_queue_depth:

```python
    def get_stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "overhead_calls": self._overhead_calls,
            "overhead_pct": (
                f"{self._overhead_calls / self._total_calls * 100:.1f}%"
                if self._total_calls > 0 else "0%"
            ),
            "swaps_prevented": self._swaps_prevented,
            "swap_budget_remaining": self.swap_budget.remaining,
        }
```

- [ ] **Step 2: Update ensure_gpu_utilized**

Replace the `ensure_gpu_utilized` method and add `_loaded_model_can_grade`:

```python
    async def ensure_gpu_utilized(self, upcoming_tasks: list[dict]):
        """Proactively load a local model if GPU is idle and there's work.

        Enhanced: when loaded model can't grade self-generated tasks,
        swap to a grader model instead of waiting for idle unload.
        """
        try:
            from src.models.local_model_manager import get_local_manager
            manager = get_local_manager()

            if upcoming_tasks:
                if not manager.current_model:
                    best_model = self._find_best_local_for_batch(upcoming_tasks)
                    if best_model:
                        logger.info(f"proactive GPU load | model={best_model} queue_depth={len(upcoming_tasks)}")
                        await manager.ensure_model(best_model, reason="proactive_load")
                return

            # No main work. Check overhead needs.
            if not await self._has_pending_overhead_needs():
                return

            if manager.current_model:
                if await self._loaded_model_can_grade():
                    return  # idle path will handle grading
                # Loaded model can't grade (self-generated) → swap
                best = self._find_fastest_general_model()
                if best and best != manager.current_model:
                    logger.info(f"grade swap | loaded={manager.current_model} → {best}")
                    await manager.ensure_model(best, reason="grade_swap")
            else:
                best = self._find_fastest_general_model()
                if best:
                    logger.info(f"overhead load | model={best}")
                    await manager.ensure_model(best, reason="overhead_load")

        except Exception as e:
            logger.debug(f"ensure_gpu_utilized failed: {e}")

    async def _loaded_model_can_grade(self) -> bool:
        """Check if loaded model can grade ANY ungraded task."""
        loaded = self._get_loaded_litellm_name()
        if not loaded:
            return False
        try:
            import json
            from src.infra.db import get_db
            db = await get_db()
            cursor = await db.execute(
                "SELECT context FROM tasks WHERE status = 'ungraded'"
            )
            rows = await cursor.fetchall()
            if not rows:
                return False
            for row in rows:
                try:
                    ctx = json.loads(row["context"] or "{}")
                except (ValueError, TypeError):
                    ctx = {}
                if ctx.get("generating_model") != loaded:
                    return True
            return False
        except Exception:
            return False
```

- [ ] **Step 3: Update `_has_pending_overhead_needs`**

```python
    async def _has_pending_overhead_needs(self) -> bool:
        """Check if there's pending work that needs a model loaded."""
        try:
            from src.infra.db import get_db
            db = await get_db()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'ungraded'"
            )
            if (await cursor.fetchone())[0] > 0:
                return True
        except Exception:
            pass

        try:
            from src.infra.db import get_todos
            todos = await get_todos(status="pending")
            return len(todos) > 0
        except Exception:
            return False
```

- [ ] **Step 4: Update tests**

In `tests/test_llm_dispatcher.py`, remove tests for `GradeQueue`, `PendingGrade`, and `request_grade`. Update `on_model_swap` tests to verify `accelerate_retries` and `drain_ungraded_tasks` calls.

- [ ] **Step 5: Run all dispatcher tests**

Run: `pytest tests/test_llm_dispatcher.py tests/test_orchestrator_routing.py -v`
Expected: All PASS (with updated test expectations).

- [ ] **Step 6: Commit**

```bash
git add src/core/llm_dispatcher.py tests/test_llm_dispatcher.py
git commit -m "refactor(dispatcher): remove GradeQueue, add grade swap in ensure_gpu_utilized"
```

---

## Task 8: Orchestrator — Watchdog Simplification

**Files:**
- Modify: `src/core/orchestrator.py`
- Modify: `tests/test_stuck_tasks.py`

- [ ] **Step 1: Update watchdog method**

Replace the watchdog method body. Key changes:

1. **Stuck processing** (keep, update to use `attempts`):
```python
        # 1. Tasks stuck in "processing" for more than 5 minutes
        cursor = await db.execute(
            """SELECT id, title, attempts, max_attempts FROM tasks
               WHERE status = 'processing'
               AND started_at < datetime('now', '-5 minutes')"""
        )
        stuck = [dict(row) for row in await cursor.fetchall()]
        for task in stuck:
            attempts = (task.get("attempts") or 0) + 1
            max_attempts = task.get("max_attempts") or 6
            if attempts >= max_attempts:
                await transition_task(
                    task["id"], "failed",
                    error="Stuck in processing — attempts exhausted (watchdog)",
                    failed_in_phase="worker",
                    attempts=attempts,
                )
            else:
                await transition_task(
                    task["id"], "pending",
                    attempts=attempts,
                    retry_reason="quality",
                )
        if stuck:
            await db.commit()
```

2. **Stuck ungraded** (new):
```python
        # 2. Ungraded tasks stuck for > 30 min — safety net
        cursor_ung = await db.execute(
            """SELECT id, context FROM tasks
               WHERE status = 'ungraded'
               AND started_at < datetime('now', '-30 minutes')"""
        )
        stuck_ungraded = [dict(row) for row in await cursor_ung.fetchall()]
        for task in stuck_ungraded:
            await transition_task(
                task["id"], "completed",
                quality_score=None,
                completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            logger.warning(f"[Watchdog] Stuck ungraded #{task['id']} promoted to completed (safety net)")
        if stuck_ungraded:
            await db.commit()
```

3. **Failed dep cascade** (fix):
```python
        # 3. Tasks blocked by ALL deps failed → cascade failure
        cursor2 = await db.execute(
            "SELECT id, title, depends_on FROM tasks "
            "WHERE status = 'pending' AND depends_on != '[]'"
        )
        blocked = [dict(row) for row in await cursor2.fetchall()]
        for task in blocked:
            try:
                deps = json.loads(task.get("depends_on", "[]"))
            except (json.JSONDecodeError, TypeError):
                deps = []
            if not deps:
                continue

            placeholders = ",".join("?" * len(deps))
            # Count non-skipped deps that are failed
            fail_cursor = await db.execute(
                f"SELECT COUNT(*) FROM tasks WHERE id IN ({placeholders}) AND status = 'failed'",
                deps,
            )
            failed_count = (await fail_cursor.fetchone())[0]
            # Count non-skipped deps that are NOT terminal-success
            non_resolved = await db.execute(
                f"SELECT COUNT(*) FROM tasks WHERE id IN ({placeholders}) AND status NOT IN ('completed', 'failed', 'cancelled', 'skipped')",
                deps,
            )
            still_pending = (await non_resolved.fetchone())[0]

            # Only cascade if ALL non-skipped deps are failed (no pending/processing/ungraded left)
            if failed_count > 0 and still_pending == 0:
                total_cursor = await db.execute(
                    f"SELECT COUNT(*) FROM tasks WHERE id IN ({placeholders}) AND status NOT IN ('skipped')",
                    deps,
                )
                total_non_skipped = (await total_cursor.fetchone())[0]
                if failed_count == total_non_skipped:
                    await transition_task(
                        task["id"], "failed",
                        error="All dependencies failed",
                        failed_in_phase="worker",
                    )
        if blocked:
            await db.commit()
```

4. **Remove** the paused handler (section 1b around line 484) and sleeping queue handlers (section 1b around line 503).

5. **Waiting subtasks** (fix all-failed case):
```python
        # 4. Missions with all children done but parent still waiting
        cursor3 = await db.execute(
            "SELECT id, title FROM tasks WHERE status = 'waiting_subtasks'"
        )
        waiting = [dict(row) for row in await cursor3.fetchall()]
        for task in waiting:
            child_cursor = await db.execute(
                """SELECT COUNT(*) as total,
                   SUM(CASE WHEN status IN ('completed','failed','cancelled','skipped')
                       THEN 1 ELSE 0 END) as done,
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
                   FROM tasks WHERE parent_task_id = ?""",
                (task["id"],),
            )
            row = await child_cursor.fetchone()
            if row and row["total"] > 0 and row["total"] == row["done"]:
                if row["completed"] > 0:
                    await transition_task(
                        task["id"], "completed",
                        completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                else:
                    await transition_task(
                        task["id"], "failed",
                        error="All subtasks failed",
                        failed_in_phase="worker",
                    )
        if waiting:
            await db.commit()
```

6. **Overdue retry** (new):
```python
        # 6. Pending tasks with next_retry_at far in the past — safety net
        cursor_overdue = await db.execute(
            """SELECT id FROM tasks
               WHERE status = 'pending'
               AND next_retry_at < datetime('now', '-1 hour')"""
        )
        overdue = [dict(row) for row in await cursor_overdue.fetchall()]
        for task in overdue:
            await db.execute(
                "UPDATE tasks SET next_retry_at = NULL WHERE id = ?",
                (task["id"],),
            )
        if overdue:
            await db.commit()
            logger.info(f"[Watchdog] Cleared overdue next_retry_at for {len(overdue)} task(s)")
```

7. **Stale waiting_human** — rename from `needs_clarification` checks. Update the query:
```python
        # 5. Escalation for waiting_human tasks
        cursor_clar = await db.execute(
            """SELECT id, title, context, started_at FROM tasks
               WHERE status = 'waiting_human'
               AND started_at < ?""",
            (threshold_24h,),
        )
```

- [ ] **Step 2: Update stuck_tasks tests**

Update `tests/test_stuck_tasks.py` to use `attempts`/`max_attempts` instead of `retry_count`/`max_retries`.

- [ ] **Step 3: Run watchdog tests**

Run: `pytest tests/test_stuck_tasks.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/core/orchestrator.py tests/test_stuck_tasks.py
git commit -m "refactor(watchdog): simplify to unified retry, add stuck-ungraded, fix dep cascade"
```

---

## Task 9: Orchestrator — Main Loop and Idle Grading

**Files:**
- Modify: `src/core/orchestrator.py`

- [ ] **Step 1: Update main loop idle path**

Replace the idle grade drain (around line 3040):

```python
                else:
                    if self.cycle_count % 20 == 0:
                        logger.info(f"[Cycle {self.cycle_count}] Idle")

                    # Grade ungraded tasks with loaded model (if compatible)
                    try:
                        from src.core.llm_dispatcher import get_dispatcher
                        from src.core.grading import drain_ungraded_tasks
                        dispatcher = get_dispatcher()
                        loaded = dispatcher._get_loaded_litellm_name()
                        if loaded:
                            await drain_ungraded_tasks(loaded)
                    except Exception as _gd_err:
                        logger.debug(f"Idle grade drain failed: {_gd_err}")

                    # Use shutdown-aware sleep
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(), timeout=3
                        )
                        break
                    except asyncio.TimeoutError:
                        pass
```

- [ ] **Step 2: Remove old grade drain calls**

Remove `drain_grades_if_full()` call (around line 2912) and `drain_grades_if_idle()` import.

- [ ] **Step 3: Update `_check_mission_completion`**

In `_check_mission_completion` (line 2615), update the status exclusion:

```python
        pending = [s for s in statuses if s not in ("completed", "failed", "cancelled", "skipped")]
```

- [ ] **Step 4: Run orchestrator tests**

Run: `pytest tests/test_orchestrator_routing.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py
git commit -m "refactor(orchestrator): idle grading via drain_ungraded_tasks, fix mission completion"
```

---

## Task 10: Agent Base — Worker Completion Flow

**Files:**
- Modify: `src/agents/base.py`

- [ ] **Step 1: Update worker phase completion**

In `base.py`, find the grading section (around line 1576-1659). Replace with:

```python
                # Grade ALL completed tasks — quality signal feeds model selection
                quality_score = None
                grader_data = {}
                _iterations = iteration + 1
                try:
                    from src.core.llm_dispatcher import get_dispatcher
                    from src.core.grading import grade_task, apply_grade_result, GradeResult

                    dispatcher = get_dispatcher()
                    loaded = dispatcher._get_loaded_litellm_name()
                    generating = used_model

                    # Can we grade immediately? (loaded model != generating)
                    can_grade_now = (
                        loaded and generating != loaded
                    ) or reqs.priority >= 8

                    if can_grade_now:
                        try:
                            verdict = await grade_task(task, loaded or "")
                            quality_score = verdict.score
                            if verdict.passed:
                                # Will be set to completed by _handle_complete
                                pass
                            else:
                                # Grade FAIL — apply immediately
                                await apply_grade_result(task_id, verdict)
                                await self._clear_checkpoint_safe(task_id)
                                return {
                                    "status": "pending",  # retry signal
                                    "result": result,
                                    "model": used_model,
                                    "quality_score": quality_score,
                                }
                        except (ValueError, RuntimeError):
                            # Grading failed — defer
                            can_grade_now = False

                    if not can_grade_now:
                        # Defer grading — set to ungraded
                        import json as _json
                        from datetime import datetime as _dt
                        _ctx = task.get("context", "{}")
                        if isinstance(_ctx, str):
                            try:
                                _ctx = _json.loads(_ctx)
                            except (ValueError, TypeError):
                                _ctx = {}
                        _ctx["generating_model"] = used_model
                        _ctx["worker_completed_at"] = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                        _ctx["tools_used_names"] = sorted(tools_used_names)

                        from src.core.state_machine import transition_task
                        await transition_task(
                            task_id, "ungraded",
                            context=_json.dumps(_ctx),
                        )
                        await self._clear_checkpoint_safe(task_id)
                        return {
                            "status": "ungraded",
                            "result": result,
                            "model": used_model,
                            "iterations": iteration + 1,
                            "tools_used_names": sorted(tools_used_names),
                        }

                except Exception as exc:
                    logger.warning(f"grading failed | task_id={task_id} error={exc}")
```

- [ ] **Step 2: Update orchestrator `process_task` to handle `ungraded` status**

In `orchestrator.py`, in the status dispatch section (around line 1695), add handling for `ungraded`:

```python
            elif status == "ungraded":
                # Task deferred to grading phase — store result, don't notify yet
                await update_task(
                    task_id, status="ungraded", result=result_text,
                    cost=cost,
                )
                logger.info("task ungraded (deferred grading)", task_id=task_id, model=model)
```

- [ ] **Step 3: Test with import check**

Run: `python -c "from src.agents.base import BaseAgent; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/agents/base.py src/core/orchestrator.py
git commit -m "feat(base): worker completion → ungraded flow with deferred grading"
```

---

## Task 11: Dead Letter Queue — Phase-Aware Retry

**Files:**
- Modify: `src/infra/dead_letter.py`
- Modify: `tests/test_dead_letter.py`

- [ ] **Step 1: Write test for phase-aware DLQ retry**

```python
# Add to tests/test_dead_letter.py
class TestPhaseAwareRetry:
    def test_worker_phase_retry_goes_to_pending(self):
        """DLQ retry for worker-phase failure → pending."""
        mock_db = _make_mock_db()
        mock_task = {
            "id": 1, "status": "failed", "failed_in_phase": "worker",
            "context": '{"excluded_models":["m1"],"last_avail_delay":300}',
        }
        mock_db.execute.return_value.fetchone = AsyncMock(return_value=mock_task)

        with patch(DB_PATCH, AsyncMock(return_value=mock_db)):
            with patch("src.infra.dead_letter.resolve_dlq_task", AsyncMock()):
                with patch("src.infra.db.update_task", AsyncMock()) as mock_update:
                    with patch("src.infra.db.get_task", AsyncMock(return_value=mock_task)):
                        run_async(retry_dlq_task(1))

        # Verify status and resets
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["status"] == "pending"

    def test_grading_phase_retry_goes_to_ungraded(self):
        """DLQ retry for grading-phase failure → ungraded."""
        mock_db = _make_mock_db()
        mock_task = {
            "id": 2, "status": "failed", "failed_in_phase": "grading",
            "context": '{"generating_model":"m1","grade_excluded_models":["m2"]}',
        }
        mock_db.execute.return_value.fetchone = AsyncMock(return_value=mock_task)

        with patch(DB_PATCH, AsyncMock(return_value=mock_db)):
            with patch("src.infra.dead_letter.resolve_dlq_task", AsyncMock()):
                with patch("src.infra.db.update_task", AsyncMock()) as mock_update:
                    with patch("src.infra.db.get_task", AsyncMock(return_value=mock_task)):
                        run_async(retry_dlq_task(2))

        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["status"] == "ungraded"
```

- [ ] **Step 2: Update `retry_dlq_task`**

```python
async def retry_dlq_task(task_id: int) -> bool:
    """Re-queue a dead-letter task for another attempt.

    Phase-aware: restores to the phase where the task failed.
    Resets exclusions and backoff, preserves attempt counters.
    """
    import json
    from src.infra.db import get_task, update_task

    await resolve_dlq_task(task_id, resolution="retry")

    task = await get_task(task_id)
    if not task:
        logger.warning(f"[DLQ] Task #{task_id} not found for retry")
        return False

    # Phase-aware status
    failed_phase = task.get("failed_in_phase")
    new_status = "ungraded" if failed_phase == "grading" else "pending"

    # Reset exclusions and backoff in context
    ctx = {}
    try:
        ctx = json.loads(task.get("context") or "{}")
    except (json.JSONDecodeError, TypeError):
        pass

    ctx["last_avail_delay"] = 0
    ctx["excluded_models"] = []
    ctx["grade_excluded_models"] = []
    # Keep generating_model (prevents self-grading)

    await update_task(
        task_id,
        status=new_status,
        next_retry_at=None,
        retry_reason=None,
        context=json.dumps(ctx),
    )
    logger.info(f"[DLQ] Task #{task_id} re-queued → {new_status} (phase={failed_phase})")
    return True
```

- [ ] **Step 3: Run DLQ tests**

Run: `pytest tests/test_dead_letter.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/infra/dead_letter.py tests/test_dead_letter.py
git commit -m "feat(dlq): phase-aware retry — grading failures restore to ungraded"
```

---

## Task 12: Workflow Hooks — Remove `_schema_retry_count`

**Files:**
- Modify: `src/workflows/engine/hooks.py`

- [ ] **Step 1: Find and replace `_schema_retry_count`**

Search for `_schema_retry_count` in hooks.py. Replace with usage of the task's `attempts` column. The retry is handled by the unified `compute_retry_timing` path now — the hook just needs to report the failure, not manage its own counter.

In the schema validation section, change from:
```python
schema_retries = ctx.get("_schema_retry_count", 0)
if schema_retries < 3:
    ctx["_schema_retry_count"] = schema_retries + 1
    # ... retry logic
```

To simply letting the failure propagate as a quality failure — the orchestrator's unified retry path handles the rest.

- [ ] **Step 2: Test with import check**

Run: `python -c "from src.workflows.engine.hooks import post_execute_workflow_step; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/workflows/engine/hooks.py
git commit -m "refactor(hooks): remove _schema_retry_count, use unified attempts"
```

---

## Task 13: Telegram Bot — Status Display Updates

**Files:**
- Modify: `src/app/telegram_bot.py`

- [ ] **Step 1: Update status display strings**

Search and replace across telegram_bot.py:
- `needs_clarification` → `waiting_human` in all status checks and display strings
- `rejected` → `cancelled` in all status checks
- Add `ungraded` display: `"Bekliyor: Derecelendirme..."`
- Remove `paused` and `sleeping` from status displays

- [ ] **Step 2: Update DLQ retry handler**

In `cmd_dlq` (around line 3108) and inline handler (around line 5129), update retry to be phase-aware — the `retry_dlq_task` function already handles phase awareness, so the Telegram handler just calls it.

- [ ] **Step 3: Remove paused/sleeping button handlers**

Remove the "Resume" handler for paused tasks. Update "Retry" button to show only for `failed` tasks.

- [ ] **Step 4: Update task status counts in `/tasks`**

Remove sleeping/paused counts, add `ungraded` count.

- [ ] **Step 5: Add grade notifications to `apply_grade_result`**

In `src/core/grading.py`, in `apply_grade_result`, after the PASS/FAIL transitions, add Telegram notifications:

```python
    # Telegram notification for grade results
    try:
        from src.app.telegram_bot import get_bot
        bot = get_bot()
        if bot:
            task_ctx_raw = task.get("context", "{}")
            if isinstance(task_ctx_raw, str):
                try:
                    _tctx = json.loads(task_ctx_raw)
                except (json.JSONDecodeError, TypeError):
                    _tctx = {}
            else:
                _tctx = task_ctx_raw

            is_silent = _tctx.get("silent", False)
            if not is_silent:
                if verdict.passed:
                    await bot.send_notification(
                        f"✅ Görev #{task_id} derecelendirildi ve tamamlandı\n"
                        f"**{task.get('title', '')[:60]}**"
                    )
                else:
                    await bot.send_notification(
                        f"🔄 Görev #{task_id} çıktısı reddedildi, farklı model ile tekrar deniyor\n"
                        f"**{task.get('title', '')[:60]}**"
                    )
    except Exception:
        pass  # non-critical
```

- [ ] **Step 6: Test with import check**

Run: `python -c "from src.app.telegram_bot import TelegramInterface; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/app/telegram_bot.py src/core/grading.py
git commit -m "refactor(telegram): update status displays, add grade notifications"
```

---

## Task 14: Update All `wake_sleeping_tasks` Callers

**Files:**
- Modify: `src/models/rate_limiter.py`
- Modify: `src/models/quota_planner.py`
- Modify: `src/models/gpu_scheduler.py`
- Modify: `src/models/local_model_manager.py`

- [ ] **Step 1: Replace all `wake_sleeping_tasks` calls with `accelerate_retries`**

In each file, change:
```python
from src.infra.db import wake_sleeping_tasks
await wake_sleeping_tasks("reason")
# or
asyncio.ensure_future(wake_sleeping_tasks("reason"))
```

To:
```python
from src.infra.db import accelerate_retries
await accelerate_retries("reason")
# or
asyncio.ensure_future(accelerate_retries("reason"))
```

Files and locations:
- `src/models/rate_limiter.py:42-43`
- `src/models/quota_planner.py:112-113`
- `src/models/gpu_scheduler.py:188-189`
- `src/models/local_model_manager.py:315-316`

- [ ] **Step 2: Remove sleeping queue imports from orchestrator**

In `src/core/orchestrator.py`, remove from the import block:
```python
    get_sleeping_tasks, wake_sleeping_tasks, make_sleep_state,
    compute_next_timer_wake, _SLEEP_TIER_INTERVALS,
```

- [ ] **Step 3: Remove sleeping queue functions from db.py**

Delete these functions from `src/infra/db.py`:
- `get_sleeping_tasks()` (line 933)
- `wake_sleeping_tasks()` (line 942 — already replaced by `accelerate_retries`)
- `compute_next_timer_wake()` (line 984)
- `make_sleep_state()` (line 994)
- `_SLEEP_TIER_INTERVALS` constant (line 927)
- `_MAX_SIGNAL_FAILURES` constant (line 930)

- [ ] **Step 4: Test all imports**

```bash
python -c "from src.models.rate_limiter import *; print('OK')"
python -c "from src.models.quota_planner import *; print('OK')"
python -c "from src.models.gpu_scheduler import *; print('OK')"
python -c "from src.models.local_model_manager import *; print('OK')"
python -c "from src.core.orchestrator import Orchestrator; print('OK')"
```

Expected: All `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/models/rate_limiter.py src/models/quota_planner.py src/models/gpu_scheduler.py src/models/local_model_manager.py src/core/orchestrator.py src/infra/db.py
git commit -m "refactor: replace wake_sleeping_tasks with accelerate_retries everywhere"
```

---

## Task 15: Integration Test — Full Lifecycle

**Files:**
- Create: `tests/integration/test_unified_lifecycle.py`

- [ ] **Step 1: Write end-to-end lifecycle test**

```python
# tests/integration/test_unified_lifecycle.py
"""Integration tests for the unified task lifecycle.

Tests the full flow: pending → processing → ungraded → completed
and failure paths through the unified retry system.
"""
import pytest
import asyncio
import json


@pytest.mark.integration
class TestUnifiedLifecycle:
    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_ungraded_blocks_dependents(self, temp_db):
        async def _test():
            from src.infra.db import add_task, update_task, get_ready_tasks

            parent = await add_task("parent", "work")
            await update_task(parent, status="ungraded",
                              context=json.dumps({"generating_model": "model_a"}))
            child = await add_task("child", "depends", depends_on=[parent])

            ready = await get_ready_tasks()
            ready_ids = [t["id"] for t in ready]
            assert child not in ready_ids

            # Complete parent → child unblocks
            await update_task(parent, status="completed")
            ready = await get_ready_tasks()
            ready_ids = [t["id"] for t in ready]
            assert child in ready_ids

        self._run(_test())

    def test_quality_failure_increments_attempts(self, temp_db):
        async def _test():
            from src.infra.db import add_task, update_task, get_task
            from src.core.grading import apply_grade_result, GradeResult

            tid = await add_task("test", "desc")
            await update_task(tid, status="ungraded",
                              context=json.dumps({"generating_model": "model_a"}))
            await apply_grade_result(tid, GradeResult(passed=False))

            task = await get_task(tid)
            assert task["status"] == "pending"
            assert task["attempts"] == 1
            ctx = json.loads(task["context"])
            assert "model_a" in ctx["failed_models"]

        self._run(_test())

    def test_availability_backoff_doubles(self, temp_db):
        async def _test():
            from src.infra.db import add_task, update_task, get_task
            from src.core.retry import compute_retry_timing
            from datetime import datetime, timedelta

            tid = await add_task("test", "desc")
            # Simulate first availability failure
            decision = compute_retry_timing("availability", last_avail_delay=0)
            assert decision.delay_seconds == 60

            # Second
            decision = compute_retry_timing("availability", last_avail_delay=60)
            assert decision.delay_seconds == 120

            # At cap
            decision = compute_retry_timing("availability", last_avail_delay=7200)
            assert decision.action == "terminal"

        self._run(_test())

    def test_dlq_retry_restores_grading_phase(self, temp_db):
        async def _test():
            from src.infra.db import add_task, update_task, get_task
            from src.infra.dead_letter import retry_dlq_task, quarantine_task

            tid = await add_task("test", "desc")
            await update_task(tid, status="failed", failed_in_phase="grading",
                              context=json.dumps({
                                  "generating_model": "model_a",
                                  "excluded_models": ["model_b"],
                                  "last_avail_delay": 300,
                              }))
            await quarantine_task(tid, None, "test error")

            await retry_dlq_task(tid)
            task = await get_task(tid)
            assert task["status"] == "ungraded"  # not pending!
            ctx = json.loads(task["context"])
            assert ctx["excluded_models"] == []
            assert ctx["last_avail_delay"] == 0
            assert ctx["generating_model"] == "model_a"  # preserved

        self._run(_test())

    def test_next_retry_at_filters_ready_tasks(self, temp_db):
        async def _test():
            from src.infra.db import add_task, get_db, get_ready_tasks

            tid = await add_task("delayed", "desc")
            db = await get_db()
            await db.execute(
                "UPDATE tasks SET next_retry_at = datetime('now', '+1 hour') WHERE id = ?",
                (tid,),
            )
            await db.commit()

            ready = await get_ready_tasks()
            assert tid not in [t["id"] for t in ready]

        self._run(_test())
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/integration/test_unified_lifecycle.py -v`
Expected: All PASS.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/shopping -m "not llm" -x`
Expected: All PASS (or only pre-existing failures).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_unified_lifecycle.py
git commit -m "test: integration tests for unified task lifecycle"
```

---

## Task 16: Remove Dead Code and Final Cleanup

**Files:**
- Various

- [ ] **Step 1: Remove old sleeping queue code from db.py**

Verify `get_sleeping_tasks`, `wake_sleeping_tasks`, `make_sleep_state`, `compute_next_timer_wake`, `_SLEEP_TIER_INTERVALS`, `_MAX_SIGNAL_FAILURES` are all removed. Should have been done in Task 14 step 3.

- [ ] **Step 2: Remove old grade queue imports**

Search for any remaining imports of `GradeQueue`, `PendingGrade`, `drain_grades_if_idle`, `drain_grades_if_full` across the codebase.

```bash
grep -rn "GradeQueue\|PendingGrade\|drain_grades_if" src/
```

Remove any remaining references.

- [ ] **Step 3: Remove old sleeping/paused references**

```bash
grep -rn "sleeping\|paused\|needs_clarification\|needs_review\b\|rejected" src/ --include="*.py" | grep -v "test\|__pycache__\|\.pyc"
```

Review each match. Some are in comments/docs (OK to leave). Status checks and logic must be updated.

- [ ] **Step 4: Stop writing deprecated columns**

Search for any remaining writes to old columns:

```bash
grep -rn "retry_count\|max_retries\|sleep_state\|error_category" src/ --include="*.py" | grep -v "test\|__pycache__\|migration\|COALESCE\|deprecated"
```

For each write site:
- `retry_count=X` → remove (or replace with `attempts=X` if not already done)
- `max_retries=X` → remove (or replace with `max_attempts=X`)
- `sleep_state=X` → remove (replaced by `next_retry_at` + `context.last_avail_delay`)
- `error_category=X` → remove (replaced by `retry_reason`)

Keep READ access to deprecated columns (backward compat for old data display).

- [ ] **Step 4: Update CLAUDE.md**

Update the Common Pitfalls section:
- Replace "sleeping queue" references with `next_retry_at` backoff
- Add: `ungraded` state blocks dependents until grading completes
- Add: All status changes go through `transition_task()` — never raw `update_task(status=...)`

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/shopping -m "not llm" -x`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove dead sleeping/grade queue code, update CLAUDE.md"
```
