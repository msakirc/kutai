# Retry Pipeline Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix iteration starvation, silent exhaustion, shared attempt budgets, blind retry scheduling, polluted escalation context, and overloaded terminology across the agent execution and retry pipeline.

**Architecture:** 12 sequenced tasks, each independently testable and rollback-safe. RetryContext dataclass centralizes all retry state. Parallel tool execution within iterations. Guards become sub-iteration corrections. Exhaustion returns distinct status with reason-aware retry.

**Tech Stack:** Python 3.10, asyncio, aiosqlite, dataclasses, pytest

**Spec:** `docs/superpowers/specs/2026-04-07-agent-iteration-exhaustion-fixes.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/core/retry.py` | RetryContext dataclass + existing retry logic (extend) |
| `src/agents/base.py` | Agent loop: parallel tools, sub-iteration guards, exhaustion status |
| `src/core/orchestrator.py` | Task lifecycle: exhaustion handler, swap-aware pickup, infra separation |
| `src/core/grading.py` | RetryContext integration for grade failures |
| `src/infra/db.py` | Schema migration, column renames, accelerate_retries expansion |
| `src/infra/dead_letter.py` | Column rename, RetryContext integration |
| `src/core/llm_dispatcher.py` | Exclusion-aware proactive loading |
| `src/app/config.py` | Constant renames |
| `tests/test_retry_context.py` | RetryContext unit tests (new) |
| `tests/test_parallel_tools.py` | Parallel tool execution tests (new) |
| `tests/test_sub_iteration_guards.py` | Guard sub-iteration tests (new) |
| `tests/test_exhaustion.py` | Exhaustion handling tests (new) |
| `tests/test_swap_aware_scheduling.py` | Swap-aware retry scheduling tests (new) |
| `tests/test_escalation_reset.py` | Escalation context trim tests (new) |
| `tests/test_db_migration.py` | DB migration tests (new) |
| `tests/test_infra_vs_quality.py` | Infra/quality separation tests (new) |
| `tests/integration/test_retry_pipeline_integration.py` | End-to-end integration test (new) |

---

### Task 1: RetryContext Dataclass

**Files:**
- Create: `tests/test_retry_context.py`
- Modify: `src/core/retry.py`

- [ ] **Step 1: Write failing tests for RetryContext**

```python
# tests/test_retry_context.py
import pytest
from src.core.retry import RetryContext, RetryDecision


class TestRetryContextCreation:
    def test_defaults(self):
        ctx = RetryContext()
        assert ctx.worker_attempts == 0
        assert ctx.infra_resets == 0
        assert ctx.max_worker_attempts == 6
        assert ctx.grade_attempts == 0
        assert ctx.max_grade_attempts == 3
        assert ctx.failed_models == []
        assert ctx.format_corrections == 0
        assert ctx.consecutive_tool_failures == 0
        assert ctx.model_escalated is False
        assert ctx.guard_burns == 0
        assert ctx.useful_iterations == 0
        assert ctx.exhaustion_reason is None

    def test_from_task_fresh(self):
        task = {"id": 1, "status": "pending", "context": "{}"}
        ctx = RetryContext.from_task(task)
        assert ctx.worker_attempts == 0
        assert ctx.infra_resets == 0

    def test_from_task_with_legacy_attempts(self):
        """Backwards compat: reads 'attempts' if 'worker_attempts' missing."""
        task = {"id": 1, "attempts": 3, "max_attempts": 6, "context": "{}"}
        ctx = RetryContext.from_task(task)
        assert ctx.worker_attempts == 3
        assert ctx.max_worker_attempts == 6

    def test_from_task_with_new_columns(self):
        task = {
            "id": 1,
            "worker_attempts": 2,
            "infra_resets": 1,
            "max_worker_attempts": 6,
            "grade_attempts": 1,
            "context": '{"failed_models": ["modelA"]}',
        }
        ctx = RetryContext.from_task(task)
        assert ctx.worker_attempts == 2
        assert ctx.infra_resets == 1
        assert ctx.failed_models == ["modelA"]

    def test_from_task_with_dict_context(self):
        task = {
            "id": 1,
            "context": {"failed_models": ["x"], "grade_excluded_models": ["y"]},
        }
        ctx = RetryContext.from_task(task)
        assert ctx.failed_models == ["x"]
        assert ctx.grade_excluded_models == ["y"]


class TestRetryContextProperties:
    def test_total_attempts(self):
        ctx = RetryContext(worker_attempts=3, infra_resets=2)
        assert ctx.total_attempts == 5

    def test_effective_difficulty_bump_below_threshold(self):
        ctx = RetryContext(worker_attempts=2)
        assert ctx.effective_difficulty_bump == 0

    def test_effective_difficulty_bump_at_threshold(self):
        ctx = RetryContext(worker_attempts=4)
        assert ctx.effective_difficulty_bump == 2

    def test_effective_difficulty_bump_high(self):
        ctx = RetryContext(worker_attempts=6)
        assert ctx.effective_difficulty_bump == 6

    def test_excluded_models_below_threshold(self):
        ctx = RetryContext(worker_attempts=2, failed_models=["a", "b"])
        assert ctx.excluded_models == []

    def test_excluded_models_at_threshold(self):
        ctx = RetryContext(worker_attempts=3, failed_models=["a", "b"])
        assert ctx.excluded_models == ["a", "b"]


class TestRetryContextRecordFailure:
    def test_record_quality_failure(self):
        ctx = RetryContext(worker_attempts=0, max_worker_attempts=6)
        decision = ctx.record_failure("quality", model="modelA")
        assert ctx.worker_attempts == 1
        assert "modelA" in ctx.failed_models
        assert decision.action == "immediate"

    def test_record_quality_failure_terminal(self):
        ctx = RetryContext(worker_attempts=5, max_worker_attempts=6)
        decision = ctx.record_failure("quality", model="modelF")
        assert ctx.worker_attempts == 6
        assert decision.action == "terminal"

    def test_record_quality_failure_delayed(self):
        ctx = RetryContext(worker_attempts=2, max_worker_attempts=6)
        decision = ctx.record_failure("quality", model="m")
        assert ctx.worker_attempts == 3
        assert decision.action == "delayed"
        assert decision.delay_seconds == 600

    def test_record_infrastructure_failure(self):
        ctx = RetryContext(infra_resets=0)
        decision = ctx.record_failure("infrastructure")
        assert ctx.infra_resets == 1
        assert ctx.worker_attempts == 0
        assert decision.action == "immediate"

    def test_record_infrastructure_terminal(self):
        ctx = RetryContext(infra_resets=2)
        decision = ctx.record_failure("infrastructure")
        assert ctx.infra_resets == 3
        assert decision.action == "terminal"

    def test_record_exhaustion_budget(self):
        ctx = RetryContext(worker_attempts=0, useful_iterations=6, max_iterations=8)
        decision = ctx.record_failure("exhaustion", model="m")
        assert ctx.worker_attempts == 1
        assert ctx.exhaustion_reason == "budget"

    def test_record_exhaustion_guards(self):
        ctx = RetryContext(worker_attempts=0, guard_burns=5, max_iterations=8)
        decision = ctx.record_failure("exhaustion", model="m")
        assert ctx.exhaustion_reason == "guards"

    def test_record_exhaustion_tool_failures(self):
        ctx = RetryContext(worker_attempts=0, guard_burns=1, max_iterations=8,
                           consecutive_tool_failures=3)
        decision = ctx.record_failure("exhaustion", model="m")
        assert ctx.exhaustion_reason == "tool_failures"


class TestRetryContextSerialization:
    def test_to_db_fields(self):
        ctx = RetryContext(worker_attempts=3, infra_resets=1, max_worker_attempts=6,
                          retry_reason="quality", failed_in_phase="worker")
        fields = ctx.to_db_fields()
        assert fields["worker_attempts"] == 3
        assert fields["infra_resets"] == 1
        assert fields["retry_reason"] == "quality"
        assert "failed_models" not in fields

    def test_to_context_patch(self):
        ctx = RetryContext(failed_models=["a"], grade_excluded_models=["b"])
        patch = ctx.to_context_patch()
        assert patch["failed_models"] == ["a"]
        assert patch["grade_excluded_models"] == ["b"]

    def test_to_checkpoint(self):
        ctx = RetryContext(iteration=3, format_corrections=1, model_escalated=True,
                          guard_burns=2, useful_iterations=3)
        cp = ctx.to_checkpoint()
        assert cp["iteration"] == 3
        assert cp["format_corrections"] == 1
        assert cp["model_escalated"] is True
        assert cp["guard_burns"] == 2

    def test_roundtrip_from_task(self):
        import json
        ctx = RetryContext(worker_attempts=2, failed_models=["m1"],
                          grade_excluded_models=["m2"])
        task = {**ctx.to_db_fields(), "context": json.dumps(ctx.to_context_patch())}
        restored = RetryContext.from_task(task)
        assert restored.worker_attempts == 2
        assert restored.failed_models == ["m1"]


class TestRetryContextGuardTracking:
    def test_record_guard_burn(self):
        ctx = RetryContext()
        ctx.record_guard_burn("hallucination")
        ctx.record_guard_burn("search_required")
        assert ctx.guard_burns == 2

    def test_record_useful_iteration(self):
        ctx = RetryContext()
        ctx.record_useful_iteration()
        ctx.record_useful_iteration()
        assert ctx.useful_iterations == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retry_context.py -v`
Expected: FAIL — `ImportError: cannot import name 'RetryContext' from 'src.core.retry'`

- [ ] **Step 3: Implement RetryContext in retry.py**

Add to `src/core/retry.py` after the existing `RetryDecision` class (line 28) and before `compute_retry_timing` (line 31). Keep all existing functions — this is purely additive.

```python
from dataclasses import dataclass, field
import json as _json


@dataclass
class RetryContext:
    """Unified retry state for a task across all lifecycle phases."""

    # ── Task-level (persisted as DB columns) ──
    worker_attempts: int = 0
    infra_resets: int = 0
    max_worker_attempts: int = 6
    grade_attempts: int = 0
    max_grade_attempts: int = 3
    next_retry_at: str | None = None
    retry_reason: str | None = None
    failed_in_phase: str | None = None

    # ── Model tracking (persisted in task.context JSON) ──
    failed_models: list[str] = field(default_factory=list)
    grade_excluded_models: list[str] = field(default_factory=list)

    # ── Iteration-level (persisted in checkpoint) ──
    iteration: int = 0
    max_iterations: int = 8
    format_corrections: int = 0
    consecutive_tool_failures: int = 0
    model_escalated: bool = False
    guard_burns: int = 0
    useful_iterations: int = 0

    # ── Exhaustion tracking ──
    exhaustion_reason: str | None = None

    @property
    def total_attempts(self) -> int:
        return self.worker_attempts + self.infra_resets

    @property
    def effective_difficulty_bump(self) -> int:
        if self.worker_attempts >= 4:
            return max(0, (self.worker_attempts - 3) * 2)
        return 0

    @property
    def excluded_models(self) -> list[str]:
        return list(self.failed_models) if self.worker_attempts >= 3 else []

    def record_failure(self, failure_type: str, model: str = "") -> "RetryDecision":
        if failure_type == "infrastructure":
            self.infra_resets += 1
            if self.infra_resets >= 3:
                self.failed_in_phase = "infrastructure"
                return RetryDecision.terminal()
            self.retry_reason = "infrastructure"
            return RetryDecision.immediate()

        if failure_type == "exhaustion":
            if self.guard_burns >= self.max_iterations * 0.5:
                self.exhaustion_reason = "guards"
            elif self.consecutive_tool_failures >= 3:
                self.exhaustion_reason = "tool_failures"
            else:
                self.exhaustion_reason = "budget"
            self.worker_attempts += 1
            if model and model not in self.failed_models:
                self.failed_models.append(model)
            self.failed_in_phase = "worker"
            return compute_retry_timing(
                "quality", attempts=self.worker_attempts,
                max_attempts=self.max_worker_attempts,
            )

        if failure_type in ("quality", "timeout"):
            self.worker_attempts += 1
            if model and model not in self.failed_models:
                self.failed_models.append(model)
            self.failed_in_phase = "worker"
            self.retry_reason = failure_type
            return compute_retry_timing(
                "quality", attempts=self.worker_attempts,
                max_attempts=self.max_worker_attempts,
            )

        if failure_type == "availability":
            self.worker_attempts += 1
            self.retry_reason = "availability"
            return compute_retry_timing("availability", last_avail_delay=0)

        raise ValueError(f"Unknown failure_type: {failure_type}")

    def record_guard_burn(self, guard_name: str) -> None:
        self.guard_burns += 1

    def record_useful_iteration(self) -> None:
        self.useful_iterations += 1

    def to_db_fields(self) -> dict:
        return {
            "worker_attempts": self.worker_attempts,
            "infra_resets": self.infra_resets,
            "max_worker_attempts": self.max_worker_attempts,
            "grade_attempts": self.grade_attempts,
            "max_grade_attempts": self.max_grade_attempts,
            "next_retry_at": self.next_retry_at,
            "retry_reason": self.retry_reason,
            "failed_in_phase": self.failed_in_phase,
            "exhaustion_reason": self.exhaustion_reason,
        }

    def to_context_patch(self) -> dict:
        return {
            "failed_models": self.failed_models,
            "grade_excluded_models": self.grade_excluded_models,
        }

    def to_checkpoint(self) -> dict:
        return {
            "iteration": self.iteration,
            "format_corrections": self.format_corrections,
            "consecutive_tool_failures": self.consecutive_tool_failures,
            "model_escalated": self.model_escalated,
            "guard_burns": self.guard_burns,
            "useful_iterations": self.useful_iterations,
        }

    @classmethod
    def from_task(cls, task: dict) -> "RetryContext":
        raw_ctx = task.get("context", {})
        if isinstance(raw_ctx, str):
            try:
                ctx = _json.loads(raw_ctx)
            except (ValueError, TypeError):
                ctx = {}
        elif isinstance(raw_ctx, dict):
            ctx = raw_ctx
        else:
            ctx = {}

        worker_attempts = task.get("worker_attempts")
        if worker_attempts is None:
            worker_attempts = task.get("attempts", 0) or 0

        max_worker_attempts = task.get("max_worker_attempts")
        if max_worker_attempts is None:
            max_worker_attempts = task.get("max_attempts", 6) or 6

        return cls(
            worker_attempts=worker_attempts,
            infra_resets=task.get("infra_resets", 0) or 0,
            max_worker_attempts=max_worker_attempts,
            grade_attempts=task.get("grade_attempts", 0) or 0,
            max_grade_attempts=task.get("max_grade_attempts", 3) or 3,
            next_retry_at=task.get("next_retry_at"),
            retry_reason=task.get("retry_reason"),
            failed_in_phase=task.get("failed_in_phase"),
            failed_models=ctx.get("failed_models", []),
            grade_excluded_models=ctx.get("grade_excluded_models", []),
            exhaustion_reason=task.get("exhaustion_reason"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retry_context.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/retry.py tests/test_retry_context.py
git commit -m "feat(retry): add RetryContext dataclass with unified state management"
```

---

### Task 2: Terminology Renames — Constants and Local Variables

**Files:**
- Modify: `src/agents/base.py:66,70,1190,1193-1194,1382,1909,1922`

- [ ] **Step 1: Rename constants**

In `src/agents/base.py` line 65-66, replace:
```python
# Max JSON format-correction retries before falling through to final_answer.
MAX_FORMAT_RETRIES: int = 2
```
with:
```python
# Max JSON format-corrections (sub-iteration) before falling through to final_answer.
MAX_FORMAT_CORRECTIONS: int = 2
```

Lines 68-70, replace:
```python
# Mid-task escalation: after this many iterations with tool failures,
# escalate to the next tier up.
ESCALATION_THRESHOLD: int = 3
```
with:
```python
# Mid-task model escalation: after this many iterations with consecutive
# tool failures, bump model quality requirements.
TOOL_FAILURE_ESCALATION_THRESHOLD: int = 3
```

- [ ] **Step 2: Rename local variables and all references**

Use find-and-replace within `src/agents/base.py`:
- `format_retries` → `format_corrections` (all ~15 occurrences)
- `MAX_FORMAT_RETRIES` → `MAX_FORMAT_CORRECTIONS` (all ~4 occurrences)
- `ESCALATION_THRESHOLD` → `TOOL_FAILURE_ESCALATION_THRESHOLD` (all ~2 occurrences)

For the `escalated` local variable (lines 1194, 1909, 1922), rename to `model_escalated`. Be careful: only rename the LOCAL variable, not method names containing "escalat".

- [ ] **Step 3: Update checkpoint compat**

At checkpoint restore (line ~1140), change:
```python
            format_retries = checkpoint.get("format_retries", 0)
```
to:
```python
            format_corrections = checkpoint.get("format_corrections",
                                                 checkpoint.get("format_retries", 0))
```

At checkpoint save (line ~2397), change `"format_retries"` key to `"format_corrections"`.

- [ ] **Step 4: Update the log message at line ~1382**

Change `f"retry {format_retries}/{MAX_FORMAT_RETRIES}"` to `f"format-correction {format_corrections}/{MAX_FORMAT_CORRECTIONS}"`.

- [ ] **Step 5: Run existing tests**

Run: `pytest tests/ -x -q --timeout=30`
Expected: ALL PASS (pure rename)

- [ ] **Step 6: Commit**

```bash
git add src/agents/base.py
git commit -m "refactor: rename format_retries, ESCALATION_THRESHOLD, escalated for clarity"
```

---

### Task 3: DB Migration — New Columns and Renames

**Files:**
- Modify: `src/infra/db.py:130-154,583-599`
- Modify: `src/infra/dead_letter.py:52-66,69-90`
- Create: `tests/test_db_migration.py`

- [ ] **Step 1: Write migration test**

```python
# tests/test_db_migration.py
import pytest
import asyncio


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_migration_adds_new_columns():
    async def _test():
        import tempfile, os, importlib
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        os.environ["DB_PATH"] = db_path
        import src.infra.db as db_mod
        importlib.reload(db_mod)
        await db_mod._init_db()
        db = await db_mod.get_db()
        cursor = await db.execute("PRAGMA table_info(tasks)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "worker_attempts" in columns
        assert "infra_resets" in columns
        assert "exhaustion_reason" in columns
        assert "max_worker_attempts" in columns
    run_async(_test())
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_db_migration.py -v`
Expected: FAIL

- [ ] **Step 3: Update CREATE TABLE in db.py**

In the tasks table CREATE statement (line ~130-154), change column names:
- `attempts INTEGER DEFAULT 0` → `worker_attempts INTEGER DEFAULT 0`
- `max_attempts INTEGER DEFAULT 6` → `max_worker_attempts INTEGER DEFAULT 6`

Add after `failed_in_phase TEXT`:
- `infra_resets INTEGER DEFAULT 0,`
- `exhaustion_reason TEXT,`

- [ ] **Step 4: Add migration block in db.py**

After existing unified lifecycle migration (line ~599), add:

```python
    # Migration: Retry Pipeline Overhaul
    if "worker_attempts" not in columns:
        for sql in [
            "ALTER TABLE tasks RENAME COLUMN attempts TO worker_attempts",
            "ALTER TABLE tasks RENAME COLUMN max_attempts TO max_worker_attempts",
            "ALTER TABLE tasks ADD COLUMN infra_resets INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN exhaustion_reason TEXT",
        ]:
            try:
                await db.execute(sql)
                await db.commit()
            except Exception:
                pass
        logger.info("Applied retry pipeline overhaul migration")
```

- [ ] **Step 5: Update dead_letter.py schema**

In CREATE TABLE (line 59): `retry_count` → `attempts_snapshot`.
In `quarantine_task` param (line 75): `retry_count` → `attempts_snapshot`.
In INSERT statement: update column name.

Add migration in `_ensure_dlq_table`:
```python
    try:
        cursor = await db.execute("PRAGMA table_info(dead_letter_tasks)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "retry_count" in cols and "attempts_snapshot" not in cols:
            await db.execute("ALTER TABLE dead_letter_tasks RENAME COLUMN retry_count TO attempts_snapshot")
            await db.commit()
    except Exception:
        pass
```

- [ ] **Step 6: Update orchestrator SQL queries for new column names**

In watchdog (line ~451): `attempts` → `worker_attempts`, `max_attempts` → `max_worker_attempts`.
Add `infra_resets` to SELECT.

All `retry_count=attempts` calls to `quarantine_task` → `attempts_snapshot=attempts` (~4 sites).

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_db_migration.py tests/test_retry.py tests/test_dead_letter.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/infra/db.py src/infra/dead_letter.py src/core/orchestrator.py tests/test_db_migration.py
git commit -m "feat(db): migrate to worker_attempts/infra_resets/attempts_snapshot columns"
```

---

### Task 4: Separate Infrastructure vs Quality Attempts

**Files:**
- Modify: `src/core/orchestrator.py:456-482,3258-3263`
- Create: `tests/test_infra_vs_quality.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_infra_vs_quality.py
from src.core.retry import RetryContext


def test_infra_does_not_increment_worker():
    ctx = RetryContext(worker_attempts=2, infra_resets=0)
    ctx.record_failure("infrastructure")
    assert ctx.worker_attempts == 2
    assert ctx.infra_resets == 1

def test_infra_terminal_at_3():
    ctx = RetryContext(infra_resets=2)
    d = ctx.record_failure("infrastructure")
    assert d.action == "terminal"
    assert ctx.failed_in_phase == "infrastructure"

def test_budgets_independent():
    ctx = RetryContext(worker_attempts=5, infra_resets=2)
    d = ctx.record_failure("quality", model="m")
    assert d.action == "terminal"  # quality at 6/6
    ctx2 = RetryContext(worker_attempts=5, infra_resets=2)
    d2 = ctx2.record_failure("infrastructure")
    assert d2.action == "terminal"  # infra at 3/3
```

- [ ] **Step 2: Run — should pass (RetryContext already implements)**

Run: `pytest tests/test_infra_vs_quality.py -v`
Expected: ALL PASS

- [ ] **Step 3: Update watchdog in orchestrator.py (lines 456-482)**

Replace with RetryContext-based logic that increments `infra_resets` instead of `attempts`. Use `retry_reason='infrastructure'` and `failed_in_phase='infrastructure'`.

- [ ] **Step 4: Update module resumption (lines 3258-3263)**

Change from incrementing `attempts` to incrementing `infra_resets`. Update SELECT to include `infra_resets`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_infra_vs_quality.py tests/test_stuck_tasks.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/core/orchestrator.py tests/test_infra_vs_quality.py
git commit -m "feat(retry): separate infrastructure resets from quality attempts"
```

---

### Task 5: Guards as Sub-Iteration Corrections

**Files:**
- Modify: `src/agents/base.py:1197-1565`
- Create: `tests/test_sub_iteration_guards.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sub_iteration_guards.py
from src.agents.base import BaseAgent, GuardCorrection


def test_hallucination_guard_returns_correction():
    agent = BaseAgent()
    agent.allowed_tools = ["shell", "read_file"]
    parsed = {"action": "final_answer", "result": "I would run ls"}
    task = {"title": "List files in /tmp"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task=task,
        search_depth="none", suppress_guards=False,
    )
    assert c is not None
    assert c.guard_name == "hallucination"

def test_no_guard_on_tool_call():
    agent = BaseAgent()
    parsed = {"action": "tool_call", "tool": "shell", "args": {}}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "t"},
        search_depth="none", suppress_guards=False,
    )
    assert c is None

def test_suppress_guards_flag():
    agent = BaseAgent()
    agent.allowed_tools = ["shell"]
    parsed = {"action": "final_answer", "result": "x"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "Do thing"},
        search_depth="none", suppress_guards=True,
    )
    assert c is None

def test_search_guard_fires():
    agent = BaseAgent()
    agent.allowed_tools = ["web_search", "shell"]
    parsed = {"action": "final_answer", "result": "answer"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "Tokyo time"},
        search_depth="standard", suppress_guards=False,
    )
    assert c is not None
    assert c.guard_name == "search_required"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_sub_iteration_guards.py -v`
Expected: FAIL — `_check_sub_iteration_guards` not found

- [ ] **Step 3: Add GuardCorrection dataclass and _check_sub_iteration_guards method**

Add `GuardCorrection` dataclass and the `_check_sub_iteration_guards` method to `BaseAgent`. Extract the hallucination guard, search-required guard, and blocked clarification guard logic from the main loop into this method. Method returns `GuardCorrection | None`.

- [ ] **Step 4: Restructure main loop with inner correction loop**

Wrap the LLM call + parse + guard check section in a `while sub_corrections <= MAX_SUB_CORRECTIONS:` inner loop. Guards that fire append correction messages and `continue` the inner loop (same iteration). The outer loop only advances for actual tool execution or accepted final_answer.

Move custom validation and task-type validation into the inner loop as well.

Remove the old inline guard blocks that used `continue` on the outer loop.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_sub_iteration_guards.py tests/test_iteration_exhaustion.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/agents/base.py tests/test_sub_iteration_guards.py
git commit -m "feat(agent): guards as sub-iteration corrections, no longer burn iterations"
```

---

### Task 6: Parallel Tool Execution

**Files:**
- Modify: `src/agents/base.py:965-1000,200-210,708-740,1734-1900`
- Create: `tests/test_parallel_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_parallel_tools.py
from src.agents.base import BaseAgent


class TestParseMultiToolCall:
    def test_single_unchanged(self):
        r = BaseAgent._parse_function_call_response([
            {"name": "read_file", "arguments": {"filepath": "a.py"}}
        ])
        assert r["action"] == "tool_call"

    def test_multiple_returns_multi(self):
        r = BaseAgent._parse_function_call_response([
            {"name": "read_file", "arguments": {"filepath": "a.py"}},
            {"name": "read_file", "arguments": {"filepath": "b.py"}},
        ])
        assert r["action"] == "multi_tool_call"
        assert len(r["tools"]) == 2

    def test_final_answer_still_works(self):
        r = BaseAgent._parse_function_call_response([
            {"name": "final_answer", "arguments": {"result": "done"}}
        ])
        assert r["action"] == "final_answer"


class TestPartitionTools:
    def test_all_read_only(self):
        from src.agents.base import _partition_tool_calls
        p, s = _partition_tool_calls([
            {"tool": "read_file", "args": {}}, {"tool": "file_tree", "args": {}},
        ])
        assert len(p) == 2 and len(s) == 0

    def test_all_side_effect(self):
        from src.agents.base import _partition_tool_calls
        p, s = _partition_tool_calls([
            {"tool": "write_file", "args": {}}, {"tool": "shell", "args": {}},
        ])
        assert len(p) == 0 and len(s) == 2

    def test_mixed(self):
        from src.agents.base import _partition_tool_calls
        p, s = _partition_tool_calls([
            {"tool": "read_file", "args": {}}, {"tool": "write_file", "args": {}},
        ])
        assert len(p) == 1 and len(s) == 1

    def test_unknown_is_side_effect(self):
        from src.agents.base import _partition_tool_calls
        p, s = _partition_tool_calls([{"tool": "unknown", "args": {}}])
        assert len(p) == 0 and len(s) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_parallel_tools.py -v`
Expected: FAIL

- [ ] **Step 3: Update _parse_function_call_response for multi-tool**

Replace the method (line 965-1000) to return `multi_tool_call` when `len(tool_calls) > 1` (filtering out pseudo-tools like final_answer/clarify).

- [ ] **Step 4: Add _partition_tool_calls module function**

Partition by `CACHEABLE_READ_TOOLS` (parallel) vs everything else (sequential).

- [ ] **Step 5: Add multi_tool_call to _normalize_action**

Passthrough: `if action == "multi_tool_call" and "tools" in parsed: return parsed`.

- [ ] **Step 6: Add multi_tool_call to system prompt**

In `_get_available_tools_prompt`, add example showing multi_tool_call JSON action.

- [ ] **Step 7: Add multi_tool_call handler in main loop**

After the `if action_type == "tool_call":` block, add `if action_type == "multi_tool_call":` with parallel/sequential execution using `asyncio.gather(return_exceptions=True)` for read-only tools.

- [ ] **Step 8: Run tests**

Run: `pytest tests/test_parallel_tools.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/agents/base.py tests/test_parallel_tools.py
git commit -m "feat(agent): parallel tool execution — read-only tools run concurrently"
```

---

### Task 7: Exhaustion Handling

**Files:**
- Modify: `src/agents/base.py:2068-2104`
- Modify: `src/core/orchestrator.py:1735-1745`
- Create: `tests/test_exhaustion.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_exhaustion.py
from src.core.retry import RetryContext


def test_exhaustion_reason_budget():
    ctx = RetryContext(useful_iterations=6, max_iterations=8)
    ctx.record_failure("exhaustion", model="m")
    assert ctx.exhaustion_reason == "budget"

def test_exhaustion_reason_guards():
    ctx = RetryContext(guard_burns=5, max_iterations=8)
    ctx.record_failure("exhaustion", model="m")
    assert ctx.exhaustion_reason == "guards"

def test_exhaustion_reason_tool_failures():
    ctx = RetryContext(guard_burns=1, max_iterations=8, consecutive_tool_failures=3)
    ctx.record_failure("exhaustion", model="m")
    assert ctx.exhaustion_reason == "tool_failures"

def test_iteration_budget_boost_cap():
    assert min(int(8 * 1.5), 12) == 12
    assert min(int(5 * 1.5), 12) == 7
```

- [ ] **Step 2: Run — should pass**

Run: `pytest tests/test_exhaustion.py -v`
Expected: ALL PASS

- [ ] **Step 3: Update exhaustion block in base.py (lines 2068-2104)**

Replace `return {"status": "completed", ...}` with `return {"status": "exhausted", ...}` including `exhaustion_reason`, `guard_burns`, `useful_iterations`.

- [ ] **Step 4: Add dynamic iteration budget at loop start**

Read `task_ctx.get("iteration_budget_boost", 1.0)`, compute `effective_max_iterations = min(int(max_iterations * boost), 12)`. Use in loop range.

- [ ] **Step 5: Add exhaustion handler in orchestrator.py**

After `elif status == "failed":` block, add `elif status == "exhausted":` with reason-aware retry strategy (budget→boost, guards→suppress, tool_failures→quality retry).

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_exhaustion.py tests/test_retry_context.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/agents/base.py src/core/orchestrator.py tests/test_exhaustion.py
git commit -m "feat: exhausted status with reason-aware retry and dynamic iteration budget"
```

---

### Task 8: Swap-Aware Retry Scheduling

**Files:**
- Modify: `src/core/orchestrator.py:140-226`
- Modify: `src/core/llm_dispatcher.py:537-594`
- Modify: `src/infra/db.py:1121-1124`
- Create: `tests/test_swap_aware_scheduling.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_swap_aware_scheduling.py
import json

def test_defer_when_model_excluded():
    from src.core.orchestrator import _should_defer_for_loaded_model
    task = {"worker_attempts": 3, "context": json.dumps({"failed_models": ["m1"]})}
    assert _should_defer_for_loaded_model(task, "m1") is True

def test_no_defer_below_threshold():
    from src.core.orchestrator import _should_defer_for_loaded_model
    task = {"worker_attempts": 2, "context": json.dumps({"failed_models": ["m1"]})}
    assert _should_defer_for_loaded_model(task, "m1") is False

def test_no_defer_different_model():
    from src.core.orchestrator import _should_defer_for_loaded_model
    task = {"worker_attempts": 3, "context": json.dumps({"failed_models": ["m1"]})}
    assert _should_defer_for_loaded_model(task, "m2") is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_swap_aware_scheduling.py -v`
Expected: FAIL

- [ ] **Step 3: Add _should_defer_for_loaded_model helper to orchestrator.py**

Module-level function that checks `worker_attempts >= 3` and loaded model in `failed_models`.

- [ ] **Step 4: Add exclusion-aware task pickup after get_ready_tasks()**

Partition into runnable/deferred based on loaded model and exclusions.

- [ ] **Step 5: Update _reorder_by_model_affinity**

Set `fit = 0.0` for tasks where `_should_defer_for_loaded_model` returns True.

- [ ] **Step 6: Update _find_best_local_for_batch in llm_dispatcher.py**

Skip models excluded by retry tasks (`worker_attempts >= 3`, model in `failed_models`).

- [ ] **Step 7: Expand accelerate_retries in db.py**

Change `retry_reason = 'availability'` to `retry_reason IN ('availability', 'quality')`.

- [ ] **Step 8: Run tests**

Run: `pytest tests/test_swap_aware_scheduling.py tests/test_orchestrator_routing.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/core/orchestrator.py src/core/llm_dispatcher.py src/infra/db.py tests/test_swap_aware_scheduling.py
git commit -m "feat: swap-aware retry scheduling — exclusion-aware pickup, affinity, proactive loading"
```

---

### Task 9: Escalation Context Reset

**Files:**
- Modify: `src/agents/base.py:1907-1928`
- Create: `tests/test_escalation_reset.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_escalation_reset.py
from src.agents.base import BaseAgent


def test_trim_keeps_system_and_task():
    agent = BaseAgent()
    msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "## Task\nDo something"},
        {"role": "assistant", "content": "reasoning..."},
        {"role": "user", "content": "## Tool Result (`shell`):\n```\nok\n```"},
        {"role": "assistant", "content": "more reasoning"},
    ]
    t = agent._trim_for_escalation(msgs, iteration=4, max_iterations=8)
    assert t[0]["role"] == "system"
    assert "## Task" in t[1]["content"]
    assert any("ok" in m["content"] for m in t)
    assert not any("more reasoning" in m.get("content", "") for m in t)
    assert "previous attempt" in t[-1]["content"].lower()

def test_trim_keeps_last_error():
    agent = BaseAgent()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Task"},
        {"role": "user", "content": "❌ timeout"},
    ]
    t = agent._trim_for_escalation(msgs, iteration=3, max_iterations=8)
    assert any("timeout" in m.get("content", "") for m in t)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_escalation_reset.py -v`
Expected: FAIL

- [ ] **Step 3: Implement _trim_for_escalation on BaseAgent**

Keep system prompt, original task, successful tool results, last error. Strip assistant messages, guard corrections, format retries. Inject escalation context message at end.

- [ ] **Step 4: Wire into escalation block**

After `model_escalated = True`, call `messages = self._trim_for_escalation(messages, iteration, effective_max_iterations)`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_escalation_reset.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/agents/base.py tests/test_escalation_reset.py
git commit -m "feat(agent): trim message context on model escalation for clean handoff"
```

---

### Task 10: Integrate RetryContext Into All Failure Paths

**Files:**
- Modify: `src/core/orchestrator.py` (~8 failure handler blocks)
- Modify: `src/core/grading.py:279-350`

- [ ] **Step 1: Replace orchestrator failure blocks**

Each block like:
```python
attempts = (task.get("attempts") or 0) + 1
max_attempts = task.get("max_attempts") or 6
update_exclusions_on_failure(task_ctx, model, attempts)
decision = compute_retry_timing("quality", attempts=attempts, max_attempts=max_attempts)
```
becomes:
```python
retry_ctx = RetryContext.from_task(task)
decision = retry_ctx.record_failure("quality", model=model)
```

Apply to: timeout (~1682), disguised failure (~1776), ungraded disguised (~1848), suppressed clarification (~1927), explicit failure (~1956), generic exception (~2081).

- [ ] **Step 2: Replace grading.py failure block (lines 279-350)**

Same pattern: `RetryContext.from_task(task)` → `record_failure("quality", model=generating_model)`.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x -q --timeout=60`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/orchestrator.py src/core/grading.py
git commit -m "refactor: replace scattered retry field manipulation with RetryContext"
```

---

### Task 11: Log Message Standardization

**Files:**
- Modify: `src/core/orchestrator.py` (log messages)
- Modify: `src/agents/base.py` (escalation log)

- [ ] **Step 1: Update orchestrator logs**

- `retry {attempts}/{max}` → `worker-retry {retry_ctx.worker_attempts}/{retry_ctx.max_worker_attempts}`
- `resetting (attempt ...)` → `infra-reset {retry_ctx.infra_resets}/3`
- Escalation log: `⬆️ Escalating:` → `⬆️ model-escalation:`

- [ ] **Step 2: Run tests**

Run: `pytest tests/ -x -q --timeout=30`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add src/core/orchestrator.py src/agents/base.py
git commit -m "refactor: standardize log messages — worker-retry, infra-reset, model-escalation"
```

---

### Task 12: Final Integration Test

**Files:**
- Create: `tests/integration/test_retry_pipeline_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_retry_pipeline_integration.py
import pytest
import asyncio
import json


@pytest.mark.integration
def test_retry_context_roundtrip_through_db():
    async def _test():
        import tempfile, os, importlib
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        os.environ["DB_PATH"] = db_path
        import src.infra.db as db_mod
        importlib.reload(db_mod)
        await db_mod._init_db()
        from src.infra.db import add_task, get_task, update_task
        from src.core.retry import RetryContext

        tid = await add_task(title="Test", description="test", agent_type="coder")
        task = await get_task(tid)
        ctx = RetryContext.from_task(task)
        assert ctx.worker_attempts == 0

        ctx.record_failure("quality", model="model-a")
        assert ctx.worker_attempts == 1

        task_ctx = json.loads(task.get("context") or "{}")
        task_ctx.update(ctx.to_context_patch())
        await update_task(tid, context=json.dumps(task_ctx), **ctx.to_db_fields())

        task2 = await get_task(tid)
        ctx2 = RetryContext.from_task(task2)
        assert ctx2.worker_attempts == 1
        assert "model-a" in ctx2.failed_models

        ctx2.record_failure("infrastructure")
        assert ctx2.infra_resets == 1
        assert ctx2.worker_attempts == 1

    asyncio.new_event_loop().run_until_complete(_test())
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/integration/test_retry_pipeline_integration.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x -q --timeout=60`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_retry_pipeline_integration.py
git commit -m "test: integration test for RetryContext DB roundtrip lifecycle"
```
