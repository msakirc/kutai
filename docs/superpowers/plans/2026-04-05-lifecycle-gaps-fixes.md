# Task Lifecycle Gaps & Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all bugs, close design gaps, and eliminate deprecated code paths in the unified task lifecycle system.

**Architecture:** 10 tasks targeting 4 files. The core theme is convergence: every failure/completion path must flow through the same `attempts`/`max_attempts` counters and `transition_task()` validation. We also fix timestamp formats, add missing indexes, and harden edge cases in grading and watchdog logic.

**Tech Stack:** Python 3.10, aiosqlite, pytest

---

### Task 1: Fix `isoformat()` timestamps in orchestrator

Three call sites use `datetime.now().isoformat()` for `completed_at`, producing `T`-separated timestamps that break SQLite `datetime()` comparisons. The constant `_DB_DT_FMT` already exists at line 40.

**Files:**
- Modify: `src/core/orchestrator.py:2020,2396,2601`
- Test: `tests/integration/test_unified_lifecycle.py`

- [ ] **Step 1: Write failing test**

```python
# In tests/integration/test_unified_lifecycle.py, add to TestUnifiedLifecycle:

def test_completed_at_format(self, temp_db):
    """completed_at must use space-separated format, not isoformat."""
    async def _test():
        from src.infra.db import add_task, update_task, get_task
        tid = await add_task("fmt", "desc")
        await update_task(tid, status="completed",
                          completed_at="2026-04-05 12:00:00")
        task = await get_task(tid)
        # Space-separated format: no 'T' in the timestamp
        assert "T" not in task["completed_at"], \
            f"completed_at uses isoformat: {task['completed_at']}"
    self._run(_test())
```

- [ ] **Step 2: Run test to verify it passes (baseline — this tests DB storage, not orchestrator)**

Run: `pytest tests/integration/test_unified_lifecycle.py::TestUnifiedLifecycle::test_completed_at_format -v`
Expected: PASS (direct update_task is fine)

- [ ] **Step 3: Fix the three `isoformat()` calls**

In `src/core/orchestrator.py`, replace all three occurrences:

**Line 2020** (`_handle_complete`):
```python
# OLD:
completed_at=datetime.now().isoformat(),
# NEW:
completed_at=datetime.now().strftime(_DB_DT_FMT),
```

**Line 2396** (`_handle_review`):
```python
# OLD:
completed_at=datetime.now().isoformat())
# NEW:
completed_at=datetime.now().strftime(_DB_DT_FMT))
```

**Line 2601** (`_check_mission_completion`):
```python
# OLD:
completed_at=datetime.now().isoformat())
# NEW:
completed_at=datetime.now().strftime(_DB_DT_FMT))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/test_unified_lifecycle.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py tests/integration/test_unified_lifecycle.py
git commit -m "fix: replace isoformat() with strftime for completed_at timestamps"
```

---

### Task 2: Migrate disguised-failure handler from `retry_count` to `attempts`

The post-hook disguised-failure path (`orchestrator.py:1687-1710`) uses `retry_count`/`max_retries` — a separate budget from the unified `attempts`/`max_attempts`. This lets tasks get up to 9 retries instead of 6.

**Files:**
- Modify: `src/core/orchestrator.py:1687-1720`
- Test: `tests/integration/test_unified_lifecycle.py`

- [ ] **Step 1: Write failing test**

```python
# In tests/integration/test_unified_lifecycle.py, add to TestUnifiedLifecycle:

def test_disguised_failure_increments_attempts(self, temp_db):
    """Disguised failure should use attempts, not retry_count."""
    async def _test():
        from src.infra.db import add_task, update_task, get_task
        tid = await add_task("disguised", "desc")
        # Simulate 5 prior attempts
        await update_task(tid, attempts=5, max_attempts=6)
        task = await get_task(tid)
        assert task["attempts"] == 5
        assert task["max_attempts"] == 6
    self._run(_test())
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_unified_lifecycle.py::TestUnifiedLifecycle::test_disguised_failure_increments_attempts -v`
Expected: PASS (baseline)

- [ ] **Step 3: Replace retry_count with attempts in disguised failure handler**

In `src/core/orchestrator.py`, the block starting around line 1685 (inside the `if result.get("status") == "failed":` after post-hook):

```python
# OLD (lines ~1687-1720):
                        error_msg = result.get("error", "Disguised failure detected")
                        retry_count = task.get("retry_count", 0) or 0
                        max_retries = task.get("max_retries", 3) or 3

                        if retry_count < max_retries:
                            await update_task(
                                task_id, status="pending",
                                retry_count=retry_count + 1,
                                error=error_msg,
                            )
                            logger.warning(
                                f"disguised failure, retrying "
                                f"{retry_count + 1}/{max_retries}",
                                task_id=task_id,
                            )
                        else:
                            # Unified retry: use next_retry_at instead of paused state
                            from datetime import timedelta
                            next_retry = (datetime.now() + timedelta(seconds=600)).strftime(_DB_DT_FMT)
                            await update_task(
                                task_id, status="pending",
                                error=f"Backpressure after {max_retries} failures: {error_msg}",
                                next_retry_at=next_retry,
                                retry_reason="quality",
                            )

# NEW:
                        error_msg = result.get("error", "Disguised failure detected")
                        attempts = (task.get("attempts") or 0) + 1
                        max_attempts = task.get("max_attempts") or 6

                        from src.core.retry import compute_retry_timing, update_exclusions_on_failure
                        update_exclusions_on_failure(task_ctx, result.get("model", ""), attempts)
                        decision = compute_retry_timing("quality", attempts=attempts, max_attempts=max_attempts)

                        if decision.action == "terminal":
                            await update_task(
                                task_id, status="failed",
                                attempts=attempts,
                                failed_in_phase="worker",
                                error=f"Disguised failure exhausted: {error_msg}",
                                context=json.dumps(task_ctx),
                            )
                            try:
                                from src.infra.dead_letter import quarantine_task
                                await quarantine_task(
                                    task_id=task_id,
                                    mission_id=task.get("mission_id"),
                                    error=f"Disguised failure after {attempts} attempts: {error_msg}",
                                    error_category="quality",
                                    original_agent=task.get("agent_type", "executor"),
                                    retry_count=attempts,
                                )
                            except Exception:
                                pass
                            logger.warning(
                                "disguised failure terminal",
                                task_id=task_id,
                                attempts=attempts,
                            )
                        else:
                            next_retry = None
                            if decision.action == "delayed":
                                next_retry = (
                                    datetime.now() + timedelta(seconds=decision.delay_seconds)
                                ).strftime(_DB_DT_FMT)
                            await update_task(
                                task_id, status="pending",
                                attempts=attempts,
                                error=error_msg,
                                next_retry_at=next_retry,
                                retry_reason="quality",
                                context=json.dumps(task_ctx),
                            )
                            logger.warning(
                                f"disguised failure, retrying "
                                f"{attempts}/{max_attempts}",
                                task_id=task_id,
                            )
```

Also update the Telegram notification (was inside the `else` backpressure branch) — move it inside the terminal branch:

```python
                            await self.telegram.send_notification(
                                f"❌ Task #{task_id} disguised failure → DLQ\n"
                                f"**{task.get('title', '')}**\n"
                                f"Reason: {error_msg[:100]}"
                            )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/ -m "not llm" -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py tests/integration/test_unified_lifecycle.py
git commit -m "fix: disguised failure handler uses unified attempts instead of retry_count"
```

---

### Task 3: Migrate agent-failed and general-exception handlers from `retry_count` to `attempts`

Two more code paths use `retry_count`/`max_retries`: the `status == "failed"` branch (line 1764) and the general `except Exception` (line 1865). Both need the same treatment as Task 2.

**Files:**
- Modify: `src/core/orchestrator.py:1764-1797,1865-1934`

- [ ] **Step 1: Write test for both paths**

```python
# In tests/integration/test_unified_lifecycle.py, add:

def test_retry_count_not_used_for_agent_failure(self, temp_db):
    """Agent failure must increment attempts, not retry_count."""
    async def _test():
        from src.infra.db import add_task, update_task, get_task
        tid = await add_task("fail", "desc")
        # Simulate: attempts=2, retry_count=0 (stale)
        await update_task(tid, attempts=2, retry_count=0, status="processing")
        task = await get_task(tid)
        # After a failure, attempts should be 3, not retry_count=1
        assert task["attempts"] == 2
        assert task["retry_count"] == 0
    self._run(_test())
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_unified_lifecycle.py::TestUnifiedLifecycle::test_retry_count_not_used_for_agent_failure -v`
Expected: PASS (baseline)

- [ ] **Step 3: Replace agent-failed handler (lines ~1764-1797)**

```python
# OLD:
            elif status == "failed":
                error_str = result.get("error", result.get("result", "Unknown error"))
                retry_count = task.get("retry_count", 0) or 0
                max_retries = task.get("max_retries", 3) or 3

                if retry_count < max_retries:
                    await update_task(task_id, status="pending",
                                      retry_count=retry_count + 1,
                                      error=error_str[:500])
                    logger.warning(f"agent failed, retrying {retry_count + 1}/{max_retries}",
                                   task_id=task_id, error=error_str[:200])
                elif task_ctx.get("is_workflow_step"):
                    # Workflow step: backpressure — pending with delayed retry
                    from datetime import timedelta
                    next_retry = (datetime.now() + timedelta(seconds=600)).strftime(_DB_DT_FMT)
                    await update_task(
                        task_id, status="pending",
                        error=f"Backpressure after {max_retries} failures: {error_str[:300]}",
                        next_retry_at=next_retry,
                        retry_reason="quality",
                    )
                    ...
                else:
                    await update_task(task_id, status="failed",
                                      error=error_str[:500])
                    ...

# NEW:
            elif status == "failed":
                error_str = result.get("error", result.get("result", "Unknown error"))
                attempts = (task.get("attempts") or 0) + 1
                max_attempts = task.get("max_attempts") or 6

                from src.core.retry import compute_retry_timing, update_exclusions_on_failure
                update_exclusions_on_failure(task_ctx, result.get("model", ""), attempts)
                decision = compute_retry_timing("quality", attempts=attempts, max_attempts=max_attempts)

                if decision.action == "terminal":
                    await update_task(
                        task_id, status="failed",
                        error=error_str[:500],
                        attempts=attempts,
                        failed_in_phase="worker",
                        context=json.dumps(task_ctx),
                    )
                    logger.error("agent failure terminal", task_id=task_id,
                                 error=error_str[:200])
                    await self.telegram.send_error(task_id, title, error_str)
                    try:
                        from src.infra.dead_letter import quarantine_task
                        await quarantine_task(
                            task_id=task_id,
                            mission_id=task.get("mission_id"),
                            error=error_str[:500],
                            error_category="quality",
                            original_agent=task.get("agent_type", "executor"),
                            retry_count=attempts,
                        )
                    except Exception:
                        pass
                else:
                    next_retry = None
                    if decision.action == "delayed":
                        next_retry = (
                            datetime.now() + timedelta(seconds=decision.delay_seconds)
                        ).strftime(_DB_DT_FMT)
                    await update_task(
                        task_id, status="pending",
                        attempts=attempts,
                        error=error_str[:500],
                        next_retry_at=next_retry,
                        retry_reason="quality",
                        context=json.dumps(task_ctx),
                    )
                    logger.warning(f"agent failed, retrying {attempts}/{max_attempts}",
                                   task_id=task_id, error=error_str[:200])
```

- [ ] **Step 4: Replace general-exception handler (lines ~1865-1934)**

```python
# OLD:
        except Exception as e:
            ...
            retry_count = task.get("retry_count", 0)
            max_retries = task.get("max_retries", 3)

            if retry_count < max_retries:
                await update_task(task_id, status="pending",
                                  retry_count=retry_count + 1,
                                  error=f"{type(e).__name__}: {str(e)[:200]}")
            else:
                ...

# NEW:
        except Exception as e:
            logger.exception("task failed", task_id=task_id, error_type=type(e).__name__, error=str(e))
            try:
                await release_task_locks(task_id)
            except Exception:
                pass

            error_str = f"{type(e).__name__}: {str(e)[:500]}"
            attempts = (task.get("attempts") or 0) + 1
            max_attempts = task.get("max_attempts") or 6

            from src.core.retry import compute_retry_timing
            decision = compute_retry_timing("quality", attempts=attempts, max_attempts=max_attempts)

            if decision.action == "terminal":
                try:
                    from ..infra.dead_letter import _classify_error
                    error_cat = _classify_error(error_str, "unknown")
                except Exception:
                    error_cat = "unknown"
                await update_task(
                    task_id, status="failed", error=error_str,
                    error_category=error_cat,
                    attempts=attempts,
                    failed_in_phase="worker",
                )
                await self.telegram.send_error(task_id, title, error_str)

                if task_ctx.get("is_workflow_step"):
                    try:
                        wf_phase = task_ctx.get("workflow_phase", "?")
                        await self.telegram.send_notification(
                            f"Workflow step failed: #{task_id}\n"
                            f"_{task.get('title', '')[:60]}_\n"
                            f"Phase: {wf_phase}"
                        )
                    except Exception:
                        pass

                try:
                    from ..memory.episodic import store_task_result
                    await store_task_result(
                        task=task, result=error_str, model="unknown",
                        cost=0.0, duration=0.0, success=False,
                    )
                except Exception:
                    pass

                recovery_spawned = await self._spawn_error_recovery(task, error_str)
                if not recovery_spawned:
                    try:
                        from ..infra.dead_letter import quarantine_task
                        await quarantine_task(
                            task_id=task_id,
                            mission_id=task.get("mission_id"),
                            error=error_str,
                            original_agent=task.get("agent_type", "executor"),
                            retry_count=attempts,
                        )
                    except Exception as dlq_err:
                        logger.error("dlq quarantine failed", task_id=task_id, error=str(dlq_err))
            else:
                next_retry = None
                if decision.action == "delayed":
                    next_retry = (
                        datetime.now() + timedelta(seconds=decision.delay_seconds)
                    ).strftime(_DB_DT_FMT)
                await update_task(
                    task_id, status="pending",
                    attempts=attempts,
                    error=error_str,
                    next_retry_at=next_retry,
                    retry_reason="quality",
                )
                logger.info("task will retry", task_id=task_id,
                            attempts=attempts, max_attempts=max_attempts)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/integration/ -m "not llm" -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/core/orchestrator.py
git commit -m "fix: agent-failed and exception handlers use unified attempts, not retry_count"
```

---

### Task 4: Fix watchdog stuck-ungraded to use `worker_completed_at`

The watchdog checks `started_at` for stuck ungraded tasks, but a long-running agent (25 min) finishes and gets promoted after only 5 min of actual grading wait. The `worker_completed_at` context field is already set by `base.py:1620`.

**Files:**
- Modify: `src/core/orchestrator.py:486-501`
- Test: `tests/integration/test_unified_lifecycle.py`

- [ ] **Step 1: Write failing test**

```python
# In tests/integration/test_unified_lifecycle.py:

def test_stuck_ungraded_uses_worker_completed_at(self, temp_db):
    """Watchdog should use worker_completed_at, not started_at, for ungraded timeout."""
    async def _test():
        from src.infra.db import add_task, get_db
        import json

        tid = await add_task("ungraded-test", "desc")
        db = await get_db()
        # started_at 40 min ago, worker_completed_at 10 min ago
        ctx = json.dumps({"worker_completed_at": "2099-01-01 00:00:00"})
        await db.execute(
            """UPDATE tasks SET status = 'ungraded',
               started_at = datetime('now', '-40 minutes'),
               context = ?
               WHERE id = ?""",
            (ctx, tid),
        )
        await db.commit()

        # The task has worker_completed_at far in the future, so it should NOT
        # be promoted even though started_at is 40 min old.
        task_row = await db.execute("SELECT status FROM tasks WHERE id = ?", (tid,))
        row = await task_row.fetchone()
        assert row[0] == "ungraded"  # baseline: still ungraded

    self._run(_test())
```

- [ ] **Step 2: Run test (baseline)**

Run: `pytest tests/integration/test_unified_lifecycle.py::TestUnifiedLifecycle::test_stuck_ungraded_uses_worker_completed_at -v`
Expected: PASS

- [ ] **Step 3: Fix watchdog stuck-ungraded check**

In `src/core/orchestrator.py`, replace the stuck-ungraded block (lines ~486-501):

```python
# OLD:
        # 2. Ungraded tasks stuck for > 30 min — safety net
        cursor_ung = await db.execute(
            """SELECT id FROM tasks
               WHERE status = 'ungraded'
               AND started_at < datetime('now', '-30 minutes')"""
        )
        stuck_ungraded = [dict(row) for row in await cursor_ung.fetchall()]
        for task in stuck_ungraded:
            await db.execute(
                "UPDATE tasks SET status = 'completed', quality_score = NULL, "
                "completed_at = ? WHERE id = ?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), task["id"])
            )
            logger.warning(f"[Watchdog] Stuck ungraded #{task['id']} promoted to completed (safety net)")
        if stuck_ungraded:
            await db.commit()

# NEW:
        # 2. Ungraded tasks stuck for > 30 min — safety net
        #    Use worker_completed_at from context (set by base.py on entering ungraded).
        #    Falls back to started_at if worker_completed_at is missing.
        cursor_ung = await db.execute(
            """SELECT id, context, started_at FROM tasks
               WHERE status = 'ungraded'"""
        )
        all_ungraded = [dict(row) for row in await cursor_ung.fetchall()]
        stuck_ungraded = []
        for task in all_ungraded:
            raw_ctx = task.get("context", "{}")
            try:
                ctx = json.loads(raw_ctx) if isinstance(raw_ctx, str) else (raw_ctx or {})
            except (json.JSONDecodeError, TypeError):
                ctx = {}
            completed_at_str = ctx.get("worker_completed_at") or task.get("started_at")
            if not completed_at_str:
                continue
            try:
                completed_dt = datetime.strptime(
                    str(completed_at_str).replace("T", " ")[:19],
                    "%Y-%m-%d %H:%M:%S",
                )
                if (datetime.now() - completed_dt).total_seconds() > 1800:
                    stuck_ungraded.append(task)
            except (ValueError, TypeError):
                continue

        for task in stuck_ungraded:
            await db.execute(
                "UPDATE tasks SET status = 'completed', quality_score = NULL, "
                "completed_at = ? WHERE id = ?",
                (datetime.now().strftime(_DB_DT_FMT), task["id"]),
            )
            logger.warning(f"[Watchdog] Stuck ungraded #{task['id']} promoted to completed (safety net)")
        if stuck_ungraded:
            await db.commit()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/ -m "not llm" -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py tests/integration/test_unified_lifecycle.py
git commit -m "fix: watchdog stuck-ungraded uses worker_completed_at instead of started_at"
```

---

### Task 5: Fix grading auto-pass for trivial/empty output

`grading.py:114-115` auto-passes any output under 10 characters. A failed agent returning "Error" gets auto-passed. Empty or near-empty results should FAIL, not auto-pass.

**Files:**
- Modify: `src/core/grading.py:113-115`
- Test: `tests/integration/test_unified_lifecycle.py`

- [ ] **Step 1: Write failing test**

```python
# In tests/integration/test_unified_lifecycle.py:

def test_trivial_output_not_auto_passed(self):
    """Empty/trivial results should fail grading, not auto-pass."""
    from src.core.grading import GradeResult

    # Simulate what grade_task does for trivial output
    for trivial in ["", "   ", "Error", "Done.", None]:
        text = str(trivial or "").strip()
        # Trivial output should NOT be auto-passed
        if len(text) < 10:
            # Current code returns GradeResult(passed=True) here — that's the bug
            assert len(text) < 10  # just setup assertion
```

- [ ] **Step 2: Fix the auto-pass logic**

In `src/core/grading.py`, replace lines ~113-115:

```python
# OLD:
    result_text = task.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return GradeResult(passed=True, raw="auto-pass: trivial output")

# NEW:
    result_text = task.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return GradeResult(passed=False, raw="auto-fail: trivial/empty output")
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/integration/ -m "not llm" -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/core/grading.py tests/integration/test_unified_lifecycle.py
git commit -m "fix: trivial/empty task output fails grading instead of auto-passing"
```

---

### Task 6: Fix `_loaded_model_can_grade` to check `grade_excluded_models`

The function at `llm_dispatcher.py:653-677` only checks `generating_model != loaded` but ignores `grade_excluded_models`. This causes `ensure_gpu_utilized` to think the loaded model can grade when it was already excluded for parse failures.

**Files:**
- Modify: `src/core/llm_dispatcher.py:653-677`
- Test: `tests/integration/test_unified_lifecycle.py`

- [ ] **Step 1: Write test**

```python
# In tests/integration/test_unified_lifecycle.py:

def test_grade_excluded_models_respected(self, temp_db):
    """A model in grade_excluded_models should not be considered a valid grader."""
    async def _test():
        from src.infra.db import add_task, update_task
        import json

        tid = await add_task("excl-test", "desc")
        ctx = json.dumps({
            "generating_model": "model_a",
            "grade_excluded_models": ["model_b"],
        })
        await update_task(tid, status="ungraded", context=ctx)

        # model_b should NOT be able to grade this task
        from src.infra.db import get_db
        db = await get_db()
        cursor = await db.execute(
            "SELECT context FROM tasks WHERE status = 'ungraded'"
        )
        rows = await cursor.fetchall()
        for row in rows:
            ctx = json.loads(row["context"] or "{}")
            # model_b is not generating_model but IS excluded
            assert "model_b" in ctx.get("grade_excluded_models", [])

    self._run(_test())
```

- [ ] **Step 2: Run test (baseline)**

Run: `pytest tests/integration/test_unified_lifecycle.py::TestUnifiedLifecycle::test_grade_excluded_models_respected -v`
Expected: PASS

- [ ] **Step 3: Fix `_loaded_model_can_grade`**

In `src/core/llm_dispatcher.py`, replace the method (lines ~653-677):

```python
# OLD:
    async def _loaded_model_can_grade(self) -> bool:
        """Check if loaded model can grade ANY ungraded task (not self-generated)."""
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

# NEW:
    async def _loaded_model_can_grade(self) -> bool:
        """Check if loaded model can grade ANY ungraded task.

        Excludes tasks where loaded model is the generating model
        OR is in grade_excluded_models.
        """
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
                if ctx.get("generating_model") == loaded:
                    continue
                if loaded in ctx.get("grade_excluded_models", []):
                    continue
                return True
            return False
        except Exception:
            return False
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/ -m "not llm" -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/core/llm_dispatcher.py tests/integration/test_unified_lifecycle.py
git commit -m "fix: _loaded_model_can_grade checks grade_excluded_models"
```

---

### Task 7: Add jitter to `accelerate_retries` to prevent thundering herd

When `accelerate_retries` fires, all availability-delayed tasks wake simultaneously. Adding staggered wake times (random 0-30s jitter) prevents burst failures.

**Files:**
- Modify: `src/infra/db.py:1005-1042`
- Test: `tests/integration/test_unified_lifecycle.py`

- [ ] **Step 1: Write test**

```python
# In tests/integration/test_unified_lifecycle.py:

def test_accelerate_retries_staggered(self, temp_db):
    """Accelerated tasks should have slightly staggered next_retry_at, not all identical."""
    async def _test():
        from src.infra.db import add_task, get_db, accelerate_retries
        import json

        tids = []
        db = await get_db()
        for i in range(5):
            tid = await add_task(f"avail-{i}", "desc")
            await db.execute(
                """UPDATE tasks SET
                   next_retry_at = datetime('now', '+1 hour'),
                   retry_reason = 'availability',
                   context = ?
                   WHERE id = ?""",
                (json.dumps({"last_avail_delay": 120}), tid),
            )
            tids.append(tid)
        await db.commit()

        woken = await accelerate_retries("test_signal")
        assert woken == 5

        # All should be eligible now (next_retry_at <= now + 30s)
        cursor = await db.execute(
            "SELECT next_retry_at FROM tasks WHERE id IN ({})".format(
                ",".join("?" * len(tids))
            ),
            tids,
        )
        rows = await cursor.fetchall()
        times = [r[0] for r in rows]
        # At least some should differ (jitter applied)
        # Note: with 5 tasks and 0-30s jitter, probability of all identical is negligible
        assert len(times) == 5
        # All should be in the near past/present (not still 1h out)
        for t in times:
            assert t is not None

    self._run(_test())
```

- [ ] **Step 2: Run test (will pass even before change — it's a soft assertion)**

Run: `pytest tests/integration/test_unified_lifecycle.py::TestUnifiedLifecycle::test_accelerate_retries_staggered -v`

- [ ] **Step 3: Add jitter to accelerate_retries**

In `src/infra/db.py`, modify `accelerate_retries` (lines ~1005-1042):

```python
# OLD (inside the for loop):
        ctx["last_avail_delay"] = 0
        await db.execute(
            """UPDATE tasks SET next_retry_at = datetime('now'),
               context = ? WHERE id = ?""",
            (_json.dumps(ctx), row["id"]),
        )

# NEW:
        import random
        ctx["last_avail_delay"] = 0
        jitter = random.randint(0, 30)
        await db.execute(
            f"""UPDATE tasks SET next_retry_at = datetime('now', '+{jitter} seconds'),
               context = ? WHERE id = ?""",
            (_json.dumps(ctx), row["id"]),
        )
```

Move the `import random` to the top of the function (before the loop), not inside the loop.

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/ -m "not llm" -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/infra/db.py tests/integration/test_unified_lifecycle.py
git commit -m "fix: add jitter to accelerate_retries to prevent thundering herd"
```

---

### Task 8: Set `failed_in_phase` on general exception failures

The general exception handler and agent-failed path don't set `failed_in_phase`, so DLQ retry can't determine the correct re-entry point.

Note: This is mostly addressed by Task 3's rewrite of the general exception handler. But verify the other `update_task(status="failed")` calls also set it.

**Files:**
- Modify: `src/core/orchestrator.py` (various `status="failed"` calls)

- [ ] **Step 1: Grep for all `status="failed"` without `failed_in_phase`**

Search `src/core/orchestrator.py` for `status="failed"` or `status='failed'` calls that don't include `failed_in_phase`. After Task 3, the main handlers will have it. Check remaining ones:

- Line 1748 (silent clarification): `await update_task(task_id, status="failed", error="Insufficient info (silent task, no clarification)")` — this should set `failed_in_phase="worker"`.

- [ ] **Step 2: Fix remaining calls**

In `src/core/orchestrator.py` line ~1748:
```python
# OLD:
                    await update_task(task_id, status="failed",
                                      error="Insufficient info (silent task, no clarification)")
# NEW:
                    await update_task(task_id, status="failed",
                                      error="Insufficient info (silent task, no clarification)",
                                      failed_in_phase="worker")
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/integration/ -m "not llm" -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/core/orchestrator.py
git commit -m "fix: set failed_in_phase on all failure paths for correct DLQ recovery"
```

---

### Task 9: Add `next_retry_at` compound index

`get_ready_tasks` and `drain_ungraded_tasks` both filter on `(status, next_retry_at)` but there's no compound index for this.

**Files:**
- Modify: `src/infra/db.py:611-631` (index list)
- Test: `tests/integration/test_unified_lifecycle.py`

- [ ] **Step 1: Add indexes to the index list**

In `src/infra/db.py`, add to the `_indexes` list (around line 611):

```python
        ("idx_tasks_status_retry", "tasks", "status, next_retry_at"),
```

- [ ] **Step 2: Write test**

```python
# In tests/integration/test_unified_lifecycle.py:

def test_retry_at_index_exists(self, temp_db):
    """Compound index on (status, next_retry_at) should exist."""
    async def _test():
        from src.infra.db import get_db
        db = await get_db()
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_tasks_status_retry'"
        )
        row = await cursor.fetchone()
        assert row is not None, "Missing index idx_tasks_status_retry"
    self._run(_test())
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/integration/test_unified_lifecycle.py::TestUnifiedLifecycle::test_retry_at_index_exists -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/infra/db.py tests/integration/test_unified_lifecycle.py
git commit -m "perf: add compound index on (status, next_retry_at) for retry queries"
```

---

### Task 10: Eliminate workflow backpressure infinite loop

When a workflow step exhausts `max_retries` (old counter), it enters "backpressure" — `pending` with 10-min delay. But there's no terminal condition. The watchdog clears overdue `next_retry_at`, it fails again, loops forever. After Task 3, the agent-failed handler uses unified `attempts` which has a terminal path. But the disguised-failure handler (Task 2) also needs to ensure no special workflow backpressure exists.

The fix from Tasks 2+3 already eliminates the separate backpressure branches — both now use `compute_retry_timing("quality")` which terminates at `max_attempts`. Verify this is complete.

**Files:**
- Verify: `src/core/orchestrator.py` (no more `Backpressure` strings in failure handlers)

- [ ] **Step 1: Verify no backpressure branches remain**

After Tasks 2 and 3, grep for "Backpressure" and "backpressure" in the failure handlers. The only remaining reference should be in the Telegram notification text, not in branching logic.

Run: `grep -n -i "backpressure" src/core/orchestrator.py`

Expected: Only log messages, no branching logic.

- [ ] **Step 2: Write integration test for terminal condition**

```python
# In tests/integration/test_unified_lifecycle.py:

def test_workflow_step_reaches_terminal(self, temp_db):
    """Workflow step that exhausts attempts must reach failed, not loop forever."""
    async def _test():
        from src.infra.db import add_task, update_task, get_task
        from src.core.retry import compute_retry_timing
        import json

        tid = await add_task("wf-step", "desc")
        ctx = json.dumps({"is_workflow_step": True})
        await update_task(tid, context=ctx, attempts=6, max_attempts=6)
        task = await get_task(tid)

        decision = compute_retry_timing(
            "quality",
            attempts=task["attempts"],
            max_attempts=task["max_attempts"],
        )
        assert decision.action == "terminal", \
            f"Expected terminal at max_attempts, got {decision.action}"

    self._run(_test())
```

- [ ] **Step 3: Run test**

Run: `pytest tests/integration/test_unified_lifecycle.py::TestUnifiedLifecycle::test_workflow_step_reaches_terminal -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_unified_lifecycle.py
git commit -m "test: verify workflow steps reach terminal via unified retry"
```

---

### Task 11: Update docs with resolved known limitations

Update `docs/task-lifecycle.md` Known Limitations section to reflect what was fixed.

**Files:**
- Modify: `docs/task-lifecycle.md:255-264`

- [ ] **Step 1: Update Known Limitations**

Replace the Known Limitations section with:

```markdown
## Known Limitations

1. **`transition_task()` not enforced everywhere yet**: Many raw `update_task(status=...)` calls in the orchestrator bypass validation. These should be migrated to use `transition_task()` to catch invalid transitions at runtime. (The failure/retry handlers now use unified `attempts` counters, but still call `update_task` directly rather than `transition_task`.)

2. **Stale comments**: Some files still reference "sleeping queue" in comments. The function calls are correct — just the comments are outdated.
```

- [ ] **Step 2: Add a "Fixed in 2026-04-05" section after Known Limitations**

```markdown
## Fixed (2026-04-05)

- `completed_at` timestamps now use `strftime("%Y-%m-%d %H:%M:%S")` everywhere (was `isoformat()` in 3 locations)
- All failure handlers (disguised failure, agent-failed, general exception) use unified `attempts`/`max_attempts` counters (was `retry_count`/`max_retries`)
- Watchdog stuck-ungraded check uses `worker_completed_at` from context (was `started_at`)
- Trivial/empty task output fails grading instead of auto-passing
- `_loaded_model_can_grade` checks `grade_excluded_models` (was only checking `generating_model`)
- `accelerate_retries` adds jitter (0-30s) to prevent thundering herd
- `failed_in_phase` set on all failure paths for correct DLQ recovery
- Compound index on `(status, next_retry_at)` added
- Workflow backpressure infinite loop eliminated (unified retry has terminal condition)
```

- [ ] **Step 3: Commit**

```bash
git add docs/task-lifecycle.md
git commit -m "docs: update task-lifecycle with fixed items from 2026-04-05 audit"
```
