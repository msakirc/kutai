# Task Lifecycle Fixes — Overhead Budget, Checkpoint Continuity, Empty Iterations

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix six structural issues in the task execution lifecycle that cause wasted work, unnecessary DLQs, and timeout budget theft by overhead calls.

**Architecture:** Move LLM-dependent post-processing (summarization, grading) out of the task's timeout budget into separate orchestrator-scheduled work. Make the agent checkpoint system actually resume across attempts. Stop counting empty LLM responses as used iterations.

**Tech Stack:** Python async, aiosqlite, existing orchestrator/agent/hooks architecture

---

## Issues Addressed

| # | Issue | Impact |
|---|-------|--------|
| 1 | LLM summarization in post-hook eats task timeout | 4-min overhead call kills a finished task |
| 2 | Inline grading in agent.execute() eats task timeout | Grade swap storm times out a completed task |
| 3 | `_prev_output` only injected on timeout failures | Schema/quality failures start fresh — agent re-reads files, wastes iterations |
| 4 | 0-char responses advance iteration counter | 5 iterations of empty responses → agent hits max_iterations with no work done |
| 5 | Checkpoints not resumed across retry attempts | Agent starts at iteration 1 every time despite checkpoint at iteration 4 |
| 6 | Todo suggestions broken (overhead LLM call during cron) | Overhead call can't get a model when system is idle |

## File Map

| File | Changes |
|------|---------|
| `src/workflows/engine/hooks.py` | Remove `_llm_summarize` LLM call from post-hook, replace with structural-only summary |
| `src/core/orchestrator.py` | Add summarization as background work in idle path; inject `_prev_output` on all failure types; schedule overhead work |
| `src/agents/base.py` | Remove inline grading; skip iteration advance on empty response; resume from checkpoint on retry |
| `tests/test_grading.py` | Update tests for grading-only-deferred |
| `tests/test_lifecycle_fixes.py` | New: tests for empty iteration skip, checkpoint resume, prev_output injection |

## Dependency Order

Tasks 1-2 can be done in parallel (independent code paths). Task 3 depends on understanding the retry flow but touches different code. Tasks 4-5 touch `base.py` and should be done together. Task 6 is independent.

---

### Task 1: Remove LLM summarization from post-hook

The `_llm_summarize` call in `post_execute_workflow_step` makes an OVERHEAD LLM call for every artifact > 3000 chars. This runs inside the task's timeout budget and can take minutes during model swap storms.

**Files:**
- Modify: `src/workflows/engine/hooks.py:1021-1041` (remove LLM summarization, keep structural)
- Modify: `src/workflows/engine/hooks.py:405-463` (`_llm_summarize` function — simplify to structural-only)

- [ ] **Step 1: Write failing test — summarization post-hook does NOT make LLM calls**

```python
# tests/test_lifecycle_fixes.py
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

class TestPostHookNoLLM:
    """post_execute_workflow_step must NEVER make LLM calls."""

    def test_large_artifact_no_llm_call(self):
        """Artifacts > 3000 chars should use structural summary, not LLM."""
        from src.workflows.engine.hooks import post_execute_workflow_step

        task = {
            "id": 999,
            "mission_id": 99,
            "context": '{"is_workflow_step": true, "workflow_step_id": "1.1", '
                       '"output_artifacts": ["test_artifact"], '
                       '"artifact_schema": {}}',
        }
        # Result with >3000 chars to trigger summarization path
        result = {"status": "completed", "result": "x" * 5000}

        with patch("src.workflows.engine.hooks.get_artifact_store") as mock_store, \
             patch("src.core.llm_dispatcher.get_dispatcher") as mock_disp:
            mock_store.return_value = AsyncMock()
            mock_store.return_value.store = AsyncMock()
            run_async(post_execute_workflow_step(task, result))
            # Dispatcher must NOT have been called — no LLM summarization
            mock_disp.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python -m pytest tests/test_lifecycle_fixes.py::TestPostHookNoLLM -v -p no:cacheprovider`
Expected: FAIL — dispatcher IS currently called for summarization

- [ ] **Step 3: Replace LLM summarization with structural-only in post-hook**

In `src/workflows/engine/hooks.py`, replace lines 1021-1041 (the auto-summarize block):

```python
    # ── Auto-summarize large artifacts ──
    # Structural summary only — no LLM calls in the post-hook.
    # LLM summarization was removed because it ate the task's timeout
    # budget and triggered model swaps during post-processing.
    _SUMMARY_THRESHOLD = 3000
    _MIN_SUMMARY_LEN = 50
    if output_value and len(output_value) > _SUMMARY_THRESHOLD:
        for name in output_names:
            summary = _structural_summary(output_value)
            if summary and len(summary) >= _MIN_SUMMARY_LEN:
                summary_name = f"{name}_summary"
                await store.store(mission_id, summary_name, summary)
                logger.info(
                    f"[Workflow Hook] Structural summary '{name}' -> '{summary_name}' "
                    f"({len(output_value)} -> {len(summary)} chars)"
                )
```

Then simplify `_llm_summarize` (lines 405-463) to just call `_structural_summary` — or delete it entirely and replace all callers with `_structural_summary`. Check for other callers first:
`grep -rn "_llm_summarize" src/`

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 .venv/Scripts/python -m pytest tests/test_lifecycle_fixes.py::TestPostHookNoLLM -v -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Run existing hooks tests**

Run: `timeout 30 .venv/Scripts/python -m pytest tests/ -k "hook or workflow" -v -p no:cacheprovider`
Expected: All pass (no behavioral change for callers — just structural summary instead of LLM)

- [ ] **Step 6: Commit**

```bash
git add src/workflows/engine/hooks.py tests/test_lifecycle_fixes.py
git commit -m "fix(hooks): remove LLM summarization from post-hook — structural only

LLM summarization in post_execute_workflow_step ate the task's timeout
budget and triggered model swaps during post-processing. Replace with
structural-only summary. Summarization never needed LLM quality for
its downstream consumers (context injection truncates to ~2K anyway)."
```

---

### Task 2: Remove inline grading from agent.execute()

Inline grading at `base.py:2055-2087` makes an LLM call inside `agent.execute()`, inside the `wait_for` timeout. If the grading model isn't loaded, it triggers a swap. The deferred grading path (status=ungraded → orchestrator drains) already exists and works correctly.

**Files:**
- Modify: `src/agents/base.py:2046-2087` (remove inline grading, always defer)
- Modify: `tests/test_grading.py` (update any tests expecting inline grading)

- [ ] **Step 1: Write failing test — agent.execute never calls grade_task**

```python
# tests/test_lifecycle_fixes.py (append to existing)

class TestNoInlineGrading:
    """agent.execute() must always defer grading — never call grade_task inline."""

    def test_execute_does_not_call_grade_task(self):
        """After final_answer, agent should return 'ungraded' without calling grade_task."""
        from src.agents.base import BaseAgent
        import json

        agent = BaseAgent.__new__(BaseAgent)
        # Minimal setup — we only care that grade_task is NOT called
        task = {
            "id": 888,
            "title": "test task",
            "description": "test",
            "context": json.dumps({"is_workflow_step": True, "workflow_step_id": "1.1"}),
        }

        with patch("src.core.grading.grade_task") as mock_grade:
            # We can't easily run execute() without a full LLM mock,
            # so instead verify that grade_task import doesn't exist
            # in the execute code path. This is a structural check.
            import inspect
            source = inspect.getsource(BaseAgent._execute_react_loop)
            assert "grade_task" not in source, \
                "agent._execute_react_loop still references grade_task — inline grading not removed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python -m pytest tests/test_lifecycle_fixes.py::TestNoInlineGrading -v -p no:cacheprovider`
Expected: FAIL — grade_task IS still referenced in execute

- [ ] **Step 3: Remove inline grading from base.py**

In `src/agents/base.py`, find the section around lines 2046-2087 where inline grading happens. Replace the entire `can_grade_now` block with a direct path to deferred grading:

```python
                    # Always defer grading to the orchestrator's drain cycle.
                    # Inline grading was removed because it made LLM calls
                    # inside the task's timeout budget and could trigger
                    # model swaps during post-processing.
                    if not can_grade_now and task_id != "?":
```

becomes simply:

```python
                    if task_id != "?":
```

Delete the entire `can_grade_now` variable, the `grade_applied` variable, the `if can_grade_now` block (lines 2055-2087), and the `grade_task` / `apply_grade_result` imports inside that block.

The deferred path (lines 2089-2118) remains unchanged — it sets status="ungraded" and the orchestrator's `drain_ungraded_tasks()` handles grading separately.

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 .venv/Scripts/python -m pytest tests/test_lifecycle_fixes.py::TestNoInlineGrading -v -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Run all grading tests**

Run: `timeout 30 .venv/Scripts/python -m pytest tests/test_grading.py tests/test_llm_dispatcher.py -v -p no:cacheprovider`
Expected: All pass. The drain_ungraded path is unchanged; only inline grading removed.

- [ ] **Step 6: Commit**

```bash
git add src/agents/base.py tests/test_lifecycle_fixes.py
git commit -m "fix(agents): remove inline grading from agent.execute()

Inline grading made an LLM call inside the task's timeout budget,
triggering model swaps and eating minutes from completed tasks.
All grading now defers to orchestrator's drain_ungraded_tasks()
which runs during idle and handles model selection independently."
```

---

### Task 3: Inject `_prev_output` on all failure types

Currently `_prev_output` and `_timeout_hint` are only injected on the timeout path (`orchestrator.py:1826`). Schema failures and quality failures go through the "disguised failure" path which doesn't inject continuation context. The agent starts fresh every retry.

**Files:**
- Modify: `src/core/orchestrator.py` — the "disguised failure" handler (around lines 1960-2045) and the ungraded→failed handler (around lines 2075-2140)
- Modify: `src/workflows/engine/hooks.py:1068-1082` — schema validation already stores `_prev_output` in context but only for schema failures, not quality failures

- [ ] **Step 1: Write failing test — schema failure injects _prev_output**

```python
# tests/test_lifecycle_fixes.py (append)
import json

class TestPrevOutputInjection:
    """All failure types must inject _prev_output into context for next attempt."""

    def test_disguised_failure_injects_prev_output(self):
        """When orchestrator detects disguised failure, _prev_output must be in context."""
        # The disguised failure path at orchestrator.py:1960+ should store
        # the agent's output in _prev_output so the next attempt can continue.
        from src.infra.db import update_task, get_task
        # This is a behavioral test — we need to trace through the code
        # to verify _prev_output is stored in the task context after
        # a disguised failure retry.
        # For now, verify the code path exists by checking the source.
        import inspect
        from src.core.orchestrator import Orchestrator
        source = inspect.getsource(Orchestrator.process_task)
        # Count how many times _prev_output is injected
        prev_output_injections = source.count('"_prev_output"')
        # Should appear in: timeout path, disguised failure path, quality failure path
        assert prev_output_injections >= 3, \
            f"_prev_output only injected {prev_output_injections} times — " \
            f"expected at least 3 (timeout, disguised failure, quality failure)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python -m pytest tests/test_lifecycle_fixes.py::TestPrevOutputInjection -v -p no:cacheprovider`
Expected: FAIL — currently only in timeout path + schema validation hook = 2 places

- [ ] **Step 3: Add _prev_output injection to disguised failure path**

In `src/core/orchestrator.py`, find the "disguised failure" handler. There are two instances:

**Instance 1** — status == "completed" disguised failure (around line 1960):
After the line that detects disguised failure and before the retry logic, add:

```python
                    # Inject partial output so next attempt knows what was done
                    result_text = result.get("result", "")
                    if result_text:
                        task_ctx["_prev_output"] = str(result_text)[:6000]
                        task_ctx["_retry_hint"] = (
                            "Your previous attempt's output failed quality checks. "
                            "Your partial work is shown in context. Build on it — "
                            "do NOT start over."
                        )
```

**Instance 2** — status == "ungraded" → post-hook overrides to "failed" (around line 2075):
Same injection before the retry logic.

The schema validation hook at `hooks.py:1076` already stores `_prev_output` — verify it's preserved through the retry path (the orchestrator re-reads context from DB at line 1948-1955).

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 .venv/Scripts/python -m pytest tests/test_lifecycle_fixes.py::TestPrevOutputInjection -v -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py tests/test_lifecycle_fixes.py
git commit -m "fix(orchestrator): inject _prev_output on all failure types

Previously only timeout failures injected _prev_output into context.
Schema and quality failures started fresh, wasting iterations re-reading
workspace files. Now all failure paths inject the previous output so the
agent can continue from where it left off."
```

---

### Task 4: Skip iteration advance on empty LLM response

When the LLM returns 0-char content, the agent advances the iteration counter and wastes a slot. Task 1023 burned 25 iterations across 5 attempts with mostly empty responses.

**Files:**
- Modify: `src/agents/base.py` — the iteration loop (around line 1797 where `Raw response` is logged)

- [ ] **Step 1: Write failing test — empty response does not advance iteration**

```python
# tests/test_lifecycle_fixes.py (append)

class TestEmptyResponseSkip:
    """0-char LLM responses must not count as used iterations."""

    def test_empty_content_does_not_advance_iteration(self):
        """If LLM returns empty content and no tool_calls, iteration should not advance."""
        # Verify the code structure: after getting empty content, there should be
        # a `continue` that skips to the next iteration without advancing the counter,
        # or the iteration should be decremented.
        import inspect
        from src.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent._execute_react_loop)
        # Look for empty response handling near the Raw response log
        assert "empty_response" in source or "0 chars" in source or "skip" in source.lower(), \
            "No empty response handling found in _execute_react_loop"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python -m pytest tests/test_lifecycle_fixes.py::TestEmptyResponseSkip -v -p no:cacheprovider`
Expected: FAIL

- [ ] **Step 3: Add empty response guard in the iteration loop**

In `src/agents/base.py`, after the LLM call returns and content is extracted (around line 1797 where `Raw response` is logged), before parsing:

```python
                # Skip empty responses — don't burn an iteration on nothing.
                # The model returned no content and no tool_calls (common with
                # thinking-only responses or streaming failures).
                if not content and not response.get("tool_calls"):
                    logger.warning(
                        f"[Task #{task_id}] Empty response (0 chars, no tool_calls) "
                        f"— not counting as iteration {iteration + 1}/{effective_max_iterations}"
                    )
                    empty_response_count = empty_response_count + 1
                    if empty_response_count >= 3:
                        # 3 consecutive empties = model is broken, fail the task
                        return {
                            "status": "failed",
                            "error": f"Model returned {empty_response_count} consecutive empty responses",
                            "model": used_model,
                            "cost": total_cost,
                        }
                    continue  # retry same iteration
```

Initialize `empty_response_count = 0` at the start of the loop (before line 1610). Reset it to 0 whenever a non-empty response is received.

- [ ] **Step 4: Run test to verify it passes** (update test assertion to match actual marker)

- [ ] **Step 5: Run full agent test suite**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/ -k "agent or base" -v -p no:cacheprovider --ignore=tests/integration`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/agents/base.py tests/test_lifecycle_fixes.py
git commit -m "fix(agents): skip iteration counter on empty LLM responses

0-char responses with no tool_calls burned iteration slots without doing
work. Now the same iteration is retried, up to 3 consecutive empties
before failing the task."
```

---

### Task 5: Resume from checkpoint across retry attempts

When a task fails and retries, `_execute_react_loop` loads the checkpoint at line 1503 and sets `start_iteration` from it. But the checkpoint gets cleared on certain paths, and the timeout handler rolls it back. The net effect: retries often start at iteration 0 instead of resuming.

**Files:**
- Modify: `src/agents/base.py` — checkpoint handling at start of `_execute_react_loop` and in the timeout/failure paths
- Modify: `src/core/orchestrator.py` — stop clearing checkpoints on retry paths

- [ ] **Step 1: Investigate current checkpoint clearing**

Find all places that clear or reset checkpoints:

```bash
grep -n "clear_checkpoint\|_clear_checkpoint\|save_task_checkpoint.*iteration.*0\|delete.*checkpoint" src/agents/base.py src/core/orchestrator.py
```

Document which paths clear the checkpoint and whether they should:
- Final answer → completed: YES, clear (task done)
- Final answer → ungraded: currently clears at line 2107 — WRONG, grading may fail and retry
- Timeout → retry: rolls back by 2 iterations — OK
- Schema failure → retry: checkpoint state unknown — should PRESERVE
- Quality failure → retry: checkpoint state unknown — should PRESERVE

- [ ] **Step 2: Remove checkpoint clear from the ungraded path**

In `src/agents/base.py`, line 2107:
```python
                        await self._clear_checkpoint_safe(task_id)
```
This clears the checkpoint when the task goes to "ungraded". But if grading fails and the task retries, the checkpoint is gone. Remove this line — let the checkpoint persist until the task is truly completed (status=completed in _handle_complete).

- [ ] **Step 3: Verify checkpoint is loaded on retry**

In `src/agents/base.py:1503-1543`, the checkpoint recovery code already loads the checkpoint and sets `start_iteration`. Verify it works when the checkpoint wasn't cleared:

```python
# tests/test_lifecycle_fixes.py (append)

class TestCheckpointResume:
    """Retried tasks must resume from their last checkpoint, not iteration 0."""

    def test_checkpoint_loaded_on_retry(self):
        """If a checkpoint exists with iteration=3, start_iteration should be 3."""
        from src.infra.db import save_task_checkpoint, load_task_checkpoint

        # Save a fake checkpoint
        run_async(save_task_checkpoint(9999, {
            "iteration": 3,
            "messages": [{"role": "user", "content": "test"}],
            "max_iterations": 7,
        }))

        # Verify it loads
        cp = run_async(load_task_checkpoint(9999))
        assert cp is not None
        assert cp["iteration"] == 3
```

- [ ] **Step 4: Only clear checkpoint on final completion**

Move checkpoint clearing to `_handle_complete` in the orchestrator (line ~2606) instead of in `agent.execute()`. This way the checkpoint persists through retries and is only cleaned up when the task is fully done.

In `src/core/orchestrator.py`, in `_handle_complete` (around line 2606), add:
```python
        # Clear checkpoint — task is fully done
        try:
            from src.infra.db import clear_task_checkpoint
            await clear_task_checkpoint(task_id)
        except Exception:
            pass
```

- [ ] **Step 5: Run tests**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/test_lifecycle_fixes.py tests/test_grading.py -v -p no:cacheprovider`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/agents/base.py src/core/orchestrator.py tests/test_lifecycle_fixes.py
git commit -m "fix(lifecycle): preserve checkpoint across retries, clear only on completion

Checkpoints were cleared when entering 'ungraded' state, so retries after
grading failure started at iteration 0. Now checkpoints persist through
all retry paths and are only cleared in _handle_complete when the task
is fully done."
```

---

### Task 6: Fix todo suggestions and overhead scheduling

`_start_todo_suggestions` makes an OVERHEAD LLM call during `check_scheduled_tasks()`. With the `call_category` scoring fix, Hoca strongly prefers the loaded model for overhead. But when the system is idle and the model was unloaded by IdleUnloader, there's no loaded model. The todo suggestion call either fails or triggers an unwanted swap.

The fix: if no model is loaded when the cron fires, skip the LLM suggestion and send the plain reminder. The structural summary approach from Task 1 shows the pattern — graceful degradation when LLM is unavailable.

**Files:**
- Modify: `src/core/orchestrator.py:1316-1391` (`_generate_suggestions` method)

- [ ] **Step 1: Write failing test — suggestions gracefully skip when no model available**

```python
# tests/test_lifecycle_fixes.py (append)

class TestTodoSuggestionsGraceful:
    """Todo suggestions must not crash or block when no model is loaded."""

    def test_suggestions_skip_when_dispatcher_fails(self):
        """If OVERHEAD call fails, suggestions are skipped and reminder still sent."""
        from src.core.orchestrator import Orchestrator

        orch = Orchestrator.__new__(Orchestrator)
        orch.telegram = MagicMock()

        # Mock DB to return todos needing suggestions
        with patch("src.core.orchestrator.get_todos", new_callable=AsyncMock) as mock_todos, \
             patch("src.core.orchestrator.send_todo_reminder", new_callable=AsyncMock) as mock_remind, \
             patch("src.core.llm_dispatcher.get_dispatcher") as mock_disp:

            mock_todos.return_value = [
                {"id": 1, "title": "Buy milk", "suggestion": None, "suggestion_at": None}
            ]
            # Dispatcher raises — model not available
            mock_disp.return_value.request = AsyncMock(
                side_effect=RuntimeError("OVERHEAD call failed")
            )

            run_async(orch._start_todo_suggestions())

            # Reminder should still be sent despite suggestion failure
            mock_remind.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails or passes** (it may already handle this via the try/except)

- [ ] **Step 3: Verify the timeout wrapper is tight enough**

In `_generate_suggestions` (line 1349), the LLM call has a 45s `asyncio.wait_for` timeout. This is fine — it won't hang. But verify the `except` blocks at lines 1383-1390 properly mark todos as attempted so they're not retried endlessly.

- [ ] **Step 4: Commit if changes needed**

```bash
git add src/core/orchestrator.py tests/test_lifecycle_fixes.py
git commit -m "fix(orchestrator): harden todo suggestion failure handling"
```

---

## Execution Notes

- **Task timeout budget**: After Tasks 1 and 2, the only things running inside the `wait_for(timeout)` are: agent iterations + tool execution. No LLM overhead calls. The timeout finally measures what it should — agent work.
- **Testing**: Run `timeout 120 .venv/Scripts/python -m pytest tests/ -x -q -p no:cacheprovider --ignore=tests/integration` after all tasks to verify no regressions.
- **Restart KutAI**: After committing all changes, restart KutAI via Telegram `/restart` to pick up the fixes. The running orchestrator has the old code.
