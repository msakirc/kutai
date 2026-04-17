import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

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
        result = {"status": "completed", "result": "x" * 5000}

        with patch("src.workflows.engine.hooks.get_artifact_store") as mock_store, \
             patch("src.workflows.engine.hooks.queue_llm_summary", new_callable=AsyncMock) as mock_queue, \
             patch("src.core.llm_dispatcher.get_dispatcher") as mock_disp:
            mock_store.return_value = AsyncMock()
            mock_store.return_value.store = AsyncMock()
            run_async(post_execute_workflow_step(task, result))
            # Post-hook must NOT make LLM calls (dispatcher untouched)
            mock_disp.assert_not_called()
            # Post-hook MUST queue the artifact for later LLM summarization
            mock_queue.assert_called()


class TestNoInlineGrading:
    """agent.execute() must always defer grading — never call grade_task inline."""

    def test_execute_does_not_reference_grade_task(self):
        """_execute_react_loop should not import or call grade_task."""
        import inspect
        from src.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent._execute_react_loop)
        assert "grade_task" not in source, \
            "agent._execute_react_loop still references grade_task — inline grading not removed"

    def test_execute_does_not_reference_can_grade_now(self):
        """can_grade_now variable should be fully removed."""
        import inspect
        from src.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent._execute_react_loop)
        assert "can_grade_now" not in source, \
            "can_grade_now still exists in _execute_react_loop"


class TestPrevOutputInjection:
    """All failure types must inject _prev_output into context for next attempt."""

    def test_all_failure_paths_inject_prev_output(self):
        """All failure paths must inject _prev_output for next attempt.

        After the Plan A refactor:
        - Timeout path stays inline in Orchestrator.process_task (1 site).
        - Both disguised-failure paths (post-completed and post-ungraded)
          funnel through result_guards._quality_retry_flow, which owns
          a single shared `_prev_output` injection site.

        So the total raw count across both modules is 2, but it covers
        all 3 logical paths.  Verify the helper + timeout path both exist.
        """
        import inspect
        from src.core.orchestrator import Orchestrator
        from src.core import result_guards
        # Timeout path lives in Orchestrator._dispatch after the refactor
        dispatch_src = inspect.getsource(Orchestrator._dispatch)
        guards = inspect.getsource(result_guards)
        assert dispatch_src.count('"_prev_output"') >= 1, \
            "timeout path in _dispatch must inject _prev_output"
        # Disguised-failure flow (shared by completed + ungraded post-hook)
        assert guards.count('"_prev_output"') >= 1, \
            "result_guards must inject _prev_output for disguised failures"
        # The shared helper must be invoked from both post-hook guards
        assert guards.count("_quality_retry_flow") >= 3, \
            "both completed and ungraded post-hooks must call _quality_retry_flow"


class TestEmptyResponseSkip:
    """0-char LLM responses must not count as used iterations."""

    def test_empty_response_guard_exists(self):
        """_execute_react_loop must have empty response handling."""
        import inspect
        from src.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent._execute_react_loop)
        assert "empty_response_count" in source, \
            "No empty_response_count guard found in _execute_react_loop"

    def test_empty_guard_has_continue(self):
        """Empty response handler must 'continue' to retry same iteration."""
        import inspect
        from src.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent._execute_react_loop)
        # Find the "not counting as iteration" warning — that's inside the guard block
        idx = source.find("not counting as iteration")
        assert idx > 0, "Empty response guard warning text not found"
        block = source[idx:idx+800]
        assert "continue" in block, "Empty response handler must use 'continue' to retry"


class TestTodoSuggestionsGraceful:
    """Todo suggestions must not crash or block when no model is loaded."""

    def test_reminder_sent_even_when_suggestions_fail(self):
        """If LLM call fails, reminder should still be sent."""
        import asyncio
        from unittest.mock import patch, AsyncMock, MagicMock
        from src.core.orchestrator import Orchestrator

        loop = asyncio.new_event_loop()

        orch = Orchestrator.__new__(Orchestrator)
        orch.telegram = MagicMock()

        with patch("src.infra.db.get_todos", new_callable=AsyncMock) as mock_todos, \
             patch("src.app.reminders.send_todo_reminder", new_callable=AsyncMock) as mock_remind:

            mock_todos.return_value = [
                {"id": 1, "title": "Buy milk", "suggestion": None, "suggestion_at": None}
            ]
            # Make _generate_suggestions raise
            orch._generate_suggestions = AsyncMock(side_effect=RuntimeError("LLM failed"))

            loop.run_until_complete(orch._start_todo_suggestions())

            # Reminder should still be sent
            mock_remind.assert_called_once()


class TestSummaryQueuePersistent:
    """LLM summary queue must survive restarts — backed by SQLite."""

    def test_pending_summaries_table_exists(self):
        """init_db must create the pending_llm_summaries table."""
        import inspect
        from src.infra import db as db_module
        source = inspect.getsource(db_module.init_db)
        assert "CREATE TABLE IF NOT EXISTS pending_llm_summaries" in source, \
            "pending_llm_summaries table missing from schema"

    def test_queue_function_is_async_and_persists(self):
        """queue_llm_summary must be async and write to SQLite."""
        import inspect
        from src.workflows.engine.hooks import queue_llm_summary
        assert inspect.iscoroutinefunction(queue_llm_summary), \
            "queue_llm_summary must be async (it writes to DB)"
        source = inspect.getsource(queue_llm_summary)
        assert "pending_llm_summaries" in source, \
            "queue_llm_summary must write to pending_llm_summaries table"

    def test_drain_reads_from_db(self):
        """drain_pending_summaries must query the pending_llm_summaries table."""
        import inspect
        from src.workflows.engine.hooks import drain_pending_summaries
        source = inspect.getsource(drain_pending_summaries)
        assert "pending_llm_summaries" in source, \
            "drain_pending_summaries must read from pending_llm_summaries table"
        assert "DELETE FROM pending_llm_summaries" in source, \
            "drain must delete completed jobs"

class TestCheckpointResume:
    """Checkpoints must persist across retries and only clear on final completion."""

    def test_no_checkpoint_clear_in_react_loop(self):
        """_execute_react_loop must NOT clear checkpoints — orchestrator handles it."""
        import inspect
        from src.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent._execute_react_loop)
        assert "_clear_checkpoint" not in source, \
            "_clear_checkpoint still called in _execute_react_loop — should only be in orchestrator"

    def test_handle_complete_clears_checkpoint(self):
        """_handle_complete must clear the checkpoint."""
        import inspect
        from src.core.orchestrator import Orchestrator
        source = inspect.getsource(Orchestrator._handle_complete)
        assert "checkpoint" in source.lower(), \
            "_handle_complete does not reference checkpoint clearing"
