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
    """post_execute_workflow_step must NEVER make LLM calls inline.

    LLM summaries are now Beckman-scheduled post-hook tasks
    (RequestPostHook('summary:<name>')), not inline calls. See
    docs/superpowers/specs/2026-04-21-posthook-task-extraction-design.md.
    """

    def test_large_artifact_no_llm_call(self):
        """Artifacts > 3000 chars: store structural summary, no inline LLM."""
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
             patch("src.core.llm_dispatcher.get_dispatcher") as mock_disp:
            mock_store.return_value = AsyncMock()
            mock_store.return_value.store = AsyncMock()
            run_async(post_execute_workflow_step(task, result))
            # Post-hook must NOT make LLM calls (dispatcher untouched).
            mock_disp.assert_not_called()


class TestNoInlineGrading:
    """The agent execution loop must always defer grading — never call
    grade_task inline.

    SP3: the inline grade_task() function is DELETED entirely (grading now runs
    as a raw_dispatch reviewer child + posthook.grade.resume continuation), so
    "no inline grade_task" is now a global guarantee, not a per-loop scan. The
    ReAct loop also moved out of the deleted BaseAgent into coulson.react.run.
    """

    def test_grade_task_symbol_is_deleted(self):
        """grade_task must no longer be importable from src.core.grading."""
        import src.core.grading as grading
        assert not hasattr(grading, "grade_task"), \
            "grade_task should be deleted in SP3 (grading is a CPS post-hook now)"

    def test_react_loop_does_not_reference_grade_task_or_can_grade_now(self):
        """The coulson ReAct loop must not reference inline-grading symbols."""
        import inspect
        from coulson import react
        source = inspect.getsource(react.run)
        assert "grade_task" not in source, \
            "coulson.react.run still references grade_task — inline grading not removed"
        assert "can_grade_now" not in source, \
            "coulson.react.run still references can_grade_now — inline grading not removed"


class TestPrevOutputInjection:
    """All failure types must inject _prev_output into context for next attempt."""

    def test_timeout_path_injects_prev_output(self):
        """Timeout path in _dispatch must inject _prev_output."""
        import inspect
        from src.core.orchestrator import Orchestrator
        dispatch_src = inspect.getsource(Orchestrator._dispatch)
        assert dispatch_src.count('"_prev_output"') >= 1, \
            "timeout path in _dispatch must inject _prev_output"


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


class TestSummaryViaBeckmanPostHook:
    """LLM summary is a Beckman-scheduled post-hook task, not a drain queue.

    Replaces the old TestSummaryQueuePersistent class — queue_llm_summary
    and drain_pending_summaries were removed in the post-hook extraction
    refactor (2026-04-21).
    """

    def test_queue_llm_summary_removed(self):
        from src.workflows.engine import hooks
        assert not hasattr(hooks, "queue_llm_summary"), \
            "queue_llm_summary should be removed (replaced by RequestPostHook)"

    def test_drain_pending_summaries_removed(self):
        from src.workflows.engine import hooks
        assert not hasattr(hooks, "drain_pending_summaries"), \
            "drain_pending_summaries should be removed (queue-driven now)"

    def test_summarizer_and_reviewer_agents_exist(self):
        """SP3: the deleted artifact_summarizer/grader/code_reviewer wrapper
        agents are replaced by the raw_dispatch ``summarizer`` (consumes
        RequestPostHook('summary:*')) and ``reviewer`` (consumes grade /
        code_review) agent configs, whose verdicts apply via the durable
        posthook.summary.resume / posthook.grade.resume continuations."""
        # summarizer is now served from the Prompt Foundry (data Profile), so it
        # is reached via get_agent rather than living in the class AGENT_REGISTRY.
        from src.agents import AGENT_REGISTRY, get_agent
        assert get_agent("summarizer").name == "summarizer", \
            "summarizer must be reachable to consume RequestPostHook('summary:*')"
        assert get_agent("reviewer").name == "reviewer", \
            "reviewer must be reachable to consume grade / code_review post-hooks"
        # The deleted wrapper agents must NOT resolve to a real agent (they fall
        # back to the executor default).
        assert "artifact_summarizer" not in AGENT_REGISTRY
        assert "grader" not in AGENT_REGISTRY
        assert "code_reviewer" not in AGENT_REGISTRY

class TestCheckpointResume:
    """Checkpoints must persist across retries and only clear on final completion."""

    def test_no_checkpoint_clear_in_react_loop(self):
        """_execute_react_loop must NOT clear checkpoints — orchestrator handles it."""
        import inspect
        from src.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent._execute_react_loop)
        assert "_clear_checkpoint" not in source, \
            "_clear_checkpoint still called in _execute_react_loop — should only be in orchestrator"
