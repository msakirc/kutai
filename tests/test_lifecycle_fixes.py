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
             patch("src.core.llm_dispatcher.get_dispatcher") as mock_disp:
            mock_store.return_value = AsyncMock()
            mock_store.return_value.store = AsyncMock()
            run_async(post_execute_workflow_step(task, result))
            mock_disp.assert_not_called()


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
