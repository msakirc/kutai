"""Tests for the mechanical-executor routing in Orchestrator._dispatch.

Post-Task-13 refactor: _dispatch routes executor='mechanical' tasks through
salako.run() and never reaches the LLM dispatch path. Result is reported to
beckman.on_task_finished() in both success and failure cases.
"""

from unittest.mock import AsyncMock, patch

import pytest

import salako
from src.core.orchestrator import Orchestrator


def _make_orch() -> Orchestrator:
    """Build an Orchestrator without starting any background loops."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.telegram = None
    orch.shutdown_event = None
    orch._shutting_down = False
    orch._running_futures = []
    orch.running = False
    return orch


@pytest.mark.asyncio
async def test_mechanical_executor_routes_to_salako():
    """executor='mechanical' → salako.run invoked, LLM path skipped, completed reported."""
    orch = _make_orch()
    task = {
        "id": 1,
        "mission_id": 2,
        "title": "snapshot",
        "agent_type": "mechanical",
        "executor": "mechanical",
        "payload": {"action": "workspace_snapshot", "workspace_path": "/ws"},
    }

    with patch("src.core.orchestrator.salako.run", new_callable=AsyncMock) as mock_run, \
         patch("general_beckman.on_task_finished", new_callable=AsyncMock) as mock_finished, \
         patch("src.core.orchestrator.get_agent") as mock_get_agent, \
         patch("src.core.orchestrator.inject_chain_context", new_callable=AsyncMock, return_value=task), \
         patch("src.core.orchestrator.release_task_locks", new_callable=AsyncMock):
        mock_run.return_value = salako.Action(
            status="completed", result={"commit_sha": "abc"}
        )
        await orch._dispatch(task)

    mock_run.assert_awaited_once()
    mock_get_agent.assert_not_called()
    mock_finished.assert_awaited_once()
    call_args = mock_finished.call_args
    assert call_args.args[0] == 1
    assert call_args.args[1]["status"] == "completed"


@pytest.mark.asyncio
async def test_mechanical_executor_failure_reports_failed():
    """Failed Action → on_task_finished called with status='failed'."""
    orch = _make_orch()
    task = {
        "id": 3,
        "mission_id": 4,
        "title": "t",
        "agent_type": "mechanical",
        "executor": "mechanical",
        "payload": {"action": "unknown"},
    }

    with patch("src.core.orchestrator.salako.run", new_callable=AsyncMock) as mock_run, \
         patch("general_beckman.on_task_finished", new_callable=AsyncMock) as mock_finished, \
         patch("src.core.orchestrator.get_agent") as mock_get_agent, \
         patch("src.core.orchestrator.inject_chain_context", new_callable=AsyncMock, return_value=task), \
         patch("src.core.orchestrator.release_task_locks", new_callable=AsyncMock):
        mock_run.return_value = salako.Action(status="failed", error="boom")
        await orch._dispatch(task)

    mock_get_agent.assert_not_called()
    result = mock_finished.call_args.args[1]
    assert result["status"] == "failed"
    assert "boom" in result["error"]


@pytest.mark.asyncio
async def test_no_executor_tag_still_takes_llm_path():
    """Regression guard: tasks without executor='mechanical' must not call salako.run."""
    orch = _make_orch()
    task = {
        "id": 5,
        "mission_id": 6,
        "title": "code",
        "agent_type": "coder",
    }

    # get_agent returns a mock whose .execute short-circuits. We only care
    # that salako.run was NOT called.
    agent_mock = AsyncMock()
    agent_mock.execute = AsyncMock(return_value={"status": "completed", "result": "ok"})

    with patch("src.core.orchestrator.salako.run", new_callable=AsyncMock) as mock_run, \
         patch("general_beckman.on_task_finished", new_callable=AsyncMock), \
         patch("src.core.orchestrator.get_agent", return_value=agent_mock), \
         patch("src.core.orchestrator.inject_chain_context", new_callable=AsyncMock, return_value=task), \
         patch("src.core.orchestrator.release_task_locks", new_callable=AsyncMock):
        await orch._dispatch(task)

    mock_run.assert_not_awaited()
