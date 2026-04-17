"""Tests for the mechanical-executor routing in Orchestrator.process_task.

Phase 2a salako integration: tasks with executor='mechanical' must flow
through salako.run() and never reach the LLM dispatch path.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import salako
from src.core.orchestrator import Orchestrator


def _make_orch() -> Orchestrator:
    """Build an Orchestrator without starting any background loops."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.telegram = None
    return orch


@pytest.mark.asyncio
async def test_mechanical_executor_routes_to_salako():
    """executor='mechanical' → salako.run invoked, LLM path skipped, task marked completed."""
    orch = _make_orch()
    task = {
        "id": 1,
        "mission_id": 2,
        "title": "snapshot",
        "agent_type": "mechanical",
        "executor": "mechanical",
        "payload": {"action": "workspace_snapshot", "workspace_path": "/ws"},
    }
    prepared = (task, "mechanical", 60)

    with patch.object(orch, "_prepare", new_callable=AsyncMock, return_value=prepared), \
         patch("src.core.orchestrator.salako.run", new_callable=AsyncMock) as mock_run, \
         patch("src.core.orchestrator.update_task", new_callable=AsyncMock) as mock_update, \
         patch("src.core.orchestrator.get_agent") as mock_get_agent:
        mock_run.return_value = salako.Action(
            status="completed", result={"commit_sha": "abc"}
        )
        await orch.process_task(task)

    mock_run.assert_awaited_once_with(task)
    mock_get_agent.assert_not_called()
    mock_update.assert_awaited_once()
    kwargs = mock_update.call_args.kwargs
    assert kwargs["status"] == "completed"
    assert json.loads(kwargs["result"]) == {"commit_sha": "abc"}


@pytest.mark.asyncio
async def test_mechanical_executor_failure_marks_task_failed():
    """Failed Action → update_task(status='failed', error=...)."""
    orch = _make_orch()
    task = {
        "id": 3,
        "mission_id": 4,
        "title": "t",
        "agent_type": "mechanical",
        "executor": "mechanical",
        "payload": {"action": "unknown"},
    }
    prepared = (task, "mechanical", 60)

    with patch.object(orch, "_prepare", new_callable=AsyncMock, return_value=prepared), \
         patch("src.core.orchestrator.salako.run", new_callable=AsyncMock) as mock_run, \
         patch("src.core.orchestrator.update_task", new_callable=AsyncMock) as mock_update, \
         patch("src.core.orchestrator.get_agent") as mock_get_agent:
        mock_run.return_value = salako.Action(status="failed", error="boom")
        await orch.process_task(task)

    mock_get_agent.assert_not_called()
    kwargs = mock_update.call_args.kwargs
    assert kwargs["status"] == "failed"
    assert kwargs["error"] == "boom"


@pytest.mark.asyncio
async def test_no_executor_tag_still_takes_llm_path():
    """Regression guard: tasks without executor='mechanical' must not call salako.run."""
    orch = _make_orch()
    task = {
        "id": 5,
        "mission_id": 6,
        "title": "code",
        "agent_type": "coder",
        # no executor tag
    }
    prepared = (task, "coder", 60)

    # Intentionally make get_agent raise to short-circuit the LLM path right
    # after routing — we only need to assert that salako.run was NOT called.
    with patch.object(orch, "_prepare", new_callable=AsyncMock, return_value=prepared), \
         patch("src.core.orchestrator.salako.run", new_callable=AsyncMock) as mock_run, \
         patch("src.core.orchestrator.update_task", new_callable=AsyncMock), \
         patch("src.core.orchestrator.get_agent", side_effect=RuntimeError("stop here")):
        try:
            await orch.process_task(task)
        except RuntimeError:
            pass  # outer handler may swallow; either way, we only care about salako

    mock_run.assert_not_awaited()
