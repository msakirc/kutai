"""Tests for general_beckman.lifecycle."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_handle_clarification_emits_salako_task():
    """handle_clarification must emit a mechanical task with action='clarify',
    NOT call Telegram directly."""
    from general_beckman.lifecycle import handle_clarification
    task = {"id": 5, "title": "plan trip", "mission_id": 2, "chat_id": 99}
    result = {"clarification": "Which dates?"}
    with patch("general_beckman.lifecycle.add_task", new=AsyncMock()) as at, \
         patch("general_beckman.lifecycle.update_task", new=AsyncMock()) as ut:
        await handle_clarification(task, result)
    at.assert_awaited_once()
    kwargs = at.await_args.kwargs
    assert kwargs["agent_type"] == "mechanical"
    assert kwargs["payload"]["action"] == "clarify"
    assert kwargs["payload"]["question"] == "Which dates?"
    assert kwargs["payload"]["chat_id"] == 99
    ut.assert_awaited_once_with(5, status="waiting_human")


@pytest.mark.asyncio
async def test_handle_clarification_uses_fallback_question():
    from general_beckman.lifecycle import handle_clarification
    task = {"id": 1, "title": "x"}
    with patch("general_beckman.lifecycle.add_task", new=AsyncMock()) as at, \
         patch("general_beckman.lifecycle.update_task", new=AsyncMock()):
        await handle_clarification(task, {})
    kwargs = at.await_args.kwargs
    assert "more information" in kwargs["payload"]["question"].lower()


@pytest.mark.asyncio
async def test_on_task_finished_missing_task_returns():
    from general_beckman.lifecycle import on_task_finished
    with patch("general_beckman.lifecycle.get_task", new=AsyncMock(return_value=None)):
        # Must not raise
        await on_task_finished(999, {"status": "completed"})


@pytest.mark.asyncio
async def test_on_task_finished_routes_complete_to_handler():
    """on_task_finished fetches task, routes result, dispatches to handle_complete."""
    from general_beckman import lifecycle
    fake_task = {"id": 1, "title": "t"}
    fake_orch = MagicMock()
    fake_orch._handle_complete = AsyncMock()
    lifecycle.set_orchestrator(fake_orch)
    try:
        with patch("general_beckman.lifecycle.get_task", new=AsyncMock(return_value=fake_task)):
            await lifecycle.on_task_finished(1, {"status": "completed", "result": "ok"})
        fake_orch._handle_complete.assert_awaited_once()
    finally:
        lifecycle.set_orchestrator(None)


@pytest.mark.asyncio
async def test_on_task_finished_routes_clarification_to_salako_emit():
    """A clarification result routes through handle_clarification, which emits
    a mechanical salako task (no orchestrator method is called)."""
    from general_beckman import lifecycle
    fake_task = {"id": 7, "title": "book", "mission_id": None, "chat_id": 55}
    # Build a result that route_result maps to RequestClarification
    result = {"status": "needs_clarification", "clarification": "Which city?"}
    with patch("general_beckman.lifecycle.get_task", new=AsyncMock(return_value=fake_task)), \
         patch("general_beckman.lifecycle.add_task", new=AsyncMock()) as at, \
         patch("general_beckman.lifecycle.update_task", new=AsyncMock()):
        await lifecycle.on_task_finished(7, result)
    at.assert_awaited_once()
    assert at.await_args.kwargs["payload"]["action"] == "clarify"


@pytest.mark.asyncio
async def test_set_orchestrator_accessor_roundtrip():
    from general_beckman import lifecycle
    obj = object()
    lifecycle.set_orchestrator(obj)
    assert lifecycle.get_orchestrator() is obj
    lifecycle.set_orchestrator(None)
    with pytest.raises(RuntimeError):
        lifecycle.get_orchestrator()
