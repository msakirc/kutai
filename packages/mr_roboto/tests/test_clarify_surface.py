"""Tests for the clarify `surface_choice` branch (5.0b surfaces_lock).

Pre-fix: surface_choice fell through to the default branch which raised
``clarify payload requires 'question'`` and DLQ'd the step. Now it sends
a reply keyboard and parks the task as needs_clarification.
"""
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_clarify_surface_choice_sends_keyboard():
    from mr_roboto import run

    task = {
        "id": 12,
        "mission_id": 4,
        "title": "Lock surfaces",
        "chat_id": 888,
        "payload": {
            "action": "clarify",
            "kind": "surface_choice",
            "options": [
                "mobile only",
                "web only",
                "mobile + web",
            ],
        },
    }

    with patch("mr_roboto.clarify.send_surface_keyboard",
               new=AsyncMock(return_value=True)) as sent, \
         patch("mr_roboto.clarify.update_task", new=AsyncMock()) as upd, \
         patch("src.collaboration.blackboard.read_blackboard",
               new=AsyncMock(return_value={})):
        action = await run(task)

    sent.assert_awaited_once_with(
        4, 12, 888, ["mobile only", "web only", "mobile + web"],
    )
    upd.assert_awaited_once()
    assert upd.call_args.kwargs.get("status") == "waiting_human"
    assert action.status == "needs_clarification"
    result = action.result or {}
    assert result.get("kind") == "surface_choice"
    assert result.get("keyboard_sent") is True


@pytest.mark.asyncio
async def test_clarify_surface_choice_no_chat_id_fails_closed():
    """No chat_id → fail-closed (no silent skip of the human gate)."""
    from mr_roboto import run

    task = {
        "id": 13,
        "mission_id": 5,
        "title": "Lock surfaces",
        "payload": {
            "action": "clarify",
            "kind": "surface_choice",
            "options": ["mobile + web"],
        },
    }

    with patch("mr_roboto.clarify.send_surface_keyboard",
               new=AsyncMock(return_value=False)) as sent, \
         patch("mr_roboto.clarify.update_task", new=AsyncMock()) as upd, \
         patch("src.collaboration.blackboard.read_blackboard",
               new=AsyncMock(return_value={})):
        action = await run(task)

    sent.assert_awaited_once()
    upd.assert_not_awaited()
    assert action.status == "failed"
    assert (action.result or {}).get("keyboard_sent") is False
