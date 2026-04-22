import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_clarify_variant_choice_sends_keyboard():
    """variant_choice kind: calls send_variant_keyboard and returns needs_clarification."""
    from salako import run

    task = {
        "id": 9,
        "mission_id": 1,
        "title": "Shopping disambiguation",
        "payload": {
            "action": "clarify",
            "kind": "variant_choice",
            "payload_from": "gate_result",
        },
        "artifacts": {
            "gate_result": {
                "gate": {"kind": "clarify"},
                "clarify_options": [
                    {"label": "Galaxy S25", "group_id": 0, "prominence": 2.0},
                    {"label": "Galaxy S25 Ultra", "group_id": 1, "prominence": 1.0},
                ],
                "base_label": "Samsung Galaxy S25",
            }
        },
    }

    with patch("salako.clarify.send_variant_keyboard", new=AsyncMock(return_value=None)) as sent, \
         patch("salako.clarify.update_task", new=AsyncMock()):
        action = await run(task)

    sent.assert_awaited_once_with(
        1, 9, "Samsung Galaxy S25",
        [
            {"label": "Galaxy S25", "group_id": 0, "prominence": 2.0},
            {"label": "Galaxy S25 Ultra", "group_id": 1, "prominence": 1.0},
        ],
    )
    assert action.status == "completed"
    result = action.result or {}
    assert result.get("status") == "needs_clarification"
    assert result.get("kind") == "variant_choice"
    assert "Samsung Galaxy S25" in result.get("prompt", "")


@pytest.mark.asyncio
async def test_clarify_variant_choice_default_payload_from():
    """payload_from defaults to 'gate_result' when omitted."""
    from salako import run

    task = {
        "id": 10,
        "mission_id": 2,
        "title": "Shopping disambiguation",
        "payload": {
            "action": "clarify",
            "kind": "variant_choice",
            # no payload_from — should default to "gate_result"
        },
        "artifacts": {
            "gate_result": {
                "clarify_options": [
                    {"label": "iPhone 15", "group_id": 0, "prominence": 1.5},
                ],
                "base_label": "Apple iPhone 15",
            }
        },
    }

    with patch("salako.clarify.send_variant_keyboard", new=AsyncMock(return_value=None)) as sent, \
         patch("salako.clarify.update_task", new=AsyncMock()):
        action = await run(task)

    sent.assert_awaited_once()
    assert action.status == "completed"
    assert action.result.get("status") == "needs_clarification"


@pytest.mark.asyncio
async def test_clarify_non_variant_still_works():
    """Default clarify (question kind) still works after refactor."""
    task = {
        "id": 42,
        "title": "Book a flight",
        "payload": {
            "action": "clarify",
            "question": "Which city?",
            "chat_id": 111,
        },
    }
    fake_tg = AsyncMock()
    with patch("salako.clarify.get_telegram", return_value=fake_tg), \
         patch("salako.clarify.update_task", new=AsyncMock()) as ut:
        from salako import run
        action = await run(task)
    assert action.status == "completed"
    fake_tg.request_clarification.assert_awaited_once_with(42, "Book a flight", "Which city?")
    ut.assert_awaited_once()
