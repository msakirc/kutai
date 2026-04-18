import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_clarify_sends_via_telegram_and_marks_pending():
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


@pytest.mark.asyncio
async def test_clarify_missing_question_fails():
    from salako import run
    action = await run({"id": 1, "payload": {"action": "clarify"}})
    assert action.status == "failed"
    assert "question" in (action.error or "")
