import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_clarify_sends_via_telegram_to_source_task():
    """Mechanical clarify should register the SOURCE (parent) task with
    Telegram, not its own row. The source is what beckman marked
    waiting_human; the mechanical row gets overwritten to completed
    on return so it's the wrong id to wait for a reply against."""
    task = {
        "id": 42,
        "parent_task_id": 100,   # source that requested clarification
        "title": "Book a flight",
        "payload": {
            "action": "clarify",
            "question": "Which city?",
            "chat_id": 111,
        },
    }
    fake_tg = AsyncMock()
    with patch("salako.clarify.get_telegram", return_value=fake_tg):
        from salako import run
        action = await run(task)
    assert action.status == "completed"
    fake_tg.request_clarification.assert_awaited_once_with(100, "Book a flight", "Which city?")
    assert action.result.get("source_task_id") == 100


@pytest.mark.asyncio
async def test_clarify_without_parent_falls_back_to_task_id():
    """Test fixtures / orphan calls without parent_task_id still work."""
    task = {
        "id": 42,
        "title": "T",
        "payload": {"action": "clarify", "question": "Q?"},
    }
    fake_tg = AsyncMock()
    with patch("salako.clarify.get_telegram", return_value=fake_tg):
        from salako import run
        action = await run(task)
    assert action.status == "completed"
    fake_tg.request_clarification.assert_awaited_once_with(42, "T", "Q?")


@pytest.mark.asyncio
async def test_clarify_missing_question_fails():
    from salako import run
    action = await run({"id": 1, "payload": {"action": "clarify"}})
    assert action.status == "failed"
    assert "question" in (action.error or "")
