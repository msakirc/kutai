import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_notify_user_sends_message():
    task = {"id": 7, "payload": {"action": "notify_user", "chat_id": 222, "text": "Mission done"}}
    fake_tg = AsyncMock()
    with patch("salako.notify_user.get_telegram", return_value=fake_tg):
        from salako import run
        action = await run(task)
    assert action.status == "completed"
    fake_tg.send_message.assert_awaited_once_with(222, "Mission done")


@pytest.mark.asyncio
async def test_notify_user_missing_text_fails():
    from salako import run
    action = await run({"id": 1, "payload": {"action": "notify_user", "chat_id": 1}})
    assert action.status == "failed"
