"""A critic-gate child task is internal veto machinery (agent_type='critic',
kind='overhead') — its completion must NOT emit a user-facing '✅ <title>'
step-progress ping.

Bug (2026-06-18): every mission-scoped notify_user/git_commit spawns an admitted
critic child titled 'critic_gate:notify_user'. `_send_step_progress` skipped
mechanical/reviewer/summarizer but NOT critic, so the user got flooded with
'✅ criticgate:notifyuser' ticks — the gate machinery announcing itself.
"""
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_critic_gate_task_emits_no_step_progress(monkeypatch):
    import general_beckman as gb

    tg = AsyncMock()
    monkeypatch.setattr("src.app.telegram_bot.get_telegram", lambda: tg, raising=False)
    # Live DB shows the critic child completed — without the skip this would
    # pass the gate and ping the user.
    monkeypatch.setattr(
        "dabidabi.get_task",
        AsyncMock(return_value={"status": "completed"}),
        raising=False,
    )

    task = {
        "id": 77, "mission_id": 86, "agent_type": "critic",
        "title": "critic_gate:notify_user", "context": "{}",
    }
    await gb._send_step_progress(task, "completed", {"result": "veto"})

    tg.send_notification.assert_not_called()
