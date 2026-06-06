"""critic_gate on notify_user must judge message CONTENT only, never chat_id.

Task #261969 (2026-06-02): a resource-health notify_user was enqueued with no
chat_id (the normal case — notify_user defaults a null recipient to the admin
chat). mr_roboto fed {"message": ..., "chat_id": None} into the critic_gate
LLM, which vetoed "chat_id is null, which will likely cause the action to
fail ... missing recipient" and DLQ'd the task. The recipient is irrelevant to
the critic's mandate (spec break / founder fury / secret leak), and null is
valid, so chat_id must not reach the critic at all.
"""
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_critic_gate_payload_excludes_chat_id(monkeypatch):
    captured = {}

    async def fake_gate(action_name, payload, **kwargs):
        captured["payload"] = payload
        return {"verdict": "pass", "reasons": []}

    import mr_roboto.critic_gate as cg

    monkeypatch.setattr(cg, "critic_gate", fake_gate)
    monkeypatch.setattr(cg, "_opt_out", lambda: False)

    # notify_user task with NO chat_id (defaults to admin in the executor).
    task = {"id": 9, "payload": {"action": "notify_user", "message": "VRAM at 95%"}}
    fake_tg = AsyncMock()
    with patch("mr_roboto.notify_user.get_telegram", return_value=fake_tg):
        from mr_roboto import run

        action = await run(task)

    assert action.status == "completed", action.error
    assert "payload" in captured, "critic_gate was not invoked"
    assert "chat_id" not in captured["payload"], (
        "critic must not receive chat_id — a null recipient baited the LLM into "
        "a spurious validity-veto"
    )
    assert captured["payload"]["message"] == "VRAM at 95%"
