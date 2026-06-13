"""critic_gate on notify_user must judge message CONTENT only, never chat_id.

Task #261969 (2026-06-02): a resource-health notify_user was enqueued with no
chat_id (the normal case — notify_user defaults a null recipient to the admin
chat). mr_roboto fed {"message": ..., "chat_id": None} into the critic_gate
LLM, which vetoed "chat_id is null, which will likely cause the action to
fail ... missing recipient" and DLQ'd the task. The recipient is irrelevant to
the critic's mandate (spec break / founder fury / secret leak), and null is
valid, so chat_id must not reach the critic at all.

SP6 T4: notify_user is now a two-pass self-park gated ONLY for mission-scoped
tasks. The chat_id-exclusion guarantee still holds: pass 1 builds the critic
spec from {"message": text} only, so chat_id can never reach the critic. This
test was repointed from the old inline single-pass call to the pass-1 enqueue
path (and given a mission_id so the gate actually fires).
"""
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_critic_gate_payload_excludes_chat_id(monkeypatch):
    captured = {}

    enq = AsyncMock(return_value=1)
    upd = AsyncMock()
    monkeypatch.setattr("mr_roboto.enqueue", enq, raising=False)
    monkeypatch.setattr("mr_roboto.update_task", upd, raising=False)

    import mr_roboto.critic_gate as cg

    def fake_build_spec(action_name, redacted):
        captured["redacted"] = redacted
        return {"title": "critic", "payload": {"action": "critic_gate"}}

    monkeypatch.setattr(cg, "_build_critic_spec", fake_build_spec)
    monkeypatch.setattr(cg, "_opt_out", lambda: False)

    # MISSION-SCOPED notify_user with a chat_id present. Pass 1 must build the
    # critic spec from {"message": ...} only — chat_id excluded.
    task = {"id": 9, "mission_id": 4, "context": "{}",
            "payload": {"action": "notify_user", "message": "VRAM at 95%",
                        "chat_id": 222}}
    fake_tg = AsyncMock()
    with patch("mr_roboto.notify_user.get_telegram", return_value=fake_tg):
        from mr_roboto import run

        action = await run(task)

    # Pass 1 parks (no send yet).
    assert action.status == "needs_clarification", action.error
    assert "redacted" in captured, "critic spec was not built (gate did not fire)"
    assert "chat_id" not in captured["redacted"], (
        "critic must not receive chat_id — a null recipient baited the LLM into "
        "a spurious validity-veto"
    )
    assert captured["redacted"]["message"] == "VRAM at 95%"
