"""A DLQ failure alert is an internal admin status ping, not outward-facing
autonomous-agent comms. It must bypass the SP6 critic gate.

Bug (2026-06-18): mission-scoped DLQ notify_user tasks were critic-gated
(mr_roboto gates notify_user whenever `mission_id is not None`). When the critic
child stalled or its continuation TTL expired, `parse_verdict_strict` fail-closed
to VETO and the notification was dropped — the user stopped receiving "task → DLQ"
alerts for mission tasks (prod: 10 vetoed/TTL-expired, 2 parked forever).

Fix: producers of internal status alerts stamp `critic_gate=False` so the gate
(which honours `payload.get("critic_gate", True)`) never fires for them.
"""
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_dlq_notify_bypasses_critic_gate(monkeypatch):
    import dabidabi

    captured: list[dict] = []

    async def fake_add_task(**kw):
        captured.append(kw)
        return 1

    monkeypatch.setattr(dabidabi, "add_task", fake_add_task, raising=False)
    monkeypatch.setattr(dabidabi, "update_task", AsyncMock(), raising=False)

    import src.infra.dead_letter as dl
    monkeypatch.setattr(dl, "quarantine_task", AsyncMock(return_value=1))
    # Fresh quarantine (no prior unresolved row) so the notify path runs.
    monkeypatch.setattr(dl, "is_unresolved_dlq", AsyncMock(return_value=False), raising=False)

    from general_beckman.apply import _dlq_write

    task = {
        "id": 555, "mission_id": 86, "title": "build feature X",
        "agent_type": "coder", "context": "{}",
    }
    await _dlq_write(task, error="boom", category="exhausted", attempts=3)

    notify = [c for c in captured if (c.get("title") or "").startswith("Notify: DLQ")]
    assert notify, "DLQ notify_user task was not enqueued"
    payload = notify[0]["context"]["payload"]
    assert payload["action"] == "notify_user"
    assert payload.get("critic_gate") is False, (
        "DLQ alert must opt out of the critic gate — a stalled/vetoing critic "
        "silences mission-scoped DLQ notifications"
    )
