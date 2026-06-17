"""A task that is ALREADY in the dead-letter queue (unresolved) must not emit a
second 'Notify: DLQ task #X' alert when _dlq_write runs again.

Bug (2026-06-18): on a fail-looping mission, the same task is re-DLQ'd on each
retry/re-pend cycle. `quarantine_task` is idempotent (UNIQUE(task_id),
INSERT OR REPLACE — one row), but `_dlq_write` sent a fresh notify (+ demand
signal + critic gate) every call. Prod mission 86: 21 notifies for 9 distinct
tasks; #459160 and #459147 each alerted 4×.

Fix: _dlq_write notifies only when this is a FRESH quarantine (no prior
unresolved DLQ row).
"""
from unittest.mock import AsyncMock

import pytest


async def _run_dlq_write(monkeypatch, *, already_in_dlq: bool):
    import dabidabi
    captured: list[dict] = []

    async def fake_add_task(**kw):
        captured.append(kw)
        return 1

    monkeypatch.setattr(dabidabi, "add_task", fake_add_task, raising=False)
    monkeypatch.setattr(dabidabi, "update_task", AsyncMock(), raising=False)

    import src.infra.dead_letter as dl
    monkeypatch.setattr(dl, "quarantine_task", AsyncMock(return_value=1))
    monkeypatch.setattr(
        dl, "is_unresolved_dlq",
        AsyncMock(return_value=already_in_dlq), raising=False,
    )

    from general_beckman.apply import _dlq_write
    task = {
        "id": 459160, "mission_id": 86, "title": "interview script",
        "agent_type": "researcher", "context": "{}",
    }
    await _dlq_write(task, error="boom", category="exhausted", attempts=5)
    return [c for c in captured if (c.get("title") or "").startswith("Notify: DLQ")]


@pytest.mark.asyncio
async def test_redlq_of_already_quarantined_task_does_not_renotify(monkeypatch):
    notifies = await _run_dlq_write(monkeypatch, already_in_dlq=True)
    assert notifies == [], (
        "task already in the DLQ was re-notified — duplicate alert flood"
    )


@pytest.mark.asyncio
async def test_fresh_quarantine_still_notifies(monkeypatch):
    notifies = await _run_dlq_write(monkeypatch, already_in_dlq=False)
    assert len(notifies) == 1, "fresh DLQ must still send exactly one alert"
