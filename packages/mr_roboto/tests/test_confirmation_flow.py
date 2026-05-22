"""Z10 §2.C — mr_roboto confirmation gate: park/resume via clarify path.

The old skeleton busy-polled ``check_confirmation`` for 60s holding the
worker slot, and hard-failed on timeout. The gate now reuses the existing
clarification park/resume machinery:

* First entry (no ``user_clarification`` in context) → send a Telegram
  question, mark the task ``waiting_human``, return
  ``Action(status="needs_clarification")``. The orchestrator special-cases
  that status for mechanical tasks and skips ``on_task_finished``, leaving
  the row parked (orchestrator.py:316-446 — no change needed there).
* Resume (founder's typed reply → ``_resume_with_clarification`` injected
  ``context["user_clarification"]`` and reset status to ``pending`` →
  Beckman re-dispatches → ``mr_roboto.run`` re-runs → gate runs again):
  approve tokens → ``None`` (proceed); reject tokens →
  ``Action(status="rejected")``; anything else → fail-closed
  ``Action(status="rejected")``.

These tests are host-path / fully isolated: ``get_telegram`` is monkeypatched
to a fake recorder and ``src.infra.db.update_task`` is monkeypatched — the
live DB is NEVER touched (it hangs on the orchestrator's SQLite lock).
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

import mr_roboto


class _FakeTelegram:
    """Records request_clarification calls without touching Telegram/DB."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def request_clarification(self, task_id, title, question):
        self.calls.append((task_id, title, question))


def _patch_clean(monkeypatch):
    """Install fake telegram + recording update_task; ban asyncio.sleep.

    Returns ``(fake_tg, update_calls)``. ``update_calls`` is a list of
    ``(task_id, kwargs)`` tuples recorded from ``update_task``.
    """
    fake_tg = _FakeTelegram()
    monkeypatch.setattr(
        "src.app.telegram_bot.get_telegram", lambda: fake_tg, raising=True
    )

    update_calls: list[tuple] = []

    async def _fake_update_task(task_id, **kwargs):
        update_calls.append((task_id, kwargs))

    monkeypatch.setattr(
        "src.infra.db.update_task", _fake_update_task, raising=True
    )

    # Busy-poll regression guard: no path may sleep. Any await on
    # asyncio.sleep blows up the test.
    async def _no_sleep(*_a, **_k):
        raise AssertionError("_await_confirmation must NOT sleep (busy-poll)")

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    return fake_tg, update_calls


# ── 1. First entry parks via clarify ────────────────────────────────────
@pytest.mark.asyncio
async def test_first_entry_parks_and_asks(monkeypatch):
    fake_tg, update_calls = _patch_clean(monkeypatch)

    task = {
        "id": 101,
        "mission_id": 7,
        "title": "Push to App Store",
        "context": {},  # no user_clarification → first entry
    }
    action = await mr_roboto._await_confirmation(
        task=task,
        verb="appstore_submit",
        reversibility="irreversible",
        payload={},
    )

    assert action is not None
    assert action.status == "needs_clarification"
    # Telegram asked exactly once, on the right task, with a real question.
    assert len(fake_tg.calls) == 1
    asked_task_id, _title, question = fake_tg.calls[0]
    assert asked_task_id == 101
    assert isinstance(question, str) and question.strip()
    # Task parked to waiting_human.
    assert any(
        tid == 101 and kw.get("status") == "waiting_human"
        for tid, kw in update_calls
    )


# ── 2. Resume approve → proceed (None) ───────────────────────────────────
@pytest.mark.parametrize("token", ["evet", "yes", "approved", "✅", "OK", "Onayla"])
@pytest.mark.asyncio
async def test_resume_approve_proceeds(monkeypatch, token):
    fake_tg, update_calls = _patch_clean(monkeypatch)

    task = {
        "id": 102,
        "mission_id": 7,
        "title": "Push to App Store",
        "context": {"user_clarification": token},
    }
    action = await mr_roboto._await_confirmation(
        task=task,
        verb="appstore_submit",
        reversibility="irreversible",
        payload={},
    )
    assert action is None  # proceed with dispatch
    # Resume path must not re-ask or re-park.
    assert fake_tg.calls == []
    assert update_calls == []


# ── 3. Resume reject → rejected ──────────────────────────────────────────
@pytest.mark.parametrize("token", ["hayır", "no", "reddet", "iptal", "❌"])
@pytest.mark.asyncio
async def test_resume_reject_blocks(monkeypatch, token):
    fake_tg, _update_calls = _patch_clean(monkeypatch)

    task = {
        "id": 103,
        "mission_id": 7,
        "title": "Push to App Store",
        "context": {"user_clarification": token},
    }
    action = await mr_roboto._await_confirmation(
        task=task,
        verb="appstore_submit",
        reversibility="irreversible",
        payload={},
    )
    assert action is not None
    assert action.status == "rejected"
    assert fake_tg.calls == []


# ── 4. Resume ambiguous → fail-closed (rejected) ─────────────────────────
@pytest.mark.asyncio
async def test_resume_ambiguous_fails_closed(monkeypatch):
    fake_tg, _update_calls = _patch_clean(monkeypatch)

    task = {
        "id": 104,
        "mission_id": 7,
        "title": "Push to App Store",
        "context": {"user_clarification": "belki sonra"},
    }
    action = await mr_roboto._await_confirmation(
        task=task,
        verb="appstore_submit",
        reversibility="irreversible",
        payload={},
    )
    assert action is not None
    assert action.status == "rejected"
    # Error must signal the fail-closed reason.
    assert action.error and (
        "fail-closed" in action.error.lower()
        or "not understood" in action.error.lower()
    )
    assert fake_tg.calls == []


# ── 5. Telegram unavailable on first entry → fail-closed ─────────────────
@pytest.mark.asyncio
async def test_first_entry_telegram_unavailable_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "src.app.telegram_bot.get_telegram", lambda: None, raising=True
    )

    update_calls: list[tuple] = []

    async def _fake_update_task(task_id, **kwargs):
        update_calls.append((task_id, kwargs))

    monkeypatch.setattr(
        "src.infra.db.update_task", _fake_update_task, raising=True
    )

    task = {
        "id": 105,
        "mission_id": 7,
        "title": "Push to App Store",
        "context": {},
    }
    action = await mr_roboto._await_confirmation(
        task=task,
        verb="appstore_submit",
        reversibility="irreversible",
        payload={},
    )
    assert action is not None
    assert action.status == "failed"
    # A human gate that can't reach the human must not be skippable, and
    # must not have parked the task.
    assert update_calls == []


# ── 6. No busy-poll: signature has no poll params + no sleep ─────────────
@pytest.mark.asyncio
async def test_no_busy_poll_signature_and_no_sleep(monkeypatch):
    sig = inspect.signature(mr_roboto._await_confirmation)
    params = set(sig.parameters)
    assert "max_wait_s" not in params
    assert "poll_interval_s" not in params
    assert "task" in params

    # And no path sleeps: _patch_clean bans asyncio.sleep. Exercise the
    # first-entry, resume-approve, resume-reject and ambiguous paths; if
    # any slept, _no_sleep would raise.
    fake_tg, _uc = _patch_clean(monkeypatch)

    base = {"id": 106, "mission_id": 7, "title": "X"}
    # first entry
    await mr_roboto._await_confirmation(
        task={**base, "context": {}},
        verb="appstore_submit", reversibility="irreversible", payload={},
    )
    # approve
    r = await mr_roboto._await_confirmation(
        task={**base, "context": {"user_clarification": "evet"}},
        verb="appstore_submit", reversibility="irreversible", payload={},
    )
    assert r is None
    # reject
    r = await mr_roboto._await_confirmation(
        task={**base, "context": {"user_clarification": "no"}},
        verb="appstore_submit", reversibility="irreversible", payload={},
    )
    assert r.status == "rejected"
    # ambiguous
    r = await mr_roboto._await_confirmation(
        task={**base, "context": {"user_clarification": "huh"}},
        verb="appstore_submit", reversibility="irreversible", payload={},
    )
    assert r.status == "rejected"
