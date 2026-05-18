"""Z8 wiring-sweep — /force_action command.

Z8 shipped the oncall_action executor + verb whitelist but no command to
manually trigger one; /force_action was on the Z8 deferred list. This is
that command. Host-path coverage: a valid call enqueues a real oncall_action
mechanical task; an unknown verb is refused without enqueue.
"""
from __future__ import annotations

import pytest


class _FakeMsg:
    def __init__(self):
        self.replies = []

        class _Chat:
            id = 909
        self.chat = _Chat()

    async def reply_text(self, text, **kw):
        self.replies.append(text)
        return self


class _FakeUpdate:
    def __init__(self):
        self.message = _FakeMsg()

    @property
    def effective_chat(self):
        return self.message.chat


class _FakeCtx:
    def __init__(self, args=None):
        self.args = args or []


def _make_tg():
    from src.app.telegram_bot import TelegramInterface
    tg = TelegramInterface.__new__(TelegramInterface)
    tg._kb_state = {}
    return tg


@pytest.mark.asyncio
async def test_force_action_enqueues_oncall_task(monkeypatch):
    import general_beckman
    enqueued = []

    async def _fake_enqueue(spec, **kw):
        enqueued.append((spec, kw))
        return 1

    monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_force_action(
        update, _FakeCtx(args=["42", "restart_service", '{"svc": "api"}']))

    assert len(enqueued) == 1, "force_action must enqueue one task"
    spec, kw = enqueued[0]
    assert spec["payload"]["action"] == "oncall_action"
    assert spec["payload"]["verb"] == "restart_service"
    assert spec["payload"]["params"] == {"svc": "api"}
    assert spec["mission_id"] == 42
    assert kw.get("lane") == "oneshot"


@pytest.mark.asyncio
async def test_force_action_rejects_unknown_verb(monkeypatch):
    import general_beckman
    enqueued = []
    monkeypatch.setattr(general_beckman, "enqueue",
                        lambda *a, **k: enqueued.append(a))

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_force_action(update, _FakeCtx(args=["42", "delete_everything"]))

    assert not enqueued, "unknown verb must not enqueue"
    assert "Unknown on-call verb" in update.message.replies[0]


@pytest.mark.asyncio
async def test_force_action_usage_on_missing_args(monkeypatch):
    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_force_action(update, _FakeCtx(args=[]))
    assert "Usage" in update.message.replies[0]
