"""Z7 wiring-sweep #1 — C9 changelog/announcement blast has a trigger.

The changelog/publish mr_roboto verb (which fans an announcement email out
to every subscriber) had no production caller: no i2p step, no Telegram
command, and the changelog_freshness posthook only emits advisory text.

The /changelog command is that missing founder-gated trigger. Host-path
coverage: the command lists real draft rows and enqueues a real
changelog/publish mechanical task.
"""
from __future__ import annotations

import pytest


class _FakeMsg:
    def __init__(self):
        self.replies = []

        class _Chat:
            id = 12345
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


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z7_b2.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


async def _insert_draft(db, product_id="12345", *, version="v1.2.0",
                        title="Faster search", published=0):
    cur = await db.execute(
        "INSERT INTO changelog_entries (product_id, version, title, body_md, "
        "published) VALUES (?, ?, ?, ?, ?)",
        (product_id, version, title, "- did stuff", published),
    )
    await db.commit()
    return cur.lastrowid


@pytest.mark.asyncio
async def test_changelog_list_shows_unpublished_drafts(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    await _insert_draft(db, version="v2.0.0", title="Dark mode")
    await _insert_draft(db, version="v1.9.0", title="Shipped", published=1)

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_changelog(update, _FakeCtx())

    body = update.message.replies[0]
    assert "Dark mode" in body, "unpublished draft not listed"
    assert "Shipped" not in body, "published entry must not be listed"


@pytest.mark.asyncio
async def test_changelog_publish_enqueues_mechanical_task(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    entry_id = await _insert_draft(db)

    import general_beckman
    enqueued = []

    async def _fake_enqueue(spec, **kw):
        enqueued.append((spec, kw))
        return 1

    monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_changelog(update, _FakeCtx(args=["publish", str(entry_id)]))

    assert len(enqueued) == 1, "publish must enqueue exactly one task"
    spec, kw = enqueued[0]
    assert spec["payload"]["action"] == "changelog/publish"
    assert spec["payload"]["entry_id"] == entry_id
    assert spec["agent_type"] == "mechanical"
    assert kw.get("lane") == "oneshot"


@pytest.mark.asyncio
async def test_changelog_publish_rejects_unknown_entry(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await db_mod.get_db()

    import general_beckman
    enqueued = []
    monkeypatch.setattr(general_beckman, "enqueue",
                        lambda *a, **k: enqueued.append(a))

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_changelog(update, _FakeCtx(args=["publish", "99999"]))

    assert not enqueued, "must not enqueue for a non-existent entry"
    assert "No changelog draft" in update.message.replies[0]
