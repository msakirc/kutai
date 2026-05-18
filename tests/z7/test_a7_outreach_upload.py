"""Z7 wiring-sweep #4 — /outreach upload is no longer a stub.

Before: /outreach upload only replied text — no list storage, no founder
card, no task. run_outreach_send had no entry point.

Now: upload persists prospects to outreach_prospects + emits a founder
approval card; approving the card drafts + dispatches outreach/draft per
prospect. Host-path coverage exercises both halves.
"""
from __future__ import annotations

import pytest


class _FakeMsg:
    def __init__(self):
        self.replies = []

        class _Chat:
            id = 70701
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
    db_path = tmp_path / "z7_a7_upload.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("OUTREACH_ENABLED", "1")
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_upload_persists_prospects_and_emits_card(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_outreach(update, _FakeCtx(
        args=["upload", "q3list", "alice@acme.com", "bob@beta.io", "junk"]))

    # Two valid emails persisted (junk skipped).
    cur = await db.execute(
        "SELECT target_email, status FROM outreach_prospects "
        "WHERE product_id='70701' AND list_id='q3list'")
    rows = await cur.fetchall()
    assert len(rows) == 2
    assert all(s == "pending" for _, s in rows)

    # A founder approval card was created.
    cur = await db.execute(
        "SELECT expected_output_schema_json FROM founder_actions "
        "WHERE title LIKE 'Approve cold-outreach batch%'")
    fa_row = await cur.fetchone()
    assert fa_row is not None, "no outreach approval card created"
    import json
    schema = json.loads(fa_row[0])
    assert schema["_outreach_approval_pending"] is True
    assert schema["list_id"] == "q3list"


@pytest.mark.asyncio
async def test_approving_card_dispatches_outreach_drafts(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()

    tg = _make_tg()
    await tg.cmd_outreach(_FakeUpdate(), _FakeCtx(
        args=["upload", "q3list", "alice@acme.com", "bob@beta.io"]))

    cur = await db.execute(
        "SELECT id FROM founder_actions "
        "WHERE title LIKE 'Approve cold-outreach batch%'")
    aid = (await cur.fetchone())[0]

    import general_beckman
    enqueued = []

    async def _fake_enqueue(spec, **kw):
        enqueued.append(spec)
        return 1

    monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

    await tg.cmd_action_done(_FakeUpdate(), _FakeCtx(args=[str(aid)]))

    draft_tasks = [e for e in enqueued
                   if e.get("payload", {}).get("action") == "outreach/draft"]
    assert len(draft_tasks) == 2, "approval must dispatch one draft per prospect"
    emails = {t["payload"]["prospect_data"]["email"] for t in draft_tasks}
    assert emails == {"alice@acme.com", "bob@beta.io"}

    # Prospects flipped to approved.
    cur = await db.execute(
        "SELECT COUNT(*) FROM outreach_prospects "
        "WHERE list_id='q3list' AND status='approved'")
    assert (await cur.fetchone())[0] == 2


@pytest.mark.asyncio
async def test_reject_payload_dispatches_nothing(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()

    tg = _make_tg()
    await tg.cmd_outreach(_FakeUpdate(), _FakeCtx(
        args=["upload", "q3list", "alice@acme.com"]))
    cur = await db.execute(
        "SELECT id FROM founder_actions "
        "WHERE title LIKE 'Approve cold-outreach batch%'")
    aid = (await cur.fetchone())[0]

    import general_beckman
    enqueued = []
    monkeypatch.setattr(general_beckman, "enqueue",
                        lambda *a, **k: enqueued.append(a))

    await tg.cmd_action_done(
        _FakeUpdate(), _FakeCtx(args=[str(aid), '{"reject": true}']))

    assert not enqueued, "rejected batch must dispatch nothing"
    cur = await db.execute(
        "SELECT status FROM outreach_prospects WHERE list_id='q3list'")
    assert (await cur.fetchone())[0] == "pending"
