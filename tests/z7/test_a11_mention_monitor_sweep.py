"""Z7 wiring-sweep #6 — A11 mention monitor gets a trigger.

The mention_polls/<source> verbs work, but the A11 monitor never ran: no
/mention_monitor command registered products, the mention_monitor.json
workflow had no loader, and it was absent from cron_seed. acted_on was
written but read by nothing.

This wires: a mention_monitors registry table, the /mention_monitor command,
an hourly mention_monitor_sweep cron, and a digest that consumes acted_on.

Host-path coverage:
  1. cron_seed carries the mention_monitor_sweep cadence.
  2. /mention_monitor add writes a real mention_monitors row.
  3. the sweep polls registered products and enqueues a digest, flipping
     acted_on on the digested mentions.
"""
from __future__ import annotations

import json

import pytest


class _FakeMsg:
    def __init__(self):
        self.replies = []

        class _Chat:
            id = 55501
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
    db_path = tmp_path / "z7_a11.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


def test_cron_seed_has_mention_monitor_cadence():
    from general_beckman.cron_seed import INTERNAL_CADENCES
    c = next((c for c in INTERNAL_CADENCES
              if c.get("title") == "mention_monitor_sweep"), None)
    assert c is not None, "mention_monitor_sweep cadence missing"
    assert c["payload"].get("_executor") == "mention_monitor_sweep"


@pytest.mark.asyncio
async def test_mention_monitor_add_registers_product(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_mention_monitor(update, _FakeCtx(args=["add", "Acme", "hn", "reddit"]))

    cur = await db.execute(
        "SELECT product_name, channels_json, enabled FROM mention_monitors "
        "WHERE product_id='55501'")
    row = await cur.fetchone()
    assert row is not None, "/mention_monitor add wrote no row"
    assert row[0] == "Acme"
    assert set(json.loads(row[1])) == {"hn", "reddit"}
    assert row[2] == 1


@pytest.mark.asyncio
async def test_sweep_polls_registered_product_and_digests(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()

    # Register a monitor.
    await db.execute(
        "INSERT INTO mention_monitors (product_id, product_name, channels_json) "
        "VALUES ('55501', 'Acme', ?)", (json.dumps(["hn"]),))
    # Two mid-tier mentions pending digest + one high-score (already acted).
    for i, (score, acted) in enumerate([(5, 0), (4, 0), (9, 1)]):
        await db.execute(
            "INSERT INTO mentions (product_id, source, source_id, text, "
            "signal_score, acted_on) VALUES ('55501', 'hn', ?, ?, ?, ?)",
            (f"id{i}", f"mention {i}", score, acted))
    await db.commit()

    # poll_source does real network — stub it.
    polled = []

    async def _fake_poll(*, source, product_id, product_name, config):
        polled.append((source, product_id))
        return {"ingested": 0}

    monkeypatch.setattr("mr_roboto.mention_polls.poll_source", _fake_poll)

    import mr_roboto
    action = await mr_roboto.run(
        {"id": 1, "agent_type": "mechanical",
         "payload": {"action": "mention_monitor_sweep"}})

    assert action.status == "completed", action.error
    assert action.result["monitors"] == 1
    assert ("hn", "55501") in polled, "registered channel not polled"
    assert action.result["digested"] == 2, "score 4-6 mentions not digested"

    # The two mid-tier mentions are now acted_on — acted_on has a consumer.
    cur = await db.execute(
        "SELECT COUNT(*) FROM mentions WHERE product_id='55501' AND acted_on=1")
    assert (await cur.fetchone())[0] == 3
