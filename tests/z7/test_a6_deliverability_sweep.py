"""Z7 wiring-sweep #3 — A6 outreach deliverability check has a trigger.

The outreach_deliverability_check verb genuinely writes an outreach_pauses
row when a sender domain degrades, but it had no production trigger: its
PostHookSpec is auto_wire_triggers=[], no workflow step declares it, and no
cron seeded it. The pause was therefore never written in production.

This wires a daily cron sweep. Host-path coverage:
  1. cron_seed.INTERNAL_CADENCES carries the outreach_deliverability_check
     cadence routed to the mr_roboto executor.
  2. A cron-shaped task (payload action only, no product_id/list_id) runs
     the sweep across every active list and writes a real pause row.
"""
from __future__ import annotations

import pytest


def test_cron_seed_has_deliverability_cadence():
    from general_beckman.cron_seed import INTERNAL_CADENCES
    cadence = next(
        (c for c in INTERNAL_CADENCES
         if c.get("title") == "outreach_deliverability_check"), None
    )
    assert cadence is not None, "outreach_deliverability_check cadence missing"
    assert cadence["payload"].get("_executor") == "outreach_deliverability_check"
    assert cadence["interval_seconds"] > 0


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z7_a6.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_cron_tick_sweeps_and_writes_pause(tmp_path, monkeypatch):
    """A cron-tick task carries no product_id/list_id — the dispatch must
    sweep every active list and pause a degraded one."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()

    # 12 sends, 3 bounced → 25% bounce rate, well over the 5% threshold.
    for i in range(12):
        bounced = "2026-05-17 10:00:00" if i < 3 else None
        await db.execute(
            "INSERT INTO outreach_sends "
            "(product_id, list_id, target_email, sent_at, bounced_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("prod-A", "list-A", f"p{i}@example.com",
             "2026-05-17 09:00:00", bounced),
        )
    await db.commit()

    # Cron-shaped mechanical task — no product_id/list_id in payload.
    import mr_roboto
    task = {"id": 1, "agent_type": "mechanical",
            "payload": {"action": "outreach_deliverability_check"}}
    action = await mr_roboto.run(task)
    assert action.status == "completed", action.error
    assert action.result.get("lists_checked") == 1
    assert action.result.get("paused") == 1

    # The real pause row exists — outreach_send Gate 2b will honor it.
    cur = await db.execute(
        "SELECT reason FROM outreach_pauses "
        "WHERE product_id='prod-A' AND list_id='list-A' AND cleared_at IS NULL"
    )
    row = await cur.fetchone()
    assert row is not None, "deliverability sweep did not write a pause row"
