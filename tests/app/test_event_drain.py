"""Z10 T2B (D4) — drain action_confirmations + mission_budget_alerts."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "drain.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_drain_confirmation_creates_event(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Demo", "x")

    # Insert a task tied to the mission (drain wants mission_id).
    db = await db_mod.get_db()
    cur = await db.execute(
        "INSERT INTO tasks (title, description, mission_id, status) "
        "VALUES (?, ?, ?, ?)", ("t1", "x", mid, "pending"),
    )
    await db.commit()
    task_id = cur.lastrowid

    cid = await db_mod.request_confirmation(
        task_id=task_id, verb="rm", reversibility="irreversible",
        payload_summary="rm /tmp",
    )

    # Patch _get_bot to a stub bot.
    fake_bot = MagicMock()
    fake_bot.send_message = AsyncMock(return_value=MagicMock(message_id=11))

    with patch("mr_roboto.mission_event_drain._get_bot", return_value=fake_bot):
        from mr_roboto.mission_event_drain import run
        res = await run({})

    assert res["confirmations_drained"] == 1

    # action_confirmations.telegram_event_id should be stamped.
    cur = await db.execute(
        "SELECT telegram_event_id FROM action_confirmations WHERE id = ?",
        (cid,),
    )
    row = await cur.fetchone()
    assert row[0] is not None

    # And a mission_events row exists with kind=confirmation_required.
    cur = await db.execute(
        "SELECT kind, mission_id FROM mission_events WHERE id = ?", (row[0],),
    )
    e = await cur.fetchone()
    assert e[0] == "confirmation_required"
    assert e[1] == mid


@pytest.mark.asyncio
async def test_drain_handles_missing_budget_alerts_table(tmp_path, monkeypatch):
    """T2A not merged → mission_budget_alerts absent → drain swallows."""
    await _setup_db(tmp_path, monkeypatch)
    fake_bot = MagicMock()
    fake_bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    with patch("mr_roboto.mission_event_drain._get_bot", return_value=fake_bot):
        from mr_roboto.mission_event_drain import run
        res = await run({})
    # No budget alerts table → returns 0, no crash.
    assert res["budget_alerts_drained"] == 0


@pytest.mark.asyncio
async def test_drain_budget_alert_when_table_present(tmp_path, monkeypatch):
    """If T2A's table exists, drain posts cost_alert events for each row."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Demo", "x")
    db = await db_mod.get_db()

    # T2A's real schema (already created via apply_migration in _setup_db).
    await db.execute(
        "INSERT INTO mission_budget_alerts (mission_id, threshold, total_usd) "
        "VALUES (?, ?, ?)",
        (mid, 0.80, 1.23),
    )
    await db.commit()

    fake_bot = MagicMock()
    fake_bot.send_message = AsyncMock(return_value=MagicMock(message_id=99))
    with patch("mr_roboto.mission_event_drain._get_bot", return_value=fake_bot):
        from mr_roboto.mission_event_drain import run
        res = await run({})
    assert res["budget_alerts_drained"] == 1

    cur = await db.execute(
        "SELECT drained_at FROM mission_budget_alerts"
    )
    row = await cur.fetchone()
    assert row[0] is not None
    cur = await db.execute(
        "SELECT kind FROM mission_events WHERE mission_id = ?", (mid,),
    )
    assert (await cur.fetchone())[0] == "cost_alert"
