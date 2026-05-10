"""Z10 T2A D4 — mission_budget_alerts cron + idempotency."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "mba.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


async def _mission_with_spend(db, mission_id, ceiling, spend):
    """Seed a running mission + spend cost via vendor_cost (simplest path)."""
    conn = await db.get_db()
    await conn.execute(
        "INSERT INTO missions (id, title, description, status) "
        "VALUES (?, ?, ?, ?)",
        (mission_id, f"M{mission_id}", "x", "running"),
    )
    await conn.commit()
    await db.ensure_mission_cost_row(mission_id, budget_ceiling_usd=ceiling)
    await db.record_vendor_cost(mission_id, "synthetic", spend, "test")


@pytest.mark.asyncio
async def test_alert_at_50_pct(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    # 60% of $10 ceiling → 0.5 threshold fires once
    await _mission_with_spend(db, 21, ceiling=10.0, spend=6.0)
    n = await db.check_and_write_mission_budget_alerts()
    assert n == 1
    alerts = await db.get_pending_cost_alerts()
    assert len(alerts) == 1
    assert alerts[0]["mission_id"] == 21
    assert alerts[0]["threshold"] == 0.5
    # Second invocation: idempotent — no new rows.
    n2 = await db.check_and_write_mission_budget_alerts()
    assert n2 == 0
    alerts2 = await db.get_pending_cost_alerts()
    assert len(alerts2) == 1


@pytest.mark.asyncio
async def test_alert_at_75_pct(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await _mission_with_spend(db, 22, ceiling=10.0, spend=8.0)
    n = await db.check_and_write_mission_budget_alerts()
    # 80% → 0.5 and 0.75 both fire (first run)
    assert n == 2
    thresholds = sorted([
        a["threshold"] for a in await db.get_pending_cost_alerts()
    ])
    assert thresholds == [0.5, 0.75]


@pytest.mark.asyncio
async def test_alert_at_90_pct(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await _mission_with_spend(db, 23, ceiling=10.0, spend=9.5)
    n = await db.check_and_write_mission_budget_alerts()
    # 95% → 0.5 + 0.75 + 0.9 all fire on first run
    assert n == 3
    thresholds = sorted([
        a["threshold"] for a in await db.get_pending_cost_alerts()
    ])
    assert thresholds == [0.5, 0.75, 0.9]
