"""Z10 T2A D8 — record_vendor_cost feeds breakdown.vendor_usd."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "vc.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_vendor_cost_in_breakdown(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await db.record_vendor_cost(47, "openai", 0.50, "embeddings")
    bd = await db.get_mission_cost_breakdown(47)
    assert abs(bd["vendor_usd"] - 0.50) < 1e-9


@pytest.mark.asyncio
async def test_vendor_costs_aggregate_across_vendors(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await db.record_vendor_cost(48, "openai", 0.30, "embeddings")
    await db.record_vendor_cost(48, "anthropic", 0.20, "messages")
    bd = await db.get_mission_cost_breakdown(48)
    assert abs(bd["vendor_usd"] - 0.50) < 1e-9


@pytest.mark.asyncio
async def test_vendor_cost_increments_existing_row(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await db.record_vendor_cost(49, "openai", 0.10, "a")
    await db.record_vendor_cost(49, "openai", 0.15, "b")
    bd = await db.get_mission_cost_breakdown(49)
    assert abs(bd["vendor_usd"] - 0.25) < 1e-9


@pytest.mark.asyncio
async def test_adapter_reexport_calls_db(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    from kuleden_donen_var.cost_adapter import record_vendor_cost
    await record_vendor_cost(50, "openai", 0.40, "x")
    bd = await db.get_mission_cost_breakdown(50)
    assert abs(bd["vendor_usd"] - 0.40) < 1e-9
