"""Z10 T2A D1 — ensure_mission_cost_row + token-write hook."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "cb.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_ensure_mission_cost_row_inserts(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await db.ensure_mission_cost_row(47, budget_ceiling_usd=10.0)
    row = await db.get_budget("mission", "47")
    assert row is not None
    assert row["scope"] == "mission"
    assert row["scope_id"] == "47"
    assert float(row["budget_ceiling_usd"]) == 10.0
    assert float(row["spent_total"]) == 0.0


@pytest.mark.asyncio
async def test_ensure_mission_cost_row_idempotent(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await db.ensure_mission_cost_row(47, budget_ceiling_usd=10.0)
    await db.ensure_mission_cost_row(47, budget_ceiling_usd=10.0)
    await db.ensure_mission_cost_row(47)  # no ceiling change
    conn = await db.get_db()
    cur = await conn.execute(
        "SELECT COUNT(*) FROM cost_budgets WHERE scope='mission' AND scope_id='47'"
    )
    row = await cur.fetchone()
    assert row[0] == 1
    row = await db.get_budget("mission", "47")
    assert float(row["budget_ceiling_usd"]) == 10.0


@pytest.mark.asyncio
async def test_token_write_hook_increments_spent(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await db.ensure_mission_cost_row(47, budget_ceiling_usd=10.0)
    # Seed a task that belongs to mission 47.
    conn = await db.get_db()
    await conn.execute(
        "INSERT INTO tasks (id, mission_id, title, agent_type, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (901, 47, "x", "coder", "completed"),
    )
    await conn.execute(
        "INSERT INTO model_call_tokens "
        "(task_id, model, provider, total_tokens, success, "
        " prompt_tokens, completion_tokens, iteration_n) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (901, "m", "p", 100, 1, 60, 40, 0),
    )
    await conn.commit()

    await db.record_call_cost(task_id=901, cost_usd=0.25)
    row = await db.get_budget("mission", "47")
    assert abs(float(row["spent_total"]) - 0.25) < 1e-9
