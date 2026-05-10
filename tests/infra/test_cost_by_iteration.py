"""Z10 T2A D2 — cost_by_iteration view + breakdown API."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "cbi.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


async def _seed_mission_with_tokens(db, mission_id, rows):
    conn = await db.get_db()
    await conn.execute(
        "INSERT INTO tasks (id, mission_id, title, agent_type, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (5000 + mission_id, mission_id, "t", "coder", "completed"),
    )
    for iteration_n, prompt_tokens, completion_tokens, cost_usd in rows:
        await conn.execute(
            "INSERT INTO model_call_tokens "
            "(task_id, model, provider, total_tokens, success, "
            " prompt_tokens, completion_tokens, iteration_n, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                5000 + mission_id,
                "m",
                "p",
                prompt_tokens + completion_tokens,
                1,
                prompt_tokens,
                completion_tokens,
                iteration_n,
                cost_usd,
            ),
        )
    await conn.commit()


@pytest.mark.asyncio
async def test_view_returns_per_iteration_sums(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await _seed_mission_with_tokens(
        db,
        mission_id=11,
        rows=[
            (0, 100, 50, 0.10),
            (0, 200, 100, 0.20),
            (1, 50, 25, 0.05),
            (2, 30, 15, 0.03),
        ],
    )
    rows = await db.get_cost_by_iteration(11)
    assert len(rows) == 3
    by_iter = {r["iteration_n"]: r for r in rows}
    assert by_iter[0]["prompt_tokens"] == 300
    assert by_iter[0]["completion_tokens"] == 150
    assert abs(by_iter[0]["cost_usd"] - 0.30) < 1e-9
    assert by_iter[0]["calls"] == 2
    assert by_iter[1]["calls"] == 1
    assert by_iter[2]["calls"] == 1


@pytest.mark.asyncio
async def test_breakdown_splits_first_pass_vs_retry(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await _seed_mission_with_tokens(
        db,
        mission_id=12,
        rows=[
            (0, 100, 50, 0.10),
            (1, 50, 25, 0.05),
            (2, 30, 15, 0.03),
        ],
    )
    bd = await db.get_mission_cost_breakdown(12)
    assert abs(bd["first_pass_usd"] - 0.10) < 1e-9
    assert abs(bd["retry_usd"] - 0.08) < 1e-9
    assert bd["vendor_usd"] == 0.0
    assert abs(bd["total_usd"] - 0.18) < 1e-9


@pytest.mark.asyncio
async def test_breakdown_includes_vendor_costs(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    await _seed_mission_with_tokens(
        db, mission_id=13, rows=[(0, 100, 50, 0.10)],
    )
    await db.record_vendor_cost(13, "openai", 0.50, "embeddings")
    bd = await db.get_mission_cost_breakdown(13)
    assert abs(bd["vendor_usd"] - 0.50) < 1e-9
    assert abs(bd["total_usd"] - 0.60) < 1e-9
