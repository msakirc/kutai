"""Z10 T2A D5 — estimate_task_cost (history avg + per-kind defaults)."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "etc.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_estimate_uses_history_when_enough_samples(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    conn = await db.get_db()
    # Seed 10 historical (coder, modelX) rows with cost_usd = 0.04 each.
    for i in range(10):
        tid = 7000 + i
        await conn.execute(
            "INSERT INTO tasks (id, title, agent_type, status) "
            "VALUES (?, ?, ?, ?)",
            (tid, "t", "coder", "completed"),
        )
        await conn.execute(
            "INSERT INTO model_call_tokens "
            "(task_id, model, provider, total_tokens, success, cost_usd, "
            " iteration_n, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tid, "modelX", "p", 1000, 1, 0.04, 0, 600, 400),
        )
    await conn.commit()
    est = await db.estimate_task_cost("modelX", "coder")
    assert abs(est - 0.04) < 1e-6


@pytest.mark.asyncio
async def test_estimate_falls_back_with_few_samples(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    # Empty history → per-kind default (coder → 0.05)
    est = await db.estimate_task_cost("modelY", "coder")
    assert est == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_estimate_unknown_kind_uses_fallback(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    est = await db.estimate_task_cost("modelZ", "exotic_kind")
    # _TASK_KIND_DEFAULT_FALLBACK_USD = 0.02
    assert est == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_finalize_task_actual_sums_token_rows(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    conn = await db.get_db()
    await conn.execute(
        "INSERT INTO tasks (id, title, agent_type, status) "
        "VALUES (?, ?, ?, ?)",
        (9000, "t", "coder", "completed"),
    )
    for cost in (0.10, 0.20, 0.05):
        await conn.execute(
            "INSERT INTO model_call_tokens "
            "(task_id, model, provider, total_tokens, success, cost_usd, "
            " iteration_n) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (9000, "m", "p", 100, 1, cost, 0),
        )
    await conn.commit()
    actual = await db.finalize_task_actual_cost(9000)
    assert abs(actual - 0.35) < 1e-9
    # Stamped on the task row.
    t = await db.get_task(9000)
    assert abs(float(t["actual_cost_usd"]) - 0.35) < 1e-9
