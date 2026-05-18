"""Tests for attention_check — Z1 T5A (A5) founder attention budget."""
from __future__ import annotations

import os

import pytest

from mr_roboto.attention_check import (
    attention_check,
    attention_debit,
    write_deferred_question,
)


@pytest.fixture(autouse=True)
async def _db_reset():
    import src.infra.db as _dbmod
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None
    yield
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None


async def _setup_db(tmp_path, monkeypatch):
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db, get_db
    await init_db()
    return await get_db()


async def _create_mission(db, mission_id: int, budget: int | None) -> None:
    if budget is None:
        await db.execute(
            "INSERT INTO missions (id, title, status) VALUES (?, ?, ?)",
            (mission_id, f"m{mission_id}", "active"),
        )
    else:
        await db.execute(
            "INSERT INTO missions (id, title, status, "
            "founder_attention_budget_minutes) VALUES (?, ?, ?, ?)",
            (mission_id, f"m{mission_id}", "active", budget),
        )
    await db.commit()


@pytest.mark.asyncio
async def test_no_budget_returns_unbounded(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 1, None)

    res = await attention_check(mission_id=1, reserve_minutes=30)
    assert res["ok"] is True
    assert res["budget_set"] is False
    assert res["remaining"] is None
    assert res["would_exceed"] is False


@pytest.mark.asyncio
async def test_budget_with_no_debits_returns_full(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 1, 60)

    res = await attention_check(mission_id=1, reserve_minutes=10)
    assert res["ok"] is True
    assert res["budget_set"] is True
    assert res["remaining"] == 60
    assert res["spent"] == 0


@pytest.mark.asyncio
async def test_debits_subtracted(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 1, 60)

    await attention_debit(mission_id=1, step_id="0.5", action="clarify", minutes_debited=20)
    await attention_debit(mission_id=1, step_id="0.6a", action="clarify", minutes_debited=15)

    res = await attention_check(mission_id=1, reserve_minutes=10)
    assert res["ok"] is True
    assert res["spent"] == 35
    assert res["remaining"] == 25


@pytest.mark.asyncio
async def test_would_exceed_when_remaining_below_reserve(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 1, 60)
    await attention_debit(mission_id=1, step_id="0.5", action="clarify", minutes_debited=58)

    res = await attention_check(mission_id=1, reserve_minutes=5)
    assert res["ok"] is False
    assert res["would_exceed"] is True
    assert res["remaining"] == 2


@pytest.mark.asyncio
async def test_unknown_mission_unbounded(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    res = await attention_check(mission_id=999, reserve_minutes=10)
    # No row → treat as unbounded so missing-row never blocks the dispatcher.
    assert res["ok"] is True
    assert res["budget_set"] is False


@pytest.mark.asyncio
async def test_write_deferred_question_creates_file(tmp_path):
    res = await write_deferred_question(
        mission_id=1,
        step_id="0.5",
        question_text="Pick a tagline",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    body = open(res["path"], encoding="utf-8").read()
    assert "Pick a tagline" in body
    assert "0.5" in body


@pytest.mark.asyncio
async def test_write_deferred_question_appends(tmp_path):
    await write_deferred_question(
        mission_id=1, step_id="0.5", question_text="Q1", workspace_path=str(tmp_path),
    )
    res = await write_deferred_question(
        mission_id=1, step_id="0.6a", question_text="Q2", workspace_path=str(tmp_path),
    )
    body = open(res["path"], encoding="utf-8").read()
    assert "Q1" in body and "Q2" in body
    # Header should appear exactly once.
    assert body.count("# Deferred clarify questions") == 1


@pytest.mark.asyncio
async def test_dispatch_through_run(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 1, 60)

    from mr_roboto import run
    task = {
        "id": 99,
        "mission_id": 1,
        "payload": {
            "action": "attention_check",
            "reserve_minutes": 5,
        },
    }
    result = await run(task)
    assert result.status == "completed"
    assert result.result["ok"] is True
    assert result.result["remaining"] == 60


@pytest.mark.asyncio
async def test_dispatch_attention_debit_through_run(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 1, 60)

    from mr_roboto import run
    task = {
        "id": 99,
        "mission_id": 1,
        "payload": {
            "action": "attention_debit",
            "step_id": "0.5",
            "debit_action": "clarify",
            "minutes_debited": 7,
        },
    }
    result = await run(task)
    assert result.status == "completed"

    res = await attention_check(mission_id=1, reserve_minutes=5)
    assert res["spent"] == 7
