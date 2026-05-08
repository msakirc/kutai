"""Tests for in-flight cost estimation ceiling backstop in next_task admission.

Uses mechanical tasks (agent_type='mechanical') to bypass the fatih_hoca
selector gate — mechanical tasks are admitted directly, so the ceiling
backstop is the only thing that can block them in these tests.
"""
import pytest
import aiosqlite
from unittest.mock import AsyncMock, patch


def _reset_db_for(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None
    return db_path


@pytest.mark.asyncio
async def test_in_flight_estimates_block_overshoot(tmp_path, monkeypatch):
    """Two parallel tasks both fit individually but combined exceed ceiling."""
    db_path = _reset_db_for(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd, lifecycle_state) "
            "VALUES ('m', 1.0, 0.0, 'active')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status, agent_type, estimated_cost_usd) "
            "VALUES (?, 't1', 'running', 'mechanical', 0.60)", (mid,),
        )
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status, agent_type, estimated_cost_usd) "
            "VALUES (?, 't2', 'pending', 'mechanical', 0.55)", (mid,),
        )
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    import general_beckman
    with patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=None), create=True):
        task = await general_beckman.next_task()

    # 0.0 spent + 0.60 in-flight + 0.55 new = 1.15 > 1.0 → must block t2 OR return None.
    if task is not None:
        assert task["title"] != "t2"


@pytest.mark.asyncio
async def test_dispatch_when_under_ceiling(tmp_path, monkeypatch):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd, lifecycle_state) "
            "VALUES ('m', 5.0, 0.50, 'active')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status, agent_type, estimated_cost_usd) "
            "VALUES (?, 't', 'pending', 'mechanical', 0.30)", (mid,),
        )
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    import general_beckman
    with patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=None), create=True):
        task = await general_beckman.next_task()

    # 0.50 + 0 in-flight + 0.30 new = 0.80 < 5.0 → dispatches.
    assert task is not None
    assert task["title"] == "t"


@pytest.mark.asyncio
async def test_no_ceiling_no_backstop(tmp_path, monkeypatch):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status, agent_type, estimated_cost_usd) "
            "VALUES (?, 't', 'pending', 'mechanical', 1000.0)", (mid,),
        )
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    import general_beckman
    with patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=None), create=True):
        task = await general_beckman.next_task()

    assert task is not None  # NULL ceiling → no enforcement
