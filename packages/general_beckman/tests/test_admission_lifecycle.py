"""Tests for mission lifecycle_state admission gate in next_task / pick_ready_top_k.

Tasks belonging to paused or killed missions must not be admitted.
Standalone tasks (mission_id IS NULL) must still dispatch.
"""
import pytest
import aiosqlite


def _reset_db(db_module, db_path: str):
    db_module._db_connection = None
    db_module._db_connection_path = None
    db_module.DB_PATH = db_path


@pytest.mark.asyncio
async def test_skips_paused_mission(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)

    import src.infra.db as db_module
    _reset_db(db_module, db_path)

    from src.infra.db import init_db
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'paused')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status) VALUES (?, 't', 'pending')",
            (mid,),
        )
        await db.commit()

    _reset_db(db_module, db_path)

    from general_beckman.queue import pick_ready_top_k
    candidates = await pick_ready_top_k(k=5)
    assert candidates == [], f"Expected no candidates for paused mission, got {candidates}"


@pytest.mark.asyncio
async def test_skips_killed_mission(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)

    import src.infra.db as db_module
    _reset_db(db_module, db_path)

    from src.infra.db import init_db
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'killed')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status) VALUES (?, 't', 'pending')",
            (mid,),
        )
        await db.commit()

    _reset_db(db_module, db_path)

    from general_beckman.queue import pick_ready_top_k
    candidates = await pick_ready_top_k(k=5)
    assert candidates == [], f"Expected no candidates for killed mission, got {candidates}"


@pytest.mark.asyncio
async def test_dispatches_active_mission(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)

    import src.infra.db as db_module
    _reset_db(db_module, db_path)

    from src.infra.db import init_db
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status) VALUES (?, 't', 'pending')",
            (mid,),
        )
        await db.commit()

    _reset_db(db_module, db_path)

    from general_beckman.queue import pick_ready_top_k
    candidates = await pick_ready_top_k(k=5)
    assert len(candidates) == 1, f"Expected 1 candidate for active mission, got {candidates}"
    assert candidates[0]["mission_id"] == mid


@pytest.mark.asyncio
async def test_dispatches_standalone_task(tmp_path, monkeypatch):
    """Standalone tasks (mission_id IS NULL) must still dispatch."""
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)

    import src.infra.db as db_module
    _reset_db(db_module, db_path)

    from src.infra.db import init_db
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status) VALUES (NULL, 'standalone', 'pending')"
        )
        await db.commit()

    _reset_db(db_module, db_path)

    from general_beckman.queue import pick_ready_top_k
    candidates = await pick_ready_top_k(k=5)
    assert len(candidates) == 1, f"Expected 1 standalone candidate, got {candidates}"
    assert candidates[0]["mission_id"] is None
