import os
import pytest
import aiosqlite

from src.infra.db import init_db


@pytest.mark.asyncio
async def test_z0_columns_added(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)

    # Reset the module-level DB_PATH after monkeypatching the env var
    import src.infra.db as db_module
    db_module.DB_PATH = db_path
    db_module._db_connection = None
    db_module._db_connection_path = None

    # Pre-seed a missions row to simulate pre-Z0 install
    async with aiosqlite.connect(db_path) as db:
        await db.execute("CREATE TABLE missions (id INTEGER PRIMARY KEY, title TEXT)")
        await db.execute("INSERT INTO missions (title) VALUES ('legacy')")
        await db.commit()

        # Verify pre-seeded table
        cur = await db.execute("PRAGMA table_info(missions)")
        cols_before = [row[1] for row in await cur.fetchall()]
        assert cols_before == ['id', 'title']

    # Now run init_db
    await init_db()

    # Reopen and check
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(missions)")
        cols = {row[1] for row in await cur.fetchall()}
        assert "cost_ceiling_usd" in cols
        assert "spent_usd" in cols
        assert "message_thread_id" in cols
        assert "lifecycle_state" in cols

        cur = await db.execute(
            "SELECT lifecycle_state, spent_usd, cost_ceiling_usd "
            "FROM missions WHERE title = 'legacy'"
        )
        row = await cur.fetchone()
        assert row == ("active", 0, None)

        cur = await db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='mission_lifecycle_log'"
        )
        assert await cur.fetchone() is not None


@pytest.mark.asyncio
async def test_z0_migration_idempotent(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)

    # Reset the module-level DB_PATH after monkeypatching the env var
    import src.infra.db as db_module
    db_module.DB_PATH = db_path
    db_module._db_connection = None
    db_module._db_connection_path = None

    await init_db()
    await init_db()  # second pass — must not raise
