import pytest
import aiosqlite

import src.infra.db as db_module


@pytest.mark.asyncio
async def test_tasks_estimated_cost_usd_column(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None

    from src.infra.db import init_db
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(tasks)")
        cols = {row[1] for row in await cur.fetchall()}
        assert "estimated_cost_usd" in cols
