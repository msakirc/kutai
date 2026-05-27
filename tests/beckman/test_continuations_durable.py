"""SP1 durable continuation substrate — host-path, DB-isolated tests."""
import asyncio
import json
import pytest

import src.infra.db as _db_mod
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp


async def _fresh_db(tmp_path, monkeypatch):
    """Isolated temp DB per test (mirrors tests/beckman/test_continuations.py)."""
    db_file = tmp_path / "cps.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    monkeypatch.setattr(_cs, "_seeded", False)
    _pp._patterns.clear()
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_continuations_table_and_index_created(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='continuations'"
        )
        assert await cur.fetchone() is not None, "continuations table missing"
        cur = await db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_continuations_pending'"
        )
        assert await cur.fetchone() is not None, "status index missing"
        cur = await db.execute("PRAGMA table_info(continuations)")
        cols = {row[1] for row in await cur.fetchall()}
        assert cols == {
            "child_task_id", "resume_name", "on_error_name",
            "state_json", "status", "created_at",
        }, f"unexpected columns: {cols}"
    finally:
        await _close_db()
