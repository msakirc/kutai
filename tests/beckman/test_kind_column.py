"""TDD tests for tasks.kind column (Task 1 — Beckman admission migration)."""
import pytest
import importlib

import src.infra.db as _db_mod


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "kind_col.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_kind_column_exists_after_init_db(tmp_path, monkeypatch):
    """PRAGMA table_info(tasks) must include a 'kind' column after init_db."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        db = await _db_mod.get_db()
        cursor = await db.execute("PRAGMA table_info(tasks)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "kind" in columns, f"'kind' column missing from tasks. Got: {sorted(columns)}"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_add_task_kind_defaults_to_main_work(tmp_path, monkeypatch):
    """A freshly inserted task row must have kind == 'main_work' by default."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        task_id = await _db_mod.add_task(title="t", description="d")
        assert task_id is not None, "add_task returned None (dedup collision?)"
        row = await _db_mod.get_task(task_id)
        assert row is not None
        assert row["kind"] == "main_work", (
            f"Expected kind='main_work', got {row['kind']!r}"
        )
    finally:
        await _close_db()
