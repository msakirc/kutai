"""Boot migration converts stale 'ungraded' rows into the post-hook shape."""
import json
import pytest


@pytest.mark.asyncio
async def test_stale_ungraded_row_gets_migrated(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task, get_db
    await init_db()
    # Simulate a pre-refactor 'ungraded' row: no _pending_posthooks, has generating_model.
    source_id = await add_task(
        title="legacy", description="", agent_type="writer", mission_id=1,
        context=json.dumps({"generating_model": "qwen-7b"}),
    )
    await update_task(source_id, status="ungraded", result="legacy output")

    # Reset migration sentinel so the function actually runs.
    from general_beckman import posthook_migration
    posthook_migration._migrated = False

    await posthook_migration.run()

    refreshed = await get_task(source_id)
    ctx = json.loads(refreshed["context"])
    assert ctx["_pending_posthooks"] == ["grade"]

    db = await get_db()
    cursor = await db.execute(
        "SELECT id, context FROM tasks WHERE agent_type='grader'"
    )
    rows = list(await cursor.fetchall())
    assert len(rows) == 1
    grader_ctx = json.loads(rows[0]["context"])
    assert grader_ctx["source_task_id"] == source_id


@pytest.mark.asyncio
async def test_already_migrated_row_unchanged(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task, get_db
    await init_db()
    source_id = await add_task(
        title="already", description="", agent_type="writer", mission_id=1,
        context=json.dumps({
            "generating_model": "qwen-7b",
            "_pending_posthooks": ["grade"],
        }),
    )
    await update_task(source_id, status="ungraded")

    from general_beckman import posthook_migration
    posthook_migration._migrated = False
    await posthook_migration.run()

    # No new grader row should spawn for an already-migrated source.
    db = await get_db()
    cursor = await db.execute(
        "SELECT id FROM tasks WHERE agent_type='grader'"
    )
    rows = list(await cursor.fetchall())
    assert len(rows) == 0
