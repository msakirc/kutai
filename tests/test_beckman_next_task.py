"""Integration tests for general_beckman.next_task().

Exercises the full sweep + cron-fire + pick pipeline through the public API.
Each test uses a fresh temp DB to avoid cross-test interference.
"""
import pytest
import src.infra.db as _db_mod

from general_beckman import next_task, enqueue
from general_beckman.paused_patterns import pause, unpause
from src.infra.db import init_db, add_task, get_task, get_db


async def _fresh(tmp_path, monkeypatch):
    """Reset DB to a fresh temp file for isolation."""
    db_file = tmp_path / "t.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    # Reset cron_seed so each test gets a clean seeder state.
    import general_beckman.cron_seed as cs_mod
    monkeypatch.setattr(cs_mod, "_seeded", False)
    # Reset paused_patterns module state.
    from general_beckman import paused_patterns as _pp
    _pp._patterns.clear()
    await init_db()
    # Push all seeded scheduled_tasks next_run into the future so cron doesn't
    # fire on first next_task() call and insert unexpected tasks.
    conn = await get_db()
    await conn.execute(
        "UPDATE scheduled_tasks SET next_run = datetime('now', '+1 hour')"
    )
    await conn.commit()


@pytest.mark.asyncio
async def test_returns_none_when_queue_empty(tmp_path, monkeypatch):
    await _fresh(tmp_path, monkeypatch)
    result = await next_task()
    assert result is None

    # Teardown
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_returns_single_pending_task(tmp_path, monkeypatch):
    await _fresh(tmp_path, monkeypatch)
    tid = await add_task(title="t", description="", agent_type="coder")
    task = await next_task()
    assert task is not None
    assert task["id"] == tid
    # Task should be claimed (set to processing)
    row = await get_task(tid)
    assert row["status"] == "processing"

    # Teardown
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_paused_pattern_excludes_matching_task(tmp_path, monkeypatch):
    await _fresh(tmp_path, monkeypatch)
    # Add a task that matches the paused pattern (has error_category="quality")
    blocked = await add_task(title="b", description="", agent_type="coder")
    # We need to set error_category on the blocked task directly in DB
    from src.infra.db import update_task
    await update_task(blocked, error_category="quality")
    ok = await add_task(title="o", description="", agent_type="coder")
    pause("category:quality")
    try:
        task = await next_task()
        assert task is not None and task["id"] == ok
    finally:
        unpause("category:quality")

    # Teardown
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
