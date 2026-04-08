# tests/test_db_migration.py
"""Tests for DB migration: retry pipeline overhaul columns."""
import pytest
import asyncio
import tempfile
import os
import importlib


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _fresh_db():
    """Create a fresh DB in a temp dir, resetting module state."""
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    os.environ["DB_PATH"] = db_path
    import src.infra.db as db_mod
    # Reset cached connection
    db_mod._db_connection = None
    importlib.reload(db_mod)
    await db_mod.init_db()
    return db_mod, await db_mod.get_db()


def test_fresh_db_has_new_columns():
    """Fresh DB should have worker_attempts, infra_resets, exhaustion_reason."""
    async def _test():
        db_mod, db = await _fresh_db()
        cursor = await db.execute("PRAGMA table_info(tasks)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "worker_attempts" in columns
        assert "infra_resets" in columns
        assert "exhaustion_reason" in columns
        assert "max_worker_attempts" in columns
        # Old columns should NOT exist in fresh DB
        assert "attempts" not in columns
        assert "max_attempts" not in columns
        await db_mod.close_db()
    run_async(_test())


def test_fresh_db_has_no_legacy_retry_count():
    """Fresh DB should not have retry_count or max_retries columns."""
    async def _test():
        db_mod, db = await _fresh_db()
        cursor = await db.execute("PRAGMA table_info(tasks)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "retry_count" not in columns
        assert "max_retries" not in columns
        await db_mod.close_db()
    run_async(_test())


def test_dlq_fresh_has_attempts_snapshot():
    """Fresh DLQ table should have attempts_snapshot, not retry_count."""
    async def _test():
        db_mod, db = await _fresh_db()

        from src.infra.dead_letter import _ensure_dlq_table
        await _ensure_dlq_table()

        cursor = await db.execute("PRAGMA table_info(dead_letter_tasks)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "attempts_snapshot" in columns
        assert "retry_count" not in columns
        await db_mod.close_db()
    run_async(_test())


def test_fresh_db_has_lifecycle_columns():
    """Fresh DB should have all unified lifecycle columns."""
    async def _test():
        db_mod, db = await _fresh_db()
        cursor = await db.execute("PRAGMA table_info(tasks)")
        columns = {row[1] for row in await cursor.fetchall()}
        # All lifecycle columns should be present
        for col in [
            "worker_attempts", "max_worker_attempts",
            "grade_attempts", "max_grade_attempts",
            "next_retry_at", "retry_reason", "failed_in_phase",
            "infra_resets", "exhaustion_reason",
        ]:
            assert col in columns, f"Missing column: {col}"
        await db_mod.close_db()
    run_async(_test())
