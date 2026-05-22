"""Schema + writer must populate model_pick_log.provider correctly.

Isolation technique (mirrors test_pick_log_task_id.py):
  - Tests that call write_pick_log_row reset the db module singleton
    (DB_PATH + _db_connection = None) to a fresh temp DB, then call
    the real init_db() to build the full production schema.
  - get_db() therefore returns the temp DB connection — no kutai.db touch,
    no WAL lock contention.
  - Tests that do NOT call write_pick_log_row (schema-only or migration-only
    tests) can still use an inline DDL for speed since they never go through
    the singleton.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import aiosqlite
import pytest

from src.infra.pick_log import write_pick_log_row


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _fresh_db():
    """Create a fresh temp DB with the real production schema via init_db()."""
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    import src.infra.db as db_mod

    db_mod.DB_PATH = db_path
    db_mod._db_connection = None
    os.environ["DB_PATH"] = db_path
    await db_mod.init_db()
    return db_mod, db_path


async def _close_db(db_mod):
    try:
        conn = getattr(db_mod, "_db_connection", None)
        if conn is not None:
            await conn.close()
        db_mod._db_connection = None
    except Exception:
        pass


# ── Legacy DDL for migration tests (no write_pick_log_row, safe to inline) ──

_DDL_LEGACY = """
CREATE TABLE model_pick_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task_name TEXT NOT NULL,
    picked_model TEXT NOT NULL,
    picked_score REAL NOT NULL,
    call_category TEXT,
    candidates_json TEXT NOT NULL DEFAULT '[]',
    snapshot_summary TEXT,
    success INTEGER,
    error_category TEXT
)
"""


async def _apply_migration(db_path: str) -> None:
    """Simulates the idempotent ALTER TABLE loop from init_db."""
    async with aiosqlite.connect(db_path) as db:
        for col_name, col_type in (
            ("pool", "TEXT"),
            ("urgency", "REAL"),
            ("success", "INTEGER"),
            ("error_category", "TEXT"),
            ("provider", "TEXT"),
        ):
            try:
                await db.execute(
                    f"ALTER TABLE model_pick_log ADD COLUMN {col_name} {col_type}"
                )
            except Exception as e:
                if "duplicate column" not in str(e).lower():
                    raise
        await db.execute(
            "UPDATE model_pick_log SET provider='local' WHERE provider IS NULL"
        )
        await db.commit()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_provider_column_exists_after_init():
    """Fresh DB created via real init_db() has provider column."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute("PRAGMA table_info(model_pick_log)")
                cols = [row[1] for row in await cur.fetchall()]
            assert "provider" in cols, f"provider missing; columns: {cols}"
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_legacy_rows_backfilled_to_local():
    """Pre-migration DB had no provider column. After ALTER + backfill,
    existing NULL rows must be set to 'local'."""
    async def _test():
        # This test only tests a migration script on an inline legacy DDL;
        # it never calls write_pick_log_row, so no singleton isolation needed.
        db_path = os.path.join(tempfile.mkdtemp(), "legacy.db")
        async with aiosqlite.connect(db_path) as db:
            await db.execute(_DDL_LEGACY)
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, picked_model, picked_score, call_category) "
                "VALUES ('t1', 'qwen-30b', 0.5, 'main_work')"
            )
            await db.commit()
        await _apply_migration(db_path)
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute(
                "SELECT provider FROM model_pick_log WHERE task_name='t1'"
            )
            row = await cur.fetchone()
        assert row[0] == "local"

    run_async(_test())


def test_writer_persists_provider():
    """write_pick_log_row with provider='groq' stores 'groq' in the column."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            await write_pick_log_row(
                db_path=db_path,
                task_name="t",
                picked_model="groq/llama-3.3-70b-versatile",
                picked_score=0.9,
                category="main_work",
                success=True,
                provider="groq",
            )
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT provider FROM model_pick_log WHERE task_name='t'"
                )
                row = await cur.fetchone()
            assert row is not None, "no row written"
            assert row[0] == "groq"
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_writer_default_provider_is_local():
    """When provider arg omitted, default to 'local' for backward compat."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            await write_pick_log_row(
                db_path=db_path,
                task_name="t2",
                picked_model="qwen",
                picked_score=0.5,
                category="main_work",
                success=True,
            )
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT provider FROM model_pick_log WHERE task_name='t2'"
                )
                row = await cur.fetchone()
            assert row is not None, "no row written"
            assert row[0] == "local"
        finally:
            await _close_db(db_mod)

    run_async(_test())
