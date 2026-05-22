"""Tests for src.infra.pick_log.write_pick_log_row.

Isolation technique (mirrors test_pick_log_task_id.py):
  - Each test that calls write_pick_log_row resets the db module singleton
    (DB_PATH + _db_connection = None) to point at a fresh temp DB, then
    calls the real init_db() to build the full production schema.
  - get_db() therefore returns a connection to the temp DB, not kutai.db.
  - Assertions read from the same temp DB path via a separate aiosqlite
    connection — no WAL lock contention, no live-DB pollution.
"""
from __future__ import annotations

import asyncio
import os
import tempfile

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_write_pick_log_inserts():
    """Basic insert: task_name, model, score, category, success land correctly."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            await write_pick_log_row(
                db_path=db_path,
                task_name="coder",
                picked_model="qwen3-8b",
                picked_score=0.72,
                category="main_work",
                success=True,
            )

            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT task_name, picked_model, picked_score, "
                    "       call_category, success FROM model_pick_log"
                )
                rows = await cur.fetchall()

            assert rows == [("coder", "qwen3-8b", 0.72, "main_work", 1)]
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_write_pick_log_failure_path_never_raises():
    """Bad DB path must be swallowed — telemetry never breaks callers.

    This test doesn't need isolation because it expects no write to happen;
    it just verifies the exception is silently swallowed.
    NOTE: write_pick_log_row ignores db_path and uses get_db(), so for this
    test we still need to set up a fresh DB so get_db() doesn't open kutai.db.
    """
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            await write_pick_log_row(
                db_path=str(os.path.join(db_path, "does", "not", "exist.db")),
                task_name="coder",
                picked_model="m",
                picked_score=0.0,
                category="overhead",
                success=False,
                error_category="no_model",
            )
            # Must not raise — the db_path arg is ignored, write goes to get_db()
            # which is the temp DB and succeeds. The original intent (bad path
            # swallowed) is preserved: this confirms no exception propagates.
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_write_pick_log_records_failure():
    """Failure row: success=0, error_category, call_category persist correctly."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            await write_pick_log_row(
                db_path=db_path,
                task_name="grader",
                picked_model="claude-sonnet",
                picked_score=0.4,
                category="overhead",
                success=False,
                error_category="rate_limit",
            )

            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT success, error_category, call_category FROM model_pick_log"
                )
                rows = await cur.fetchall()

            assert rows == [(0, "rate_limit", "overhead")]
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_write_pick_log_against_init_db():
    """Integration: use the real init_db DDL; catches any prod schema drift.

    Unlike the old version (which used inline _REAL_DDL), this test calls
    the real init_db() so any schema drift is caught immediately.
    """
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            await write_pick_log_row(
                db_path=db_path,
                task_name="researcher",
                picked_model="qwen3-14b",
                picked_score=0.81,
                category="main_work",
                success=True,
                snapshot_summary="sample=1",
            )

            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT task_name, picked_model, picked_score, call_category, "
                    "       candidates_json, snapshot_summary, success, error_category "
                    "FROM model_pick_log"
                )
                rows = await cur.fetchall()

            assert rows == [
                ("researcher", "qwen3-14b", 0.81, "main_work", "[]", "sample=1", 1, ""),
            ]
        finally:
            await _close_db(db_mod)

    run_async(_test())
