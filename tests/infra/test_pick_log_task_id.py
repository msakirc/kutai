"""TDD tests for model_pick_log.task_id column (Z9 reinforce-loop fix).

Verifies:
  (a) migration  — schema-init on a fresh temp DB yields model_pick_log WITH
                   task_id; running init twice does not raise (idempotent).
  (b) write      — write_pick_log_row(..., task_id=42) persists task_id=42.
  (c) join correctness — _reinforce_winning_model resolves via task_id even
                   when task_name != tasks.title, and does NOT fall through
                   to the cross-mission global-fallback row.
  (d) legacy fallback — a row with task_id=NULL but matching task_name=title
                   still resolves via the title-join path.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import patch

import aiosqlite
import pytest


# ---------------------------------------------------------------------------
# DB helper — mirrors _fresh_db from tests/test_hypothesis_verdict.py
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _fresh_db():
    """Create a fresh DB in a temp dir, resetting module-level singleton."""
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
# (a) Migration — task_id column appears after init_db; idempotent on re-init
# ---------------------------------------------------------------------------

def test_migration_adds_task_id_column():
    """A fresh DB via init_db must have model_pick_log.task_id."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute("PRAGMA table_info(model_pick_log)")
                cols = [row[1] for row in await cur.fetchall()]
            assert "task_id" in cols, f"task_id missing from columns: {cols}"
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_migration_is_idempotent():
    """Running init_db twice on the same DB must not raise."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            # second call — idempotent ALTER must silently skip
            await db_mod.init_db()
        finally:
            await _close_db(db_mod)

    run_async(_test())


# ---------------------------------------------------------------------------
# (b) Write — write_pick_log_row with task_id=42 persists it
# ---------------------------------------------------------------------------

def test_write_pick_log_row_persists_task_id():
    """write_pick_log_row(..., task_id=42) stores 42 in the task_id column."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            from src.infra.pick_log import write_pick_log_row

            await write_pick_log_row(
                db_path=db_path,
                task_name="build feature",
                picked_model="qwen3-8b",
                picked_score=0.9,
                category="main_work",
                success=True,
                task_id=42,
            )

            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT task_id FROM model_pick_log WHERE task_name='build feature'"
                )
                row = await cur.fetchone()
            assert row is not None, "no row written"
            assert row[0] == 42
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_write_pick_log_row_task_id_defaults_to_null():
    """Omitting task_id leaves the column NULL (backward compat)."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            from src.infra.pick_log import write_pick_log_row

            await write_pick_log_row(
                db_path=db_path,
                task_name="grader_task",
                picked_model="gpt-4o",
                picked_score=0.7,
                category="overhead",
                success=True,
            )

            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT task_id FROM model_pick_log WHERE task_name='grader_task'"
                )
                row = await cur.fetchone()
            assert row is not None, "no row written"
            assert row[0] is None, f"expected NULL task_id, got {row[0]}"
        finally:
            await _close_db(db_mod)

    run_async(_test())


# ---------------------------------------------------------------------------
# (c) Join correctness — task_id path wins over the cross-mission fallback
# ---------------------------------------------------------------------------

def test_reinforce_resolves_by_task_id_not_cross_mission():
    """_reinforce_winning_model uses task_id join, not cross-mission fallback.

    Setup:
      mission M, task T (title="step A")
      pick row P1: task_id=T, task_name="unrelated" (mismatch on title)
      pick row P2: task_id=NULL, different mission (what the old global-
                   fallback would pick — newer timestamp, different mission)

    Expected: function resolves model from P1 (task_id join), NOT P2.
    """
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            from src.infra.db import get_db, add_mission

            # Create mission M
            mid = await add_mission(title="Test Mission", description="d")

            db = await get_db()

            # Insert task T for mission M
            cur = await db.execute(
                "INSERT INTO tasks (title, description, agent_type, mission_id, status) "
                "VALUES (?, ?, ?, ?, 'completed')",
                ("step A", "desc", "coder", mid),
            )
            task_id = cur.lastrowid

            # P1: correct pick — task_id=T but task_name doesn't match title
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, picked_model, picked_score, call_category, "
                " candidates_json, provider, task_id, timestamp) "
                "VALUES (?, ?, 80.0, 'main_work', '[]', 'local', ?, '2026-05-01 10:00:00')",
                ("unrelated", "correct-model", task_id),
            )

            # P2: decoy pick — newer timestamp, no task_id, different mission_id
            # This is what the old global-fallback would pick (most-recent non-reinforce).
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, picked_model, picked_score, call_category, "
                " candidates_json, provider, task_id, timestamp) "
                "VALUES (?, ?, 90.0, 'main_work', '[]', 'cloud', NULL, '2026-05-02 12:00:00')",
                ("other mission task", "wrong-model"),
            )
            await db.commit()

            from mr_roboto.executors.record_verdict import _reinforce_winning_model

            nudge_calls = []

            async def fake_nudge(model, task_name="", provider="local",
                                 hypothesis_id=None):
                nudge_calls.append(model)

            with patch(
                "src.infra.db.record_reinforce_nudge",
                side_effect=fake_nudge,
            ):
                result = await _reinforce_winning_model(
                    mission_id=mid, hypothesis_id=None, feature="f"
                )

            assert result == "correct-model", (
                f"expected 'correct-model' from task_id join, got {result!r}"
            )
            assert nudge_calls == ["correct-model"]
        finally:
            await _close_db(db_mod)

    run_async(_test())


# ---------------------------------------------------------------------------
# (d) Legacy fallback — NULL task_id, matching title still resolves
# ---------------------------------------------------------------------------

def test_reinforce_falls_back_to_title_join_for_legacy_rows():
    """A NULL task_id row with task_name matching tasks.title uses title path.

    Setup:
      mission M, task T (title="build widget")
      pick row: task_id=NULL, task_name="build widget" (matches title)

    Expected: function resolves the model via the title-join path.
    """
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            from src.infra.db import get_db, add_mission

            mid = await add_mission(title="Legacy Mission", description="d")
            db = await get_db()

            await db.execute(
                "INSERT INTO tasks (title, description, agent_type, mission_id, status) "
                "VALUES (?, ?, ?, ?, 'completed')",
                ("build widget", "desc", "coder", mid),
            )

            # Legacy row: task_id=NULL, task_name matches title
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, picked_model, picked_score, call_category, "
                " candidates_json, provider, task_id) "
                "VALUES (?, ?, 75.0, 'main_work', '[]', 'local', NULL)",
                ("build widget", "legacy-model"),
            )
            await db.commit()

            from mr_roboto.executors.record_verdict import _reinforce_winning_model

            nudge_calls = []

            async def fake_nudge(model, task_name="", provider="local",
                                 hypothesis_id=None):
                nudge_calls.append(model)

            with patch(
                "src.infra.db.record_reinforce_nudge",
                side_effect=fake_nudge,
            ):
                result = await _reinforce_winning_model(
                    mission_id=mid, hypothesis_id=None, feature="legacy"
                )

            assert result == "legacy-model", (
                f"expected 'legacy-model' from title-join fallback, got {result!r}"
            )
            assert nudge_calls == ["legacy-model"]
        finally:
            await _close_db(db_mod)

    run_async(_test())
