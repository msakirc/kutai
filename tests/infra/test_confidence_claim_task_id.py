"""TDD tests for record_confidence_claim task_id-first attribution.

Verifies:
  (a) task_id wins — when model_pick_log has a row with task_id matching the
      task AND a NEWER row matching only by title, the task_id row wins.
  (b) legacy fallback — a model_pick_log row with task_id=NULL but task_name
      matching the task's title resolves via the title path.

The fix is in src/infra/db.py::record_confidence_claim (~line 8509).
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import aiosqlite
import pytest


# ---------------------------------------------------------------------------
# Isolation helpers — mirrors tests/infra/test_pick_log_task_id.py exactly
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _fresh_db():
    """Create a fresh isolated DB in a temp dir, resetting the singleton."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_conf_claim.db")
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
# (a) task_id wins over a newer title-matched row
# ---------------------------------------------------------------------------

def test_record_confidence_claim_prefers_task_id_over_title():
    """record_confidence_claim should attribute model_by_id (task_id row),
    NOT model_by_title (newer row matched only by task_name=title).

    Setup:
      task T: title="step X", confidence_categorical="high"
      pick row P1: task_id=T, task_name="WRONG_NAME", picked_model="model_by_id"
                   timestamp older
      pick row P2: task_id=NULL, task_name="step X" (matches title),
                   picked_model="model_by_title", timestamp NEWER

    Without the fix, tier-0 is absent and P2 (title match, newer) wins.
    After the fix, tier-0 finds P1 via task_id and P2 is never consulted.
    """
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            db = await db_mod.get_db()

            # Insert task T
            cur = await db.execute(
                "INSERT INTO tasks "
                "(title, description, agent_type, status, "
                " confidence_categorical, confidence_numeric, mission_id) "
                "VALUES (?, ?, ?, 'pending', ?, ?, NULL)",
                ("step X", "desc", "coder", "high", None),
            )
            task_id = cur.lastrowid

            # P1: task_id-matched row — older timestamp, wrong task_name
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, picked_model, picked_score, call_category, "
                " candidates_json, provider, task_id, timestamp) "
                "VALUES (?, ?, 80.0, 'main_work', '[]', 'local', ?, '2026-05-01 10:00:00')",
                ("WRONG_NAME", "model_by_id", task_id),
            )

            # P2: title-matched row — NEWER timestamp, task_id=NULL
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, picked_model, picked_score, call_category, "
                " candidates_json, provider, task_id, timestamp) "
                "VALUES (?, ?, 90.0, 'main_work', '[]', 'cloud', NULL, '2026-05-02 12:00:00')",
                ("step X", "model_by_title"),
            )
            await db.commit()

            claim_id = await db_mod.record_confidence_claim(task_id)
            assert claim_id is not None and claim_id > 0, (
                "record_confidence_claim returned None — task likely lacked confidence signal"
            )

            # Read the inserted row to check which model was attributed
            async with aiosqlite.connect(db_path) as raw:
                cur2 = await raw.execute(
                    "SELECT model_id FROM confidence_outcomes WHERE id = ?",
                    (claim_id,),
                )
                row = await cur2.fetchone()

            assert row is not None, "no confidence_outcomes row was inserted"
            assert row[0] == "model_by_id", (
                f"expected 'model_by_id' (task_id-matched), got {row[0]!r} — "
                "tier-0 lookup is missing"
            )
        finally:
            await _close_db(db_mod)

    run_async(_test())


# ---------------------------------------------------------------------------
# (b) Legacy fallback — NULL task_id, title match resolves correctly
# ---------------------------------------------------------------------------

def test_record_confidence_claim_falls_back_to_title_for_legacy_rows():
    """When no task_id-matched row exists, fall back to task_name=title.

    Setup:
      task T: title="legacy task", confidence_categorical="medium"
      pick row P: task_id=NULL, task_name="legacy task", picked_model="legacy-model"

    Expected: record_confidence_claim(T) attributes "legacy-model".
    """
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            db = await db_mod.get_db()

            cur = await db.execute(
                "INSERT INTO tasks "
                "(title, description, agent_type, status, "
                " confidence_categorical, confidence_numeric, mission_id) "
                "VALUES (?, ?, ?, 'pending', ?, ?, NULL)",
                ("legacy task", "desc", "researcher", "medium", None),
            )
            task_id = cur.lastrowid

            # Legacy row: task_id=NULL, task_name matches title
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, picked_model, picked_score, call_category, "
                " candidates_json, provider, task_id) "
                "VALUES (?, ?, 75.0, 'main_work', '[]', 'local', NULL)",
                ("legacy task", "legacy-model"),
            )
            await db.commit()

            claim_id = await db_mod.record_confidence_claim(task_id)
            assert claim_id is not None and claim_id > 0

            async with aiosqlite.connect(db_path) as raw:
                cur2 = await raw.execute(
                    "SELECT model_id FROM confidence_outcomes WHERE id = ?",
                    (claim_id,),
                )
                row = await cur2.fetchone()

            assert row is not None, "no confidence_outcomes row inserted"
            assert row[0] == "legacy-model", (
                f"expected 'legacy-model' from title fallback, got {row[0]!r}"
            )
        finally:
            await _close_db(db_mod)

    run_async(_test())
