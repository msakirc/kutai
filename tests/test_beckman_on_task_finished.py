"""Integration tests for general_beckman.on_task_finished end-to-end pipeline.

Verifies the new route_result -> rewrite_actions -> apply_actions flow:
- completed result marks task completed
- mission task completion spawns a salako workflow_advance mechanical task
- clarification spawns a salako clarify mechanical task
- silent task + clarify request -> rewrite turns into Failed, retry path
  leaves the task in 'pending' (first retry).
"""
import json
import pytest

import src.infra.db as _db_mod
from general_beckman import on_task_finished
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp
from src.infra.db import init_db, add_task, get_task, get_db


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "otf.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    monkeypatch.setattr(_cs, "_seeded", False)
    _pp._patterns.clear()
    await init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_completed_result_marks_task_completed(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        tid = await add_task(title="t", description="", agent_type="coder")
        await on_task_finished(tid, {"status": "completed", "result": "ok"})
        row = await get_task(tid)
        assert row["status"] == "completed"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_mission_task_complete_spawns_workflow_advance(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        tid = await add_task(title="mt", description="", agent_type="coder",
                             mission_id=9)
        await on_task_finished(tid, {"status": "completed", "result": "ok"})
        # Parent task should be completed
        parent = await get_task(tid)
        assert parent["status"] == "completed"
        # A sibling mechanical workflow_advance task should exist in the mission
        conn = await get_db()
        cursor = await conn.execute(
            """SELECT agent_type, context FROM tasks
               WHERE mission_id = 9 AND id != ?""", (tid,),
        )
        child = await cursor.fetchone()
        assert child is not None, "expected workflow_advance sibling task"
        assert child["agent_type"] == "mechanical"
        ctx = json.loads(child["context"])
        # Canonical mechanical shape: ctx["executor"] == "mechanical" and
        # ctx["payload"]["action"] is the salako action name.
        assert ctx["executor"] == "mechanical"
        assert ctx["payload"]["action"] == "workflow_advance"
        assert ctx["payload"]["mission_id"] == 9
        assert ctx["payload"]["completed_task_id"] == tid
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_clarify_spawns_salako_clarify_task(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        tid = await add_task(title="t", description="", agent_type="coder")
        await on_task_finished(tid, {"status": "needs_clarification",
                                     "question": "What?"})
        row = await get_task(tid)
        assert row["status"] == "waiting_human"
        conn = await get_db()
        cursor = await conn.execute(
            """SELECT agent_type, context FROM tasks
               WHERE parent_task_id = ?""", (tid,),
        )
        child = await cursor.fetchone()
        assert child is not None
        assert child["agent_type"] == "mechanical"
        ctx = json.loads(child["context"])
        assert ctx["executor"] == "mechanical"
        assert ctx["payload"]["action"] == "clarify"
        assert ctx["payload"]["question"] == "What?"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_silent_task_clarify_becomes_failure(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        tid = await add_task(title="t", description="", agent_type="coder",
                             context={"silent": True})
        await on_task_finished(tid, {"status": "needs_clarification",
                                     "question": "?"})
        row = await get_task(tid)
        # rewrite converted clarify -> Failed; apply ran _retry_or_dlq which
        # on first failure (attempts=1, max=3) returns RetryDecision(immediate),
        # so task ends up status='pending' with worker_attempts=1.
        assert row["status"] == "pending"
        assert int(row.get("worker_attempts") or 0) == 1
        assert row.get("error_category") in ("worker", None) or isinstance(
            row.get("error_category"), str
        )
    finally:
        await _close_db()
