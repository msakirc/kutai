"""Tests for general_beckman.apply — action → DB row dispatch."""
import json
import pytest
import src.infra.db as _db_mod

from general_beckman.apply import apply_actions
from general_beckman.result_router import (
    Complete, SpawnSubtasks, RequestClarification, RequestReview,
    Exhausted, Failed, MissionAdvance, CompleteWithReusedAnswer,
)
from src.infra.db import init_db, add_task, get_task, get_db


async def _fresh_db(tmp_path, monkeypatch):
    """Reset DB to a fresh temp file for isolation."""
    db_file = tmp_path / "apply.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    await init_db()


@pytest.mark.asyncio
async def test_complete_marks_task_completed(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    tid = await add_task(title="t", description="", agent_type="coder")
    await apply_actions(
        await get_task(tid),
        [Complete(task_id=tid, result="done", raw={})],
    )
    row = await get_task(tid)
    assert row["status"] == "completed"

    # Teardown
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_subtasks_spawns_child_rows(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    parent = await add_task(title="p", description="", agent_type="planner")
    subs = [{"title": "c1", "description": "", "agent_type": "coder"}]
    await apply_actions(
        await get_task(parent),
        [SpawnSubtasks(parent_task_id=parent, subtasks=subs, raw={})],
    )
    row = await get_task(parent)
    assert row["status"] == "waiting_subtasks"
    conn = await get_db()
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE parent_task_id = ?", (parent,)
    )
    (n,) = await cursor.fetchone()
    assert n == 1

    # Teardown
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_clarification_spawns_salako_task(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    tid = await add_task(title="t", description="", agent_type="coder")
    await apply_actions(
        await get_task(tid),
        [RequestClarification(task_id=tid, question="Why?", chat_id=42, raw={})],
    )
    row = await get_task(tid)
    assert row["status"] == "waiting_human"
    conn = await get_db()
    cursor = await conn.execute(
        """SELECT agent_type, context FROM tasks WHERE parent_task_id = ?""", (tid,),
    )
    child = await cursor.fetchone()
    assert child is not None
    assert child["agent_type"] == "mechanical"
    # Canonical mechanical shape: {executor: mechanical, payload: {action: ...}}
    ctx = json.loads(child["context"])
    assert ctx["executor"] == "mechanical"
    assert ctx["payload"]["action"] == "clarify"
    assert ctx["payload"]["question"] == "Why?"

    # Teardown
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_mission_advance_spawns_workflow_advance_task(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    parent = await add_task(title="mt", description="", agent_type="coder",
                            mission_id=7)
    await apply_actions(
        await get_task(parent),
        [MissionAdvance(task_id=parent, mission_id=7,
                        completed_task_id=parent, raw={})],
    )
    conn = await get_db()
    cursor = await conn.execute(
        """SELECT agent_type, context FROM tasks
           WHERE mission_id = 7 AND id != ?""", (parent,),
    )
    child = await cursor.fetchone()
    assert child is not None
    assert child["agent_type"] == "mechanical"
    # Canonical mechanical shape: {executor: mechanical, payload: {action: ...}}
    ctx = json.loads(child["context"])
    assert ctx["executor"] == "mechanical"
    assert ctx["payload"]["action"] == "workflow_advance"
    assert ctx["payload"]["mission_id"] == 7
    assert ctx["payload"]["completed_task_id"] == parent

    # Teardown
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
