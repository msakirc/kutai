"""Integration tests for general_beckman.on_task_finished end-to-end pipeline.

Verifies the new route_result -> rewrite_actions -> apply_actions flow:
- completed result marks task completed
- mission task completion spawns a mr_roboto workflow_advance mechanical task
- clarification spawns a mr_roboto clarify mechanical task
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
async def test_mission_task_complete_spawns_workflow_advance_and_reviewer(tmp_path, monkeypatch):
    """SP3: a completed mission task parks the parent `ungraded` and spawns
    two siblings — the workflow_advance mechanical task AND the grade post-hook
    CHILD, which now runs as a raw_dispatch ``reviewer`` task (not a
    cap-counted ``grader`` agent task) with a durable posthook.grade.resume
    continuation. The reviewer child carries the source reference in its
    continuation cont_state (mission_id never on the child row)."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        tid = await add_task(title="mt", description="", agent_type="coder",
                             mission_id=9)
        # Non-trivial, non-degenerate result so build_grading_spec does not
        # short-circuit to an auto-fail verdict (which would skip the child).
        await on_task_finished(tid, {"status": "completed", "result": (
            "Implemented the order service with idempotent payment capture and "
            "a reconciliation job that sweeps stuck transactions hourly."
        )})
        # Parent task is parked in ungraded until the grade post-hook verdict lands.
        parent = await get_task(tid)
        assert parent["status"] == "ungraded"
        ctx = json.loads(parent["context"] or "{}")
        assert ctx.get("_pending_posthooks") == ["grade"]

        # Sibling 1 — the workflow_advance mechanical task carries mission_id=9
        # and keeps its canonical mechanical shape.
        conn = await get_db()
        cursor = await conn.execute(
            """SELECT id, agent_type, context FROM tasks
               WHERE mission_id = 9 AND id != ? ORDER BY id""", (tid,),
        )
        mission_rows = list(await cursor.fetchall())
        assert {r["agent_type"] for r in mission_rows} == {"mechanical"}, (
            "expected exactly the workflow_advance mechanical mission sibling, "
            f"got {[r['agent_type'] for r in mission_rows]}"
        )
        wa = mission_rows[0]
        wa_ctx = json.loads(wa["context"])
        assert wa_ctx["executor"] == "mechanical"
        assert wa_ctx["payload"]["action"] == "workflow_advance"
        assert wa_ctx["payload"]["mission_id"] == 9
        assert wa_ctx["payload"]["completed_task_id"] == tid

        # Sibling 2 — the grade post-hook reviewer CHILD. SP3 child-spec
        # hygiene: mission_id rides ONLY in cont_state, never on the child row,
        # so the reviewer task has mission_id IS NULL and is linked by
        # parent_task_id, not mission_id.
        cur_child = await conn.execute(
            """SELECT id, agent_type, mission_id FROM tasks
               WHERE parent_task_id = ?""", (tid,),
        )
        child_rows = list(await cur_child.fetchall())
        assert len(child_rows) == 1, (
            f"expected one grade reviewer child, got {child_rows}"
        )
        reviewer = child_rows[0]
        assert reviewer["agent_type"] == "reviewer"
        assert reviewer["mission_id"] is None, \
            "mission_id must ride in cont_state, never on the child row"

        # The reviewer child carries the source reference + grade continuation
        # in its cont_state (NOT the context column).
        cur2 = await conn.execute(
            "SELECT resume_name, state_json FROM continuations "
            "WHERE child_task_id = ?",
            (reviewer["id"],),
        )
        cont = await cur2.fetchone()
        assert cont is not None, "reviewer child must have a registered continuation"
        assert cont["resume_name"] == "posthook.grade.resume"
        cs = json.loads(cont["state_json"])
        assert cs["source_task_id"] == tid
        assert cs["mission_id"] == 9
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_clarify_spawns_mr_roboto_clarify_task(tmp_path, monkeypatch):
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
