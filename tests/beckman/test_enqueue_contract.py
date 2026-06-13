"""TDD tests for widened general_beckman.enqueue() signature (Task 2)."""
import asyncio
import json
import pytest

import src.infra.db as _db_mod
import general_beckman as _gb
from general_beckman import enqueue
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "enqueue_contract.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    monkeypatch.setattr(_cs, "_seeded", False)
    _pp._patterns.clear()
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_enqueue_default_returns_task_id(tmp_path, monkeypatch):
    """Old behaviour: plain enqueue(spec) returns an int task_id."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        spec = {"title": "hello", "description": "world", "agent_type": "coder"}
        result = await enqueue(spec)
        assert isinstance(result, int), f"Expected int task_id, got {type(result)}"
        assert result > 0
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_enqueue_propagates_kind_to_db_row(tmp_path, monkeypatch):
    """spec['kind'] must be persisted to the tasks.kind column."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        spec = {
            "title": "overhead task",
            "description": "d",
            "agent_type": "grader",
            "kind": "overhead",
        }
        task_id = await enqueue(spec)
        assert task_id is not None
        row = await _db_mod.get_task(task_id)
        assert row["kind"] == "overhead", f"Expected 'overhead', got {row['kind']!r}"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_enqueue_propagates_parent_id_to_db_row(tmp_path, monkeypatch):
    """parent_id kwarg must be stored in tasks.parent_task_id."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        parent_id = await enqueue({"title": "parent", "description": "p", "agent_type": "coder"})
        child_id = await enqueue(
            {"title": "child", "description": "c", "agent_type": "coder"},
            parent_id=parent_id,
        )
        assert child_id is not None
        row = await _db_mod.get_task(child_id)
        assert row["parent_task_id"] == parent_id, (
            f"Expected parent_task_id={parent_id}, got {row['parent_task_id']}"
        )
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_enqueue_stores_continuation_in_context_envelope(tmp_path, monkeypatch):
    """on_complete goes to the continuations table (NOT context); next_task_spec
    stays in context['beckman'] (fire-and-forget chain, not the durable substrate).
    """
    await _fresh_db(tmp_path, monkeypatch)
    try:
        next_spec = {"title": "followup", "description": "f"}
        task_id = await enqueue(
            {"title": "main", "description": "m", "agent_type": "coder"},
            on_complete="agent.resume",
            next_task_spec=next_spec,
        )
        assert task_id is not None

        # on_complete must be in the continuations table, NOT context
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT resume_name FROM continuations WHERE child_task_id = ?",
            (task_id,),
        )
        cont_row = await cur.fetchone()
        assert cont_row is not None, "continuations row missing"
        assert cont_row[0] == "agent.resume", f"resume_name wrong: {cont_row[0]}"

        # next_task_spec still travels via context['beckman']
        row = await _db_mod.get_task(task_id)
        ctx_raw = row.get("context") or "{}"
        ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
        beckman = ctx.get("beckman", {})
        assert "on_complete" not in beckman, (
            f"on_complete must NOT be in context (now lives in continuations table): {beckman}"
        )
        assert beckman.get("next_task_spec") == next_spec, (
            f"next_task_spec missing/wrong: {beckman}"
        )
    finally:
        await _close_db()
