"""Tests for beckman task write API:
  beckman.add_task, update_task, update_task_by_context_field,
  add_subtasks, propagate_skips, cancel_task, reprioritize_task,
  save_task_checkpoint, clear_task_checkpoint.

No guard tests in this file — guards come in part 5b when src/app and
src/core call sites are also migrated.
"""
from __future__ import annotations

import json

import pytest
import aiosqlite


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (same pattern as test_mission_write_api.py)
# ──────────────────────────────────────────────────────────────────────────────


def _reset_db(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None
    return db_path


async def _close_db(db_mod) -> None:
    """Close and reset the shared DB connection to avoid cross-test leaks."""
    if db_mod._db_connection is not None:
        await db_mod._db_connection.close()
        db_mod._db_connection = None


async def _fetch_task(db_path: str, task_id: int) -> dict | None:
    """Direct aiosqlite read for test verification."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def _fetch_all_tasks(db_path: str) -> list[dict]:
    """Return all tasks rows."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM tasks")
        return [dict(r) for r in await cur.fetchall()]


async def _fetch_tasks_for_mission(db_path: str, mission_id: int) -> list[dict]:
    """Return all tasks for a given mission."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM tasks WHERE mission_id = ?", (mission_id,)
        )
        return [dict(r) for r in await cur.fetchall()]


async def _add_mission(db_module, title: str = "Test Mission") -> int:
    """Helper: insert a minimal mission row, return its id."""
    db = await db_module.get_db()
    cur = await db.execute(
        "INSERT INTO missions (title, description) VALUES (?, ?)",
        (title, "test"),
    )
    await db.commit()
    return cur.lastrowid


# ──────────────────────────────────────────────────────────────────────────────
# add_task
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_task_returns_id_and_persists(tmp_path, monkeypatch):
    """add_task returns a positive int id and the row lands in DB."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task
        task_id = await add_task(
            title="My Task",
            description="does something",
            agent_type="executor",
        )

        assert isinstance(task_id, int)
        assert task_id > 0

        row = await _fetch_task(db_path, task_id)
        assert row is not None
        assert row["title"] == "My Task"
        assert row["description"] == "does something"
        assert row["agent_type"] == "executor"
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# update_task
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_task_modifies_fields(tmp_path, monkeypatch):
    """update_task changes status and error fields, readable back from DB."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, update_task
        task_id = await add_task(title="T", description="d")

        db_module._db_connection = None
        await update_task(task_id, status="failed", error="something went wrong")

        row = await _fetch_task(db_path, task_id)
        assert row["status"] == "failed"
        assert row["error"] == "something went wrong"
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# update_task_by_context_field
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_task_by_context_field_matches_json(tmp_path, monkeypatch):
    """Insert task with context={workflow_step_id: '1.1'}, update by that field, verify status changed."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, update_task_by_context_field

        mid = await _add_mission(db_module)
        db_module._db_connection = None

        task_id = await add_task(
            title="Step 1.1",
            description="d",
            mission_id=mid,
            context={"workflow_step_id": "1.1"},
        )

        db_module._db_connection = None
        await update_task_by_context_field(
            mission_id=mid,
            field="workflow_step_id",
            value="1.1",
            status="skipped",
        )

        row = await _fetch_task(db_path, task_id)
        assert row["status"] == "skipped"
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# add_subtasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_subtasks_creates_children_and_updates_parent(tmp_path, monkeypatch):
    """add_subtasks creates children and sets parent status to waiting_subtasks."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, add_subtasks

        mid = await _add_mission(db_module)
        db_module._db_connection = None

        parent_id = await add_task(
            title="Parent Task",
            description="parent",
            mission_id=mid,
        )
        db_module._db_connection = None

        child_ids = await add_subtasks(
            parent_task_id=parent_id,
            subtasks=[
                {
                    "title": "Child 1",
                    "description": "c1",
                    "agent_type": "executor",
                    "tier": "auto",
                    "priority": 5,
                    "depends_on": [],
                },
                {
                    "title": "Child 2",
                    "description": "c2",
                    "agent_type": "executor",
                    "tier": "auto",
                    "priority": 5,
                    "depends_on": [],
                },
            ],
            mission_id=mid,
        )

        assert isinstance(child_ids, list)
        assert len(child_ids) == 2
        assert all(isinstance(cid, int) and cid > 0 for cid in child_ids)

        parent_row = await _fetch_task(db_path, parent_id)
        assert parent_row["status"] == "waiting_subtasks"

        all_tasks = await _fetch_tasks_for_mission(db_path, mid)
        child_tasks = [t for t in all_tasks if t["parent_task_id"] == parent_id]
        assert len(child_tasks) == 2
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# propagate_skips
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_propagate_skips_cascades(tmp_path, monkeypatch):
    """Insert parent task (skipped) + dependent child (pending), propagate_skips returns 1, child is now skipped."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, update_task, propagate_skips

        mid = await _add_mission(db_module)
        db_module._db_connection = None

        parent_id = await add_task(
            title="Parent",
            description="p",
            mission_id=mid,
        )
        db_module._db_connection = None

        # Mark parent as skipped
        await update_task(parent_id, status="skipped", error="dependency_skipped")
        db_module._db_connection = None

        # Add a child that depends on the parent
        child_id = await add_task(
            title="Child",
            description="c",
            mission_id=mid,
            depends_on=[parent_id],
        )
        db_module._db_connection = None

        count = await propagate_skips(mid)
        assert count == 1

        child_row = await _fetch_task(db_path, child_id)
        assert child_row["status"] == "skipped"
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# cancel_task
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_task_sets_cancelled(tmp_path, monkeypatch):
    """add task, cancel_task returns True, row status=cancelled."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, cancel_task

        task_id = await add_task(title="To Cancel", description="d")

        db_module._db_connection = None
        result = await cancel_task(task_id)

        assert result is True

        row = await _fetch_task(db_path, task_id)
        assert row["status"] == "cancelled"
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# reprioritize_task
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reprioritize_task_updates_priority(tmp_path, monkeypatch):
    """add task, reprioritize to 9, read back priority=9."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, reprioritize_task

        task_id = await add_task(title="Reprio Task", description="d", priority=5)

        db_module._db_connection = None
        result = await reprioritize_task(task_id, new_priority=9)

        assert result is True

        row = await _fetch_task(db_path, task_id)
        assert row["priority"] == 9
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# save_task_checkpoint / clear_task_checkpoint
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_and_clear_task_checkpoint(tmp_path, monkeypatch):
    """save_task_checkpoint stores state; clear_task_checkpoint sets task_state=NULL."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, save_task_checkpoint, clear_task_checkpoint

        task_id = await add_task(title="Checkpoint Task", description="d")

        db_module._db_connection = None
        checkpoint_state = {"step": 3, "retries": 1, "partial_result": "foo"}
        await save_task_checkpoint(task_id, checkpoint_state)

        # Verify via direct aiosqlite read (bypass beckman)
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute(
                "SELECT task_state FROM tasks WHERE id = ?", (task_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] is not None
            stored = json.loads(row[0])
            assert stored == checkpoint_state

        db_module._db_connection = None
        await clear_task_checkpoint(task_id)

        # Verify checkpoint is cleared
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute(
                "SELECT task_state FROM tasks WHERE id = ?", (task_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] is None
    finally:
        await _close_db(db_module)
