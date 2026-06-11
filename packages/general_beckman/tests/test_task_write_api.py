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


# ──────────────────────────────────────────────────────────────────────────────
# reset_failed_tasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_failed_tasks(tmp_path, monkeypatch):
    """reset_failed_tasks resets all failed tasks to pending."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, update_task, reset_failed_tasks

        t1 = await add_task(title="F1", description="d")
        db_module._db_connection = None
        t2 = await add_task(title="F2", description="d")
        db_module._db_connection = None
        t3 = await add_task(title="P1", description="d")

        db_module._db_connection = None
        await update_task(t1, status="failed", error="err")
        db_module._db_connection = None
        await update_task(t2, status="failed", error="err")
        # t3 stays pending

        db_module._db_connection = None
        count = await reset_failed_tasks()
        assert count == 2

        r1 = await _fetch_task(db_path, t1)
        r2 = await _fetch_task(db_path, t2)
        r3 = await _fetch_task(db_path, t3)
        assert r1["status"] == "pending"
        assert r2["status"] == "pending"
        assert r3["status"] == "pending"  # unaffected
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# reset_stuck_tasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_stuck_tasks(tmp_path, monkeypatch):
    """reset_stuck_tasks resets processing tasks to pending."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, update_task, reset_stuck_tasks

        t1 = await add_task(title="S1", description="d")
        db_module._db_connection = None
        t2 = await add_task(title="P1", description="d")

        db_module._db_connection = None
        await update_task(t1, status="processing")

        db_module._db_connection = None
        count = await reset_stuck_tasks()
        assert count == 1

        r1 = await _fetch_task(db_path, t1)
        r2 = await _fetch_task(db_path, t2)
        assert r1["status"] == "pending"
        assert r2["status"] == "pending"  # unaffected
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# reset_blocked_tasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_blocked_tasks(tmp_path, monkeypatch):
    """reset_blocked_tasks clears depends_on on pending tasks."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, reset_blocked_tasks

        t1 = await add_task(title="B1", description="d", depends_on=[99])
        db_module._db_connection = None
        t2 = await add_task(title="P1", description="d")  # no deps

        db_module._db_connection = None
        count = await reset_blocked_tasks()
        assert count == 1

        r1 = await _fetch_task(db_path, t1)
        assert r1["depends_on"] == "[]"
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# cancel_pending_tasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_pending_tasks(tmp_path, monkeypatch):
    """cancel_pending_tasks cancels only pending tasks for the given mission."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, update_task, cancel_pending_tasks

        mid = await _add_mission(db_module)
        db_module._db_connection = None
        mid2 = await _add_mission(db_module, title="Other")
        db_module._db_connection = None

        t1 = await add_task(title="T1", description="d", mission_id=mid)
        db_module._db_connection = None
        t2 = await add_task(title="T2", description="d", mission_id=mid)
        db_module._db_connection = None
        t3 = await add_task(title="T3", description="d", mission_id=mid)
        db_module._db_connection = None
        t_other = await add_task(title="Other", description="d", mission_id=mid2)

        # t3 is already processing — should not be cancelled
        db_module._db_connection = None
        await update_task(t3, status="processing")

        db_module._db_connection = None
        count = await cancel_pending_tasks(mid)
        assert count == 2  # t1 and t2 only

        r1 = await _fetch_task(db_path, t1)
        r2 = await _fetch_task(db_path, t2)
        r3 = await _fetch_task(db_path, t3)
        r_other = await _fetch_task(db_path, t_other)
        assert r1["status"] == "cancelled"
        assert r2["status"] == "cancelled"
        assert r3["status"] == "processing"  # unaffected
        assert r_other["status"] == "pending"  # different mission
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# reset_workflow_step
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_workflow_step(tmp_path, monkeypatch):
    """reset_workflow_step resets writer, verify sibling, and confirm task."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, update_task, reset_workflow_step

        mid = await _add_mission(db_module)
        db_module._db_connection = None

        # Writer step
        writer_id = await add_task(
            title="Writer", description="d", mission_id=mid,
            context={"workflow_step_id": "3.draft"},
        )
        db_module._db_connection = None
        # Verify sibling
        verify_id = await add_task(
            title="Verify", description="d", mission_id=mid,
            context={"workflow_step_id": "3.draft.verify"},
        )
        db_module._db_connection = None
        # Confirm task (by id)
        confirm_id = await add_task(title="Confirm", description="d", mission_id=mid)

        # Mark all as completed
        db_module._db_connection = None
        await update_task(writer_id, status="completed")
        db_module._db_connection = None
        await update_task(verify_id, status="completed")
        db_module._db_connection = None
        await update_task(confirm_id, status="completed")

        db_module._db_connection = None
        await reset_workflow_step(mid, "3.draft", confirm_task_id=confirm_id)

        r_writer = await _fetch_task(db_path, writer_id)
        r_verify = await _fetch_task(db_path, verify_id)
        r_confirm = await _fetch_task(db_path, confirm_id)
        assert r_writer["status"] == "pending"
        assert r_writer["worker_attempts"] == 0
        assert r_verify["status"] == "pending"
        assert r_confirm["status"] == "pending"
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# recover_startup_tasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recover_startup_tasks(tmp_path, monkeypatch):
    """recover_startup_tasks resets processing→pending and clears backoff."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, update_task, recover_startup_tasks

        # One processing task
        t1 = await add_task(title="Processing", description="d")
        db_module._db_connection = None
        await update_task(t1, status="processing")

        # One pending task with a future next_retry_at
        db_module._db_connection = None
        t2 = await add_task(title="Backoff", description="d")
        db_module._db_connection = None
        # Set next_retry_at to a future time
        db = await db_module.get_db()
        await db.execute(
            "UPDATE tasks SET next_retry_at = datetime('now', '+1 hour') WHERE id = ?",
            (t2,)
        )
        await db.commit()

        db_module._db_connection = None
        result = await recover_startup_tasks()

        assert result["interrupted"] == 1
        assert result["backoff_cleared"] == 1

        r1 = await _fetch_task(db_path, t1)
        r2 = await _fetch_task(db_path, t2)
        assert r1["status"] == "pending"
        assert r1["retry_reason"] == "infrastructure"
        assert r2["next_retry_at"] is None
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# reset_cascade_failed_dependents
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_cascade_failed_dependents(tmp_path, monkeypatch):
    """reset_cascade_failed_dependents resets cascade-failed tasks."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_task, update_task, reset_cascade_failed_dependents

        # Primary task that was in DLQ
        primary_id = await add_task(title="Primary", description="d")
        db_module._db_connection = None

        # Dependent that was cascade-failed because primary failed
        dep_id = await add_task(
            title="Dependent", description="d", depends_on=[primary_id]
        )
        db_module._db_connection = None
        await update_task(
            dep_id, status="failed", error="All dependencies failed"
        )

        # Another failed task with a different error — should NOT be reset
        other_id = await add_task(title="Other", description="d")
        db_module._db_connection = None
        await update_task(other_id, status="failed", error="different error")

        db_module._db_connection = None
        count = await reset_cascade_failed_dependents(primary_id)
        assert count == 1

        r_dep = await _fetch_task(db_path, dep_id)
        r_other = await _fetch_task(db_path, other_id)
        assert r_dep["status"] == "pending"
        assert r_dep["error"] is None
        assert r_other["status"] == "failed"  # unaffected
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# Guard tests: no raw task SQL or helper imports outside sanctioned modules
# ──────────────────────────────────────────────────────────────────────────────


def test_no_raw_tasks_sql_outside_db():
    """No source file outside src/infra/db.py may contain raw
    INSERT INTO tasks, UPDATE tasks, or DELETE FROM tasks SQL.

    After migration all former raw-SQL sites call beckman APIs instead.
    src/infra/db.py is the sole SQL owner.
    """
    import re
    import os
    from pathlib import Path

    root = Path(__file__).parents[3]  # repo root (worktree)

    sql_re = re.compile(
        r'(INSERT\s+INTO\s+tasks|UPDATE\s+tasks\s+SET|DELETE\s+FROM\s+tasks)',
        re.IGNORECASE,
    )

    # Allowed: src/infra/db.py (SQL owner) and all of general_beckman/src/
    # (beckman is the write-owner — its internal modules may write tasks SQL).
    allowed = {
        (root / "src" / "infra" / "db.py").resolve(),
    }
    allowed_dirs = {
        (root / "packages" / "general_beckman" / "src" / "general_beckman").resolve(),
    }
    # Also allow this test file and any test helpers that reference these strings.
    allowed.add(Path(__file__).resolve())

    # Migration scripts are one-time DB tools, not production callers — exclude.
    skip_prefixes = {
        (root / "scripts").resolve(),
    }

    violations: list[str] = []
    skip_dirs = {".venv", "__pycache__", ".git", ".benchmark_cache", "node_modules", "worktrees"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname.startswith("test_") or fname.endswith("_test.py"):
                continue
            if "tests" in Path(dirpath).parts:
                continue
            filepath = (Path(dirpath) / fname).resolve()
            if filepath in allowed:
                continue
            # Allow all of general_beckman/src/general_beckman/ — beckman is the write-owner.
            if any(str(filepath).startswith(str(d)) for d in allowed_dirs):
                continue
            # Skip migration scripts (one-time DB tools, not prod callers).
            if any(str(filepath).startswith(str(p)) for p in skip_prefixes):
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if sql_re.search(line):
                    rel = filepath.relative_to(root.resolve())
                    violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert violations == [], (
        "Raw tasks SQL found outside src/infra/db.py and general_beckman — "
        "use general_beckman task write API instead:\n"
        + "\n".join(violations)
    )


def test_no_raw_db_task_imports_outside_infra_beckman():
    """No source file outside src/infra/db.py itself and general_beckman may
    import task-write helpers directly from src.infra.db.

    Covered helpers:
      add_task, update_task, update_task_by_context_field,
      add_subtasks_atomically, insert_tasks_atomically,
      propagate_skips, claim_task, cancel_task, reprioritize_task,
      save_task_checkpoint, clear_task_checkpoint,
      reset_failed_tasks, reset_stuck_tasks, reset_blocked_tasks,
      cancel_pending_tasks, reset_workflow_step,
      recover_startup_tasks, reset_cascade_failed_dependents.

    IMPORTANT: src/infra/dead_letter.py is NOT exempt — it must route via
    beckman like everyone else. Only src/infra/db.py itself is exempt.
    """
    import re
    import os
    from pathlib import Path

    root = Path(__file__).parents[3]  # repo root

    # Matches: from src.infra.db import add_task  (or update_task, etc.)
    import_re = re.compile(
        r'from\s+src\.infra\.db\s+import\s+.*?\b('
        r'add_task|update_task|update_task_by_context_field'
        r'|add_subtasks_atomically|insert_tasks_atomically'
        r'|propagate_skips|claim_task|cancel_task|reprioritize_task'
        r'|save_task_checkpoint|clear_task_checkpoint'
        r'|reset_failed_tasks|reset_stuck_tasks|reset_blocked_tasks'
        r'|cancel_pending_tasks|reset_workflow_step'
        r'|recover_startup_tasks|reset_cascade_failed_dependents'
        r')\b',
    )
    # Also catch relative imports: from ..infra.db import add_task
    rel_import_re = re.compile(
        r'from\s+\.+infra\.db\s+import\s+.*?\b('
        r'add_task|update_task|update_task_by_context_field'
        r'|add_subtasks_atomically|insert_tasks_atomically'
        r'|propagate_skips|claim_task|cancel_task|reprioritize_task'
        r'|save_task_checkpoint|clear_task_checkpoint'
        r'|reset_failed_tasks|reset_stuck_tasks|reset_blocked_tasks'
        r'|cancel_pending_tasks|reset_workflow_step'
        r'|recover_startup_tasks|reset_cascade_failed_dependents'
        r')\b',
    )

    # Only src/infra/db.py itself is exempt (not all of src/infra/)
    allowed_files = {
        (root / "src" / "infra" / "db.py").resolve(),
        Path(__file__).resolve(),
    }
    allowed_dirs = {
        (root / "packages" / "general_beckman" / "src" / "general_beckman").resolve(),
        # Coulson is deferred — it owns posthook/checkpoint helpers that will
        # be migrated in a separate session.
        (root / "packages" / "coulson" / "src" / "coulson").resolve(),
    }
    # Migration scripts are one-time DB tools — not prod callers.
    skip_prefixes = {
        (root / "scripts").resolve(),
    }

    violations: list[str] = []
    skip_dirs = {".venv", "__pycache__", ".git", ".benchmark_cache", "node_modules", "worktrees"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname.startswith("test_") or fname.endswith("_test.py"):
                continue
            if "tests" in Path(dirpath).parts:
                continue
            filepath = (Path(dirpath) / fname).resolve()
            if filepath in allowed_files:
                continue
            if any(str(filepath).startswith(str(d)) for d in allowed_dirs):
                continue
            if any(str(filepath).startswith(str(p)) for p in skip_prefixes):
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if import_re.search(line) or rel_import_re.search(line):
                    rel = filepath.relative_to(root.resolve())
                    violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert violations == [], (
        "Direct db task-write helper import found outside src/infra/db.py and "
        "general_beckman — use general_beckman task write API instead:\n"
        + "\n".join(violations)
    )
