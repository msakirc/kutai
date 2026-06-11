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
# File-local helpers (direct DB reads for verification; NOT shared DB setup)
# DB setup is handled by the fresh_db fixture in conftest.py.
# ──────────────────────────────────────────────────────────────────────────────


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
async def test_add_task_returns_id_and_persists(fresh_db):
    """add_task returns a positive int id and the row lands in DB."""
    db_path = fresh_db
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


# ──────────────────────────────────────────────────────────────────────────────
# update_task
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_task_modifies_fields(fresh_db):
    """update_task changes status and error fields, readable back from DB."""
    db_path = fresh_db
    from general_beckman import add_task, update_task
    task_id = await add_task(title="T", description="d")

    await update_task(task_id, status="failed", error="something went wrong")

    row = await _fetch_task(db_path, task_id)
    assert row["status"] == "failed"
    assert row["error"] == "something went wrong"


# ──────────────────────────────────────────────────────────────────────────────
# update_task_by_context_field
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_task_by_context_field_matches_json(fresh_db):
    """Insert task with context={workflow_step_id: '1.1'}, update by that field, verify status changed."""
    db_path = fresh_db
    import src.infra.db as db_module
    from general_beckman import add_task, update_task_by_context_field

    mid = await _add_mission(db_module)

    task_id = await add_task(
        title="Step 1.1",
        description="d",
        mission_id=mid,
        context={"workflow_step_id": "1.1"},
    )

    await update_task_by_context_field(
        mission_id=mid,
        field="workflow_step_id",
        value="1.1",
        status="skipped",
    )

    row = await _fetch_task(db_path, task_id)
    assert row["status"] == "skipped"


# ──────────────────────────────────────────────────────────────────────────────
# add_subtasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_subtasks_creates_children_and_updates_parent(fresh_db):
    """add_subtasks creates children and sets parent status to waiting_subtasks."""
    db_path = fresh_db
    import src.infra.db as db_module
    from general_beckman import add_task, add_subtasks

    mid = await _add_mission(db_module)

    parent_id = await add_task(
        title="Parent Task",
        description="parent",
        mission_id=mid,
    )

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


# ──────────────────────────────────────────────────────────────────────────────
# propagate_skips
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_propagate_skips_cascades(fresh_db):
    """Insert parent task (skipped) + dependent child (pending), propagate_skips returns 1, child is now skipped."""
    db_path = fresh_db
    import src.infra.db as db_module
    from general_beckman import add_task, update_task, propagate_skips

    mid = await _add_mission(db_module)

    parent_id = await add_task(
        title="Parent",
        description="p",
        mission_id=mid,
    )

    # Mark parent as skipped
    await update_task(parent_id, status="skipped", error="dependency_skipped")

    # Add a child that depends on the parent
    child_id = await add_task(
        title="Child",
        description="c",
        mission_id=mid,
        depends_on=[parent_id],
    )

    count = await propagate_skips(mid)
    assert count == 1

    child_row = await _fetch_task(db_path, child_id)
    assert child_row["status"] == "skipped"


# ──────────────────────────────────────────────────────────────────────────────
# cancel_task
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_task_sets_cancelled(fresh_db):
    """add task, cancel_task returns True, row status=cancelled."""
    db_path = fresh_db
    from general_beckman import add_task, cancel_task

    task_id = await add_task(title="To Cancel", description="d")

    result = await cancel_task(task_id)

    assert result is True

    row = await _fetch_task(db_path, task_id)
    assert row["status"] == "cancelled"


# ──────────────────────────────────────────────────────────────────────────────
# reprioritize_task
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reprioritize_task_updates_priority(fresh_db):
    """add task, reprioritize to 9, read back priority=9."""
    db_path = fresh_db
    from general_beckman import add_task, reprioritize_task

    task_id = await add_task(title="Reprio Task", description="d", priority=5)

    result = await reprioritize_task(task_id, new_priority=9)

    assert result is True

    row = await _fetch_task(db_path, task_id)
    assert row["priority"] == 9


# ──────────────────────────────────────────────────────────────────────────────
# save_task_checkpoint / clear_task_checkpoint
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_and_clear_task_checkpoint(fresh_db):
    """save_task_checkpoint stores state; clear_task_checkpoint sets task_state=NULL."""
    db_path = fresh_db
    from general_beckman import add_task, save_task_checkpoint, clear_task_checkpoint

    task_id = await add_task(title="Checkpoint Task", description="d")

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

    await clear_task_checkpoint(task_id)

    # Verify checkpoint is cleared
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT task_state FROM tasks WHERE id = ?", (task_id,)
        )
        row = await cur.fetchone()
        assert row is not None
        assert row[0] is None


# ──────────────────────────────────────────────────────────────────────────────
# reset_failed_tasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_failed_tasks(fresh_db):
    """reset_failed_tasks resets all failed tasks to pending."""
    db_path = fresh_db
    from general_beckman import add_task, update_task, reset_failed_tasks

    t1 = await add_task(title="F1", description="d")
    t2 = await add_task(title="F2", description="d")
    t3 = await add_task(title="P1", description="d")

    await update_task(t1, status="failed", error="err")
    await update_task(t2, status="failed", error="err")
    # t3 stays pending

    count = await reset_failed_tasks()
    assert count == 2

    r1 = await _fetch_task(db_path, t1)
    r2 = await _fetch_task(db_path, t2)
    r3 = await _fetch_task(db_path, t3)
    assert r1["status"] == "pending"
    assert r2["status"] == "pending"
    assert r3["status"] == "pending"  # unaffected


# ──────────────────────────────────────────────────────────────────────────────
# reset_stuck_tasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_stuck_tasks(fresh_db):
    """reset_stuck_tasks resets processing tasks to pending."""
    db_path = fresh_db
    from general_beckman import add_task, update_task, reset_stuck_tasks

    t1 = await add_task(title="S1", description="d")
    t2 = await add_task(title="P1", description="d")

    await update_task(t1, status="processing")

    count = await reset_stuck_tasks()
    assert count == 1

    r1 = await _fetch_task(db_path, t1)
    r2 = await _fetch_task(db_path, t2)
    assert r1["status"] == "pending"
    assert r2["status"] == "pending"  # unaffected


# ──────────────────────────────────────────────────────────────────────────────
# reset_blocked_tasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_blocked_tasks(fresh_db):
    """reset_blocked_tasks clears depends_on on pending tasks."""
    db_path = fresh_db
    from general_beckman import add_task, reset_blocked_tasks

    t1 = await add_task(title="B1", description="d", depends_on=[99])
    t2 = await add_task(title="P1", description="d")  # no deps

    count = await reset_blocked_tasks()
    assert count == 1

    r1 = await _fetch_task(db_path, t1)
    assert r1["depends_on"] == "[]"


# ──────────────────────────────────────────────────────────────────────────────
# cancel_pending_tasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_pending_tasks(fresh_db):
    """cancel_pending_tasks cancels only pending tasks for the given mission."""
    db_path = fresh_db
    import src.infra.db as db_module
    from general_beckman import add_task, update_task, cancel_pending_tasks

    mid = await _add_mission(db_module)
    mid2 = await _add_mission(db_module, title="Other")

    t1 = await add_task(title="T1", description="d", mission_id=mid)
    t2 = await add_task(title="T2", description="d", mission_id=mid)
    t3 = await add_task(title="T3", description="d", mission_id=mid)
    t_other = await add_task(title="Other", description="d", mission_id=mid2)

    # t3 is already processing — should not be cancelled
    await update_task(t3, status="processing")

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


# ──────────────────────────────────────────────────────────────────────────────
# reset_workflow_step
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_workflow_step(fresh_db):
    """reset_workflow_step resets writer, verify sibling, and confirm task."""
    db_path = fresh_db
    import src.infra.db as db_module
    from general_beckman import add_task, update_task, reset_workflow_step

    mid = await _add_mission(db_module)

    # Writer step
    writer_id = await add_task(
        title="Writer", description="d", mission_id=mid,
        context={"workflow_step_id": "3.draft"},
    )
    # Verify sibling
    verify_id = await add_task(
        title="Verify", description="d", mission_id=mid,
        context={"workflow_step_id": "3.draft.verify"},
    )
    # Confirm task (by id)
    confirm_id = await add_task(title="Confirm", description="d", mission_id=mid)

    # Mark all as completed
    await update_task(writer_id, status="completed")
    await update_task(verify_id, status="completed")
    await update_task(confirm_id, status="completed")

    await reset_workflow_step(mid, "3.draft", confirm_task_id=confirm_id)

    r_writer = await _fetch_task(db_path, writer_id)
    r_verify = await _fetch_task(db_path, verify_id)
    r_confirm = await _fetch_task(db_path, confirm_id)
    assert r_writer["status"] == "pending"
    assert r_writer["worker_attempts"] == 0
    assert r_verify["status"] == "pending"
    assert r_confirm["status"] == "pending"


# ──────────────────────────────────────────────────────────────────────────────
# recover_startup_tasks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recover_startup_tasks(fresh_db):
    """recover_startup_tasks resets processing→pending and clears backoff."""
    db_path = fresh_db
    import src.infra.db as db_module
    from general_beckman import add_task, update_task, recover_startup_tasks

    # One processing task
    t1 = await add_task(title="Processing", description="d")
    await update_task(t1, status="processing")

    # One pending task with a future next_retry_at
    t2 = await add_task(title="Backoff", description="d")
    # Set next_retry_at to a future time
    db = await db_module.get_db()
    await db.execute(
        "UPDATE tasks SET next_retry_at = datetime('now', '+1 hour') WHERE id = ?",
        (t2,)
    )
    await db.commit()

    result = await recover_startup_tasks()

    assert result["interrupted"] == 1
    assert result["backoff_cleared"] == 1

    r1 = await _fetch_task(db_path, t1)
    r2 = await _fetch_task(db_path, t2)
    assert r1["status"] == "pending"
    assert r1["retry_reason"] == "infrastructure"
    assert r2["next_retry_at"] is None


# ──────────────────────────────────────────────────────────────────────────────
# reset_cascade_failed_dependents
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_cascade_failed_dependents(fresh_db):
    """reset_cascade_failed_dependents resets cascade-failed tasks."""
    db_path = fresh_db
    from general_beckman import add_task, update_task, reset_cascade_failed_dependents

    # Primary task that was in DLQ
    primary_id = await add_task(title="Primary", description="d")

    # Dependent that was cascade-failed because primary failed
    dep_id = await add_task(
        title="Dependent", description="d", depends_on=[primary_id]
    )
    await update_task(
        dep_id, status="failed", error="All dependencies failed"
    )

    # Another failed task with a different error — should NOT be reset
    other_id = await add_task(title="Other", description="d")
    await update_task(other_id, status="failed", error="different error")

    count = await reset_cascade_failed_dependents(primary_id)
    assert count == 1

    r_dep = await _fetch_task(db_path, dep_id)
    r_other = await _fetch_task(db_path, other_id)
    assert r_dep["status"] == "pending"
    assert r_dep["error"] is None
    assert r_other["status"] == "failed"  # unaffected


# ──────────────────────────────────────────────────────────────────────────────
# Guard tests: no raw task SQL or helper imports outside sanctioned modules
# ──────────────────────────────────────────────────────────────────────────────


def test_no_raw_tasks_sql_outside_db(repo_source_texts):
    """No source file outside src/infra/db.py may contain raw
    INSERT INTO tasks, UPDATE tasks, or DELETE FROM tasks SQL.

    After migration all former raw-SQL sites call beckman APIs instead.
    src/infra/db.py is the sole SQL owner.
    """
    import re
    from pathlib import Path

    root = Path(__file__).parents[3].resolve()

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

    for filepath, text in repo_source_texts.items():
        if filepath in allowed:
            continue
        # Allow all of general_beckman/src/general_beckman/ — beckman is the write-owner.
        if any(str(filepath).startswith(str(d)) for d in allowed_dirs):
            continue
        # Skip migration scripts (one-time DB tools, not prod callers).
        if any(str(filepath).startswith(str(p)) for p in skip_prefixes):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if sql_re.search(line):
                rel = filepath.relative_to(root)
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert violations == [], (
        "Raw tasks SQL found outside src/infra/db.py and general_beckman — "
        "use general_beckman task write API instead:\n"
        + "\n".join(violations)
    )


def test_no_raw_db_task_imports_outside_infra_beckman(repo_source_texts, ast_db_write_imports_fn):
    """No source file outside src/infra/db.py itself and general_beckman may
    import task-write helpers directly from src.infra.db.

    Covered helpers:
      add_task, update_task, update_task_by_context_field,
      add_subtasks_atomically, insert_tasks_atomically,
      propagate_skips, claim_task, cancel_task, reprioritize_task,
      save_task_checkpoint, clear_task_checkpoint,
      reset_failed_tasks, reset_stuck_tasks, reset_blocked_tasks,
      cancel_pending_tasks, reset_workflow_step,
      recover_startup_tasks, reset_cascade_failed_dependents,
      reset_blocked_on_founder_tasks.

    IMPORTANT: src/infra/dead_letter.py is NOT exempt — it must route via
    beckman like everyone else. Only src/infra/db.py itself is exempt.

    Detection uses ast.parse so parenthesised multi-line imports
    (e.g. ``from src.infra.db import (\\n  add_task,\\n)``) are caught.
    Falls back to line-regex on SyntaxError.
    """
    import re
    from pathlib import Path

    root = Path(__file__).parents[3].resolve()

    guarded_names = frozenset({
        "add_task", "update_task", "update_task_by_context_field",
        "add_subtasks_atomically", "insert_tasks_atomically",
        "propagate_skips", "claim_task", "cancel_task", "reprioritize_task",
        "save_task_checkpoint", "clear_task_checkpoint",
        "reset_failed_tasks", "reset_stuck_tasks", "reset_blocked_tasks",
        "cancel_pending_tasks", "reset_workflow_step",
        "recover_startup_tasks", "reset_cascade_failed_dependents",
        "reset_blocked_on_founder_tasks",
    })

    # Fallback line-regex for unparseable files (SyntaxError path).
    _names = "|".join(sorted(guarded_names))
    import_re = re.compile(
        rf'from\s+src\.infra\.db\s+import\s+.*?\b({_names})\b',
    )
    rel_import_re = re.compile(
        rf'from\s+\.+infra\.db\s+import\s+.*?\b({_names})\b',
    )

    # Only src/infra/db.py itself is exempt (not all of src/infra/)
    allowed_files = {
        (root / "src" / "infra" / "db.py").resolve(),
        Path(__file__).resolve(),
    }
    allowed_dirs = {
        (root / "packages" / "general_beckman" / "src" / "general_beckman").resolve(),
    }
    # Migration scripts are one-time DB tools — not prod callers.
    skip_prefixes = {
        (root / "scripts").resolve(),
    }

    violations: list[str] = []

    for filepath, text in repo_source_texts.items():
        if filepath in allowed_files:
            continue
        if any(str(filepath).startswith(str(d)) for d in allowed_dirs):
            continue
        if any(str(filepath).startswith(str(p)) for p in skip_prefixes):
            continue

        # AST-based detection (catches multi-line imports).
        ast_hits = ast_db_write_imports_fn(filepath, text, guarded_names)
        if ast_hits:
            rel = filepath.relative_to(root)
            for lineno, name in ast_hits:
                violations.append(f"{rel}:{lineno}: import of '{name}' from src.infra.db")
            continue  # already reported; skip line-regex for this file

        # Fallback: line-regex (handles SyntaxError files).
        for lineno, line in enumerate(text.splitlines(), 1):
            if import_re.search(line) or rel_import_re.search(line):
                rel = filepath.relative_to(root)
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert violations == [], (
        "Direct db task-write helper import found outside src/infra/db.py and "
        "general_beckman — use general_beckman task write API instead:\n"
        + "\n".join(violations)
    )
