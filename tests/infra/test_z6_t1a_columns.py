"""Z6 T1A — tasks.needs_real_tools + reversibility hoist tests."""
from __future__ import annotations

import json

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "z6_t1a.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_columns_exist_after_init(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cur = await db.execute("PRAGMA table_info(tasks)")
    cols = [row[1] for row in await cur.fetchall()]
    assert "needs_real_tools" in cols
    assert "reversibility" in cols


@pytest.mark.asyncio
async def test_migration_idempotent(tmp_path, monkeypatch):
    """Second init_db() call must not raise."""
    _, db_mod = await _setup(tmp_path, monkeypatch)
    # Re-run init_db; both migrations are guarded by schema_migrations.
    await db_mod.init_db()
    # Sanity: columns still present (no duplicate add).
    db = await db_mod.get_db()
    cur = await db.execute("PRAGMA table_info(tasks)")
    cols = [row[1] for row in await cur.fetchall()]
    assert cols.count("needs_real_tools") == 1
    assert cols.count("reversibility") == 1


@pytest.mark.asyncio
async def test_add_task_persists_explicit_flags(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    tid = await db_mod.add_task(
        title="explicit flags",
        description="",
        needs_real_tools=True,
        reversibility="irreversible",
    )
    assert tid
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT needs_real_tools, reversibility FROM tasks WHERE id = ?",
        (tid,),
    )
    row = await cur.fetchone()
    assert int(row[0]) == 1
    assert row[1] == "irreversible"


@pytest.mark.asyncio
async def test_add_task_hoists_from_context(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    tid = await db_mod.add_task(
        title="hoist from ctx",
        description="",
        context={
            "needs_real_tools": True,
            "reversibility": "partial",
            "workflow_step_id": "13.1",
        },
    )
    assert tid
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT needs_real_tools, reversibility, context "
        "FROM tasks WHERE id = ?",
        (tid,),
    )
    row = await cur.fetchone()
    assert int(row[0]) == 1
    assert row[1] == "partial"
    # context still contains the original flag — admission gate may read it
    # for diagnostics, but the column is the gate.
    ctx = json.loads(row[2])
    assert ctx.get("needs_real_tools") is True


@pytest.mark.asyncio
async def test_add_task_defaults_zero_null(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    tid = await db_mod.add_task(title="plain", description="")
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT needs_real_tools, reversibility FROM tasks WHERE id = ?",
        (tid,),
    )
    row = await cur.fetchone()
    # Default 0 from migration; reversibility is NULL.
    assert int(row[0] or 0) == 0
    assert row[1] is None


@pytest.mark.asyncio
async def test_insert_tasks_atomically_hoists(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    # Create a mission first so mission_id resolves.
    mid = await db_mod.add_mission("test mission", "")
    ids = await db_mod.insert_tasks_atomically(
        tasks=[
            {
                "title": "real-world step",
                "description": "",
                "agent_type": "executor",
                "context": {
                    "needs_real_tools": True,
                    "reversibility": "irreversible",
                    "workflow_step_id": "13.1",
                },
            },
        ],
        mission_id=mid,
    )
    assert ids and ids[0] > 0
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT needs_real_tools, reversibility FROM tasks WHERE id = ?",
        (ids[0],),
    )
    row = await cur.fetchone()
    assert int(row[0]) == 1
    assert row[1] == "irreversible"


@pytest.mark.asyncio
async def test_expander_propagates_real_world_fields(tmp_path, monkeypatch):
    """Expander must surface needs_real_tools+reversibility+real_tool_kind
    +cost_estimate_usd onto task.context."""
    _, _db_mod = await _setup(tmp_path, monkeypatch)
    from src.workflows.engine.expander import expand_steps_to_tasks

    steps = [
        {
            "id": "13.1",
            "name": "production_infra",
            "phase": "phase_13",
            "agent": "executor",
            "instruction": "deploy",
            "needs_real_tools": True,
            "reversibility": "irreversible",
            "real_tool_kind": "vercel|railway",
            "cost_estimate_usd": 50,
        }
    ]
    tasks = expand_steps_to_tasks(steps, mission_id=1, initial_context={})
    assert len(tasks) == 1
    ctx = tasks[0]["context"]
    assert ctx.get("needs_real_tools") is True
    assert ctx.get("reversibility") == "irreversible"
    assert ctx.get("real_tool_kind") == "vercel|railway"
    assert ctx.get("cost_estimate_usd") == 50
