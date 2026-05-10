"""Z10 T3A D3 — claim_task stamps step_started_at."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "claim.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_claim_task_populates_step_started_at(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    conn = await db.get_db()
    await conn.execute(
        "INSERT INTO missions (id, title, description, status) "
        "VALUES (1, 't', '', 'active')",
    )
    cur = await conn.execute(
        "INSERT INTO tasks (mission_id, title, description, agent_type, "
        "                    status) "
        "VALUES (1, 'x', '', 'executor', 'pending')",
    )
    await conn.commit()
    task_id = cur.lastrowid

    ok = await db.claim_task(task_id)
    assert ok is True
    cur = await conn.execute(
        "SELECT status, started_at, step_started_at FROM tasks WHERE id = ?",
        (task_id,),
    )
    row = dict(await cur.fetchone())
    assert row["status"] == "processing"
    assert row["started_at"] is not None
    assert row["step_started_at"] is not None
    assert row["started_at"] == row["step_started_at"]


@pytest.mark.asyncio
async def test_add_task_extracts_phase_id_from_context(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    conn = await db.get_db()
    await conn.execute(
        "INSERT INTO missions (id, title, description, status) "
        "VALUES (2, 't', '', 'active')",
    )
    await conn.commit()
    tid = await db.add_task(
        title="seed", description="d", mission_id=2,
        context={"workflow_phase": "phase_7"},
    )
    assert tid is not None
    cur = await conn.execute(
        "SELECT phase_id FROM tasks WHERE id = ?", (tid,),
    )
    row = dict(await cur.fetchone())
    assert row["phase_id"] == "phase_7"
