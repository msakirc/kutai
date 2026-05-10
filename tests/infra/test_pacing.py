"""Z10 T3A D2 — mission pacing computation + snapshot persistence."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "pacing.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


async def _seed_mission(
    db, mission_id, *,
    title="m", status="active",
    time_budget=None, target=None, created_at=None,
):
    conn = await db.get_db()
    if created_at is None:
        await conn.execute(
            "INSERT INTO missions (id, title, description, status, "
            "                       time_budget_hours, target_launch) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (mission_id, title, "", status, time_budget, target),
        )
    else:
        await conn.execute(
            "INSERT INTO missions (id, title, description, status, "
            "                       time_budget_hours, target_launch, "
            "                       created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mission_id, title, "", status, time_budget, target, created_at),
        )
    await conn.commit()


async def _seed_task(
    db, mission_id, *,
    status="completed",
    started_at=None, completed_at=None,
    step_started_at=None,
    phase_id=None,
    agent_type="executor",
    title="t",
):
    conn = await db.get_db()
    cur = await conn.execute(
        "INSERT INTO tasks "
        "(mission_id, title, description, agent_type, status, "
        " started_at, completed_at, step_started_at, phase_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (mission_id, title, "", agent_type, status,
         started_at, completed_at, step_started_at, phase_id),
    )
    await conn.commit()
    return cur.lastrowid


@pytest.mark.asyncio
async def test_elapsed_hours_one_task_no_budget(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    await _seed_mission(db, 1, time_budget=None)
    # Two-hour completed task.
    await _seed_task(
        db, 1,
        status="completed",
        started_at="2026-05-10 10:00:00",
        completed_at="2026-05-10 12:00:00",
    )
    from src.infra.pacing import compute_mission_pacing
    p = await compute_mission_pacing(1)
    assert abs(p["elapsed_hours"] - 2.0) < 1e-3
    assert p["percent_burn"] is None
    assert p["projected_finish_at"] is None
    assert p["tradeoff_due"] is False


@pytest.mark.asyncio
async def test_percent_burn_below_threshold(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    await _seed_mission(db, 2, time_budget=4.0)
    await _seed_task(
        db, 2,
        status="completed",
        started_at="2026-05-10 10:00:00",
        completed_at="2026-05-10 12:00:00",
    )
    from src.infra.pacing import compute_mission_pacing
    p = await compute_mission_pacing(2)
    assert abs(p["percent_burn"] - 0.5) < 1e-3
    assert p["tradeoff_due"] is False


@pytest.mark.asyncio
async def test_tradeoff_due_when_burn_and_scope_high(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    await _seed_mission(db, 3, time_budget=4.0)
    # 3.5h elapsed (87.5%) + 2 of 4 tasks pending (50%) → tradeoff_due.
    await _seed_task(
        db, 3,
        status="completed",
        started_at="2026-05-10 10:00:00",
        completed_at="2026-05-10 13:30:00",
    )
    await _seed_task(db, 3, status="completed",
                     started_at="2026-05-10 10:00:00",
                     completed_at="2026-05-10 10:00:00")
    await _seed_task(db, 3, status="pending")
    await _seed_task(db, 3, status="pending")
    from src.infra.pacing import compute_mission_pacing
    p = await compute_mission_pacing(3)
    assert p["percent_burn"] is not None and p["percent_burn"] > 0.75
    assert p["scope_remaining_pct"] > 0.25
    assert p["tradeoff_due"] is True


@pytest.mark.asyncio
async def test_take_pacing_snapshot_writes_row(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    await _seed_mission(db, 47, time_budget=10.0)
    await _seed_task(
        db, 47,
        status="completed",
        started_at="2026-05-10 09:00:00",
        completed_at="2026-05-10 11:00:00",
    )
    from src.infra.pacing import take_pacing_snapshot
    snap_id = await take_pacing_snapshot(47)
    assert snap_id > 0
    conn = await db.get_db()
    cur = await conn.execute(
        "SELECT mission_id, elapsed_hours, percent_burn "
        "FROM mission_pacing_snapshots WHERE id = ?",
        (snap_id,),
    )
    row = dict(await cur.fetchone())
    assert row["mission_id"] == 47
    assert abs(float(row["elapsed_hours"]) - 2.0) < 1e-3
    assert abs(float(row["percent_burn"]) - 0.2) < 1e-3


@pytest.mark.asyncio
async def test_no_tasks_yet(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    await _seed_mission(db, 5, time_budget=4.0)
    from src.infra.pacing import compute_mission_pacing
    p = await compute_mission_pacing(5)
    assert p["elapsed_hours"] == 0.0
    assert p["scope_remaining_pct"] == 0.0
    assert p["tradeoff_due"] is False
