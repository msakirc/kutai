"""Z10 T3A D2 — per-phase breakdown in pacing.

Also covers D6: maintenance tasks (phase_id IS NULL, agent_type=mechanical)
count toward total elapsed_hours.
"""
from __future__ import annotations

import json

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "pace_phase.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_phase_breakdown_sums(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    conn = await db.get_db()
    # Phase budget for phase_5 = 6h, phase_6 = 8h.
    pb = json.dumps({"phase_5": 6.0, "phase_6": 8.0})
    await conn.execute(
        "INSERT INTO missions (id, title, description, status, "
        "                       time_budget_hours, phase_budget_json) "
        "VALUES (10, 't', '', 'active', 20.0, ?)",
        (pb,),
    )
    await conn.commit()

    async def _t(phase, end_ts):
        await conn.execute(
            "INSERT INTO tasks (mission_id, title, description, "
            " agent_type, status, started_at, completed_at, phase_id) "
            "VALUES (10, 'x', '', 'executor', 'completed', "
            "'2026-05-10 10:00:00', ?, ?)",
            (end_ts, phase),
        )
    await _t("phase_5", "2026-05-10 11:00:00")   # 1h
    await _t("phase_5", "2026-05-10 10:30:00")   # 0.5h
    await _t("phase_6", "2026-05-10 11:00:00")   # 1h
    await conn.commit()

    from src.infra.pacing import compute_mission_pacing
    p = await compute_mission_pacing(10)
    breakdown = {b["phase_id"]: b for b in p["phase_breakdown"]}
    assert abs(breakdown["phase_5"]["elapsed_h"] - 1.5) < 1e-3
    assert abs(breakdown["phase_6"]["elapsed_h"] - 1.0) < 1e-3
    assert breakdown["phase_5"]["budget_h"] == 6.0
    assert breakdown["phase_6"]["budget_h"] == 8.0


@pytest.mark.asyncio
async def test_maintenance_tasks_count_toward_elapsed(tmp_path, monkeypatch):
    """D6: mechanical task with phase_id IS NULL still contributes."""
    db = await _setup(tmp_path, monkeypatch)
    conn = await db.get_db()
    await conn.execute(
        "INSERT INTO missions (id, title, description, status, "
        "                       time_budget_hours) "
        "VALUES (11, 't', '', 'active', 5.0)",
    )
    # Phase 5 task: 1h
    await conn.execute(
        "INSERT INTO tasks (mission_id, title, description, agent_type, "
        " status, started_at, completed_at, phase_id) "
        "VALUES (11, 'a', '', 'executor', 'completed', "
        " '2026-05-10 10:00:00', '2026-05-10 11:00:00', 'phase_5')",
    )
    # Mechanical maintenance (no phase_id): 30min
    await conn.execute(
        "INSERT INTO tasks (mission_id, title, description, agent_type, "
        " status, started_at, completed_at, phase_id) "
        "VALUES (11, 'maint', '', 'mechanical', 'completed', "
        " '2026-05-10 12:00:00', '2026-05-10 12:30:00', NULL)",
    )
    await conn.commit()
    from src.infra.pacing import compute_mission_pacing
    p = await compute_mission_pacing(11)
    # Total elapsed includes both: 1.0 + 0.5 = 1.5h
    assert abs(p["elapsed_hours"] - 1.5) < 1e-3
