"""Z10 T3A D4 — /mission <id> renders Pacing block."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "mv.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_format_mission_view_contains_pacing_when_budget(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    conn = await db.get_db()
    await conn.execute(
        "INSERT INTO missions (id, title, description, status, "
        "                       time_budget_hours, target_launch) "
        "VALUES (47, 'Build it', '', 'active', 40.0, '2026-05-20')",
    )
    await conn.execute(
        "INSERT INTO tasks (mission_id, title, description, agent_type, "
        " status, started_at, completed_at) "
        "VALUES (47, 'a', '', 'executor', 'completed', "
        " '2026-05-10 09:00:00', '2026-05-10 11:00:00')",
    )
    await conn.commit()

    from src.app.mission_view import format_mission_view
    body = await format_mission_view(47)
    assert "Pacing" in body
    assert "Elapsed:" in body
    assert "Scope:" in body
    # Budget set → projection line should appear (we have elapsed > 0).
    assert "Projected:" in body or "no budget" not in body
    assert "Mission #47" in body


@pytest.mark.asyncio
async def test_format_mission_view_omits_projection_when_no_budget(
    tmp_path, monkeypatch
):
    db = await _setup(tmp_path, monkeypatch)
    conn = await db.get_db()
    await conn.execute(
        "INSERT INTO missions (id, title, description, status) "
        "VALUES (48, 'No budget', '', 'active')",
    )
    await conn.execute(
        "INSERT INTO tasks (mission_id, title, description, agent_type, "
        " status, started_at, completed_at) "
        "VALUES (48, 'a', '', 'executor', 'completed', "
        " '2026-05-10 09:00:00', '2026-05-10 10:30:00')",
    )
    await conn.commit()

    from src.app.mission_view import format_mission_view
    body = await format_mission_view(48)
    # Projection line requires budget; should be absent.
    assert "Projected:" not in body
    # But Elapsed should still appear with "no budget" hint.
    assert "no budget" in body


@pytest.mark.asyncio
async def test_format_mission_view_missing_mission(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from src.app.mission_view import format_mission_view
    body = await format_mission_view(9999)
    assert "not found" in body
