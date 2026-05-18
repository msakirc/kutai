"""Z10 T4A — missions.demo_required flag behavior."""
from __future__ import annotations

import pytest


async def _init_db(tmp_path, monkeypatch):
    db_path = tmp_path / "demo_required.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_demo_required_column_exists_with_default(tmp_path, monkeypatch):
    """Migration adds demo_required INTEGER DEFAULT 1."""
    db_mod = await _init_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status) VALUES (?, ?, ?)",
        (1, "default mission", "active"),
    )
    await db.commit()
    cur = await db.execute(
        "SELECT demo_required FROM missions WHERE id = 1"
    )
    row = await cur.fetchone()
    assert row is not None
    assert int(row[0]) == 1  # default-1 = strict


@pytest.mark.asyncio
async def test_record_demo_skips_silently_when_demo_required_zero(tmp_path, monkeypatch):
    """demo_required=0 with no specs → skips silently (no blocker post)."""
    db_mod = await _init_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status, demo_required) "
        "VALUES (?, ?, ?, ?)",
        (10, "lenient", "active", 0),
    )
    await db.commit()

    from mr_roboto import record_demo as rd

    blockers = []

    async def _fake_blocker(mid):
        blockers.append(mid)

    monkeypatch.setattr(rd, "_post_no_e2e_blocker", _fake_blocker)

    workspace = tmp_path / "ws10"
    workspace.mkdir()

    res = await rd.run(
        mission_id=10,
        scenario_path="tests/e2e/golden_path.spec.ts",
        workspace_root=str(workspace),
    )
    assert res["skipped"] is True
    assert res["reason"] == "no_e2e_specs"
    assert res["demo_required"] is False
    assert res["blocker_posted"] is False
    assert blockers == []


@pytest.mark.asyncio
async def test_record_demo_posts_blocker_when_strict(tmp_path, monkeypatch):
    """demo_required=1 (default) with no specs → posts [blocker]."""
    db_mod = await _init_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status) VALUES (?, ?, ?)",
        (11, "strict", "active"),
    )
    await db.commit()

    from mr_roboto import record_demo as rd

    blockers = []

    async def _fake_blocker(mid):
        blockers.append(mid)

    monkeypatch.setattr(rd, "_post_no_e2e_blocker", _fake_blocker)

    workspace = tmp_path / "ws11"
    workspace.mkdir()

    res = await rd.run(
        mission_id=11,
        scenario_path="tests/e2e/golden_path.spec.ts",
        workspace_root=str(workspace),
    )
    assert res["skipped"] is True
    assert res["reason"] == "no_e2e_specs"
    assert res["demo_required"] is True
    assert res["blocker_posted"] is True
    assert blockers == [11]
