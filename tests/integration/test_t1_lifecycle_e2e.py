"""Z8 T1E — e2e: ongoing mission survives simulated restart, then revoke."""
from __future__ import annotations

import pytest


async def _fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "t1e.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


async def _activate_ongoing(db_mod, mid: int, cursor_json: str) -> None:
    db = await db_mod.get_db()
    await db.execute(
        "UPDATE missions SET kind='ongoing', lifecycle_state='active', "
        "cursor=? WHERE id=?",
        (cursor_json, mid),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_ongoing_mission_survives_simulated_restart(tmp_path, monkeypatch):
    """find_resumable() returns the active ongoing mission, with cursor parsed.

    Simulates a restart by calling ``find_resumable()`` fresh after the
    mission is committed — equivalent to a new process opening the same
    DB file.
    """
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission(title="watch app", description="sentry watch", priority=5)
    await _activate_ongoing(db_mod, mid, '{"sentry": "e1"}')

    from general_beckman.resumption import find_resumable
    resumed = await find_resumable()
    survivor = next((r for r in resumed if r.id == mid), None)
    assert survivor is not None
    assert survivor.cursor == {"sentry": "e1"}


@pytest.mark.asyncio
async def test_restart_then_revoke_then_restart(tmp_path, monkeypatch):
    """Full e2e: boot → revoke → boot again → mission no longer surfaces."""
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission(title="watch app", description="watch", priority=5)
    await _activate_ongoing(db_mod, mid, '{"sentry":"e1"}')

    from general_beckman.resumption import find_resumable, revoke, update_cursor

    # Boot 1: mission surfaces.
    resumed_1 = await find_resumable()
    assert any(r.id == mid for r in resumed_1)

    # Handler advances cursor.
    await update_cursor(mid, {"sentry": "e2"})
    resumed_mid = await find_resumable()
    survivor = next(r for r in resumed_mid if r.id == mid)
    assert survivor.cursor == {"sentry": "e2"}

    # Revoke (the /stop_ops path).
    ok = await revoke(mid)
    assert ok is True

    # Boot 2: revoked mission must not surface.
    resumed_2 = await find_resumable()
    assert not any(r.id == mid for r in resumed_2)

    # And revoked_at is stamped.
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT lifecycle_state, revoked_at FROM missions WHERE id=?", (mid,)
    )
    row = await cur.fetchone()
    assert row[0] == "revoked"
    assert row[1] is not None


@pytest.mark.asyncio
async def test_mixed_lifecycle_only_active_ongoing_resumes(tmp_path, monkeypatch):
    """Sibling missions in different states must not bleed into resumption."""
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid_active = await db_mod.add_mission(title="a", description="a", priority=5)
    mid_revoked = await db_mod.add_mission(title="r", description="r", priority=5)
    mid_oneshot = await db_mod.add_mission(title="o", description="o", priority=5)
    mid_pending = await db_mod.add_mission(title="p", description="p", priority=5)

    db = await db_mod.get_db()
    await db.execute(
        "UPDATE missions SET kind='ongoing', lifecycle_state='active', "
        "cursor='{}' WHERE id=?",
        (mid_active,),
    )
    await db.execute(
        "UPDATE missions SET kind='ongoing', lifecycle_state='revoked', "
        "revoked_at=datetime('now') WHERE id=?",
        (mid_revoked,),
    )
    await db.execute(
        "UPDATE missions SET kind='ongoing', lifecycle_state='pending' "
        "WHERE id=?",
        (mid_pending,),
    )
    # mid_oneshot keeps default ('oneshot','terminal')
    await db.commit()

    from general_beckman.resumption import find_resumable
    resumed = await find_resumable()
    ids = {r.id for r in resumed}
    assert mid_active in ids
    assert mid_revoked not in ids
    assert mid_oneshot not in ids
    assert mid_pending not in ids
