"""Z8 T1C — orchestrator resumption + revocation tests."""
from __future__ import annotations

import pytest


async def _fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "resume.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


async def _make_mission(db_mod, *, kind="ongoing", state="active",
                       cursor=None, revoked=False) -> int:
    mid = await db_mod.add_mission(
        title="watch app", description="watch", priority=5,
    )
    db = await db_mod.get_db()
    if revoked:
        await db.execute(
            "UPDATE missions SET kind=?, lifecycle_state='revoked', "
            "revoked_at=datetime('now'), cursor=? WHERE id=?",
            (kind, cursor, mid),
        )
    else:
        await db.execute(
            "UPDATE missions SET kind=?, lifecycle_state=?, cursor=? WHERE id=?",
            (kind, state, cursor, mid),
        )
    await db.commit()
    return mid


@pytest.mark.asyncio
async def test_find_resumable_returns_active_ongoing(tmp_path, monkeypatch):
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await _make_mission(
        db_mod, kind="ongoing", state="active",
        cursor='{"sentry": "evt_123"}',
    )
    from general_beckman.resumption import find_resumable
    resumed = await find_resumable()
    ids = [m.id for m in resumed]
    assert mid in ids
    m = next(r for r in resumed if r.id == mid)
    assert m.cursor == {"sentry": "evt_123"}


@pytest.mark.asyncio
async def test_find_resumable_skips_revoked(tmp_path, monkeypatch):
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await _make_mission(db_mod, kind="ongoing", revoked=True)
    from general_beckman.resumption import find_resumable
    resumed = await find_resumable()
    assert mid not in [m.id for m in resumed]


@pytest.mark.asyncio
async def test_find_resumable_skips_oneshot(tmp_path, monkeypatch):
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await _make_mission(db_mod, kind="oneshot", state="active")
    from general_beckman.resumption import find_resumable
    resumed = await find_resumable()
    assert mid not in [m.id for m in resumed]


@pytest.mark.asyncio
async def test_find_resumable_skips_pending_ongoing(tmp_path, monkeypatch):
    """Ongoing mission still in 'pending' state isn't resumed — only 'active'."""
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await _make_mission(db_mod, kind="ongoing", state="pending")
    from general_beckman.resumption import find_resumable
    resumed = await find_resumable()
    assert mid not in [m.id for m in resumed]


@pytest.mark.asyncio
async def test_update_cursor_persists(tmp_path, monkeypatch):
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await _make_mission(db_mod, kind="ongoing", state="active",
                              cursor='{}')
    from general_beckman.resumption import update_cursor, find_resumable
    await update_cursor(mid, {"github": "evt_999"})
    resumed = await find_resumable()
    m = next(r for r in resumed if r.id == mid)
    assert m.cursor == {"github": "evt_999"}


@pytest.mark.asyncio
async def test_revoke_transitions_state_and_sets_revoked_at(tmp_path, monkeypatch):
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await _make_mission(db_mod, kind="ongoing", state="active")
    from general_beckman.resumption import revoke
    ok = await revoke(mid)
    assert ok is True
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT lifecycle_state, revoked_at FROM missions WHERE id=?", (mid,)
    )
    row = await cur.fetchone()
    assert row[0] == "revoked"
    assert row[1] is not None  # timestamp populated


@pytest.mark.asyncio
async def test_revoke_oneshot_is_noop(tmp_path, monkeypatch):
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await _make_mission(db_mod, kind="oneshot", state="active")
    from general_beckman.resumption import revoke
    ok = await revoke(mid)
    assert ok is False
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT lifecycle_state FROM missions WHERE id=?", (mid,)
    )
    row = await cur.fetchone()
    assert row[0] == "active"  # untouched
