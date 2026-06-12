"""Z6 T1B — founder_actions repo + table tests."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "fa.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    # Reset lifecycle cache between tests so the schema probe re-runs.
    import src.founder_actions as fa
    return db_path, db_mod, fa


@pytest.mark.asyncio
async def test_table_and_indexes_exist(tmp_path, monkeypatch):
    _, db_mod, _ = await _setup(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='founder_actions'"
    )
    assert await cur.fetchone() is not None
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name LIKE 'idx_founder_actions_%'"
    )
    idx_names = {row[0] for row in await cur.fetchall()}
    assert "idx_founder_actions_mission" in idx_names
    assert "idx_founder_actions_status" in idx_names


@pytest.mark.asyncio
async def test_create_returns_pending(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    action = await fa.create(
        mission_id=mid,
        kind="credential_paste",
        title="Paste Stripe key",
        why="Phase 13 needs Stripe API access",
        instructions=["Go to dashboard.stripe.com/apikeys", "Copy secret_key"],
        blocking_step_id="13.12",
        expected_output_kind="credential",
        notify_telegram=False,
    )
    assert action.id > 0
    assert action.status == "pending"
    assert action.kind == "credential_paste"
    assert action.blocking_step_id == "13.12"
    assert len(action.instructions) == 2


@pytest.mark.asyncio
async def test_list_by_mission_orders_desc(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a1 = await fa.create(
        mid, "generic", "first", "why", ["do x"], notify_telegram=False,
    )
    a2 = await fa.create(
        mid, "vendor_enroll", "second", "why2", [], notify_telegram=False,
    )
    rows = await fa.list_by_mission(mid)
    assert len(rows) == 2
    # Most recent first.
    assert rows[0].id == a2.id
    assert rows[1].id == a1.id


@pytest.mark.asyncio
async def test_list_by_mission_status_filter(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a1 = await fa.create(mid, "generic", "t1", "w", [], notify_telegram=False)
    a2 = await fa.create(mid, "generic", "t2", "w", [], notify_telegram=False)
    await fa.resolve(a1.id)
    pending = await fa.list_by_mission(mid, status_filter="pending")
    assert len(pending) == 1
    assert pending[0].id == a2.id
    done = await fa.list_by_mission(mid, status_filter=["done"])
    assert len(done) == 1
    assert done[0].id == a1.id


@pytest.mark.asyncio
async def test_list_pending_crosses_missions(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    m1 = await db_mod.add_mission("m1", "")
    m2 = await db_mod.add_mission("m2", "")
    await fa.create(m1, "generic", "t1", "w", [], notify_telegram=False)
    a = await fa.create(m2, "generic", "t2", "w", [], notify_telegram=False)
    await fa.resolve(a.id)
    pending = await fa.list_pending()
    assert len(pending) == 1
    assert pending[0].mission_id == m1


@pytest.mark.asyncio
async def test_update_status_transition_valid(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a = await fa.create(mid, "generic", "t", "w", [], notify_telegram=False)
    a2 = await fa.update_status(a.id, "in_progress")
    assert a2.status == "in_progress"
    a3 = await fa.update_status(a.id, "done", response_payload={"ok": True})
    assert a3.status == "done"
    assert a3.resolved_at is not None
    assert a3.response_payload == {"ok": True}


@pytest.mark.asyncio
async def test_update_status_invalid_transition_raises(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a = await fa.create(mid, "generic", "t", "w", [], notify_telegram=False)
    await fa.resolve(a.id)
    with pytest.raises(ValueError):
        await fa.update_status(a.id, "pending")  # done has no outgoing


@pytest.mark.asyncio
async def test_resolve_is_shortcut_for_done(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a = await fa.create(mid, "generic", "t", "w", [], notify_telegram=False)
    out = await fa.resolve(a.id, {"value": "abc"})
    assert out.status == "done"
    assert out.response_payload == {"value": "abc"}


@pytest.mark.asyncio
async def test_get_missing_returns_none(tmp_path, monkeypatch):
    _, _db_mod, fa = await _setup(tmp_path, monkeypatch)
    assert await fa.get(99999) is None


@pytest.mark.asyncio
async def test_indexes_used_by_query(tmp_path, monkeypatch):
    """Verify EXPLAIN QUERY PLAN hits the indexes (smoke test)."""
    _, db_mod, _ = await _setup(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cur = await db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM founder_actions "
        "WHERE mission_id = 1"
    )
    plan = " ".join(
        " ".join(str(v) for v in tuple(row)) for row in await cur.fetchall()
    )
    assert "idx_founder_actions_mission" in plan
