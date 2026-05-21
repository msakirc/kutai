"""Z6 T1E — mission lifecycle coordination with founder_actions."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "z6_lc.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()
    return db_path, db_mod, fa


async def _mission_status(db_mod, mid: int) -> str:
    """Read the mission's lifecycle gate column. Post-Z0 the gate is
    ``lifecycle_state`` (Z8 T1A migration always adds it on init_db); the
    legacy ``status`` fallback only applies to installs predating it."""
    import src.founder_actions as fa
    col = await fa._missions_lifecycle_column()
    db = await db_mod.get_db()
    cur = await db.execute(
        f"SELECT {col} FROM missions WHERE id = ?", (mid,),
    )
    row = await cur.fetchone()
    return row[0]


@pytest.mark.asyncio
async def test_lifecycle_column_uses_lifecycle_state_when_present(
    tmp_path, monkeypatch,
):
    """Post-Z0 main: init_db always adds missions.lifecycle_state (Z8 T1A
    migration), so the coordinator targets it, not legacy ``status``."""
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    col = await fa._missions_lifecycle_column()
    assert col == "lifecycle_state"


@pytest.mark.asyncio
async def test_creating_action_blocks_mission(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    assert (await _mission_status(db_mod, mid)) == "active"
    await fa.create(mid, "generic", "t", "w", [], notify_telegram=False)
    assert (
        await _mission_status(db_mod, mid)
    ) == "blocked_on_founder_action"


@pytest.mark.asyncio
async def test_resolving_only_action_unblocks_mission(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a = await fa.create(mid, "generic", "t", "w", [], notify_telegram=False)
    assert (
        await _mission_status(db_mod, mid)
    ) == "blocked_on_founder_action"
    await fa.resolve(a.id)
    assert (await _mission_status(db_mod, mid)) == "active"


@pytest.mark.asyncio
async def test_multiple_actions_keep_mission_blocked_until_last(
    tmp_path, monkeypatch,
):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a1 = await fa.create(mid, "generic", "t1", "w", [], notify_telegram=False)
    a2 = await fa.create(mid, "generic", "t2", "w", [], notify_telegram=False)
    await fa.resolve(a1.id)
    assert (
        await _mission_status(db_mod, mid)
    ) == "blocked_on_founder_action"
    await fa.resolve(a2.id)
    assert (await _mission_status(db_mod, mid)) == "active"


@pytest.mark.asyncio
async def test_block_idempotent_on_already_blocked(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    await fa.create(mid, "generic", "t1", "w", [], notify_telegram=False)
    # Already blocked; calling block_if_needed again is a no-op.
    flipped = await fa.block_mission_if_needed(mid)
    assert flipped is False


@pytest.mark.asyncio
async def test_unblock_idempotent_on_already_active(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    flipped = await fa.unblock_mission_if_clear(mid)
    assert flipped is False


@pytest.mark.asyncio
async def test_sweep_unblock_all(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    m1 = await db_mod.add_mission("m1", "")
    m2 = await db_mod.add_mission("m2", "")
    a1 = await fa.create(m1, "generic", "t", "w", [], notify_telegram=False)
    a2 = await fa.create(m2, "generic", "t", "w", [], notify_telegram=False)
    # Resolve directly via DB to skip the on-resolve hook, simulating
    # external resolution (e.g. Yaşar Usta bot, /action_done from
    # another process).
    db = await db_mod.get_db()
    await db.execute(
        "UPDATE founder_actions SET status = 'done', resolved_at = "
        "datetime('now') WHERE id IN (?, ?)",
        (a1.id, a2.id),
    )
    await db.commit()
    # Missions are still flagged blocked.
    assert (
        await _mission_status(db_mod, m1)
    ) == "blocked_on_founder_action"
    n = await fa.sweep_unblock_all()
    assert n == 2
    assert (await _mission_status(db_mod, m1)) == "active"
    assert (await _mission_status(db_mod, m2)) == "active"


@pytest.mark.asyncio
async def test_unblock_flips_blocked_tasks_back_to_pending(
    tmp_path, monkeypatch,
):
    """When mission unblocks, parked tasks return to pending so the next
    pump cycle re-evaluates them."""
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    tid = await db_mod.add_task(
        title="real-tools", description="", mission_id=mid,
        context={"workflow_step_id": "13.1"},
    )
    # Park the task as beckman would.
    await db_mod.update_task(tid, status="blocked_on_founder_action")
    a = await fa.create(mid, "generic", "t", "w", [], notify_telegram=False)
    # Resolve the action — unblock_mission flips both mission and task.
    await fa.resolve(a.id)
    db = await db_mod.get_db()
    cur = await db.execute("SELECT status FROM tasks WHERE id = ?", (tid,))
    assert (await cur.fetchone())[0] == "pending"


@pytest.mark.asyncio
async def test_lifecycle_state_path_when_z0_merged(tmp_path, monkeypatch):
    """Post-Z0: lifecycle_state is the gate; legacy status stays 'active'."""
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    mid = await db_mod.add_mission("m", "")
    await fa.create(mid, "generic", "t", "w", [], notify_telegram=False)
    cur = await db.execute(
        "SELECT lifecycle_state, status FROM missions WHERE id = ?", (mid,),
    )
    row = await cur.fetchone()
    # lifecycle_state is the gate; status remains 'active'.
    assert row[0] == "blocked_on_founder_action"
    assert row[1] == "active"
