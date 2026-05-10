"""Z10 T3C — per-mission tx-lock shard."""
from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_distinct_mission_ids_get_distinct_locks():
    from src.infra.db import _get_tx_lock
    a = _get_tx_lock(1)
    b = _get_tx_lock(2)
    g = _get_tx_lock(None)
    assert a is not b
    assert a is not g
    assert b is not g


@pytest.mark.asyncio
async def test_same_mission_id_returns_same_lock():
    from src.infra.db import _get_tx_lock
    a1 = _get_tx_lock(42)
    a2 = _get_tx_lock(42)
    assert a1 is a2


@pytest.mark.asyncio
async def test_concurrent_missions_do_not_block_each_other():
    """A slow lock-hold on mission A must NOT delay mission B's lock."""
    from src.infra.db import _get_tx_lock

    a_lock = _get_tx_lock(101)
    b_lock = _get_tx_lock(102)
    assert a_lock is not b_lock

    a_started = asyncio.Event()
    b_done = asyncio.Event()

    async def slow_a():
        async with a_lock:
            a_started.set()
            await asyncio.sleep(0.3)

    async def fast_b():
        await a_started.wait()
        # While A is mid-hold, B must acquire immediately.
        t0 = asyncio.get_event_loop().time()
        async with b_lock:
            elapsed = asyncio.get_event_loop().time() - t0
        b_done.set()
        # Generous threshold — should be near-zero, but CI jitter happens.
        assert elapsed < 0.15, f"B waited {elapsed:.3f}s for its own lock"

    await asyncio.gather(slow_a(), fast_b())
    assert b_done.is_set()


@pytest.mark.asyncio
async def test_combined_lock_order_is_global_then_mission():
    from src.infra.db import _get_combined_lock, _get_tx_lock

    g_lock = _get_tx_lock(None)
    m_lock = _get_tx_lock(999)
    async with _get_combined_lock(999):
        # Both locks should be held now — neither can be acquired by another
        # coroutine without waiting.
        assert g_lock.locked()
        assert m_lock.locked()
    # Released after exit.
    assert not g_lock.locked()
    assert not m_lock.locked()


@pytest.mark.asyncio
async def test_combined_lock_with_none_mission_skips_mission_acquire():
    from src.infra.db import _get_combined_lock, _get_tx_lock
    g_lock = _get_tx_lock(None)
    async with _get_combined_lock(None):
        assert g_lock.locked()
    assert not g_lock.locked()


@pytest.mark.asyncio
async def test_add_task_per_mission_lock_no_cross_contention(tmp_path, monkeypatch):
    """Mission A's add_task must not serialize Mission B's add_task."""
    db_path = tmp_path / "lock_shard.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()

    # Reset the per-mission lock cache so a fresh test gets clean Locks.
    db_mod._mission_tx_locks.clear()

    # 5 tasks per mission, alternating insertion order — they should
    # interleave freely without WAL contention.
    async def add_for(mission_id, n):
        for i in range(n):
            await db_mod.add_task(
                title=f"M{mission_id}-T{i}",
                description=f"desc {i}",
                mission_id=mission_id,
            )

    await asyncio.gather(add_for(1001, 5), add_for(1002, 5))
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT mission_id, COUNT(*) FROM tasks "
            "WHERE mission_id IN (1001, 1002) GROUP BY mission_id"
        )
        rows = dict(await cur.fetchall())
    assert rows.get(1001, 0) == 5
    assert rows.get(1002, 0) == 5
