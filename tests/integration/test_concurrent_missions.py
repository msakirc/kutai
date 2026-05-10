"""Z10 T3C — concurrent-mission write smoke test.

Skip by default to keep the default suite fast. Run with:
    pytest tests/integration/test_concurrent_missions.py -m integration
"""
from __future__ import annotations

import asyncio

import aiosqlite
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_two_missions_in_parallel_no_database_locked(tmp_path, monkeypatch):
    db_path = tmp_path / "concurrent.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()

    db_mod._mission_tx_locks.clear()

    async def burst(mission_id, n):
        for i in range(n):
            await db_mod.add_task(
                title=f"M{mission_id}-{i}",
                description="x",
                mission_id=mission_id,
            )

    # 20 inserts each, parallel.
    t0 = asyncio.get_event_loop().time()
    await asyncio.gather(burst(7001, 20), burst(7002, 20))
    elapsed = asyncio.get_event_loop().time() - t0

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT mission_id, COUNT(*) FROM tasks "
            "WHERE mission_id IN (7001, 7002) GROUP BY mission_id"
        )
        rows = dict(await cur.fetchall())
    assert rows.get(7001) == 20
    assert rows.get(7002) == 20
    # Loose upper bound — 40 inserts well under 60s WAL timeout.
    assert elapsed < 30, f"40 inserts took {elapsed:.2f}s — likely lock contention"
