"""Tests for general_beckman.cron_seed — internal cadence seeder."""
from __future__ import annotations

import pytest
import src.infra.db as _db_mod

from general_beckman.cron_seed import seed_internal_cadences, INTERNAL_CADENCES
from src.infra.db import get_db, init_db


@pytest.mark.asyncio
async def test_seed_internal_cadences_inserts_expected_rows(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    # Reset _seeded so this test is not affected by prior runs in the same process.
    import general_beckman.cron_seed as cs_mod
    monkeypatch.setattr(cs_mod, "_seeded", False)

    await init_db()
    await seed_internal_cadences()

    conn = await get_db()
    cursor = await conn.execute(
        "SELECT title, interval_seconds, kind FROM scheduled_tasks WHERE kind='internal'"
    )
    rows = [dict(r) for r in await cursor.fetchall()]
    titles = {r["title"] for r in rows}
    expected_titles = {c["title"] for c in INTERNAL_CADENCES}
    assert titles == expected_titles
    for r in rows:
        assert r["interval_seconds"] is not None

    # Teardown: close connection so monkeypatch's DB_PATH restore takes effect.
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_seed_is_idempotent(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    # Reset _seeded so each test gets a clean seeder state.
    import general_beckman.cron_seed as cs_mod
    monkeypatch.setattr(cs_mod, "_seeded", False)

    await init_db()
    await seed_internal_cadences()
    await seed_internal_cadences()  # second call must not duplicate

    conn = await get_db()
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM scheduled_tasks WHERE kind='internal'"
    )
    (count,) = await cursor.fetchone()
    assert count == len(INTERNAL_CADENCES)

    # Teardown: close connection so monkeypatch's DB_PATH restore takes effect.
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
