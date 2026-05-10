"""Z10 T1C — schema_migrations ledger tests."""
from __future__ import annotations

import aiosqlite
import pytest


async def _fresh_db(tmp_path, monkeypatch):
    """Point DB_PATH at a tmp DB and pre-create the ledger table."""
    db_path = tmp_path / "migrations.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)

    # Pre-create the ledger so apply_migration() can use it directly without
    # running the full init_db() pipeline.
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version TEXT PRIMARY KEY,"
            " applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " sql TEXT NOT NULL,"
            " reversal_sql TEXT,"
            " description TEXT"
            ")"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS sample (id INTEGER PRIMARY KEY)"
        )
        await db.commit()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_apply_migration_idempotent(tmp_path, monkeypatch):
    db_path, db_mod = await _fresh_db(tmp_path, monkeypatch)

    first = await db_mod.apply_migration(
        version="v-test-1",
        sql="ALTER TABLE sample ADD COLUMN x TEXT",
        reversal_sql=None,
        description="add x",
    )
    second = await db_mod.apply_migration(
        version="v-test-1",
        sql="ALTER TABLE sample ADD COLUMN x TEXT",
        reversal_sql=None,
        description="add x",
    )
    assert first is True
    assert second is False

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 'v-test-1'"
        )
        n = (await cur.fetchone())[0]
    assert n == 1


@pytest.mark.asyncio
async def test_apply_migration_rolls_back_on_failure(tmp_path, monkeypatch):
    db_path, db_mod = await _fresh_db(tmp_path, monkeypatch)

    # Second statement references a column that doesn't exist on a fresh
    # sample table — the whole transaction (incl. the first ADD COLUMN and
    # the ledger insert) must roll back.
    bad_sql = (
        "ALTER TABLE sample ADD COLUMN y TEXT;"
        "ALTER TABLE sample ADD COLUMN y TEXT"  # duplicate -> error
    )
    with pytest.raises(Exception):
        await db_mod.apply_migration(
            version="v-test-bad",
            sql=bad_sql,
            reversal_sql=None,
            description="should fail",
        )

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 'v-test-bad'"
        )
        n_ledger = (await cur.fetchone())[0]
        cur = await db.execute("PRAGMA table_info(sample)")
        cols = [r[1] for r in await cur.fetchall()]
    assert n_ledger == 0
    assert "y" not in cols


@pytest.mark.asyncio
async def test_breadcrumb_absorption(tmp_path, monkeypatch):
    """init_db should absorb _migrations_pending.txt and delete it."""
    import os
    # Run from a tmp working dir so the breadcrumb path is isolated.
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "absorb.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)

    # Write a breadcrumb row that pretends T1A already applied.
    bc = tmp_path / "_migrations_pending.txt"
    bc.write_text(
        "z10_t1a_file_locks_expires_at\t"
        "ALTER TABLE file_locks ADD COLUMN expires_at TIMESTAMP\n",
        encoding="utf-8",
    )
    assert bc.exists()

    await db_mod.init_db()

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT version, description FROM schema_migrations "
            "WHERE version = 'z10_t1a_file_locks_expires_at'"
        )
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == "z10_t1a_file_locks_expires_at"
    assert "breadcrumb" in (row[1] or "").lower()
    assert not bc.exists(), "breadcrumb file must be deleted after absorb"
