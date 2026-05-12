"""Z8 T1A — mission lifecycle column migration tests."""
from __future__ import annotations

import sqlite3

import pytest


async def _setup_fresh(tmp_path, monkeypatch):
    db_path = tmp_path / "lifecycle.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    # Reset connection singleton so init_db opens the new path.
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_missions_has_lifecycle_columns(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cur = await db.execute("PRAGMA table_info(missions)")
    cols = {row[1] for row in await cur.fetchall()}
    assert "kind" in cols
    assert "lifecycle_state" in cols
    assert "cursor" in cols
    assert "product_id" in cols
    assert "revoked_at" in cols


@pytest.mark.asyncio
async def test_existing_missions_backfilled(tmp_path, monkeypatch):
    db_path = tmp_path / "backfill.db"
    # Simulate pre-migration DB with two mission rows (no lifecycle cols).
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE missions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "title TEXT, description TEXT, status TEXT DEFAULT 'pending', "
        "priority INTEGER DEFAULT 5, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute("INSERT INTO missions (title, description) VALUES ('m1','d1')")
    conn.execute("INSERT INTO missions (title, description) VALUES ('m2','d2')")
    conn.commit()
    conn.close()

    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()

    db = await db_mod.get_db()
    cur = await db.execute("SELECT kind, lifecycle_state FROM missions")
    rows = await cur.fetchall()
    assert len(rows) == 2
    assert all(r[0] == "oneshot" for r in rows)
    assert all(r[1] == "terminal" for r in rows)


@pytest.mark.asyncio
async def test_lifecycle_index_exists(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name='idx_missions_kind_state'"
    )
    row = await cur.fetchone()
    assert row is not None
