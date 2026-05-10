"""Z10 T1A — file_locks orphan sweeper tests.

Cases covered:
1. Lock owned by a *failed* task is released by sweep_file_locks().
2. Lock with expired ``expires_at`` is released even if task still running.
3. Lock owned by a pending/running task with future TTL is *kept*.
"""
from __future__ import annotations

import aiosqlite
import pytest


_DDL = """
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE file_locks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL,
    mission_id INTEGER,
    task_id INTEGER,
    agent_type TEXT,
    acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(filepath)
);
"""


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "locks.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    # get_db() detects DB_PATH change and reopens; nothing else to reset.

    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_DDL)
        await db.commit()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_sweeper_releases_lock_owned_by_failed_task(tmp_path, monkeypatch):
    db_path, db_mod = await _setup(tmp_path, monkeypatch)

    async with aiosqlite.connect(db_path) as db:
        # Task 1 = failed; Task 2 = running.
        await db.execute("INSERT INTO tasks (id, status) VALUES (1, 'failed')")
        await db.execute("INSERT INTO tasks (id, status) VALUES (2, 'running')")
        # Both locks have a far-future expires_at — only the dead-task rule
        # should reap lock 1.
        await db.execute(
            "INSERT INTO file_locks (filepath, task_id, expires_at) "
            "VALUES (?, ?, datetime('now', '+1 hour'))",
            ("a.py", 1),
        )
        await db.execute(
            "INSERT INTO file_locks (filepath, task_id, expires_at) "
            "VALUES (?, ?, datetime('now', '+1 hour'))",
            ("b.py", 2),
        )
        await db.commit()

    released = await db_mod.sweep_file_locks()
    assert released == 1

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT filepath FROM file_locks ORDER BY filepath")
        rows = [r[0] for r in await cur.fetchall()]
    assert rows == ["b.py"], f"only running-task lock should remain, got {rows}"


@pytest.mark.asyncio
async def test_sweeper_releases_expired_lock(tmp_path, monkeypatch):
    db_path, db_mod = await _setup(tmp_path, monkeypatch)

    async with aiosqlite.connect(db_path) as db:
        await db.execute("INSERT INTO tasks (id, status) VALUES (1, 'running')")
        # expires_at is in the past — should be reaped despite task still
        # running. This catches the "lock holder never called release"
        # variant.
        await db.execute(
            "INSERT INTO file_locks (filepath, task_id, expires_at) "
            "VALUES (?, ?, datetime('now', '-5 minutes'))",
            ("c.py", 1),
        )
        await db.commit()

    released = await db_mod.sweep_file_locks()
    assert released == 1


@pytest.mark.asyncio
async def test_sweeper_keeps_healthy_lock(tmp_path, monkeypatch):
    db_path, db_mod = await _setup(tmp_path, monkeypatch)

    async with aiosqlite.connect(db_path) as db:
        await db.execute("INSERT INTO tasks (id, status) VALUES (1, 'running')")
        await db.execute(
            "INSERT INTO file_locks (filepath, task_id, expires_at) "
            "VALUES (?, ?, datetime('now', '+1 hour'))",
            ("d.py", 1),
        )
        await db.commit()

    released = await db_mod.sweep_file_locks()
    assert released == 0


@pytest.mark.asyncio
async def test_acquire_stamps_expires_at(tmp_path, monkeypatch):
    """Z10 T1A — every acquire must set expires_at so future sweeps work."""
    db_path, db_mod = await _setup(tmp_path, monkeypatch)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("INSERT INTO tasks (id, status) VALUES (1, 'running')")
        await db.commit()

    ok = await db_mod.acquire_file_lock("e.py", task_id=1, ttl_seconds=600)
    assert ok is True

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT expires_at FROM file_locks WHERE filepath='e.py'"
        )
        (expires_at,) = await cur.fetchone()
    assert expires_at is not None and expires_at != ""
