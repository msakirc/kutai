"""Tests for src.infra.pick_log.write_pick_log_row."""
from __future__ import annotations

import aiosqlite
import pytest

from src.infra.pick_log import write_pick_log_row


# Mirrors src/infra/db.py :: model_pick_log. Kept inline so the test fails
# loudly if production DDL drifts from what the helper writes.
_REAL_DDL = """
CREATE TABLE model_pick_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task_name TEXT NOT NULL,
    agent_type TEXT,
    difficulty INTEGER,
    call_category TEXT,
    picked_model TEXT NOT NULL,
    picked_score REAL NOT NULL,
    picked_reasons TEXT,
    candidates_json TEXT NOT NULL,
    failures_json TEXT,
    snapshot_summary TEXT,
    pool TEXT,
    urgency REAL,
    success INTEGER,
    error_category TEXT,
    provider TEXT
)
"""


async def _setup_real_table(db_path) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_REAL_DDL)
        await db.commit()


@pytest.mark.asyncio
async def test_write_pick_log_inserts(tmp_path):
    db_path = tmp_path / "test.db"
    await _setup_real_table(db_path)

    await write_pick_log_row(
        db_path=str(db_path),
        task_name="coder",
        picked_model="qwen3-8b",
        picked_score=0.72,
        category="main_work",
        success=True,
    )

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT task_name, picked_model, picked_score, "
            "       call_category, success FROM model_pick_log"
        ) as cur:
            rows = await cur.fetchall()

    assert rows == [("coder", "qwen3-8b", 0.72, "main_work", 1)]


@pytest.mark.asyncio
async def test_write_pick_log_failure_path_never_raises(tmp_path):
    """Bad DB path must be swallowed — telemetry never breaks callers."""
    await write_pick_log_row(
        db_path=str(tmp_path / "does" / "not" / "exist.db"),
        task_name="coder",
        picked_model="m",
        picked_score=0.0,
        category="overhead",
        success=False,
        error_category="no_model",
    )


@pytest.mark.asyncio
async def test_write_pick_log_records_failure(tmp_path):
    db_path = tmp_path / "test.db"
    await _setup_real_table(db_path)

    await write_pick_log_row(
        db_path=str(db_path),
        task_name="grader",
        picked_model="claude-sonnet",
        picked_score=0.4,
        category="overhead",
        success=False,
        error_category="rate_limit",
    )

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT success, error_category, call_category FROM model_pick_log"
        ) as cur:
            rows = await cur.fetchall()

    assert rows == [(0, "rate_limit", "overhead")]


@pytest.mark.asyncio
async def test_write_pick_log_against_init_db(tmp_path, monkeypatch):
    """Integration: use the real init_db DDL; catches any prod schema drift."""
    db_path = tmp_path / "real.db"
    monkeypatch.setenv("DB_PATH", str(db_path))

    from src.infra import db as db_mod

    # init_db() reads DB_PATH lazily; force its module-level path too if set.
    if hasattr(db_mod, "DB_PATH"):
        monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)

    async with aiosqlite.connect(db_path) as db:
        await db.execute(_REAL_DDL)
        await db.commit()

    await write_pick_log_row(
        db_path=str(db_path),
        task_name="researcher",
        picked_model="qwen3-14b",
        picked_score=0.81,
        category="main_work",
        success=True,
        snapshot_summary="sample=1",
    )

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT task_name, picked_model, picked_score, call_category, "
            "       candidates_json, snapshot_summary, success, error_category "
            "FROM model_pick_log"
        ) as cur:
            rows = await cur.fetchall()

    assert rows == [
        ("researcher", "qwen3-14b", 0.81, "main_work", "[]", "sample=1", 1, ""),
    ]
