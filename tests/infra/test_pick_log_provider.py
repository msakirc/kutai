"""Schema + writer must populate model_pick_log.provider correctly."""
from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from src.infra.pick_log import write_pick_log_row

# ── helpers ──────────────────────────────────────────────────────────────────

_DDL_FULL = """
CREATE TABLE IF NOT EXISTS model_pick_log (
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

_DDL_LEGACY = """
CREATE TABLE model_pick_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task_name TEXT NOT NULL,
    picked_model TEXT NOT NULL,
    picked_score REAL NOT NULL,
    call_category TEXT,
    candidates_json TEXT NOT NULL DEFAULT '[]',
    snapshot_summary TEXT,
    success INTEGER,
    error_category TEXT
)
"""

# Simulates what init_db does: ALTER + backfill.  We don't call init_db()
# directly because it uses a module-level DB singleton (DB_PATH from config)
# and takes no arguments.
async def _apply_migration(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        for col_name, col_type in (
            ("pool", "TEXT"),
            ("urgency", "REAL"),
            ("success", "INTEGER"),
            ("error_category", "TEXT"),
            ("provider", "TEXT"),
        ):
            try:
                await db.execute(
                    f"ALTER TABLE model_pick_log ADD COLUMN {col_name} {col_type}"
                )
            except Exception as e:
                if "duplicate column" not in str(e).lower():
                    raise
        await db.execute(
            "UPDATE model_pick_log SET provider='local' WHERE provider IS NULL"
        )
        await db.commit()


# ── tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_column_exists_after_init(tmp_path: Path):
    """Fresh DB created with full DDL (as init_db does) has provider column."""
    db_path = str(tmp_path / "k.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_DDL_FULL)
        await db.commit()
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(model_pick_log)")
        cols = [row[1] for row in await cur.fetchall()]
    assert "provider" in cols


@pytest.mark.asyncio
async def test_legacy_rows_backfilled_to_local(tmp_path: Path):
    """Pre-migration DB had no provider column. After ALTER + backfill,
    existing NULL rows must be set to 'local'."""
    db_path = str(tmp_path / "k.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_DDL_LEGACY)
        await db.execute(
            "INSERT INTO model_pick_log "
            "(task_name, picked_model, picked_score, call_category) "
            "VALUES ('t1', 'qwen-30b', 0.5, 'main_work')"
        )
        await db.commit()
    await _apply_migration(db_path)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT provider FROM model_pick_log WHERE task_name='t1'")
        row = await cur.fetchone()
    assert row[0] == "local"


@pytest.mark.asyncio
async def test_writer_persists_provider(tmp_path: Path):
    db_path = str(tmp_path / "k.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_DDL_FULL)
        await db.commit()
    await write_pick_log_row(
        db_path=db_path,
        task_name="t",
        picked_model="groq/llama-3.3-70b-versatile",
        picked_score=0.9,
        category="main_work",
        success=True,
        provider="groq",
    )
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT provider FROM model_pick_log WHERE task_name='t'")
        row = await cur.fetchone()
    assert row[0] == "groq"


@pytest.mark.asyncio
async def test_writer_default_provider_is_local(tmp_path: Path):
    """When provider arg omitted, default to 'local' for backward compat."""
    db_path = str(tmp_path / "k.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_DDL_FULL)
        await db.commit()
    await write_pick_log_row(
        db_path=db_path,
        task_name="t2",
        picked_model="qwen",
        picked_score=0.5,
        category="main_work",
        success=True,
    )
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT provider FROM model_pick_log WHERE task_name='t2'")
        row = await cur.fetchone()
    assert row[0] == "local"
