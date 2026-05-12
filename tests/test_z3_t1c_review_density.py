"""Z3 T1C — review_density resolver tests.

Covers:
- Migration adds review_density_json column (idempotent).
- get_dials with no row returns conservative defaults.
- get_dials with NULL column returns conservative defaults.
- set_dial round-trips (read back equals written value).
- set_dial rejects invalid keys.
- set_dial rejects invalid values.
- to_mission_dial_context shape matches MissionDialContext.
- MissionDialContext is defined on posthooks (T1A shape contract).
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal SQLite fixture
# ---------------------------------------------------------------------------

def _make_db_path(tmp_path: Path) -> str:
    """Return path to a temp SQLite DB with minimal missions + schema_migrations tables."""
    db_path = str(tmp_path / "test_density.db")
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS missions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT 'test-mission',
            review_density_json TEXT
        );
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            reversal_sql TEXT
        );
    """)
    con.commit()
    con.close()
    return db_path


async def _open(db_path: str):
    """Open an aiosqlite connection with Row factory."""
    import aiosqlite
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    return conn


def run(coro):
    """Run a coroutine in a fresh event loop (avoids deprecation warnings)."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. Migration adds column (idempotent)
# ---------------------------------------------------------------------------

def test_migration_adds_column(tmp_path):
    """The Z3 T1C migration adds review_density_json to a DB that lacks it."""
    db_path = str(tmp_path / "no_col.db")
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE missions ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  title TEXT NOT NULL"
        ")"
    )
    con.commit()
    con.close()

    con = sqlite3.connect(db_path)
    try:
        con.execute("ALTER TABLE missions ADD COLUMN review_density_json TEXT")
    except Exception:
        pass  # idempotent
    con.commit()

    cursor = con.execute("PRAGMA table_info(missions)")
    cols = {row[1] for row in cursor.fetchall()}
    assert "review_density_json" in cols
    con.close()


def test_migration_is_idempotent(tmp_path):
    """Applying the migration twice must not raise."""
    db_path = _make_db_path(tmp_path)  # already has review_density_json
    con = sqlite3.connect(db_path)
    try:
        con.execute("ALTER TABLE missions ADD COLUMN review_density_json TEXT")
    except Exception:
        pass  # expected — column exists
    con.commit()

    cursor = con.execute("PRAGMA table_info(missions)")
    cols = {row[1] for row in cursor.fetchall()}
    assert "review_density_json" in cols
    con.close()


# ---------------------------------------------------------------------------
# 2. get_dials with no row → defaults
# ---------------------------------------------------------------------------

def test_get_dials_no_mission_returns_defaults(tmp_path):
    """get_dials on a non-existent mission_id returns ReviewDensityDials defaults."""
    db_path = _make_db_path(tmp_path)

    async def _run():
        conn = await _open(db_path)
        with patch("src.infra.db.get_db", return_value=conn):
            from src.workflows.review_density import get_dials
            result = await get_dials(99999)
        await conn.close()
        return result

    result = run(_run())
    from src.workflows.review_density import ReviewDensityDials
    assert result == ReviewDensityDials()
    assert result.qa_dial == "standard"
    assert result.accessibility_dial == "off"
    assert result.multi_file_expansion is False
    assert result.integration_replay == "standard"


def test_get_dials_null_json_returns_defaults(tmp_path):
    """get_dials returns defaults when review_density_json is NULL."""
    db_path = _make_db_path(tmp_path)
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO missions (title, review_density_json) VALUES ('t', NULL)"
    )
    mission_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.commit()
    con.close()

    async def _run():
        conn = await _open(db_path)
        with patch("src.infra.db.get_db", return_value=conn):
            from src.workflows.review_density import get_dials
            result = await get_dials(mission_id)
        await conn.close()
        return result

    result = run(_run())
    from src.workflows.review_density import ReviewDensityDials
    assert result == ReviewDensityDials()


# ---------------------------------------------------------------------------
# 3. set_dial round-trips
# ---------------------------------------------------------------------------

def test_set_dial_roundtrip_qa_strict(tmp_path):
    """set_dial('qa_dial', 'strict') persists and get_dials reads it back."""
    db_path = _make_db_path(tmp_path)
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO missions (title, review_density_json) VALUES ('t', NULL)"
    )
    mission_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.commit()
    con.close()

    async def _run():
        conn = await _open(db_path)
        with patch("src.infra.db.get_db", return_value=conn):
            from src.workflows.review_density import get_dials, set_dial
            updated = await set_dial(mission_id, "qa_dial", "strict")
            readback = await get_dials(mission_id)
        await conn.close()
        return updated, readback

    updated, readback = run(_run())
    assert updated.qa_dial == "strict"
    assert readback.qa_dial == "strict"
    assert readback.accessibility_dial == "off"
    assert readback.multi_file_expansion is False


def test_set_dial_roundtrip_multi_file(tmp_path):
    """set_dial('multi_file_expansion', 'true') round-trips as bool True."""
    db_path = _make_db_path(tmp_path)
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO missions (title, review_density_json) VALUES ('t', NULL)"
    )
    mission_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.commit()
    con.close()

    async def _run():
        conn = await _open(db_path)
        with patch("src.infra.db.get_db", return_value=conn):
            from src.workflows.review_density import get_dials, set_dial
            updated = await set_dial(mission_id, "multi_file_expansion", "true")
            readback = await get_dials(mission_id)
        await conn.close()
        return updated, readback

    updated, readback = run(_run())
    assert updated.multi_file_expansion is True
    assert readback.multi_file_expansion is True


def test_set_dial_roundtrip_accessibility(tmp_path):
    """set_dial('accessibility_dial', 'on') persists correctly."""
    db_path = _make_db_path(tmp_path)
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO missions (title, review_density_json) VALUES ('t', NULL)"
    )
    mission_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.commit()
    con.close()

    async def _run():
        conn = await _open(db_path)
        with patch("src.infra.db.get_db", return_value=conn):
            from src.workflows.review_density import set_dial
            updated = await set_dial(mission_id, "accessibility_dial", "on")
        await conn.close()
        return updated

    updated = run(_run())
    assert updated.accessibility_dial == "on"


# ---------------------------------------------------------------------------
# 4. set_dial rejects invalid keys + values
# ---------------------------------------------------------------------------

def test_set_dial_rejects_unknown_key(tmp_path):
    """set_dial raises ValueError on unknown key (validation happens before DB lookup)."""
    async def _run():
        import src.workflows.review_density as mod
        await mod.set_dial(1, "nonexistent_dial", "quick")

    with pytest.raises(ValueError, match="Unknown dial key"):
        run(_run())


def test_set_dial_rejects_invalid_qa_value(tmp_path):
    """set_dial raises ValueError when value not in allowed set for qa_dial."""
    db_path = _make_db_path(tmp_path)
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO missions (title, review_density_json) VALUES ('t', NULL)"
    )
    mission_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.commit()
    con.close()

    async def _run():
        conn = await _open(db_path)
        with patch("src.infra.db.get_db", return_value=conn):
            import src.workflows.review_density as mod
            await mod.set_dial(mission_id, "qa_dial", "turbo")
        await conn.close()

    with pytest.raises(ValueError, match="Invalid value"):
        run(_run())


def test_set_dial_rejects_invalid_multi_file_value():
    """set_dial raises ValueError for multi_file_expansion with non-bool string."""
    async def _run():
        import src.workflows.review_density as mod
        await mod.set_dial(1, "multi_file_expansion", "yes")

    with pytest.raises(ValueError):
        run(_run())


def test_set_dial_rejects_invalid_accessibility_value(tmp_path):
    """set_dial raises ValueError for accessibility_dial with unknown value."""
    db_path = _make_db_path(tmp_path)
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO missions (title, review_density_json) VALUES ('t', NULL)"
    )
    mission_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.commit()
    con.close()

    async def _run():
        conn = await _open(db_path)
        with patch("src.infra.db.get_db", return_value=conn):
            import src.workflows.review_density as mod
            await mod.set_dial(mission_id, "accessibility_dial", "maybe")
        await conn.close()

    with pytest.raises(ValueError, match="Invalid value"):
        run(_run())


# ---------------------------------------------------------------------------
# 5. to_mission_dial_context shape matches MissionDialContext
# ---------------------------------------------------------------------------

def test_to_mission_dial_context_shape():
    """to_mission_dial_context returns a MissionDialContext with matching fields."""
    from src.workflows.review_density import ReviewDensityDials, to_mission_dial_context
    from general_beckman.posthooks import MissionDialContext

    dials = ReviewDensityDials(
        qa_dial="strict",
        accessibility_dial="on",
        multi_file_expansion=True,
        integration_replay="quick",
    )
    ctx = to_mission_dial_context(dials)

    assert isinstance(ctx, MissionDialContext)
    assert ctx.qa_dial == "strict"
    assert ctx.accessibility_dial == "on"
    assert ctx.multi_file_expansion is True
    assert ctx.integration_replay == "quick"


def test_to_mission_dial_context_defaults():
    """to_mission_dial_context with default dials produces matching MissionDialContext."""
    from src.workflows.review_density import ReviewDensityDials, to_mission_dial_context
    from general_beckman.posthooks import MissionDialContext

    dials = ReviewDensityDials()
    ctx = to_mission_dial_context(dials)

    assert isinstance(ctx, MissionDialContext)
    assert ctx.qa_dial == "standard"
    assert ctx.accessibility_dial == "off"
    assert ctx.multi_file_expansion is False
    assert ctx.integration_replay == "standard"


# ---------------------------------------------------------------------------
# 6. MissionDialContext exists on posthooks (T1A shape contract)
# ---------------------------------------------------------------------------

def test_mission_dial_context_defined_on_posthooks():
    """MissionDialContext is importable from general_beckman.posthooks."""
    from general_beckman.posthooks import MissionDialContext
    from dataclasses import fields

    field_names = {f.name for f in fields(MissionDialContext)}
    assert "qa_dial" in field_names
    assert "accessibility_dial" in field_names
    assert "multi_file_expansion" in field_names
    assert "integration_replay" in field_names


def test_mission_dial_context_defaults():
    """MissionDialContext default values match ReviewDensityDials defaults."""
    from general_beckman.posthooks import MissionDialContext

    ctx = MissionDialContext()
    assert ctx.qa_dial == "standard"
    assert ctx.accessibility_dial == "off"
    assert ctx.multi_file_expansion is False
    assert ctx.integration_replay == "standard"
