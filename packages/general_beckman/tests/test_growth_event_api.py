"""Tests for beckman.record_growth_event, beckman.supersede_growth_event, and
beckman.update_growth_event_properties.

Verifies:
  1. record_growth_event inserts a row and returns its id.
  2. supersede_growth_event marks open rows superseded, skips consumed/already-superseded.
  3. update_growth_event_properties overwrites stored properties_json.
  4. Guard: no module outside src/infra/db.py and general_beckman references
     insert_growth_event (all callers must use the beckman API).
"""
from __future__ import annotations

import pytest
import aiosqlite


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _reset_db(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None
    return db_path


async def _close_db(db_mod) -> None:
    """Close and reset the shared DB connection to avoid cross-test leaks."""
    if db_mod._db_connection is not None:
        await db_mod._db_connection.close()
        db_mod._db_connection = None


async def _fetch_events(db_path: str, kind: str) -> list[dict]:
    """Direct aiosqlite read so tests stay independent of beckman internals."""
    import json
    rows = []
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM growth_events WHERE kind = ? ORDER BY id",
            (kind,),
        )
        for r in await cur.fetchall():
            d = dict(r)
            try:
                d["properties"] = json.loads(d.get("properties_json") or "{}")
            except Exception:
                d["properties"] = {}
            rows.append(d)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_growth_event_inserts_row(tmp_path, monkeypatch):
    """record_growth_event returns a positive int id and persists the row."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import record_growth_event
        event_id = await record_growth_event(
            mission_id=None,
            kind="test_kind",
            properties={"foo": "bar"},
        )

        assert isinstance(event_id, int)
        assert event_id > 0

        # Verify the row landed in the DB.
        rows = await _fetch_events(db_path, "test_kind")
        assert len(rows) == 1
        assert rows[0]["properties"]["foo"] == "bar"
        assert rows[0]["id"] == event_id
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_record_growth_event_with_segment(tmp_path, monkeypatch):
    """segment column is persisted when provided."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import record_growth_event
        await record_growth_event(None, "seg_test", {"x": 1}, segment="cohort_a")

        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute("SELECT segment FROM growth_events WHERE kind='seg_test'")
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == "cohort_a"
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_supersede_growth_event_marks_open_rows(tmp_path, monkeypatch):
    """supersede_growth_event flips superseded=True on all open rows."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import record_growth_event, supersede_growth_event

        # Insert two open rows.
        await record_growth_event(1, "backlog_candidate", {"score": 0.9})
        await record_growth_event(1, "backlog_candidate", {"score": 0.7})

        db_module._db_connection = None

        count = await supersede_growth_event(mission_id=1, kind="backlog_candidate")
        assert count == 2

        rows = await _fetch_events(db_path, "backlog_candidate")
        assert len(rows) == 2
        for r in rows:
            assert r["properties"].get("superseded") is True
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_supersede_skips_consumed_and_already_superseded(tmp_path, monkeypatch):
    """Rows with consumed=True or superseded=True are left untouched."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import record_growth_event, supersede_growth_event

        # consumed row
        await record_growth_event(1, "northstar_review", {"consumed": True, "score": 1})
        # already-superseded row
        await record_growth_event(1, "northstar_review", {"superseded": True, "score": 2})
        # open row — should be superseded
        open_id = await record_growth_event(1, "northstar_review", {"score": 3})

        db_module._db_connection = None

        count = await supersede_growth_event(mission_id=1, kind="northstar_review")
        assert count == 1  # only the one open row

        rows = await _fetch_events(db_path, "northstar_review")
        for r in rows:
            if r["id"] == open_id:
                assert r["properties"].get("superseded") is True
            elif r["properties"].get("consumed"):
                # consumed row must not have been mutated to also carry superseded
                assert not r["properties"].get("superseded")
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_supersede_returns_zero_when_all_already_closed(tmp_path, monkeypatch):
    """Returns 0 when there are no open rows to supersede."""
    _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import record_growth_event, supersede_growth_event

        await record_growth_event(1, "sunset_candidate", {"superseded": True})
        db_module._db_connection = None

        count = await supersede_growth_event(mission_id=1, kind="sunset_candidate")
        assert count == 0
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_supersede_handles_null_properties_json(tmp_path, monkeypatch):
    """supersede_growth_event handles rows with NULL properties_json (no crash, marks superseded)."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import supersede_growth_event

        # Insert a row with NULL properties_json directly (simulates legacy rows).
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute(
                "INSERT INTO growth_events (mission_id, kind, properties_json) VALUES (?, ?, NULL)",
                (42, "null_props_kind"),
            )
            event_id = cur.lastrowid
            await db.commit()

        db_module._db_connection = None
        count = await supersede_growth_event(mission_id=42, kind="null_props_kind")
        assert count == 1

        # Verify the row now has superseded=True in properties_json.
        rows = await _fetch_events(db_path, "null_props_kind")
        assert len(rows) == 1
        assert rows[0]["properties"].get("superseded") is True
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_update_growth_event_properties_overwrites(tmp_path, monkeypatch):
    """update_growth_event_properties replaces stored properties_json in the DB."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import record_growth_event, update_growth_event_properties

        event_id = await record_growth_event(
            mission_id=None,
            kind="prop_update_test",
            properties={"original": True, "value": 1},
        )

        await update_growth_event_properties(event_id, {"replaced": True, "value": 99})

        rows = await _fetch_events(db_path, "prop_update_test")
        assert len(rows) == 1
        props = rows[0]["properties"]
        assert props.get("replaced") is True
        assert props.get("value") == 99
        # old key must not survive the overwrite
        assert "original" not in props
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# Guard test: insert_growth_event must have zero callers outside the
# sanctioned modules (src/infra/db.py and general_beckman package).
# ──────────────────────────────────────────────────────────────────────────────


def test_no_raw_insert_growth_event_callers():
    """No source file outside src/infra/db.py and general_beckman may call
    insert_growth_event directly.

    After the migration all former call sites import record_growth_event from
    general_beckman instead.  This test is the static guard.
    """
    import re
    import os
    from pathlib import Path

    root = Path(__file__).parents[3]  # repo root (worktree)

    # Pattern: any line that contains 'insert_growth_event' but is NOT
    # a pure function definition (def insert_growth_event) — i.e., it's
    # a call or import.
    call_re = re.compile(r"insert_growth_event")
    defn_re = re.compile(r"^\s*(async\s+)?def\s+insert_growth_event")

    # Allowed files (the definition site + the beckman delegate).
    allowed = {
        (root / "src" / "infra" / "db.py").resolve(),
        (root / "packages" / "general_beckman" / "src" / "general_beckman" / "__init__.py").resolve(),
    }

    # Also allow this test file itself (contains the string in comments/strings).
    allowed.add(Path(__file__).resolve())

    violations: list[str] = []
    skip_dirs = {".venv", "__pycache__", ".git", ".benchmark_cache", "node_modules", "worktrees"}

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in-place.
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            # Test files may call insert_growth_event for DB setup.
            if fname.startswith("test_") or fname.endswith("_test.py"):
                continue
            if "tests" in Path(dirpath).parts:
                continue
            filepath = (Path(dirpath) / fname).resolve()
            if filepath in allowed:
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if call_re.search(line) and not defn_re.match(line):
                    rel = filepath.relative_to(root.resolve())
                    violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert violations == [], (
        "insert_growth_event called outside sanctioned modules — "
        "use general_beckman.record_growth_event instead:\n"
        + "\n".join(violations)
    )
