"""Tests for beckman mission write API:
  beckman.add_mission, update_mission, update_mission_fields,
  block_mission, unblock_mission, purge_all_missions, purge_all,
  increment_mission_rework_loops.

Guard test: no INSERT INTO missions / UPDATE missions / DELETE FROM missions
SQL outside src/infra/db.py; no db.add_mission / db.update_mission imports
outside src/infra + general_beckman (tests exempt).
"""
from __future__ import annotations

import pytest
import aiosqlite


# ──────────────────────────────────────────────────────────────────────────────
# File-local helpers (direct DB reads for verification; NOT shared DB setup)
# DB setup is handled by the fresh_db fixture in conftest.py.
# ──────────────────────────────────────────────────────────────────────────────


async def _fetch_mission(db_path: str, mission_id: int) -> dict | None:
    """Direct aiosqlite read for test verification."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM missions WHERE id = ?", (mission_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def _fetch_all_missions(db_path: str) -> list[dict]:
    """Return all missions rows."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM missions")
        return [dict(r) for r in await cur.fetchall()]


async def _fetch_all_tasks(db_path: str) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM tasks")
        return [dict(r) for r in await cur.fetchall()]


# ──────────────────────────────────────────────────────────────────────────────
# add_mission
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_mission_returns_id_and_persists(fresh_db):
    """add_mission returns a positive int id and the row lands in DB."""
    db_path = fresh_db
    from general_beckman import add_mission
    mid = await add_mission(title="Test Mission", description="test desc")

    assert isinstance(mid, int)
    assert mid > 0

    row = await _fetch_mission(db_path, mid)
    assert row is not None
    assert row["title"] == "Test Mission"
    assert row["description"] == "test desc"


@pytest.mark.asyncio
async def test_add_mission_with_optional_fields(fresh_db):
    """Optional params are stored correctly."""
    db_path = fresh_db
    from general_beckman import add_mission
    mid = await add_mission(
        title="Opt Mission",
        description="opt",
        priority=3,
        workflow="i2p",
        repo_path="/tmp/repo",
    )
    row = await _fetch_mission(db_path, mid)
    assert row["priority"] == 3
    assert row["workflow"] == "i2p"
    assert row["repo_path"] == "/tmp/repo"


# ──────────────────────────────────────────────────────────────────────────────
# update_mission
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_mission_modifies_row(fresh_db):
    """update_mission persists changes to whitelisted columns."""
    db_path = fresh_db
    from general_beckman import add_mission, update_mission
    mid = await add_mission(title="Before", description="d")

    await update_mission(mid, title="After", description="updated")

    row = await _fetch_mission(db_path, mid)
    assert row["title"] == "After"
    assert row["description"] == "updated"


@pytest.mark.asyncio
async def test_update_mission_rejects_unknown_column(fresh_db):
    """update_mission raises ValueError for columns not in the whitelist."""
    from general_beckman import add_mission, update_mission
    mid = await add_mission(title="T", description="d")

    with pytest.raises(ValueError, match="missions"):
        await update_mission(mid, nonexistent_col="bad")


# ──────────────────────────────────────────────────────────────────────────────
# update_mission_fields
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_mission_fields_accepted_column(fresh_db):
    """update_mission_fields writes a whitelisted column to the DB."""
    db_path = fresh_db
    from general_beckman import add_mission, update_mission_fields
    mid = await add_mission(title="T", description="d")

    await update_mission_fields(mid, founder_attention_budget_minutes=90)

    row = await _fetch_mission(db_path, mid)
    assert row["founder_attention_budget_minutes"] == 90


@pytest.mark.asyncio
async def test_update_mission_fields_rejects_unknown_column(fresh_db):
    """update_mission_fields raises ValueError for unknown columns."""
    from general_beckman import add_mission, update_mission_fields
    mid = await add_mission(title="T", description="d")

    with pytest.raises(ValueError, match="unknown column"):
        await update_mission_fields(mid, totally_bogus_col="value")


@pytest.mark.asyncio
async def test_update_mission_fields_noop_on_empty(fresh_db):
    """Calling with no fields is a no-op (no error)."""
    from general_beckman import add_mission, update_mission_fields
    mid = await add_mission(title="T", description="d")
    # Should not raise
    await update_mission_fields(mid)


@pytest.mark.asyncio
async def test_update_mission_fields_multiple_columns(fresh_db):
    """Multiple whitelisted columns can be updated in one call."""
    db_path = fresh_db
    from general_beckman import add_mission, update_mission_fields
    mid = await add_mission(title="T", description="d")

    await update_mission_fields(
        mid,
        telegram_thread_id=42,
        review_density_json='{"density": 0.5}',
    )

    row = await _fetch_mission(db_path, mid)
    assert row["telegram_thread_id"] == 42
    assert row["review_density_json"] == '{"density": 0.5}'


# ──────────────────────────────────────────────────────────────────────────────
# block_mission / unblock_mission
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_block_mission_sets_blocked_state(fresh_db):
    """block_mission flips lifecycle_state to blocked_on_founder_action."""
    db_path = fresh_db
    from general_beckman import add_mission, block_mission
    mid = await add_mission(title="T", description="d")

    result = await block_mission(mid)
    assert result is True

    row = await _fetch_mission(db_path, mid)
    state = row.get("lifecycle_state") or row.get("status")
    assert state == "blocked_on_founder_action"


@pytest.mark.asyncio
async def test_block_mission_noop_if_already_blocked(fresh_db):
    """block_mission returns False if mission is already blocked."""
    from general_beckman import add_mission, block_mission
    mid = await add_mission(title="T", description="d")

    await block_mission(mid)
    result = await block_mission(mid)
    assert result is False


@pytest.mark.asyncio
async def test_unblock_mission_restores_active_and_resets_tasks(fresh_db):
    """unblock_mission flips mission to active and resets blocked tasks to pending."""
    db_path = fresh_db
    import src.infra.db as db_module
    from general_beckman import add_mission, block_mission, unblock_mission

    mid = await add_mission(title="T", description="d")

    # Insert a task that is blocked_on_founder_action for this mission.
    db = await db_module.get_db()
    await db.execute(
        "INSERT INTO tasks (mission_id, title, status, agent_type) "
        "VALUES (?, 'task1', 'blocked_on_founder_action', 'coder')",
        (mid,),
    )
    await db.commit()

    # Block first.
    await block_mission(mid)

    # Unblock.
    result = await unblock_mission(mid)
    assert result is True

    # Mission is active again.
    row = await _fetch_mission(db_path, mid)
    state = row.get("lifecycle_state") or row.get("status")
    assert state == "active"

    # The blocked task was reset to pending.
    tasks = await _fetch_all_tasks(db_path)
    assert all(t["status"] == "pending" for t in tasks if t["mission_id"] == mid)


@pytest.mark.asyncio
async def test_unblock_mission_noop_if_not_blocked(fresh_db):
    """unblock_mission returns False if mission is not blocked."""
    from general_beckman import add_mission, unblock_mission
    mid = await add_mission(title="T", description="d")

    result = await unblock_mission(mid)
    assert result is False


@pytest.mark.asyncio
async def test_block_unblock_mission_missing_row(fresh_db):
    """block/unblock return False for non-existent mission id."""
    from general_beckman import block_mission, unblock_mission
    assert await block_mission(99999) is False
    assert await unblock_mission(99999) is False


# ──────────────────────────────────────────────────────────────────────────────
# purge_all_missions / purge_all
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_purge_all_missions_clears_missions_and_tasks(fresh_db):
    """purge_all_missions deletes missions and dependent rows."""
    db_path = fresh_db
    import src.infra.db as db_module
    from general_beckman import add_mission, purge_all_missions

    mid1 = await add_mission(title="M1", description="d")
    mid2 = await add_mission(title="M2", description="d")

    # Insert a task for one of the missions.
    db = await db_module.get_db()
    await db.execute(
        "INSERT INTO tasks (mission_id, title, status, agent_type) "
        "VALUES (?, 'task', 'pending', 'coder')",
        (mid1,),
    )
    await db.commit()

    await purge_all_missions()

    assert await _fetch_all_missions(db_path) == []
    assert await _fetch_all_tasks(db_path) == []


@pytest.mark.asyncio
async def test_purge_all_clears_missions_and_conversations(fresh_db):
    """purge_all wipes missions, tasks, conversations, and memory."""
    db_path = fresh_db
    import src.infra.db as db_module
    from general_beckman import add_mission, purge_all

    await add_mission(title="M", description="d")

    # Seed a conversations row (if table exists).
    db = await db_module.get_db()
    try:
        await db.execute(
            "INSERT INTO conversations (chat_id, message, role) VALUES (1, 'hi', 'user')"
        )
        await db.commit()
    except Exception:
        pass  # table may not exist in minimal test schema

    await purge_all()

    assert await _fetch_all_missions(db_path) == []
    assert await _fetch_all_tasks(db_path) == []


# ──────────────────────────────────────────────────────────────────────────────
# increment_mission_rework_loops
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_increment_mission_rework_loops(fresh_db):
    """increment_mission_rework_loops atomically bumps counter and returns new value."""
    db_path = fresh_db
    from general_beckman import add_mission, increment_mission_rework_loops

    mid = await add_mission(title="T", description="d")

    count1 = await increment_mission_rework_loops(mid)
    count2 = await increment_mission_rework_loops(mid)

    assert count1 == 1
    assert count2 == 2

    row = await _fetch_mission(db_path, mid)
    assert row["phase_7_rework_loops"] == 2


@pytest.mark.asyncio
async def test_increment_mission_rework_loops_returns_zero_missing(fresh_db):
    """Returns 0 for a non-existent mission id (defensive, telemetry must not crash)."""
    from general_beckman import increment_mission_rework_loops
    count = await increment_mission_rework_loops(99999)
    assert count == 0


# ──────────────────────────────────────────────────────────────────────────────
# Guard test: no raw SQL against missions outside src/infra/db.py
# and no db.add_mission / db.update_mission outside src/infra + general_beckman
# ──────────────────────────────────────────────────────────────────────────────


def test_no_raw_missions_sql_outside_db(repo_source_texts):
    """No source file outside src/infra/db.py may contain raw
    INSERT INTO missions, UPDATE missions, or DELETE FROM missions SQL.

    After migration all former raw-SQL sites call beckman APIs instead.
    src/infra/db.py is the sole SQL owner.
    """
    import re
    from pathlib import Path

    root = Path(__file__).parents[3].resolve()

    sql_re = re.compile(
        r'(INSERT\s+INTO\s+missions|UPDATE\s+missions\s+SET|DELETE\s+FROM\s+missions)',
        re.IGNORECASE,
    )

    # Allowed: src/infra/db.py (SQL owner) and all of general_beckman/src/
    # (beckman is the write-owner — its internal modules may write missions SQL).
    allowed = {
        (root / "src" / "infra" / "db.py").resolve(),
    }
    allowed_dirs = {
        (root / "packages" / "general_beckman" / "src" / "general_beckman").resolve(),
    }
    # Also allow this test file and any test helpers that reference these strings.
    allowed.add(Path(__file__).resolve())

    violations: list[str] = []

    for filepath, text in repo_source_texts.items():
        if filepath in allowed:
            continue
        # Allow all of general_beckman/src/general_beckman/ — beckman is the write-owner.
        if any(str(filepath).startswith(str(d)) for d in allowed_dirs):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if sql_re.search(line):
                rel = filepath.relative_to(root)
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert violations == [], (
        "Raw missions SQL found outside src/infra/db.py and general_beckman — "
        "use general_beckman mission write API instead:\n"
        + "\n".join(violations)
    )


def test_no_raw_db_mission_imports_outside_infra_beckman(repo_source_texts, ast_db_write_imports_fn):
    """No source file outside src/infra + general_beckman may import
    db.add_mission, db.update_mission, db.update_mission_fields,
    db.increment_mission_rework_loops, db.purge_all_missions, or db.purge_all
    directly.

    After migration all callers use general_beckman's API.  Banning
    ``update_mission_fields`` here closes the latent gap: callers must go
    through ``beckman.update_mission_fields`` (which itself delegates to
    ``src.infra.db.update_mission_fields``), not bypass beckman directly.

    Detection uses ast.parse so parenthesised multi-line imports
    (e.g. ``from src.infra.db import (\\n  add_mission,\\n)``) are caught.
    Falls back to line-regex on SyntaxError.  Call-site patterns (db.add_mission)
    remain line-regex (they are always single-line tokens).
    """
    import re
    from pathlib import Path

    root = Path(__file__).parents[3].resolve()

    guarded_names = frozenset({
        "add_mission", "update_mission", "update_mission_fields",
        "increment_mission_rework_loops", "purge_all_missions", "purge_all",
    })

    # Fallback line-regex for import statements in unparseable files.
    _names = "|".join(sorted(guarded_names))
    import_re = re.compile(
        rf'from\s+src\.infra\.db\s+import\s+.*?\b({_names})\b',
    )
    # Also catch: import src.infra.db; db.add_mission( / db.update_mission_fields(
    # These are always single-line, so line-regex is correct here.
    call_re = re.compile(
        r'\b(?:db|infra\.db)\.(add_mission|update_mission'
        r'|increment_mission_rework_loops|purge_all_missions|purge_all\b)'
    )

    allowed_dirs = {
        (root / "src" / "infra").resolve(),
        (root / "packages" / "general_beckman" / "src" / "general_beckman").resolve(),
    }
    allowed_files = {Path(__file__).resolve()}

    violations: list[str] = []

    for filepath, text in repo_source_texts.items():
        if filepath in allowed_files:
            continue
        if any(
            str(filepath).startswith(str(d)) for d in allowed_dirs
        ):
            continue

        # AST-based import detection (catches multi-line parenthesised imports).
        ast_hits = ast_db_write_imports_fn(filepath, text, guarded_names)
        if ast_hits:
            rel = filepath.relative_to(root)
            for lineno, name in ast_hits:
                violations.append(f"{rel}:{lineno}: import of '{name}' from src.infra.db")
            # Still run call_re for this file (call patterns are line-level).
            for lineno, line in enumerate(text.splitlines(), 1):
                if call_re.search(line):
                    violations.append(f"{rel}:{lineno}: {line.strip()}")
            continue

        # Fallback: line-regex (handles SyntaxError files or missed imports).
        for lineno, line in enumerate(text.splitlines(), 1):
            if import_re.search(line) or call_re.search(line):
                rel = filepath.relative_to(root)
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert violations == [], (
        "Direct db.add_mission / db.update_mission (etc.) call or import "
        "found outside src/infra + general_beckman — "
        "use general_beckman mission write API instead:\n"
        + "\n".join(violations)
    )
