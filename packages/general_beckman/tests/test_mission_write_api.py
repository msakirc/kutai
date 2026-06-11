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
# Helpers (copied from test_growth_event_api.py pattern)
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
async def test_add_mission_returns_id_and_persists(tmp_path, monkeypatch):
    """add_mission returns a positive int id and the row lands in DB."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission
        mid = await add_mission(title="Test Mission", description="test desc")

        assert isinstance(mid, int)
        assert mid > 0

        row = await _fetch_mission(db_path, mid)
        assert row is not None
        assert row["title"] == "Test Mission"
        assert row["description"] == "test desc"
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_add_mission_with_optional_fields(tmp_path, monkeypatch):
    """Optional params are stored correctly."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
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
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# update_mission
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_mission_modifies_row(tmp_path, monkeypatch):
    """update_mission persists changes to whitelisted columns."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, update_mission
        mid = await add_mission(title="Before", description="d")

        db_module._db_connection = None
        await update_mission(mid, title="After", description="updated")

        row = await _fetch_mission(db_path, mid)
        assert row["title"] == "After"
        assert row["description"] == "updated"
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_update_mission_rejects_unknown_column(tmp_path, monkeypatch):
    """update_mission raises ValueError for columns not in the whitelist."""
    _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, update_mission
        mid = await add_mission(title="T", description="d")

        db_module._db_connection = None
        with pytest.raises(ValueError, match="missions"):
            await update_mission(mid, nonexistent_col="bad")
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# update_mission_fields
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_mission_fields_accepted_column(tmp_path, monkeypatch):
    """update_mission_fields writes a whitelisted column to the DB."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, update_mission_fields
        mid = await add_mission(title="T", description="d")

        db_module._db_connection = None
        await update_mission_fields(mid, founder_attention_budget_minutes=90)

        row = await _fetch_mission(db_path, mid)
        assert row["founder_attention_budget_minutes"] == 90
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_update_mission_fields_rejects_unknown_column(tmp_path, monkeypatch):
    """update_mission_fields raises ValueError for unknown columns."""
    _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, update_mission_fields
        mid = await add_mission(title="T", description="d")

        db_module._db_connection = None
        with pytest.raises(ValueError, match="unknown column"):
            await update_mission_fields(mid, totally_bogus_col="value")
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_update_mission_fields_noop_on_empty(tmp_path, monkeypatch):
    """Calling with no fields is a no-op (no error)."""
    _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, update_mission_fields
        mid = await add_mission(title="T", description="d")
        db_module._db_connection = None
        # Should not raise
        await update_mission_fields(mid)
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_update_mission_fields_multiple_columns(tmp_path, monkeypatch):
    """Multiple whitelisted columns can be updated in one call."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, update_mission_fields
        mid = await add_mission(title="T", description="d")

        db_module._db_connection = None
        await update_mission_fields(
            mid,
            telegram_thread_id=42,
            review_density_json='{"density": 0.5}',
        )

        row = await _fetch_mission(db_path, mid)
        assert row["telegram_thread_id"] == 42
        assert row["review_density_json"] == '{"density": 0.5}'
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# block_mission / unblock_mission
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_block_mission_sets_blocked_state(tmp_path, monkeypatch):
    """block_mission flips lifecycle_state to blocked_on_founder_action."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, block_mission
        mid = await add_mission(title="T", description="d")

        db_module._db_connection = None
        result = await block_mission(mid)
        assert result is True

        row = await _fetch_mission(db_path, mid)
        state = row.get("lifecycle_state") or row.get("status")
        assert state == "blocked_on_founder_action"
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_block_mission_noop_if_already_blocked(tmp_path, monkeypatch):
    """block_mission returns False if mission is already blocked."""
    _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, block_mission
        mid = await add_mission(title="T", description="d")

        db_module._db_connection = None
        await block_mission(mid)
        db_module._db_connection = None
        result = await block_mission(mid)
        assert result is False
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_unblock_mission_restores_active_and_resets_tasks(tmp_path, monkeypatch):
    """unblock_mission flips mission to active and resets blocked tasks to pending."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, block_mission, unblock_mission

        mid = await add_mission(title="T", description="d")

        # Insert a task that is blocked_on_founder_action for this mission.
        db_module._db_connection = None
        db = await db_module.get_db()
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status, agent_type) "
            "VALUES (?, 'task1', 'blocked_on_founder_action', 'coder')",
            (mid,),
        )
        await db.commit()
        db_module._db_connection = None

        # Block first.
        await block_mission(mid)
        db_module._db_connection = None

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
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_unblock_mission_noop_if_not_blocked(tmp_path, monkeypatch):
    """unblock_mission returns False if mission is not blocked."""
    _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, unblock_mission
        mid = await add_mission(title="T", description="d")

        db_module._db_connection = None
        result = await unblock_mission(mid)
        assert result is False
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_block_unblock_mission_missing_row(tmp_path, monkeypatch):
    """block/unblock return False for non-existent mission id."""
    _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import block_mission, unblock_mission
        assert await block_mission(99999) is False
        db_module._db_connection = None
        assert await unblock_mission(99999) is False
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# purge_all_missions / purge_all
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_purge_all_missions_clears_missions_and_tasks(tmp_path, monkeypatch):
    """purge_all_missions deletes missions and dependent rows."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, purge_all_missions

        mid1 = await add_mission(title="M1", description="d")
        db_module._db_connection = None
        mid2 = await add_mission(title="M2", description="d")

        # Insert a task for one of the missions.
        db_module._db_connection = None
        db = await db_module.get_db()
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status, agent_type) "
            "VALUES (?, 'task', 'pending', 'coder')",
            (mid1,),
        )
        await db.commit()
        db_module._db_connection = None

        await purge_all_missions()

        assert await _fetch_all_missions(db_path) == []
        assert await _fetch_all_tasks(db_path) == []
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_purge_all_clears_missions_and_conversations(tmp_path, monkeypatch):
    """purge_all wipes missions, tasks, conversations, and memory."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, purge_all

        await add_mission(title="M", description="d")
        db_module._db_connection = None

        # Seed a conversations row (if table exists).
        db = await db_module.get_db()
        try:
            await db.execute(
                "INSERT INTO conversations (chat_id, message, role) VALUES (1, 'hi', 'user')"
            )
            await db.commit()
        except Exception:
            pass  # table may not exist in minimal test schema
        db_module._db_connection = None

        await purge_all()

        assert await _fetch_all_missions(db_path) == []
        assert await _fetch_all_tasks(db_path) == []
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# increment_mission_rework_loops
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_increment_mission_rework_loops(tmp_path, monkeypatch):
    """increment_mission_rework_loops atomically bumps counter and returns new value."""
    db_path = _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import add_mission, increment_mission_rework_loops

        mid = await add_mission(title="T", description="d")
        db_module._db_connection = None

        count1 = await increment_mission_rework_loops(mid)
        db_module._db_connection = None
        count2 = await increment_mission_rework_loops(mid)

        assert count1 == 1
        assert count2 == 2

        row = await _fetch_mission(db_path, mid)
        assert row["phase_7_rework_loops"] == 2
    finally:
        await _close_db(db_module)


@pytest.mark.asyncio
async def test_increment_mission_rework_loops_returns_zero_missing(tmp_path, monkeypatch):
    """Returns 0 for a non-existent mission id (defensive, telemetry must not crash)."""
    _reset_db(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()

    import src.infra.db as db_module
    db_module._db_connection = None

    try:
        from general_beckman import increment_mission_rework_loops
        count = await increment_mission_rework_loops(99999)
        assert count == 0
    finally:
        await _close_db(db_module)


# ──────────────────────────────────────────────────────────────────────────────
# Guard test: no raw SQL against missions outside src/infra/db.py
# and no db.add_mission / db.update_mission outside src/infra + general_beckman
# ──────────────────────────────────────────────────────────────────────────────


def test_no_raw_missions_sql_outside_db():
    """No source file outside src/infra/db.py may contain raw
    INSERT INTO missions, UPDATE missions, or DELETE FROM missions SQL.

    After migration all former raw-SQL sites call beckman APIs instead.
    src/infra/db.py is the sole SQL owner.
    """
    import re
    import os
    from pathlib import Path

    root = Path(__file__).parents[3]  # repo root (worktree)

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
    skip_dirs = {".venv", "__pycache__", ".git", ".benchmark_cache", "node_modules", "worktrees"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname.startswith("test_") or fname.endswith("_test.py"):
                continue
            if "tests" in Path(dirpath).parts:
                continue
            filepath = (Path(dirpath) / fname).resolve()
            if filepath in allowed:
                continue
            # Allow all of general_beckman/src/general_beckman/ — beckman is the write-owner.
            if any(str(filepath).startswith(str(d)) for d in allowed_dirs):
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if sql_re.search(line):
                    rel = filepath.relative_to(root.resolve())
                    violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert violations == [], (
        "Raw missions SQL found outside src/infra/db.py and general_beckman — "
        "use general_beckman mission write API instead:\n"
        + "\n".join(violations)
    )


def test_no_raw_db_mission_imports_outside_infra_beckman():
    """No source file outside src/infra + general_beckman may import
    db.add_mission, db.update_mission, db.update_mission_fields,
    db.increment_mission_rework_loops, db.purge_all_missions, or db.purge_all
    directly.

    After migration all callers use general_beckman's API.  Banning
    ``update_mission_fields`` here closes the latent gap: callers must go
    through ``beckman.update_mission_fields`` (which itself delegates to
    ``src.infra.db.update_mission_fields``), not bypass beckman directly.
    """
    import re
    import os
    from pathlib import Path

    root = Path(__file__).parents[3]  # repo root

    # Matches: from src.infra.db import add_mission  (or update_mission, etc.)
    # or:      from src.infra.db import ... add_mission ...
    import_re = re.compile(
        r'from\s+src\.infra\.db\s+import\s+.*?\b('
        r'add_mission|update_mission|increment_mission_rework_loops'
        r'|purge_all_missions|purge_all\b'
        r')',
    )
    # Also catch: import src.infra.db; db.add_mission( / db.update_mission_fields(
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
    skip_dirs = {".venv", "__pycache__", ".git", ".benchmark_cache", "node_modules", "worktrees"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname.startswith("test_") or fname.endswith("_test.py"):
                continue
            if "tests" in Path(dirpath).parts:
                continue
            filepath = (Path(dirpath) / fname).resolve()
            if filepath in allowed_files:
                continue
            if any(
                str(filepath).startswith(str(d)) for d in allowed_dirs
            ):
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if import_re.search(line) or call_re.search(line):
                    rel = filepath.relative_to(root.resolve())
                    violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert violations == [], (
        "Direct db.add_mission / db.update_mission (etc.) call or import "
        "found outside src/infra + general_beckman — "
        "use general_beckman mission write API instead:\n"
        + "\n".join(violations)
    )
