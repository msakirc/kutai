"""Tests for Z2 T3A — migration_apply post-hook kind.

Covers:
- Registry row with correct triggers.
- Auto-wire on migration file patterns.
- SQLite path: clean .sql → ok=True.
- SQLite path: bad .sql → ok=False with error string.
- Postgres path without enable_testcontainers → skipped=True.
- Unknown stack + alembic offline path (subprocess mocked).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_migration_apply():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "migration_apply" in POST_HOOK_REGISTRY


def test_registry_row_shape():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["migration_apply"]
    assert spec.kind == "migration_apply"
    assert spec.verb == "apply_migration"
    assert spec.default_severity == "blocker"
    assert isinstance(spec.auto_wire_triggers, list)
    assert len(spec.auto_wire_triggers) > 0


def test_registry_triggers():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    triggers = POST_HOOK_REGISTRY["migration_apply"].auto_wire_triggers
    assert "migrations/*.py" in triggers
    assert "migrations/*.sql" in triggers
    assert "alembic/versions/*.py" in triggers
    assert "*.sql" in triggers


def test_registry_in_post_hook_kinds():
    from general_beckman.posthooks import POST_HOOK_KINDS
    assert "migration_apply" in POST_HOOK_KINDS


# ---------------------------------------------------------------------------
# Auto-wire (via expander triggers)
# ---------------------------------------------------------------------------

def test_autowire_on_migration_py():
    """migration_apply should auto-wire when produces includes migrations/*.py."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    import fnmatch
    spec = POST_HOOK_REGISTRY["migration_apply"]
    produces = ["migrations/001_init.py"]
    matched = any(
        fnmatch.fnmatch(p, pattern)
        for p in produces
        for pattern in spec.auto_wire_triggers
    )
    assert matched, f"Expected trigger match for {produces}"


def test_autowire_on_schema_sql():
    """migration_apply should auto-wire when produces includes schema.sql."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    import fnmatch
    spec = POST_HOOK_REGISTRY["migration_apply"]
    produces = ["schema.sql"]
    matched = any(
        fnmatch.fnmatch(p, pattern)
        for p in produces
        for pattern in spec.auto_wire_triggers
    )
    assert matched, f"Expected trigger match for {produces}"


def test_no_autowire_on_plain_py():
    """migration_apply should NOT match a plain *.py file (no migration path)."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    import fnmatch
    spec = POST_HOOK_REGISTRY["migration_apply"]
    produces = ["src/app/main.py"]
    # *.sql won't match; migrations/*.py won't match; alembic/versions/*.py won't match
    migration_triggers = [t for t in spec.auto_wire_triggers if "migration" in t or "alembic" in t]
    matched = any(
        fnmatch.fnmatch(p, pattern)
        for p in produces
        for pattern in migration_triggers
    )
    assert not matched, "Non-migration .py should not match migration-specific triggers"


# ---------------------------------------------------------------------------
# Verb: SQLite path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sqlite_clean_sql():
    """SQLite path with a valid CREATE TABLE SQL → ok=True."""
    from mr_roboto.apply_migration import apply_migration

    with tempfile.TemporaryDirectory() as tmpdir:
        sql_file = os.path.join(tmpdir, "001_init.sql")
        with open(sql_file, "w") as f:
            f.write("CREATE TABLE foo (id INTEGER PRIMARY KEY);\n")

        result = await apply_migration(
            mission_id=None,
            target_files=[sql_file],
            workspace_path=tmpdir,
            stack_hint="sqlite",
            timeout_s=30.0,
        )

    assert result["ok"] is True
    assert result["skipped"] is False
    assert result["stack_used"] in ("sqlite_direct", "sqlite_alembic")
    assert not result["error"]


@pytest.mark.asyncio
async def test_sqlite_bad_sql_syntax():
    """SQLite path with bad SQL syntax → ok=False with error string."""
    from mr_roboto.apply_migration import apply_migration

    with tempfile.TemporaryDirectory() as tmpdir:
        sql_file = os.path.join(tmpdir, "002_bad.sql")
        with open(sql_file, "w") as f:
            f.write("THIS IS NOT VALID SQL;;;")

        result = await apply_migration(
            mission_id=None,
            target_files=[sql_file],
            workspace_path=tmpdir,
            stack_hint="sqlite",
            timeout_s=30.0,
        )

    assert result["ok"] is False
    assert result["error"]
    assert "sqlite3 error" in result["error"].lower() or "error" in result["error"].lower()


@pytest.mark.asyncio
async def test_sqlite_missing_file():
    """SQLite path with a non-existent file → ok=False."""
    from mr_roboto.apply_migration import apply_migration

    with tempfile.TemporaryDirectory() as tmpdir:
        result = await apply_migration(
            mission_id=None,
            target_files=[os.path.join(tmpdir, "nonexistent.sql")],
            workspace_path=tmpdir,
            stack_hint="sqlite",
            timeout_s=30.0,
        )

    assert result["ok"] is False
    assert result["error"]


# ---------------------------------------------------------------------------
# Verb: Postgres path (no testcontainers, no opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_postgres_without_enable_testcontainers_skips():
    """Postgres stack without enable_testcontainers=True → skipped=True, ok=True."""
    from mr_roboto.apply_migration import apply_migration

    with tempfile.TemporaryDirectory() as tmpdir:
        sql_file = os.path.join(tmpdir, "001_init.sql")
        with open(sql_file, "w") as f:
            f.write("CREATE TABLE foo (id INTEGER);\n")

        result = await apply_migration(
            mission_id=None,
            target_files=[sql_file],
            workspace_path=tmpdir,
            stack_hint="postgres+fastapi",
            enable_testcontainers=False,
            timeout_s=30.0,
        )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert "enable_testcontainers" in result.get("reason", "")


@pytest.mark.asyncio
@pytest.mark.skipif(
    True,
    reason="Requires Docker + testcontainers — skip in CI without Docker",
)
async def test_postgres_with_testcontainers_clean():
    """Postgres stack with enable_testcontainers=True + Docker → ok=True."""
    from mr_roboto.apply_migration import apply_migration

    with tempfile.TemporaryDirectory() as tmpdir:
        sql_file = os.path.join(tmpdir, "001_init.sql")
        with open(sql_file, "w") as f:
            f.write("CREATE TABLE foo (id INTEGER);\n")

        result = await apply_migration(
            mission_id=None,
            target_files=[sql_file],
            workspace_path=tmpdir,
            stack_hint="postgres",
            enable_testcontainers=True,
            timeout_s=120.0,
        )

    assert result["ok"] is True
    assert result["skipped"] is False


# ---------------------------------------------------------------------------
# Verb: Unknown stack → alembic offline (mocked subprocess)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_stack_offline_no_alembic_ini():
    """Unknown stack without alembic.ini → offline parse (file-existence check)."""
    from mr_roboto.apply_migration import apply_migration

    with tempfile.TemporaryDirectory() as tmpdir:
        sql_file = os.path.join(tmpdir, "001_init.sql")
        with open(sql_file, "w") as f:
            f.write("CREATE TABLE foo (id INTEGER);\n")

        result = await apply_migration(
            mission_id=None,
            target_files=[sql_file],
            workspace_path=tmpdir,
            stack_hint="unknown_stack",
            timeout_s=30.0,
        )

    # File exists + UTF-8 readable → ok=True (offline parse passes)
    assert result["ok"] is True
    assert result["stack_used"] == "offline_parse"


@pytest.mark.asyncio
async def test_unknown_stack_offline_alembic_invoked():
    """Unknown stack with alembic.ini → alembic offline mode is invoked.

    Verifies that when alembic.ini is present, _apply_offline calls
    ``alembic upgrade head --sql``. We intercept via ``asyncio.
    create_subprocess_exec`` mock so no alembic binary is required.
    """
    import sys
    import asyncio

    invoked_cmds: list[list[str]] = []

    # Intercept at asyncio subprocess level — no alembic binary needed.
    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"-- SQL output", b""

        async def wait(self):
            return 0

    original_exec = asyncio.create_subprocess_exec

    async def fake_exec(*args, **kwargs):
        invoked_cmds.append(list(args))
        return FakeProc()

    asyncio.create_subprocess_exec = fake_exec  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "alembic.ini"), "w") as f:
                f.write("[alembic]\nscript_location = alembic\n")

            sql_file = os.path.join(tmpdir, "alembic", "versions", "001_init.py")
            os.makedirs(os.path.dirname(sql_file), exist_ok=True)
            with open(sql_file, "w") as f:
                f.write("# migration stub\n")

            from mr_roboto.apply_migration import apply_migration
            result = await apply_migration(
                mission_id=None,
                target_files=[sql_file],
                workspace_path=tmpdir,
                stack_hint="unknown_db",
                timeout_s=30.0,
            )
    finally:
        asyncio.create_subprocess_exec = original_exec  # type: ignore[assignment]

    assert result["ok"] is True
    assert len(invoked_cmds) >= 1
    flat = [str(a) for a in invoked_cmds[0]]
    assert any("alembic" in a for a in flat)
    assert "--sql" in flat


# ---------------------------------------------------------------------------
# Verb: empty target_files
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_target_files_skips():
    """No target files → skipped=True, ok=True."""
    from mr_roboto.apply_migration import apply_migration

    with tempfile.TemporaryDirectory() as tmpdir:
        result = await apply_migration(
            mission_id=None,
            target_files=[],
            workspace_path=tmpdir,
            stack_hint="sqlite",
            timeout_s=30.0,
        )

    assert result["ok"] is True
    assert result["skipped"] is True


# ---------------------------------------------------------------------------
# Reversibility registry
# ---------------------------------------------------------------------------

def test_reversibility_entry():
    from mr_roboto.reversibility import VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY.get("apply_migration") == "full"
