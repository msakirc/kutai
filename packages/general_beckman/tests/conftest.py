"""Per-test cleanup for the in-flight registry.

Beckman's admission tick now overlays the local in_flight registry from
src.core.in_flight onto the snapshot before pressure_for runs. Tests
that don't explicitly populate that registry MUST start with it empty —
otherwise leftovers from a prior test (especially test_admission_local_
inflight which adds dispatcher slots) leak into the next test's
admission decision and produce unexpected REJECTs.

Autouse fixture clears both _task_slots and _call_entries before each
test. Cleanup-only — never adds entries.

Also clears general_beckman.apply._source_verdict_locks between tests.
SP3b FIX 2 intentionally does NOT evict lock objects from the dict
(doing so reopens the lost-update race under 3+ concurrent appliers —
see apply.py _source_verdict_guard). Without this test-only cleanup,
asyncio.Lock objects created in test N's event loop would persist into
test N+1's event loop and raise "bound to a different event loop" on
the first lock acquisition in any concurrency test.

A2: fresh_db — per-function fixture shared across the three write-API
test files. Sets DB_PATH to a temp file, resets the module-level
_db_connection / _db_connection_path globals, runs init_db(), then
closes the connection in teardown. Eliminates the copy-pasted
_reset_db / _close_db helpers and the ~100 inline _db_connection=None
cache-busts that were scattered through individual tests.

A3: repo_source_texts — session-scoped fixture used by the five guard
tests. Walks the repo root once and returns a dict[Path, str] of source
texts (same exclusion rules the guards previously applied individually).
Each guard applies its own filtering on top.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# In-flight registry + verdict-lock cleanup (existing)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_in_flight_registry():
    try:
        import src.core.in_flight as in_flight_mod
        in_flight_mod._task_slots.clear()
        in_flight_mod._call_entries.clear()
    except Exception:
        pass
    # Clear per-source verdict locks so asyncio.Lock objects bound to the
    # previous test's event loop do not bleed into the next test.
    try:
        import general_beckman.apply as apply_mod
        apply_mod._source_verdict_locks.clear()
    except Exception:
        pass
    yield
    try:
        import src.core.in_flight as in_flight_mod
        in_flight_mod._task_slots.clear()
        in_flight_mod._call_entries.clear()
    except Exception:
        pass
    try:
        import general_beckman.apply as apply_mod
        apply_mod._source_verdict_locks.clear()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# A2: fresh_db — shared DB fixture for the three write-API test files
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    """Set up a fresh per-test SQLite DB and tear it down afterwards.

    Yields the db_path string so tests can use it for direct verification
    via aiosqlite without going through beckman.

    Replaces the copy-pasted _reset_db / _close_db helpers and eliminates
    the ~100 inline ``db_module._db_connection = None`` cache-busts that
    previously appeared between beckman API calls.  Because get_db()
    already detects DB_PATH changes and auto-reconnects, and because
    DB_PATH is stable for the lifetime of the test, no per-call reset is
    needed once the fixture has set up the module state correctly.
    """
    import src.infra.db as db_module
    from src.infra.db import init_db

    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    # Reset cached connection so get_db() opens a fresh connection to the
    # new path (not a stale connection to a previous test's DB).
    db_module._db_connection = None
    db_module._db_connection_path = None

    await init_db()

    yield db_path

    # Teardown: close the shared connection so aiosqlite does not warn
    # about unclosed resources between tests.
    if db_module._db_connection is not None:
        try:
            await db_module._db_connection.close()
        except Exception:
            pass
        db_module._db_connection = None
        db_module._db_connection_path = None


# ──────────────────────────────────────────────────────────────────────────────
# A3: repo_source_texts — session-scoped scan shared by the five guard tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def repo_source_texts() -> dict:
    """Walk the repo root once and return a dict mapping resolved Path →
    source text for every non-test, non-excluded .py file.

    This is consumed by the five guard tests (raw-SQL + import-guard tests
    in test_task_write_api, test_mission_write_api, test_growth_event_api).
    Each guard applies its own allow/deny rules on top; this fixture only
    pre-reads the file set so the repo is walked once per test session
    rather than five times.

    Exclusion rules are the union of what all five guards previously applied
    individually (they are identical: skip_dirs + non-.py + test files +
    files inside a "tests" directory component).  Individual guards may
    further narrow with their own allowed_files / allowed_dirs checks.
    """
    # Resolve repo root the same way the guards do: three parents up from
    # this conftest file (conftest lives in packages/general_beckman/tests/).
    root = Path(__file__).parents[3].resolve()

    skip_dirs = {
        ".venv", "__pycache__", ".git", ".benchmark_cache",
        "node_modules", "worktrees",
    }

    source_texts: dict[Path, str] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            # Skip test files (guards only police production code).
            if fname.startswith("test_") or fname.endswith("_test.py"):
                continue
            if "tests" in Path(dirpath).parts:
                continue
            filepath = (Path(dirpath) / fname).resolve()
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            source_texts[filepath] = text

    return source_texts


# ──────────────────────────────────────────────────────────────────────────────
# A1: _ast_db_write_imports — shared AST import-guard helper
# ──────────────────────────────────────────────────────────────────────────────


def _ast_db_write_imports(filepath, text, guarded_names):
    """Return list of (lineno, name) pairs where a guarded write-helper name
    is imported from src.infra.db (absolute) or a relative ...infra.db path.

    Uses ast.parse so parenthesised multi-line imports are detected
    correctly.  Falls back to an empty list if the file is not valid Python
    (SyntaxError).

    This is the single shared implementation for both the task-write and
    mission-write import guards (previously duplicated as
    _ast_task_write_imports / _ast_mission_write_imports with identical
    bodies).
    """
    import ast

    try:
        tree = ast.parse(text, filename=str(filepath))
    except SyntaxError:
        return []

    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        # Absolute: from dabidabi import ... (the engine package) or the
        # legacy from src.infra.db import ... (sys.modules alias to it).
        is_abs = module in ("dabidabi", "src.infra.db")
        # Relative: from ..infra.db import ... (any level)
        is_rel = node.level > 0 and module == "infra.db"
        if not (is_abs or is_rel):
            continue
        for alias in node.names:
            name = alias.name
            if name in guarded_names:
                hits.append((node.lineno, name))
    return hits


@pytest.fixture(scope="session")
def ast_db_write_imports_fn():
    """Expose _ast_db_write_imports as a pytest fixture so guard tests in
    the three write-API test files can call it without needing a direct
    module import (conftest is not normally importable as a module due to
    the root-level conftest.py taking precedence on sys.path).
    """
    return _ast_db_write_imports
