"""conftest.py — test isolation for the tests/shopping/ suite.

Root causes
-----------
Two distinct isolation problems affect the suite when run together:

1. DB singleton leakage
   ``src.shopping.cache`` and ``src.shopping.resilience.rate_budget`` each keep
   an open aiosqlite connection in a module-level variable (_cache_db,
   _budget_db).  A test class that opens a connection to a temp DB leaves that
   handle in the global singleton, so the next test class connects to the
   *previous* class's temp file rather than its own.

2. Module-level patch leakage from test_scrapers.py
   ``test_scrapers.py`` calls ``patch(...).start()`` at import time (outside any
   TestCase class) to mock ``src.shopping.config.get_rate_limit`` with
   ``{"daily_budget": 9999}`` — and never calls ``.stop()``.  Because pytest
   imports all test modules before running any tests, this mock is already
   active when ``test_phase0::TestRequestTracker`` runs, causing
   ``get_rate_limit("unknown_domain")`` to return 9999 instead of the real
   default of 50.

Fix
---
* A session-scoped fixture snapshots the set of active ``unittest.mock``
  patches that exist after all modules are collected (i.e. the leaked
  module-level patches from test_scrapers.py).
* A function-scoped autouse fixture temporarily stops those leaked patches
  before each test, yields, and then restarts them afterward.  This makes
  every test run against real implementations unless that test itself installs
  its own patches.
* The same function-scoped fixture also resets DB singletons before and after
  every test.
"""

from __future__ import annotations

import asyncio
import sys
from typing import List
from unittest.mock import _patch  # type: ignore[attr-defined]

import pytest


# ---------------------------------------------------------------------------
# Custom mark registrations
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "performance: performance benchmarks — measure timing thresholds",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine in a fresh event loop (safe inside sync pytest fixtures)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _close_db_singletons() -> None:
    """Close any open aiosqlite singletons and null their module variables."""
    # ── src.shopping.cache._cache_db ──────────────────────────────────────
    cache_mod = sys.modules.get("src.shopping.cache")
    if cache_mod is not None:
        conn = getattr(cache_mod, "_cache_db", None)
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass
            cache_mod._cache_db = None

    # ── src.shopping.resilience.rate_budget._budget_db ────────────────────
    rb_mod = sys.modules.get("src.shopping.resilience.rate_budget")
    if rb_mod is not None:
        conn = getattr(rb_mod, "_budget_db", None)
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass
            rb_mod._budget_db = None


# ---------------------------------------------------------------------------
# Session fixture — capture leaked module-level patches after collection
# ---------------------------------------------------------------------------

# This list is populated once by the session fixture below. It holds the patch
# objects that test_scrapers.py (and any other module) started at import time.
_module_level_patches: List = []


@pytest.fixture(autouse=True, scope="session")
def _capture_module_level_patches():
    """Record any patches already active at session start (leaked from imports).

    pytest collects/imports all test modules before running any fixtures, so by
    the time this session fixture runs, leaked ``patch.start()`` calls from
    module-level code in test_scrapers.py are already in
    ``_patch._active_patches``.  We snapshot them here so the per-test fixture
    can temporarily stop them.
    """
    _module_level_patches.extend(list(_patch._active_patches))
    yield
    # Session teardown: stop leaked patches (they were never meant to persist
    # beyond the test session).
    for p in reversed(_module_level_patches):
        try:
            p.stop()
        except RuntimeError:
            pass  # already stopped


# ---------------------------------------------------------------------------
# Per-test fixture — stop leaked patches + reset DB singletons
# ---------------------------------------------------------------------------

def _sync_request_tracker_bindings() -> None:
    """Sync request_tracker.get_rate_limit to whatever src.shopping.config.get_rate_limit is now.

    When ``patch("src.shopping.config.get_rate_limit", ...)`` is started before
    ``src.shopping.request_tracker`` is first imported, the ``from ... import
    get_rate_limit`` statement in request_tracker binds to the *mock* object.
    Stopping the patch later restores ``src.shopping.config.get_rate_limit`` to
    the real function, but the stale binding in request_tracker still holds the
    old mock handle.

    Calling this function after stopping OR restarting patches ensures that
    request_tracker always calls whatever is currently in config.
    """
    rt_mod = sys.modules.get("src.shopping.request_tracker")
    cfg_mod = sys.modules.get("src.shopping.config")
    if rt_mod is not None and cfg_mod is not None:
        current = getattr(cfg_mod, "get_rate_limit", None)
        if current is not None:
            rt_mod.get_rate_limit = current


@pytest.fixture(autouse=True)
def _isolate_each_test():
    """Before each test: stop leaked module-level patches and reset DB state.

    After the test completes (including tearDown for unittest.TestCase), the
    leaked patches are restarted so the next scrapers test sees the mocks it
    expects.
    """
    # 1. Close any open DB connections from a previous test
    _run(_close_db_singletons())

    # 2. Temporarily stop all module-level (leaked) patches so this test runs
    #    against real implementations unless it installs its own patches.
    stopped: List = []
    for p in reversed(_module_level_patches):
        try:
            p.stop()
            stopped.append(p)
        except RuntimeError:
            pass  # already stopped — nothing to do

    # 3. Fix stale bindings in modules that imported the mock at import time
    _sync_request_tracker_bindings()

    yield

    # 4. Close any DB connections this test opened
    _run(_close_db_singletons())

    # 5. Restart the module-level patches so subsequent scrapers tests still
    #    see their expected mocks.
    for p in reversed(stopped):
        try:
            p.start()
        except Exception:
            pass  # patch target may no longer exist; ignore

    # 6. After restarting patches, re-sync stale bindings to the current value
    _sync_request_tracker_bindings()


# ---------------------------------------------------------------------------
# Convenience fixture — isolated cache DB path (opt-in)
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_cache_db(tmp_path):
    """Patch cache.CACHE_DB_PATH to a unique temp file for one test."""
    from unittest.mock import patch as mock_patch
    import src.shopping.cache as cache_mod

    db_path = str(tmp_path / "isolated_cache.db")
    with mock_patch.object(cache_mod, "CACHE_DB_PATH", db_path):
        yield db_path

    _run(_close_db_singletons())


# ---------------------------------------------------------------------------
# Convenience fixture — isolated rate-budget DB path (opt-in)
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_budget_db(tmp_path):
    """Patch rate_budget.BUDGET_DB_PATH to a unique temp file for one test."""
    from unittest.mock import patch as mock_patch
    import src.shopping.resilience.rate_budget as rb_mod

    db_path = str(tmp_path / "isolated_budget.db")
    with mock_patch.object(rb_mod, "BUDGET_DB_PATH", db_path):
        rb_mod._budget_db = None
        yield db_path

    _run(_close_db_singletons())
