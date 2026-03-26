"""conftest.py — shared fixtures for the tests/integration/ suite.

Design principles
-----------------
1. DB isolation: every test class gets its own temporary SQLite file.
   The db.py singleton (_db_connection) is reset before and after each
   test so successive tests never share a connection.

2. Speed-first model selection: LLM tests set prefer_speed=True and
   difficulty=2 so the router picks the fastest/smallest available model.
   The smallest GGUF in MODEL_DIR is selected at session start.

3. Generous timeouts: local models can be slow.  Individual tests are
   expected to use pytest-timeout with 120-300 s.

4. Cleanup: missions and tasks created during tests are cleaned up in
   teardown to prevent DB bloat in repeated runs against a real DB.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import AsyncGenerator

import pytest

# ---------------------------------------------------------------------------
# Path setup (mirrors the existing test pattern)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests requiring DB/infra")
    config.addinivalue_line("markers", "llm: tests that make real LLM calls (require a loaded model)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async coroutine in a fresh event loop (safe inside sync pytest)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _reset_db_singleton() -> None:
    """Close and null the db.py singleton connection."""
    import src.infra.db as db_mod
    if db_mod._db_connection is not None:
        try:
            await db_mod._db_connection.close()
        except Exception:
            pass
        db_mod._db_connection = None


# ---------------------------------------------------------------------------
# Per-test: isolated temporary DB
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path):
    """Create a temporary DB, patch db.py, init schema, yield path, teardown.

    Works for both async and sync tests — uses run_async() internally.
    """
    import src.infra.db as db_mod
    import src.app.config as config_mod

    db_path = str(tmp_path / "test_kutai.db")

    # Save originals
    orig_db_path_mod = db_mod.DB_PATH
    orig_db_path_cfg = config_mod.DB_PATH

    # Reset any existing connection
    run_async(_reset_db_singleton())

    # Patch both copies of DB_PATH
    db_mod.DB_PATH = db_path
    config_mod.DB_PATH = db_path

    # Init schema
    run_async(db_mod.init_db())

    yield db_path

    # Teardown: close connection and restore paths
    run_async(_reset_db_singleton())
    db_mod.DB_PATH = orig_db_path_mod
    config_mod.DB_PATH = orig_db_path_cfg

    # Clean up WAL/SHM files
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(db_path + suffix)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Session: find the fastest/smallest local model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fastest_local_model() -> str | None:
    """Return the litellm_name of the smallest available local model.

    Returns None if no local models are registered (e.g. CI without MODEL_DIR).
    Tests marked @pytest.mark.llm should skip when this returns None.
    """
    try:
        from src.models.model_registry import get_registry
        registry = get_registry()

        local_models = [
            m for m in registry.models.values()
            if m.location == "local" and not m.demoted
        ]
        if not local_models:
            return None

        # Sort by active parameter count ascending (smallest = fastest)
        local_models.sort(
            key=lambda m: (m.active_params_b or m.total_params_b or 999, m.file_size_mb or 0)
        )
        chosen = local_models[0]
        return chosen.litellm_name
    except Exception:
        return None


@pytest.fixture(scope="session")
def speed_model_reqs(fastest_local_model):
    """Return a ModelRequirements instance configured for maximum speed.

    Uses model_override to pin to the fastest local model when available.
    Falls back to prefer_speed=True without pinning (cloud models).
    """
    from src.core.router import ModelRequirements
    reqs = ModelRequirements(
        task="assistant",
        agent_type="assistant",
        difficulty=2,
        prefer_speed=True,
        estimated_input_tokens=200,
        estimated_output_tokens=100,
        priority=1,
    )
    if fastest_local_model:
        reqs.model_override = fastest_local_model
    return reqs


# ---------------------------------------------------------------------------
# Async event loop for async tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def event_loop():
    """Provide a fresh event loop for each async test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Shutdown event fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def shutdown_event():
    """A fresh asyncio.Event for shutdown tests."""
    return asyncio.Event()
