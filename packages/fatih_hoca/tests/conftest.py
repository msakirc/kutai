"""Pytest configuration for fatih_hoca tests.

Adds the tests/ directory to sys.path so that sim.state (and future
tests/sim/* modules) are importable without polluting the runtime package.
"""
import sys
import pathlib

import pytest

# Allow `from sim.state import ...` in tests/sim/
_tests_dir = pathlib.Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))


@pytest.fixture(autouse=True)
def _isolated_registry_store(tmp_path):
    """Isolate src.infra.registry_store per-test. ModelRegistry now
    delegates mark_dead/is_dead/revive (and provider variants) to the
    SQLite-backed store; without isolation, dead-set state leaks across
    tests via the module-level singleton connection.

    Each test gets a fresh tmp DB; teardown closes the singleton."""
    from src.infra import registry_store
    db_path = tmp_path / "registry_test.db"
    registry_store.set_db_path(str(db_path))
    yield
    registry_store.close()
