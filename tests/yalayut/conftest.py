"""Shared fixtures for yalayut tests."""
import os
import aiosqlite
import pytest
import pytest_asyncio

from yalayut.schema import ensure_yalayut_schema


def pytest_configure(config):
    """Ensure YALAYUT_SECRET_KEY is set for tests that exercise set_secret."""
    if not os.getenv("YALAYUT_SECRET_KEY"):
        try:
            from cryptography.fernet import Fernet
            os.environ["YALAYUT_SECRET_KEY"] = Fernet.generate_key().decode()
        except ImportError:
            pass  # cryptography not installed; secret tests will skip/fail naturally


@pytest_asyncio.fixture
async def yalayut_db():
    """In-memory SQLite connection with the full yalayut schema applied.

    isolation_level=None matches src/infra/db.py (autocommit + WAL in prod).
    """
    db = await aiosqlite.connect(":memory:", isolation_level=None)
    db.row_factory = aiosqlite.Row
    await ensure_yalayut_schema(db)
    yield db
    await db.close()


@pytest.fixture(autouse=False)
def clean_yalayut_index(loop):
    """Wipe yalayut_index and related tables before admin tests to ensure
    isolation when the real shared DB is used across test runs."""
    async def _clean():
        from src.infra.db import init_db, get_db
        await init_db()
        db = await get_db()
        await db.execute("DELETE FROM yalayut_index")
        await db.execute("DELETE FROM yalayut_source_candidates")
        await db.execute("DELETE FROM yalayut_policy_proposals")
        await db.execute("DELETE FROM yalayut_policy")
        await db.execute("DELETE FROM yalayut_secrets")
        await db.commit()
    loop.run_until_complete(_clean())
    yield


@pytest.fixture(autouse=False)
def clean_yalayut_sources(loop):
    """Wipe yalayut_sources before each cron-discovery test so rows seeded
    by a prior test session do not leak into sources_scanned counts."""
    async def _clean():
        from src.infra.db import init_db, get_db
        await init_db()
        db = await get_db()
        await db.execute("DELETE FROM yalayut_sources")
        await db.commit()
    loop.run_until_complete(_clean())
    yield


@pytest.fixture(autouse=False)
def clean_demand_signals(loop):
    """Wipe yalayut_demand_signals before each Phase 4 demand test so that
    the 7-day cooldown does not leak across test runs on the real DB."""
    async def _clean():
        from src.infra.db import init_db, get_db
        await init_db()
        db = await get_db()
        await db.execute("DELETE FROM yalayut_demand_signals")
        await db.commit()
    loop.run_until_complete(_clean())
    yield
