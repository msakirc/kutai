"""Shared fixtures for yalayut tests."""
import aiosqlite
import pytest
import pytest_asyncio

from yalayut.schema import ensure_yalayut_schema


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
