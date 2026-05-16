"""Shared fixtures for yalayut tests."""
import aiosqlite
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
