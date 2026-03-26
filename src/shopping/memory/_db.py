"""Shared database connection for the shopping-memory subsystem."""

import os

import aiosqlite

from src.app.config import DB_PATH
from src.infra.logging_config import get_logger

logger = get_logger("shopping.memory.db")

# ─── Singleton Connection ────────────────────────────────────────────────────
_memory_db: aiosqlite.Connection | None = None

MEMORY_DB_PATH = os.path.join(os.path.dirname(DB_PATH), "shopping_memory.db")


async def get_memory_db() -> aiosqlite.Connection:
    """Return the shared memory DB connection, creating it on first call."""
    global _memory_db
    if _memory_db is None:
        _memory_db = await aiosqlite.connect(MEMORY_DB_PATH)
        _memory_db.row_factory = aiosqlite.Row
        await _memory_db.execute("PRAGMA journal_mode=WAL")
        await _memory_db.execute("PRAGMA synchronous=NORMAL")
        await _memory_db.execute("PRAGMA busy_timeout=5000")
    return _memory_db


async def close_memory_db() -> None:
    """Close the shared memory connection (call on shutdown)."""
    global _memory_db
    if _memory_db is not None:
        try:
            await _memory_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        await _memory_db.close()
        _memory_db = None
        logger.info("Memory database connection closed")
