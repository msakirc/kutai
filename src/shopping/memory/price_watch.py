"""Price-watch memory — track products and notify when prices drop."""

import json
import time

import aiosqlite

from src.shopping.memory._db import get_memory_db
from src.infra.logging_config import get_logger

logger = get_logger("shopping.memory.price_watch")

# ─── Schema ──────────────────────────────────────────────────────────────────


async def init_price_watch_db() -> None:
    """Create price-watch tables if they don't exist."""
    db = await get_memory_db()

    await db.execute("""
        CREATE TABLE IF NOT EXISTS price_watches (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            product_name    TEXT    NOT NULL,
            current_price   REAL    NOT NULL,
            target_price    REAL,
            source          TEXT,
            historical_low  REAL,
            created_at      REAL    NOT NULL,
            updated_at      REAL    NOT NULL,
            triggered       INTEGER NOT NULL DEFAULT 0,
            expired         INTEGER NOT NULL DEFAULT 0
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_pw_user ON price_watches(user_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_pw_active ON price_watches(triggered, expired)"
    )

    # Migration: add product_url column if missing
    try:
        await db.execute("ALTER TABLE price_watches ADD COLUMN product_url TEXT")
    except Exception:
        pass  # column already exists

    await db.execute("""
        CREATE TABLE IF NOT EXISTS price_watch_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_id    INTEGER NOT NULL,
            price       REAL    NOT NULL,
            source      TEXT,
            observed_at REAL    NOT NULL,
            FOREIGN KEY (watch_id) REFERENCES price_watches(id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_pwh_watch ON price_watch_history(watch_id)"
    )

    await db.commit()
    logger.info("Price-watch tables initialised")


# ─── Public API ──────────────────────────────────────────────────────────────


async def add_price_watch(
    user_id: int,
    product_name: str,
    current_price: float,
    target_price: float = None,
    source: str = None,
    historical_low: float = None,
    product_url: str = None,
) -> int:
    """Create a new price watch and return its watch_id."""
    db = await get_memory_db()
    now = time.time()
    cur = await db.execute(
        """
        INSERT INTO price_watches
            (user_id, product_name, current_price, target_price,
             source, historical_low, product_url, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, product_name, current_price, target_price,
         source, historical_low, product_url, now, now),
    )
    watch_id = cur.lastrowid

    # Seed first history entry
    await db.execute(
        "INSERT INTO price_watch_history (watch_id, price, source, observed_at) VALUES (?, ?, ?, ?)",
        (watch_id, current_price, source, now),
    )
    await db.commit()
    logger.info("Created price watch #%d for user %s: %s", watch_id, user_id, product_name)
    return watch_id


async def get_active_watches(user_id: int) -> list[dict]:
    """Return all active (non-triggered, non-expired) watches for a user."""
    db = await get_memory_db()
    cur = await db.execute(
        """
        SELECT id, product_name, current_price, target_price,
               source, historical_low, created_at, updated_at
        FROM price_watches
        WHERE user_id = ? AND triggered = 0 AND expired = 0
        ORDER BY created_at DESC
        """,
        (user_id,),
    )
    return [dict(r) for r in await cur.fetchall()]


async def get_all_active_watches() -> list[dict]:
    """Return all active watches across all users (for the scheduled checker)."""
    db = await get_memory_db()
    cur = await db.execute(
        """
        SELECT id, user_id, product_name, current_price, target_price,
               source, historical_low, product_url, created_at, updated_at
        FROM price_watches
        WHERE triggered = 0 AND expired = 0
        ORDER BY updated_at ASC
        """
    )
    return [dict(r) for r in await cur.fetchall()]


async def update_watch_price(watch_id: int, current_price: float, source: str) -> None:
    """Record a new observed price for a watch."""
    db = await get_memory_db()
    now = time.time()

    # Update current price and historical low
    cur = await db.execute(
        "SELECT historical_low FROM price_watches WHERE id = ?", (watch_id,)
    )
    row = await cur.fetchone()
    if row is None:
        return

    historical_low = row["historical_low"]
    if historical_low is None or current_price < historical_low:
        historical_low = current_price

    await db.execute(
        """
        UPDATE price_watches
        SET current_price = ?, historical_low = ?, source = ?, updated_at = ?
        WHERE id = ?
        """,
        (current_price, historical_low, source, now, watch_id),
    )
    await db.execute(
        "INSERT INTO price_watch_history (watch_id, price, source, observed_at) VALUES (?, ?, ?, ?)",
        (watch_id, current_price, source, now),
    )
    await db.commit()


async def trigger_watch(watch_id: int) -> None:
    """Mark a watch as triggered (target price reached)."""
    db = await get_memory_db()
    await db.execute(
        "UPDATE price_watches SET triggered = 1, updated_at = ? WHERE id = ?",
        (time.time(), watch_id),
    )
    await db.commit()
    logger.info("Price watch #%d triggered", watch_id)


async def expire_old_watches(days: int = 90) -> None:
    """Auto-expire watches older than *days* that haven't been triggered."""
    db = await get_memory_db()
    cutoff = time.time() - days * 86400
    cur = await db.execute(
        """
        UPDATE price_watches
        SET expired = 1, updated_at = ?
        WHERE triggered = 0 AND expired = 0 AND created_at < ?
        """,
        (time.time(), cutoff),
    )
    if cur.rowcount:
        logger.info("Expired %d old price watches (older than %d days)", cur.rowcount, days)
    await db.commit()


async def remove_watch(watch_id: int) -> None:
    """Delete a watch and its price history."""
    db = await get_memory_db()
    await db.execute("DELETE FROM price_watch_history WHERE watch_id = ?", (watch_id,))
    await db.execute("DELETE FROM price_watches WHERE id = ?", (watch_id,))
    await db.commit()
    logger.info("Removed price watch #%d", watch_id)
