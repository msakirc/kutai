"""Product cache with SQLite and TTL-based invalidation."""

import hashlib
import json
import os
import time

import aiosqlite

from src.app.config import DB_PATH
from src.infra.logging_config import get_logger

logger = get_logger("shopping.cache")

# ─── TTL Constants (seconds) ─────────────────────────────────────────────────
SPECS_TTL = 30 * 86400       # 30 days
PRICES_TTL = 24 * 3600       # 24 hours
REVIEWS_TTL = 7 * 86400      # 7 days
SEARCH_TTL = 12 * 3600       # 12 hours

_TTL_MAP = {
    "specs": SPECS_TTL,
    "prices": PRICES_TTL,
    "reviews": REVIEWS_TTL,
    "search": SEARCH_TTL,
}

# ─── Singleton Connection ────────────────────────────────────────────────────
_cache_db: aiosqlite.Connection | None = None

CACHE_DB_PATH = os.path.join(os.path.dirname(DB_PATH), "shopping_cache.db")


async def get_cache_db() -> aiosqlite.Connection:
    """Return the shared cache DB connection, creating it on first call."""
    global _cache_db
    if _cache_db is None:
        _cache_db = await aiosqlite.connect(CACHE_DB_PATH)
        _cache_db.row_factory = aiosqlite.Row
        await _cache_db.execute("PRAGMA journal_mode=WAL")
        await _cache_db.execute("PRAGMA synchronous=NORMAL")
        await _cache_db.execute("PRAGMA busy_timeout=5000")
    return _cache_db


async def close_cache_db() -> None:
    """Close the shared cache connection (call on shutdown)."""
    global _cache_db
    if _cache_db is not None:
        try:
            await _cache_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        await _cache_db.close()
        _cache_db = None
        logger.info("Cache database connection closed")


# ─── Schema ──────────────────────────────────────────────────────────────────

async def init_cache_db():
    """Create cache tables if they don't exist."""
    db = await get_cache_db()

    await db.execute("""
        CREATE TABLE IF NOT EXISTS products (
            url_hash      TEXT PRIMARY KEY,
            product_json  TEXT    NOT NULL,
            source        TEXT    NOT NULL,
            fetched_at    REAL    NOT NULL,
            ttl_category  TEXT    NOT NULL DEFAULT 'specs'
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            url_hash    TEXT NOT NULL,
            review_json TEXT NOT NULL,
            source      TEXT NOT NULL,
            fetched_at  REAL NOT NULL,
            PRIMARY KEY (url_hash, source)
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            url_hash    TEXT NOT NULL,
            price       REAL NOT NULL,
            source      TEXT NOT NULL,
            observed_at REAL NOT NULL
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query_hash  TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            source      TEXT NOT NULL,
            searched_at REAL NOT NULL
        )
    """)

    await db.commit()
    logger.info("Cache database initialised")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode()).hexdigest()


def _is_expired(timestamp: float, ttl_category: str) -> bool:
    ttl = _TTL_MAP.get(ttl_category, SPECS_TTL)
    return (time.time() - timestamp) > ttl


# ─── Products ────────────────────────────────────────────────────────────────

async def cache_product(
    url: str,
    product_dict: dict,
    source: str,
    ttl_category: str = "specs",
) -> None:
    """Insert or replace a cached product entry."""
    db = await get_cache_db()
    await db.execute(
        """
        INSERT OR REPLACE INTO products
            (url_hash, product_json, source, fetched_at, ttl_category)
        VALUES (?, ?, ?, ?, ?)
        """,
        (_hash(url), json.dumps(product_dict, ensure_ascii=False), source, time.time(), ttl_category),
    )
    await db.commit()


async def get_cached_product(url: str) -> dict | None:
    """Return cached product data, or None if missing / expired."""
    db = await get_cache_db()
    cursor = await db.execute(
        "SELECT product_json, fetched_at, ttl_category FROM products WHERE url_hash = ?",
        (_hash(url),),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    if _is_expired(row["fetched_at"], row["ttl_category"]):
        return None
    return json.loads(row["product_json"])


# ─── Search Cache ────────────────────────────────────────────────────────────

async def cache_search(query: str, source: str, results: list[dict]) -> None:
    """Cache search results for a query + source pair."""
    db = await get_cache_db()
    await db.execute(
        """
        INSERT OR REPLACE INTO search_cache
            (query_hash, result_json, source, searched_at)
        VALUES (?, ?, ?, ?)
        """,
        (_hash(f"{query}:{source}"), json.dumps(results, ensure_ascii=False), source, time.time()),
    )
    await db.commit()


async def get_cached_search(query: str, source: str) -> list[dict] | None:
    """Return cached search results, or None if missing / expired."""
    db = await get_cache_db()
    cursor = await db.execute(
        "SELECT result_json, searched_at FROM search_cache WHERE query_hash = ?",
        (_hash(f"{query}:{source}"),),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    if _is_expired(row["searched_at"], "search"):
        return None
    return json.loads(row["result_json"])


# ─── Reviews ─────────────────────────────────────────────────────────────────

async def cache_reviews(url: str, reviews: list[dict], source: str) -> None:
    """Cache reviews for a product URL + source pair."""
    db = await get_cache_db()
    await db.execute(
        """
        INSERT OR REPLACE INTO reviews
            (url_hash, review_json, source, fetched_at)
        VALUES (?, ?, ?, ?)
        """,
        (_hash(url), json.dumps(reviews, ensure_ascii=False), source, time.time()),
    )
    await db.commit()


async def get_cached_reviews(url: str, source: str) -> list[dict] | None:
    """Return cached reviews, or None if missing / expired."""
    db = await get_cache_db()
    cursor = await db.execute(
        "SELECT review_json, fetched_at FROM reviews WHERE url_hash = ? AND source = ?",
        (_hash(url), source),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    if _is_expired(row["fetched_at"], "reviews"):
        return None
    return json.loads(row["review_json"])


# ─── Price History ───────────────────────────────────────────────────────────

async def add_price_history(url: str, price: float, source: str) -> None:
    """Append a price observation (never overwrites)."""
    db = await get_cache_db()
    await db.execute(
        """
        INSERT INTO price_history (url_hash, price, source, observed_at)
        VALUES (?, ?, ?, ?)
        """,
        (_hash(url), price, source, time.time()),
    )
    await db.commit()


async def get_price_history(url: str) -> list[dict]:
    """Return all price observations for a URL, oldest first."""
    db = await get_cache_db()
    cursor = await db.execute(
        "SELECT price, source, observed_at FROM price_history WHERE url_hash = ? ORDER BY observed_at",
        (_hash(url),),
    )
    rows = await cursor.fetchall()
    return [{"price": r["price"], "source": r["source"], "observed_at": r["observed_at"]} for r in rows]


# ─── Cleanup ─────────────────────────────────────────────────────────────────

async def cleanup_expired() -> None:
    """Delete all expired entries from every cache table."""
    db = await get_cache_db()
    now = time.time()

    # Products — check each ttl_category
    for category, ttl in _TTL_MAP.items():
        await db.execute(
            "DELETE FROM products WHERE ttl_category = ? AND (? - fetched_at) > ?",
            (category, now, ttl),
        )

    # Reviews
    await db.execute(
        "DELETE FROM reviews WHERE (? - fetched_at) > ?",
        (now, REVIEWS_TTL),
    )

    # Search cache
    await db.execute(
        "DELETE FROM search_cache WHERE (? - searched_at) > ?",
        (now, SEARCH_TTL),
    )

    # Price history — keep last 90 days
    await db.execute(
        "DELETE FROM price_history WHERE (? - observed_at) > ?",
        (now, 90 * 86400),
    )

    await db.commit()
    logger.info("Expired cache entries cleaned up")
