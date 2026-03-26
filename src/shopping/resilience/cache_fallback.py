"""Serve stale cached data when live fetch fails.

Unlike the main cache module which enforces TTL, this module intentionally
returns *expired* entries as a last resort so the user still gets something
useful.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from src.infra.logging_config import get_logger

logger = get_logger("shopping.resilience.cache_fallback")


def _hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode()).hexdigest()


async def get_stale_product(
    query: str,
    max_age_hours: int = 72,
) -> list[dict] | None:
    """Return cached product results for *query* even if expired.

    Parameters
    ----------
    query:
        The original search query.
    max_age_hours:
        Maximum staleness in hours.  Entries older than this are not returned.

    Returns
    -------
    List of product dicts, or ``None`` if nothing is cached within the
    staleness window.
    """
    from src.shopping.cache import get_cache_db

    db = await get_cache_db()
    cutoff = time.time() - (max_age_hours * 3600)
    query_hash = _hash(query)

    # Try search cache first (matches query -> results)
    cursor = await db.execute(
        """
        SELECT result_json, searched_at FROM search_cache
        WHERE query_hash = ? AND searched_at >= ?
        ORDER BY searched_at DESC LIMIT 1
        """,
        (query_hash, cutoff),
    )
    row = await cursor.fetchone()
    if row is not None:
        logger.info("Stale search cache hit for '%s' (age %.1fh)",
                     query, (time.time() - row["searched_at"]) / 3600)
        return json.loads(row["result_json"])

    # Try broader match: any search cache entry whose query_hash starts similarly
    # (this won't match by prefix since we hash, so skip and try products table)
    cursor = await db.execute(
        """
        SELECT product_json, fetched_at FROM products
        WHERE fetched_at >= ?
        ORDER BY fetched_at DESC LIMIT 20
        """,
        (cutoff,),
    )
    rows = await cursor.fetchall()
    if not rows:
        return None

    # Filter products whose cached JSON mentions the query terms
    query_lower = query.lower()
    matches = []
    for row in rows:
        product = json.loads(row["product_json"])
        product_text = json.dumps(product, ensure_ascii=False).lower()
        if any(term in product_text for term in query_lower.split()):
            matches.append(product)

    if matches:
        logger.info("Stale product cache hit for '%s': %d results", query, len(matches))
        return matches

    return None


async def get_stale_price(
    product_name: str,
    max_age_hours: int = 72,
) -> float | None:
    """Return the most recent cached price for *product_name*, even if expired.

    Parameters
    ----------
    product_name:
        The product name to look up.
    max_age_hours:
        Maximum staleness in hours.

    Returns
    -------
    The most recent price as a float, or ``None``.
    """
    from src.shopping.cache import get_cache_db

    db = await get_cache_db()
    cutoff = time.time() - (max_age_hours * 3600)

    # Search products table for a name match
    cursor = await db.execute(
        """
        SELECT product_json, fetched_at FROM products
        WHERE fetched_at >= ?
        ORDER BY fetched_at DESC
        """,
        (cutoff,),
    )
    rows = await cursor.fetchall()
    name_lower = product_name.lower()

    for row in rows:
        product = json.loads(row["product_json"])
        cached_name = product.get("name", "").lower()
        if name_lower in cached_name or cached_name in name_lower:
            price = product.get("price")
            if price is not None:
                logger.info("Stale price for '%s': %.2f (age %.1fh)",
                            product_name, price,
                            (time.time() - row["fetched_at"]) / 3600)
                return float(price)

    # Try price_history table
    cursor = await db.execute(
        """
        SELECT price, observed_at FROM price_history
        WHERE observed_at >= ?
        ORDER BY observed_at DESC
        """,
        (cutoff,),
    )
    row = await cursor.fetchone()
    if row is not None:
        logger.info("Stale price history for '%s': %.2f", product_name, row["price"])
        return float(row["price"])

    return None


async def warmup_cache(popular_queries: list[str]) -> None:
    """Pre-populate the cache with results for common queries.

    This is best called during startup or a quiet period to ensure
    the fallback cache has fresh data for the most common products.

    Parameters
    ----------
    popular_queries:
        List of query strings to pre-fetch.
    """
    from src.shopping.resilience.fallback_chain import get_product_with_fallback

    logger.info("Cache warmup started for %d queries", len(popular_queries))
    succeeded = 0

    for query in popular_queries:
        try:
            results = await get_product_with_fallback(query)
            if results:
                succeeded += 1
        except Exception as exc:
            logger.warning("Cache warmup failed for '%s': %s", query, exc)
        # Be gentle with sources
        await asyncio.sleep(2.0)

    logger.info("Cache warmup complete: %d/%d succeeded", succeeded, len(popular_queries))
