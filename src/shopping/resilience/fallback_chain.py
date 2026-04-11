"""Graceful degradation when scrapers fail.

Implements a fallback chain pattern: try the primary function, then each
fallback in order until one succeeds.  The product-specific chain is
dedicated scraper -> Perplexica -> generic web search -> cached results.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from src.infra.logging_config import get_logger

logger = get_logger("shopping.resilience.fallback_chain")


async def execute_with_fallback(
    primary_fn: Callable,
    fallback_fns: list[Callable],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Try *primary_fn*, then each fallback in order until one succeeds.

    Parameters
    ----------
    primary_fn:
        The preferred callable (sync or async).
    fallback_fns:
        Ordered list of alternatives to try when earlier functions fail.
    *args, **kwargs:
        Forwarded to every callable in the chain.

    Returns
    -------
    The first successful result.

    Raises
    ------
    RuntimeError
        If every function in the chain fails.
    """
    chain = [primary_fn, *fallback_fns]
    last_error: Exception | None = None

    for idx, fn in enumerate(chain):
        label = getattr(fn, "__name__", f"fallback_{idx}")
        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)
            if result is not None:
                if idx > 0:
                    logger.info("Fallback succeeded via %s (attempt %d)", label, idx + 1)
                return result
        except Exception as exc:
            last_error = exc
            logger.warning("Chain step %d (%s) failed: %s", idx, label, exc)

    raise RuntimeError(
        f"All {len(chain)} fallback steps exhausted. Last error: {last_error}"
    )


async def _search_scraper(source: str, query: str) -> list:
    """Run a single scraper, return results or empty list on failure."""
    try:
        from src.shopping.scrapers import get_scraper
        scraper_cls = get_scraper(source)
        if scraper_cls is None:
            return []
        scraper = scraper_cls()
        results = await scraper.search(query)
        if results and isinstance(results, list):
            return results
    except Exception as exc:
        logger.warning("Scraper %s failed for '%s': %s", source, query, exc)
    return []


async def _search_parallel(sources: list[str], query: str) -> list:
    """Search multiple scrapers in parallel, return first non-empty result."""
    tasks = [asyncio.create_task(_search_scraper(s, query)) for s in sources]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            # Cancel remaining tasks
            for t in tasks:
                t.cancel()
            return result
    return []


# Scraper tiers — ordered by breadth and likelihood of having results.
# Tier 1: aggregators (search across many retailers)
_AGGREGATORS = ["akakce", "epey"]
# Tier 2: major retailers (large catalogues)
_MAJOR_RETAILERS = ["trendyol", "hepsiburada", "amazon_tr"]
# Tier 3: specialty retailers (narrower catalogues, still worth trying)
_SPECIALTY_RETAILERS = [
    "kitapyurdu", "dr", "decathlon", "direnc",
    "koctas", "ikea", "migros", "getir",
]
# Forum/review sites excluded — they don't return product/price data.


async def get_product_with_fallback(
    product_query: str,
    sources: list[str] | None = None,
) -> list:
    """Search for products using a tiered parallel strategy.

    Order:
      1. Aggregators in parallel (akakce, epey)
      2. Major retailers in parallel (trendyol, hepsiburada, amazon_tr)
      3. Specialty retailers in parallel
      4. Google CSE
      5. Stale cache

    Parameters
    ----------
    product_query:
        User search string.
    sources:
        Optional list of preferred source names (e.g. ``["akakce", "trendyol"]``).
        When ``None``, all tiers are searched.

    Returns
    -------
    List of Product dataclass instances from the first tier that yields results.
    """
    from src.shopping.resilience.cache_fallback import get_stale_product

    if sources:
        effective = [s for s in sources if s != "default"]
        if effective:
            results = await _search_parallel(effective, product_query)
            if results:
                return results

    # Tier 1: aggregators (parallel)
    results = await _search_parallel(_AGGREGATORS, product_query)
    if results:
        return results

    # Tier 2: major retailers (parallel)
    results = await _search_parallel(_MAJOR_RETAILERS, product_query)
    if results:
        return results

    # Tier 3: specialty retailers (parallel)
    results = await _search_parallel(_SPECIALTY_RETAILERS, product_query)
    if results:
        return results

    # Tier 4: Google CSE
    try:
        from src.shopping.scrapers.google_cse import GoogleCSEScraper
        cse = GoogleCSEScraper()
        results = await cse.search(product_query)
        if results:
            return results
    except Exception as exc:
        logger.warning("Google CSE failed for '%s': %s", product_query, exc)

    # Tier 5: stale cache
    stale = await get_stale_product(product_query, max_age_hours=72)
    if stale:
        logger.info("Serving stale cache for '%s'", product_query)
        return stale

    return []


def build_fallback_chain(source: str) -> list[Callable]:
    """Build an ordered fallback chain for *source*.

    Parameters
    ----------
    source:
        The primary source name (e.g. ``"akakce"``, ``"trendyol"``).
        ``"default"`` builds a generic chain.

    Returns
    -------
    List of callables, each accepting a query string.
    """
    async def _scraper_search(query: str) -> list[dict]:
        """Try the dedicated scraper for the given source."""
        try:
            from src.shopping.scrapers import get_scraper
            scraper_cls = get_scraper(source)
            if scraper_cls is None:
                return []
            scraper = scraper_cls()
            return await scraper.search(query)
        except Exception:
            return []

    async def _google_cse_search(query: str) -> list[dict]:
        """Fallback to Google CSE."""
        try:
            from src.shopping.scrapers.google_cse import GoogleCSEScraper
            scraper = GoogleCSEScraper()
            return await scraper.search(query)
        except Exception:
            return []

    chain: list[Callable] = []

    if source != "default":
        chain.append(_scraper_search)

    # Perplexica is intentionally excluded: it returns synthesis text
    # (answer + sources dict), not structured Product data.
    chain.append(_google_cse_search)

    return chain
