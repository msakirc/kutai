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


async def get_product_with_fallback(
    product_query: str,
    sources: list[str] | None = None,
) -> list:
    """Search for products using a degradation chain.

    Order: dedicated scraper(s) for each source -> Google CSE -> cache.

    Parameters
    ----------
    product_query:
        User search string.
    sources:
        Optional list of preferred source names (e.g. ``["akakce", "trendyol"]``).
        When ``None``, the default chain is used.

    Returns
    -------
    List of Product dataclass instances from the first source that yields results.
    """
    from src.shopping.resilience.cache_fallback import get_stale_product

    # Default general-purpose scrapers tried when caller provides no sources.
    # These cover the major Turkish e-commerce sites with broad product catalogues.
    DEFAULT_GENERAL_SOURCES = ["akakce", "trendyol", "hepsiburada", "amazon_tr", "epey"]

    # Phase 1: Try each source's dedicated scraper
    effective_sources = [s for s in (sources or []) if s != "default"]
    if not effective_sources:
        effective_sources = DEFAULT_GENERAL_SOURCES
    for source in effective_sources:
        try:
            from src.shopping.scrapers import get_scraper
            scraper_cls = get_scraper(source)
            if scraper_cls is not None:
                scraper = scraper_cls()
                results = await scraper.search(product_query)
                if results:
                    return results
        except Exception as exc:
            logger.warning("Scraper %s failed for '%s': %s", source, product_query, exc)

    # Phase 2: Shared fallbacks (Google CSE — Perplexica skipped because it
    # returns synthesis text, not structured product data)
    shared_chain = build_fallback_chain("default")
    for fn in shared_chain:
        try:
            if asyncio.iscoroutinefunction(fn):
                results = await fn(product_query)
            else:
                results = fn(product_query)
            if results:
                return results
        except Exception as exc:
            logger.warning("Fallback step %s failed for '%s': %s",
                           getattr(fn, "__name__", "?"), product_query, exc)

    # Last resort: stale cache
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
