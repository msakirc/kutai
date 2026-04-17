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


# Top-K per site — mature e-commerce search already ranks by relevance;
# the tail is mostly spare parts and accessories. Keep the head, drop
# the rest. Tune by site if needed later.
_TOP_K_PER_SITE = 5


async def _search_scraper(source: str, query: str) -> list:
    """Run a single scraper, return top-K results in site order.

    Each scraper gets its own 20s timeout so one blocked site can't
    eat the entire budget. Results are truncated to the first
    ``_TOP_K_PER_SITE`` and stamped with ``site_rank`` (0-indexed
    position within the site's response) so downstream ranking can
    preserve per-site relevance.
    """
    try:
        from src.shopping.scrapers import get_scraper
        scraper_cls = get_scraper(source)
        if scraper_cls is None:
            return []
        scraper = scraper_cls()
        results = await asyncio.wait_for(scraper.search(query), timeout=20)
        if not (results and isinstance(results, list)):
            return []
        trimmed = results[:_TOP_K_PER_SITE]
        for i, p in enumerate(trimmed):
            # Works for dataclass Products AND for any scraper that
            # happens to return plain dicts. Ignore if neither.
            if hasattr(p, "site_rank"):
                try:
                    p.site_rank = i
                except Exception:
                    pass
            elif isinstance(p, dict):
                p["site_rank"] = i
        return trimmed
    except asyncio.TimeoutError:
        logger.warning("Scraper %s timed out for '%s'", source, query)
    except Exception as exc:
        logger.warning("Scraper %s failed for '%s': %s", source, query, exc)
    return []


async def _search_parallel(sources: list[str], query: str) -> list:
    """Search multiple scrapers in parallel, collect ALL results.

    For price comparison we need products from every source, not just the
    first one that responds.
    """
    tasks = [asyncio.create_task(_search_scraper(s, query)) for s in sources]
    all_results: list = []
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            if result:
                all_results.extend(result)
        except Exception:
            pass
    return all_results


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
# Community sources — forums, complaints, discussions (not product/price data)
_COMMUNITY_SOURCES = ["technopat", "sikayetvar", "donanimhaber", "eksisozluk"]


async def get_community_data(query: str) -> list:
    """Search forum/complaint/discussion sites for community feedback.

    Runs all community scrapers in parallel and collects all results
    (not first-wins like product search — we want breadth here).
    """
    tasks = [
        asyncio.create_task(_search_scraper(s, query))
        for s in _COMMUNITY_SOURCES
    ]
    all_results: list = []
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            if result:
                all_results.extend(result)
        except Exception:
            pass
    return all_results


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

    # Search ALL tiers in parallel — collect products from every source
    # for price comparison. Tiers are searched concurrently.
    all_products: list = []

    tier_tasks = [
        asyncio.create_task(_search_parallel(_AGGREGATORS, product_query)),
        asyncio.create_task(_search_parallel(_MAJOR_RETAILERS, product_query)),
        asyncio.create_task(_search_parallel(_SPECIALTY_RETAILERS, product_query)),
    ]
    for coro in asyncio.as_completed(tier_tasks):
        try:
            result = await coro
            if result:
                all_products.extend(result)
        except Exception:
            pass

    if all_products:
        return all_products

    # Fallback: Google CSE
    try:
        from src.shopping.scrapers.google_cse import GoogleCSEScraper
        cse = GoogleCSEScraper()
        results = await cse.search(product_query)
        if results:
            return results
    except Exception as exc:
        logger.warning("Google CSE failed for '%s': %s", product_query, exc)

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
