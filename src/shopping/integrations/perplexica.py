# shopping/integrations/perplexica.py
"""
Perplexica integration for broad shopping web research.

Interfaces with a local Perplexica instance to perform shopping-focused
web searches, query optimization, and result quality assessment.

NOTE — PARTIAL DEPRECATION (plan item #70):
  `search_perplexica` in this module is a duplicate of the Perplexica
  support already in `src/tools/web_search.py`.  New code should call
  `src.tools.web_search.web_search` instead, which handles model
  discovery, circuit-breaking, and DuckDuckGo fallback automatically.

  The shopping agents already have `web_search` listed in their
  `allowed_tools`, so no wiring change is needed to start using it.

  `format_shopping_query` and `assess_result_quality` below are
  shopping-specific and have no equivalent in web_search.py; they
  remain canonical here.
"""
from __future__ import annotations

import time

from src.infra.logging_config import get_logger

logger = get_logger("shopping.integrations.perplexica")


# Thin wrapper: delegates to web_search._search_perplexica which handles
# model discovery, circuit-breaking, and the correct Vane API format.
# Kept for backward compatibility with fallback_chain.py imports.
async def search_perplexica(query: str, focus: str = "shopping") -> dict:
    """
    Call the Perplexica/Vane API for broad web research.

    Delegates to `src.tools.web_search._search_perplexica` which handles
    model discovery, circuit-breaking, and the correct Vane API format.

    Args:
        query: The search query string.
        focus: Focus mode — "shopping", "web", "academic", etc.

    Returns:
        dict with keys: answer (str), sources (list[dict]).
        On failure returns {"answer": "", "sources": [], "error": str}.
    """
    # Map shopping focus to web_search focus modes
    focus_map = {
        "shopping": "web",
        "web": "web",
        "academic": "academic",
    }
    search_type = focus_map.get(focus, "web")

    try:
        from src.tools.web_search import _search_perplexica

        result = await _search_perplexica(query, max_results=10, focus_mode=search_type)
        if result:
            logger.debug(
                "perplexica shopping search ok (via web_search)",
                answer_len=len(result.get("answer", "")),
                source_count=len(result.get("sources", [])),
            )
            return {"answer": result["answer"], "sources": result["sources"]}

        # Perplexica unavailable or returned nothing — return empty result
        logger.debug("perplexica returned no result for shopping query")
        return {"answer": "", "sources": [], "error": "no result from perplexica"}

    except Exception as e:
        logger.warning("perplexica shopping search error", error=str(e))
        return {"answer": "", "sources": [], "error": str(e)}


async def format_shopping_query(
    query: str,
    category: str | None = None,
    location: str = "Turkey",
) -> str:
    """
    Optimize a user query for shopping context.

    Adds location, category, and price-related terms to improve
    search relevance for shopping queries.

    Args:
        query: Raw user query.
        category: Optional product category (e.g. "electronics", "clothing").
        location: Market location for price context. Defaults to Turkey.

    Returns:
        Optimized query string.
    """
    parts = [query.strip()]

    if category:
        parts.append(category)

    # Add location context for price relevance
    if location:
        parts.append(f"price in {location}")

    # Add current year for freshness
    parts.append(str(time.localtime().tm_year))

    optimized = " ".join(parts)
    logger.debug("formatted shopping query", original=query, optimized=optimized)
    return optimized


async def assess_result_quality(result: dict) -> dict:
    """
    Score a Perplexica search result for shopping relevance.

    Evaluates freshness, reliability, and price validity of the result.

    Args:
        result: A search result dict (from search_perplexica).

    Returns:
        dict with scores: freshness (0-1), reliability (0-1),
        price_validity (0-1), overall (0-1), and notes (list[str]).
    """
    scores = {
        "freshness": 0.5,
        "reliability": 0.5,
        "price_validity": 0.5,
        "overall": 0.5,
        "notes": [],
    }

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # Freshness: more sources and longer answers suggest richer data
    if len(sources) >= 3:
        scores["freshness"] = 0.8
    elif len(sources) >= 1:
        scores["freshness"] = 0.6
    else:
        scores["freshness"] = 0.2
        scores["notes"].append("No sources found — data may be stale")

    # Reliability: check for known shopping domains in sources
    trusted_domains = [
        "amazon", "hepsiburada", "trendyol", "n11", "mediamarkt",
        "teknosa", "vatan", "itopya", "akakce", "cimri",
    ]
    trusted_count = sum(
        1 for s in sources
        if any(d in s.get("url", "").lower() for d in trusted_domains)
    )
    if trusted_count >= 2:
        scores["reliability"] = 0.9
    elif trusted_count >= 1:
        scores["reliability"] = 0.7
    else:
        scores["reliability"] = 0.4
        scores["notes"].append("No trusted shopping sources found")

    # Price validity: check if answer contains price-like patterns
    import re
    price_patterns = [
        r"\d+[\.,]\d+\s*(?:TL|₺|USD|\$|EUR|€)",
        r"(?:TL|₺)\s*\d+",
        r"\d+\s*(?:lira|dolar|euro)",
    ]
    has_prices = any(re.search(p, answer, re.IGNORECASE) for p in price_patterns)
    if has_prices:
        scores["price_validity"] = 0.8
    else:
        scores["price_validity"] = 0.3
        scores["notes"].append("No price data found in answer")

    # Overall score
    scores["overall"] = round(
        (scores["freshness"] + scores["reliability"] + scores["price_validity"]) / 3,
        2,
    )

    return scores
