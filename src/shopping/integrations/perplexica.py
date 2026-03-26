# shopping/integrations/perplexica.py
"""
Perplexica integration for broad shopping web research.

Interfaces with a local Perplexica instance to perform shopping-focused
web searches, query optimization, and result quality assessment.
"""
from __future__ import annotations

import os
import time

import aiohttp

from src.infra.logging_config import get_logger

logger = get_logger("shopping.integrations.perplexica")

_PERPLEXICA_URL = os.getenv("PERPLEXICA_URL", "http://localhost:3001")


async def search_perplexica(query: str, focus: str = "shopping") -> dict:
    """
    Call the Perplexica API for broad web research.

    Args:
        query: The search query string.
        focus: Focus mode — "shopping", "web", "academic", etc.

    Returns:
        dict with keys: answer (str), sources (list[dict]), raw (dict).
        On failure returns {"answer": "", "sources": [], "error": str}.
    """
    url = f"{_PERPLEXICA_URL}/api/search"

    # Map shopping focus to Perplexica's focusMode
    focus_map = {
        "shopping": "webSearch",
        "web": "webSearch",
        "academic": "academicSearch",
    }

    payload = {
        "query": query,
        "focusMode": focus_map.get(focus, "webSearch"),
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "perplexica shopping search failed",
                        status=resp.status,
                        body=body[:300],
                    )
                    return {"answer": "", "sources": [], "error": f"HTTP {resp.status}"}

                data = await resp.json()
                answer = data.get("message", data.get("answer", ""))
                sources = []
                for src in data.get("sources", []):
                    meta = src.get("metadata", {})
                    sources.append({
                        "title": meta.get("title", ""),
                        "url": meta.get("url", ""),
                        "snippet": src.get("content", "")[:300],
                    })

                logger.debug(
                    "perplexica shopping search ok",
                    answer_len=len(answer),
                    source_count=len(sources),
                )
                return {"answer": answer, "sources": sources, "raw": data}

    except aiohttp.ClientError as e:
        logger.warning("perplexica connection error", error=str(e))
        return {"answer": "", "sources": [], "error": str(e)}
    except Exception as e:
        logger.warning("perplexica search error", error=str(e))
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
