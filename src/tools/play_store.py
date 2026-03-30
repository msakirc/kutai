"""Google Play Store search, app details, and reviews."""

import asyncio
from functools import partial

from src.infra.logging_config import get_logger

logger = get_logger("tools.play_store")


async def play_store_search(query: str, count: int = 10, language: str = "tr") -> list[dict]:
    """Search Google Play Store for apps."""
    from google_play_scraper import search
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, partial(search, query, n_hits=count, lang=language)
    )
    # Slim down the results
    return [
        {
            "app_id": r.get("appId", ""),
            "title": r.get("title", ""),
            "score": r.get("score", 0),
            "installs": r.get("installs", ""),
            "developer": r.get("developer", ""),
            "free": r.get("free", True),
            "price": r.get("price", 0),
            "summary": (r.get("summary") or "")[:200],
        }
        for r in results
    ]


async def play_store_app(app_id: str, language: str = "tr") -> dict:
    """Get detailed info for a specific app."""
    from google_play_scraper import app
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, partial(app, app_id, lang=language)
    )
    return {
        "app_id": result.get("appId", ""),
        "title": result.get("title", ""),
        "score": result.get("score", 0),
        "ratings": result.get("ratings", 0),
        "reviews": result.get("reviews", 0),
        "installs": result.get("installs", ""),
        "developer": result.get("developer", ""),
        "genre": result.get("genre", ""),
        "description": (result.get("description") or "")[:500],
        "updated": result.get("updated", ""),
        "version": result.get("version", ""),
        "free": result.get("free", True),
        "price": result.get("price", 0),
    }


async def play_store_reviews(app_id: str, count: int = 20, language: str = "tr") -> list[dict]:
    """Get reviews for a specific app."""
    from google_play_scraper import reviews, Sort
    loop = asyncio.get_event_loop()
    result, _ = await loop.run_in_executor(
        None, partial(reviews, app_id, lang=language, count=count, sort=Sort.NEWEST)
    )
    return [
        {
            "score": r.get("score", 0),
            "text": (r.get("content") or "")[:300],
            "thumbs_up": r.get("thumbsUpCount", 0),
            "date": str(r.get("at", "")),
        }
        for r in result
    ]


async def play_store_similar(app_id: str, language: str = "tr") -> list[dict]:
    """Find similar/competitor apps."""
    from google_play_scraper import similar
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, partial(similar, app_id, lang=language)
    )
    return [
        {
            "app_id": r.get("appId", ""),
            "title": r.get("title", ""),
            "score": r.get("score", 0),
            "installs": r.get("installs", ""),
            "developer": r.get("developer", ""),
        }
        for r in results[:10]
    ]
