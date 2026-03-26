# shopping/integrations/web_tools_bridge.py
"""
Bridge to existing web tools (web_search, web_extract).

Provides fallback search and extraction when dedicated shopping scrapers
fail or are unavailable.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("shopping.integrations.web_tools_bridge")


async def fallback_web_search(query: str) -> list[dict]:
    """
    Fallback web search using the existing web_search tool.

    Used when dedicated shopping scrapers fail or return no results.

    Args:
        query: Search query string.

    Returns:
        List of result dicts with keys: title, url, snippet.
        Returns empty list on failure.
    """
    try:
        from src.tools.web_search import web_search

        raw = await web_search(query, max_results=5, search_type="web")

        # Parse the formatted string output into structured results
        results = []
        if not raw or "error" in raw.lower()[:50]:
            return results

        # The web_search tool returns a formatted string; extract what we can
        lines = raw.split("\n")
        current: dict | None = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Lines starting with a number are result headers
            if line and line[0].isdigit() and ". " in line[:5]:
                if current:
                    results.append(current)
                current = {"title": line.split(". ", 1)[-1].strip("*"), "url": "", "snippet": ""}
            elif current is not None:
                if line.startswith("http"):
                    current["url"] = line
                elif not current["snippet"]:
                    current["snippet"] = line.strip("*").strip()

        if current:
            results.append(current)

        logger.debug("fallback web search results", count=len(results), query=query[:50])
        return results

    except Exception as e:
        logger.warning("fallback web search failed", error=str(e), query=query[:50])
        return []


async def fallback_web_extract(url: str) -> dict:
    """
    Fallback web page extraction using the existing web_extract tool.

    Used when dedicated shopping scrapers fail to parse a product page.

    Args:
        url: The URL to extract content from.

    Returns:
        dict with keys: content (str), url (str), success (bool), error (str|None).
    """
    try:
        from src.tools.web_extract import extract_url

        content = await extract_url(url)

        if content and not content.startswith("Error"):
            logger.debug("fallback web extract ok", url=url, chars=len(content))
            return {
                "content": content,
                "url": url,
                "success": True,
                "error": None,
            }
        else:
            return {
                "content": content or "",
                "url": url,
                "success": False,
                "error": content if content else "Empty extraction",
            }

    except Exception as e:
        logger.warning("fallback web extract failed", url=url, error=str(e))
        return {
            "content": "",
            "url": url,
            "success": False,
            "error": str(e),
        }
