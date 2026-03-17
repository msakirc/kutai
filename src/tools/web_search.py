# tools/web_search.py
"""
Web search using Perplexica (primary) — with DuckDuckGo and curl fallback.
"""
import json
import os
import urllib.parse

import aiohttp

from src.infra.logging_config import get_logger
from src.tools import run_shell

logger = get_logger("tools.web_search")

# Try to use duckduckgo-search package first
_DDGS = None
try:
    from duckduckgo_search import DDGS
    _DDGS = DDGS
    logger.info("web_search: using duckduckgo-search package")
except Exception as e:
    logger.warning(f"web_search: duckduckgo-search unavailable ({e}), using curl fallback")


async def _search_perplexica(query: str, max_results: int, focus_mode: str):
    """
    Search using Perplexica API.

    Returns list of result dicts with 'title', 'url', 'snippet' keys, or None on error.
    """
    perplexica_url = os.getenv("PERPLEXICA_URL", "").strip()
    if not perplexica_url:
        return None

    # Map focus_mode to Perplexica's camelCase format
    focus_mode_map = {
        "web": "webSearch",
        "academic": "academicSearch",
        "code": "webSearch",  # Perplexica doesn't have code search, use web
    }
    perplexica_focus = focus_mode_map.get(focus_mode, "webSearch")

    payload = {
        "query": query,
        "focusMode": perplexica_focus,
        "optimizationMode": "speed",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{perplexica_url}/api/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "perplexica search failed",
                        status=resp.status,
                        query=query,
                    )
                    return None

                data = await resp.json()

                # Convert Perplexica response to standard format
                results = []
                for item in data.get("results", [])[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                    })

                if results:
                    logger.debug("perplexica search results", count=len(results))

                return results if results else None

    except Exception as e:
        logger.warning("perplexica search error", error=str(e), query=query)
        return None


async def web_search(query: str, max_results: int = 5, search_type: str = "web") -> str:
    """
    Search the web using Perplexica (primary), with DuckDuckGo fallback.

    search_type: "web", "academic", or "code"
    """
    logger.info("web search query", query=query, max_results=max_results, search_type=search_type)

    # Check for degraded capability
    try:
        from src.infra.runtime_state import runtime_state
        is_degraded = "web_search" in runtime_state.get("degraded_capabilities", [])
    except Exception:
        is_degraded = False

    # Method 0: Try Perplexica first (if available and not degraded)
    if not is_degraded:
        perplexica_results = await _search_perplexica(query, max_results, search_type)
        if perplexica_results:
            logger.debug("using perplexica backend for web search")
            lines = []
            for i, r in enumerate(perplexica_results, 1):
                title = r.get("title", "No title")
                snippet = r.get("snippet", "")[:200]
                url = r.get("url", "")
                lines.append(f"{i}. **{title}**\n   {snippet}\n   {url}")
            return f"Search results for '{query}':\n\n" + "\n\n".join(lines)

    # Method 1: duckduckgo-search package
    if _DDGS is not None:
        try:
            results = []
            with _DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)
            if results:
                logger.debug("using duckduckgo-search backend for web search", count=len(results))
                lines = []
                for i, r in enumerate(results, 1):
                    title = r.get("title", "No title")
                    body = r.get("body", "")[:200]
                    href = r.get("href", "")
                    lines.append(f"{i}. **{title}**\n   {body}\n   {href}")
                return f"Search results for '{query}':\n\n" + "\n\n".join(lines)
            else:
                return f"No results found for '{query}'"
        except Exception as e:
            logger.warning("duckduckgo search failed, using curl fallback", error=str(e))

    # Method 2: curl via shell (Docker container has internet)
    try:
        safe_query = urllib.parse.quote_plus(query)
        # DuckDuckGo Instant Answer API (free, no key needed)
        url = f"https://api.duckduckgo.com/?q={safe_query}&format=json&no_html=1&no_redirect=1"

        result = await run_shell(
            f'curl -s --max-time 10 "{url}"',
            timeout=15,
        )

        # Strip the ✅ prefix from shell output
        if result.startswith("✅"):
            result = result[1:].strip()

        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return f"Search returned non-JSON response for '{query}':\n{result[:1000]}"

        lines = []

        # Abstract (direct answer)
        if data.get("Abstract"):
            lines.append(f"**Summary:** {data['Abstract']}")
            if data.get("AbstractURL"):
                lines.append(f"Source: {data['AbstractURL']}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                text = topic["Text"][:200]
                url = topic.get("FirstURL", "")
                lines.append(f"- {text}\n  {url}")

        if lines:
            logger.debug("curl search results", count=len(lines))
            return f"Search results for '{query}':\n\n" + "\n\n".join(lines)

        # If instant answer API returned nothing useful, try HTML scraping
        scrape_url = f"https://html.duckduckgo.com/html/?q={safe_query}"
        scrape_result = await run_shell(
            f'curl -s --max-time 10 "{scrape_url}" | grep -oP \'<a rel="nofollow" class="result__a" href="[^"]*">[^<]*</a>\' | head -5',
            timeout=15,
        )

        if scrape_result.startswith("✅"):
            scrape_result = scrape_result[1:].strip()

        if scrape_result and "❌" not in scrape_result:
            logger.debug("using curl scraping backend for web search")
            return f"Search results for '{query}':\n\n{scrape_result}"

        logger.debug("no results found from curl fallback")
        return f"No results found for '{query}'. DuckDuckGo returned empty response."

    except Exception as e:
        logger.exception("web search curl fallback failed", error=str(e))
        return f"Search error: {e}"
