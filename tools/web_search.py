# tools/web_search.py
import asyncio
import logging

logger = logging.getLogger(__name__)

async def search_web(query: str, num_results: int = 5) -> str:
    """Search web using DuckDuckGo (no API key needed)."""
    try:
        from duckduckgo_search import AsyncDDGS
        async with AsyncDDGS() as ddgs:
            results = []
            async for r in ddgs.atext(query, max_results=num_results):
                results.append(f"**{r['title']}**\n{r['body']}\nURL: {r['href']}")
            if results:
                return "\n\n---\n\n".join(results)
            return "No results found."
    except ImportError:
        return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {e}"
