# tools/web_search.py
"""
Web search using DuckDuckGo — with curl fallback.
"""
import json
import logging
import urllib.parse

logger = logging.getLogger(__name__)

# Try to use duckduckgo-search package first
_DDGS = None
try:
    from duckduckgo_search import DDGS
    _DDGS = DDGS
    logger.info("web_search: using duckduckgo-search package")
except Exception as e:
    logger.warning(f"web_search: duckduckgo-search unavailable ({e}), using curl fallback")


async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""

    # Method 1: duckduckgo-search package
    if _DDGS is not None:
        try:
            results = []
            with _DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)
            if results:
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
            logger.warning(f"DDGS search failed: {e}, falling back to curl")

    # Method 2: curl via shell (Docker container has internet)
    try:
        from tools.shell import run_shell

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
            return f"Search results for '{query}':\n\n{scrape_result}"

        return f"No results found for '{query}'. DuckDuckGo returned empty response."

    except Exception as e:
        logger.error(f"Web search curl fallback failed: {e}")
        return f"Search error: {e}"
