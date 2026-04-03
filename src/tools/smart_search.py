"""smart_search — unified search tool that routes through APIs, MCP, then web.

Agents see one tool: smart_search(query). Internally it tries:
1. API registry (keyword match -> call_api)
2. MCP tools (if connected, for URL extraction etc.)
3. Web search (Brave/GCSE/DuckDuckGo fallback)
"""

import logging
import time

logger = logging.getLogger(__name__)


async def smart_search(query: str) -> str:
    """Search for information using the best available source."""
    if not query or not query.strip():
        return "Error: empty query"

    start = time.time()

    # 1. Try API registry
    result = await _try_api_registry(query)
    if result:
        await _log(query, layer=2, source="api_registry", success=True, start=start)
        return result

    # 2. Try MCP tools (Fetch for URLs)
    result = await _try_mcp(query)
    if result:
        await _log(query, layer=2, source="mcp", success=True, start=start)
        return result

    # 3. Fall back to web search
    result = await _try_web_search(query)
    if result:
        await _log(query, layer=2, source="web_search", success=True, start=start)
        return result

    await _log(query, layer=2, source=None, success=False, start=start)
    return f"No results found for: {query}"


async def _try_api_registry(query: str) -> str | None:
    """Try to answer via free API registry."""
    try:
        from src.core.fast_resolver import _find_best_match, _extract_params, _call_best_api, _format_response

        match = await _find_best_match(query)
        if not match or match["score"] < 0.3:
            return None

        api = match["api"]
        params = _extract_params(query, match["category"])
        raw = await _call_best_api(api, params)

        if not raw:
            return None

        formatted = _format_response(raw, match["category"], api.name)

        try:
            from src.infra.db import record_api_call
            await record_api_call(api.name, success=True)
        except Exception:
            pass

        return f"{formatted}\n(Source: {api.name} API)"

    except Exception as exc:
        logger.debug("API registry lookup failed: %s", exc)
        return None


async def _try_mcp(query: str) -> str | None:
    """Try MCP tools — currently Fetch for URL extraction."""
    import re
    url_match = re.search(r"https?://\S+", query)
    if not url_match:
        return None

    try:
        from src.tools import TOOL_REGISTRY
        fetch_tool = TOOL_REGISTRY.get("mcp_fetch_fetch")
        if not fetch_tool or not fetch_tool.get("function"):
            return None

        result = await fetch_tool["function"](url=url_match.group())
        if result:
            return f"{str(result)[:2000]}\n(Source: MCP Fetch)"
    except Exception as exc:
        logger.debug("MCP fetch failed: %s", exc)

    return None


async def _try_web_search(query: str) -> str | None:
    """Fall back to existing web_search tool."""
    try:
        from src.tools import TOOL_REGISTRY
        web_search_fn = TOOL_REGISTRY.get("web_search", {}).get("function")
        if not web_search_fn:
            return None

        result = await web_search_fn(query=query)
        if result:
            return f"{str(result)[:3000]}\n(Source: web search)"
    except Exception as exc:
        logger.debug("Web search failed: %s", exc)

    return None


async def _log(query: str, layer: int, source: str | None, success: bool, start: float):
    """Log to smart_search_log table."""
    try:
        from src.infra.db import log_smart_search
        elapsed_ms = int((time.time() - start) * 1000)
        await log_smart_search(query, layer, source, success, elapsed_ms)
    except Exception:
        pass
