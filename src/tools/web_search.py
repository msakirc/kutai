# tools/web_search.py
"""
Web search using Perplexica/Vane (primary) — with DuckDuckGo and curl fallback.
"""
import asyncio
import json
import os
import urllib.parse

import aiohttp

from src.infra.logging_config import get_logger
from src.tools import run_shell

logger = get_logger("tools.web_search")

# Try to use ddgs package (formerly duckduckgo-search)
_DDGS = None
try:
    from ddgs import DDGS
    _DDGS = DDGS
    logger.info("web_search: using ddgs package")
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _DDGS = DDGS
        logger.info("web_search: using duckduckgo-search (legacy) package")
    except ImportError:
        logger.warning("web_search: no ddgs/duckduckgo-search package, using curl fallback")

# Cached Perplexica provider config (populated on first call)
_perplexica_models: dict | None = None
_perplexica_fail_count: int = 0
_PERPLEXICA_MAX_FAILURES = 3  # Disable Perplexica after this many consecutive failures


async def _discover_perplexica_models(base_url: str) -> dict | None:
    """
    Fetch providers from Perplexica/Vane and pick the first chat model
    and first embedding model available.

    Returns dict with 'chatModel' and 'embeddingModel' keys, or None.
    """
    global _perplexica_models
    if _perplexica_models is not None:
        return _perplexica_models

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/api/providers",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    logger.warning("perplexica providers fetch failed", status=resp.status)
                    return None
                data = await resp.json()

        providers = data.get("providers", [])
        chat_model = None
        embedding_model = None

        # Preferred chat models in priority order.
        # Groq compound models route to appropriate backends and tend to
        # handle structured output (response_format) better than raw models.
        preferred_chat = [
            "groq/compound",
            "groq/compound-mini",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ]
        # Models to skip (not actual chat models)
        skip_models = {
            "whisper-large-v3", "whisper-large-v3-turbo",  # audio models
            "meta-llama/llama-prompt-guard-2-22m",  # classifier
            "meta-llama/llama-prompt-guard-2-86m",  # classifier
            "openai/gpt-oss-safeguard-20b",  # safety model
            "canopylabs/orpheus-arabic-saudi",  # TTS
            "canopylabs/orpheus-v1-english",  # TTS
        }

        # Collect all available chat models with their provider IDs
        all_chat_models = []
        for provider in providers:
            pid = provider.get("id", "")
            for cm in provider.get("chatModels", []):
                key = cm.get("key", "")
                if key and key != "error" and key not in skip_models:
                    all_chat_models.append({"providerId": pid, "key": key})
            for em in provider.get("embeddingModels", []):
                if em.get("key") and em.get("key") != "error" and not embedding_model:
                    embedding_model = {"providerId": pid, "key": em["key"]}

        # Select chat model: prefer known-good models, fall back to first available
        if all_chat_models:
            for pref in preferred_chat:
                for cm in all_chat_models:
                    if cm["key"] == pref:
                        chat_model = cm
                        break
                if chat_model:
                    break
            if not chat_model:
                chat_model = all_chat_models[0]

        if chat_model and embedding_model:
            _perplexica_models = {
                "chatModel": chat_model,
                "embeddingModel": embedding_model,
            }
            logger.info(
                "perplexica models discovered",
                chat=chat_model["key"],
                embedding=embedding_model["key"],
            )
            return _perplexica_models

        logger.warning(
            "perplexica: missing models",
            has_chat=chat_model is not None,
            has_embedding=embedding_model is not None,
        )
        return None

    except Exception as e:
        logger.warning("perplexica provider discovery error", error=str(e))
        return None


async def _search_perplexica(query: str, max_results: int, focus_mode: str):
    """
    Search using Perplexica/Vane API.

    The Vane API requires:
      - query (str)
      - sources (non-empty list, e.g. ["web"])
      - chatModel ({providerId, key})
      - embeddingModel ({providerId, key})
      - optimizationMode (str, default "speed")

    Returns the AI-generated answer string, or None on error.
    """
    global _perplexica_fail_count

    perplexica_url = os.getenv("PERPLEXICA_URL", "").strip()
    if not perplexica_url:
        return None

    # Skip if too many consecutive failures
    if _perplexica_fail_count >= _PERPLEXICA_MAX_FAILURES:
        logger.debug("perplexica: disabled after repeated failures, using fallback")
        return None

    # Discover available models
    models = await _discover_perplexica_models(perplexica_url)
    if not models:
        logger.debug("perplexica: no models available, skipping")
        return None

    # Map focus_mode to source types
    source_map = {
        "web": ["web"],
        "academic": ["web"],
        "code": ["web"],
    }

    payload = {
        "query": query,
        "sources": source_map.get(focus_mode, ["web"]),
        "chatModel": models["chatModel"],
        "embeddingModel": models["embeddingModel"],
        "optimizationMode": "speed",
        "stream": False,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{perplexica_url}/api/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "perplexica search failed",
                        status=resp.status,
                        body=body[:300],
                        query=query,
                    )
                    _perplexica_fail_count += 1
                    return None

                data = await resp.json()

                # Vane returns {message: "answer text", sources: [...]}
                answer = data.get("message", "")
                sources = data.get("sources", [])

                if not answer:
                    _perplexica_fail_count += 1
                    return None

                # Success - reset failure counter
                _perplexica_fail_count = 0

                # Build result with answer + source citations
                result = {"answer": answer, "sources": []}
                for src in sources[:max_results]:
                    result["sources"].append({
                        "title": src.get("metadata", {}).get("title", ""),
                        "url": src.get("metadata", {}).get("url", ""),
                        "snippet": src.get("content", "")[:200],
                    })

                logger.debug(
                    "perplexica search ok",
                    answer_len=len(answer),
                    source_count=len(result["sources"]),
                )
                return result

    except asyncio.TimeoutError:
        _perplexica_fail_count += 1
        logger.warning("perplexica search timeout (15s)", query=query)
    except Exception as e:
        _perplexica_fail_count += 1
        logger.warning("perplexica search error", error=f"{type(e).__name__}: {e}", query=query)
        return None


async def web_search(query: str, max_results: int = 5, search_type: str = "web") -> str:
    """
    Search the web using Perplexica/Vane (primary), with DuckDuckGo fallback.

    search_type: "web", "academic", or "code"
    """
    logger.info("web search query", query=query, max_results=max_results, search_type=search_type)

    # Check for degraded capability
    try:
        from src.infra.runtime_state import runtime_state
        is_degraded = "web_search" in runtime_state.get("degraded_capabilities", [])
    except Exception:
        is_degraded = False

    # Method 0: Try Perplexica/Vane first (if available and not degraded)
    if not is_degraded:
        perplexica_result = await _search_perplexica(query, max_results, search_type)
        if perplexica_result:
            logger.debug("using perplexica backend for web search")
            lines = [perplexica_result["answer"]]
            if perplexica_result["sources"]:
                lines.append("\n**Sources:**")
                for i, src in enumerate(perplexica_result["sources"], 1):
                    title = src.get("title", "Untitled")
                    url = src.get("url", "")
                    lines.append(f"{i}. [{title}]({url})")
            return "\n".join(lines)

    # Method 1: duckduckgo-search package (ddgs 9.x)
    if _DDGS is not None:
        try:
            # ddgs 9.x: DDGS().text() returns a list directly
            results = _DDGS().text(query, max_results=max_results)
            if results:
                logger.debug("using duckduckgo-search backend", count=len(results))
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

        # Strip the prefix from shell output
        if result.startswith("\u2705"):
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

        if scrape_result.startswith("\u2705"):
            scrape_result = scrape_result[1:].strip()

        if scrape_result and "\u274c" not in scrape_result:
            logger.debug("using curl scraping backend for web search")
            return f"Search results for '{query}':\n\n{scrape_result}"

        logger.debug("no results found from curl fallback")
        return f"No results found for '{query}'. DuckDuckGo returned empty response."

    except Exception as e:
        logger.exception("web search curl fallback failed", error=str(e))
        return f"Search error: {e}"
