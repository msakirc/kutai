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
_perplexica_disabled_at: float = 0.0
_PERPLEXICA_MAX_FAILURES = 3      # Disable after N consecutive failures
_PERPLEXICA_RETRY_AFTER = 300.0   # Re-enable after 5 minutes

# Phrases that indicate Perplexica couldn't find real data
_NO_DATA_PHRASES = [
    "cannot provide", "no data available", "no specific", "purely speculative",
    "not available", "unable to find", "no confirmed", "no information",
    "do not contain", "does not contain", "no relevant", "couldn't find",
    "could not find", "no results found",
]


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

        # Models to skip (not actual chat models)
        skip_models = {
            "whisper-large-v3", "whisper-large-v3-turbo",
            "meta-llama/llama-prompt-guard-2-22m",
            "meta-llama/llama-prompt-guard-2-86m",
            "openai/gpt-oss-safeguard-20b",
            "canopylabs/orpheus-arabic-saudi",
            "canopylabs/orpheus-v1-english",
            "groq/compound", "groq/compound-mini",  # hangs on Vane
        }

        # Collect all available chat models with their provider IDs
        all_chat_models = []
        for provider in providers:
            pid = provider.get("id", "")
            for cm in provider.get("chatModels", []):
                # Vane chat models may have "key" (cloud) or only "id"/"name" (local-openai).
                # Use key if present, otherwise fall back to name then id.
                key = cm.get("key") or cm.get("name") or cm.get("id", "")
                if key and key != "error" and key not in skip_models:
                    all_chat_models.append({"providerId": pid, "key": key})
            for em in provider.get("embeddingModels", []):
                em_key = em.get("key") or em.get("name") or em.get("id", "")
                if em_key and em_key != "error" and not embedding_model:
                    embedding_model = {"providerId": pid, "key": em_key}

        # Prefer local llama-server if it actually has models loaded.
        local_provider = next(
            (p for p in providers if "local" in p.get("id", "").lower()),
            None,
        )
        if local_provider and local_provider.get("chatModels"):
            cm0 = local_provider["chatModels"][0]
            chat_model = {
                "providerId": local_provider["id"],
                "key": cm0.get("key") or cm0.get("name") or cm0.get("id", "local-model"),
            }
        elif all_chat_models:
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

        # Don't cache failures — allow retry on next call
        logger.warning(
            "perplexica: missing models",
            has_chat=chat_model is not None,
            has_embedding=embedding_model is not None,
        )
        return None

    except Exception as e:
        logger.warning("perplexica provider discovery error", error=str(e))
        return None


async def _search_searxng_direct(
    query: str, max_results: int = 10
) -> str | None:
    """Search SearXNG directly, bypassing Vane's LLM synthesis.

    SearXNG runs inside the Vane Docker container. When exposed on a
    separate port (SEARXNG_URL), we can query it for raw search results
    in 6-10s instead of waiting 40-75s for Vane's full LLM pipeline.

    Falls back gracefully if SearXNG is not exposed or unavailable.
    """
    searxng_url = os.getenv("SEARXNG_URL", "").strip()
    if not searxng_url:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "engines": "duckduckgo,google,bing,brave",
                },
                timeout=aiohttp.ClientTimeout(total=12),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                results = data.get("results", [])
                if not results:
                    return None

                # Format as readable text for the agent
                parts = []
                for r in results[:max_results]:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = r.get("content", "")[:300]
                    if title and url:
                        parts.append(f"**{title}**\n{snippet}\n{url}")

                if not parts:
                    return None

                formatted = "\n\n".join(parts)
                logger.info(
                    "searxng direct search ok",
                    result_count=len(parts),
                    query=query[:50],
                )
                return formatted

    except asyncio.TimeoutError:
        logger.debug("searxng direct search timeout (12s)", query=query[:50])
    except Exception as e:
        logger.debug("searxng direct search failed", error=str(e))
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

    # Skip if too many consecutive failures (re-enable after 5 min)
    global _perplexica_disabled_at
    import time as _time
    if _perplexica_fail_count >= _PERPLEXICA_MAX_FAILURES:
        if _perplexica_disabled_at == 0:
            _perplexica_disabled_at = _time.time()
        if _time.time() - _perplexica_disabled_at < _PERPLEXICA_RETRY_AFTER:
            logger.debug("perplexica: disabled after repeated failures, using fallback")
            return None
        else:
            # Reset and retry
            _perplexica_fail_count = 0
            _perplexica_disabled_at = 0.0
            logger.info("perplexica: re-enabling after cooldown")

    # Skip Perplexica if loaded model is too slow or uses thinking mode.
    # Perplexica's LLM synthesis takes 50-70s even with fast models;
    # slow or thinking models would take 500s+ and always timeout.
    _MIN_PERPLEXICA_TPS = 5.0
    try:
        from src.core.llm_dispatcher import get_dispatcher
        dispatcher = get_dispatcher()
        model_speed = dispatcher.get_loaded_model_speed()
        is_thinking = dispatcher.is_loaded_model_thinking()
        if model_speed > 0 and model_speed < _MIN_PERPLEXICA_TPS:
            logger.debug(
                "perplexica: skipping, model too slow",
                speed=f"{model_speed:.1f}",
                min_required=_MIN_PERPLEXICA_TPS,
            )
            return None
        if is_thinking:
            logger.debug("perplexica: skipping, thinking model wastes tokens")
            return None
    except Exception:
        pass  # can't check speed — proceed anyway

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
                timeout=aiohttp.ClientTimeout(total=45),
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

                # Quality gate: reject "I don't know" answers
                if not sources:  # 0 sources = Perplexica's SearXNG found nothing useful
                    _perplexica_fail_count += 1
                    logger.debug("perplexica: rejecting answer with 0 sources", answer_preview=answer[:100])
                    return None

                answer_lower = answer.lower()
                if any(phrase in answer_lower for phrase in _NO_DATA_PHRASES):
                    _perplexica_fail_count += 1
                    logger.debug("perplexica: rejecting 'no data' answer", answer_preview=answer[:100])
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
        logger.warning("perplexica search timeout (45s)", query=query)
    except Exception as e:
        _perplexica_fail_count += 1
        logger.warning("perplexica search error", error=f"{type(e).__name__}: {e}", query=query)
        return None


async def _embed_web_results(query: str, results_text: str) -> None:
    """Phase D: Embed web search results into web_knowledge collection."""
    try:
        from src.memory.vector_store import embed_and_store, is_ready
        if not is_ready():
            return

        import hashlib
        import time as _t

        # Truncate to reasonable size for embedding
        text = f"Web search: {query}\n{results_text[:1500]}"
        doc_id = f"web-{hashlib.sha256(f'{query}:{_t.time()}'.encode()).hexdigest()[:16]}"

        await embed_and_store(
            text=text,
            metadata={
                "data_type": "web_result",
                "query": query[:200],
                "source_url": "",
                "timestamp": _t.time(),
                "ttl_days": 7,
            },
            collection="web_knowledge",
            doc_id=doc_id,
        )
        logger.debug("Embedded web search results for: %s", query[:60])
    except Exception as e:
        logger.debug("Web result embedding skipped: %s", e)


async def web_search(query: str, max_results: int = 5, search_type: str = "web") -> str:
    """
    Search the web using Perplexica/Vane (primary), with SearXNG direct and DuckDuckGo fallbacks.

    Fallback chain: Perplexica (45s, AI-synthesized) → SearXNG direct (12s, raw results)
    → DuckDuckGo package → curl/DuckDuckGo API.

    Before issuing a live search, checks the web_knowledge vector store
    for recent relevant cached results (Phase D knowledge accumulation).

    search_type: "web", "academic", or "code"
    """
    logger.info("web search query", query=query, max_results=max_results, search_type=search_type)

    # Phase D: Check existing web knowledge before live search
    try:
        from src.memory.vector_store import query as vquery, is_ready as vs_ready
        import time as _t
        if vs_ready():
            cached = await vquery(
                text=query,
                collection="web_knowledge",
                top_k=3,
            )
            # Use cached results if they are recent (< 12 hours) and relevant
            fresh_results = [
                r for r in cached
                if r.get("distance", 1.0) < 0.5
                and (_t.time() - r.get("metadata", {}).get("timestamp", 0)) < 43200
            ]
            if fresh_results:
                lines = [f"**Cached web knowledge for '{query}':**\n"]
                for r in fresh_results:
                    text = r.get("text", "")[:500]
                    lines.append(f"- {text}")
                lines.append(
                    "\n[Retrieved from cached web knowledge. "
                    "The information may be up to 12 hours old.]"
                )
                logger.debug("Returning cached web knowledge for: %s", query[:60])
                return "\n".join(lines)
    except Exception:
        pass  # Fall through to live search

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
            lines = [
                "## AI-Synthesized Answer (from Perplexica)\n",
                perplexica_result["answer"],
            ]
            if perplexica_result["sources"]:
                lines.append("\n### Sources")
                for i, src in enumerate(perplexica_result["sources"], 1):
                    title = src.get("title", "Untitled")
                    url = src.get("url", "")
                    lines.append(f"- [{title}]({url})")
            lines.append(
                "\n**Note: This answer is already synthesized from multiple "
                "sources. Use it as your final answer unless something "
                "specific is missing.**"
            )
            result_text = "\n".join(lines)

            # Phase D: Embed Perplexica results (comprehensive, higher value)
            await _embed_web_results(query, result_text)

            return result_text

    # Method 1: SearXNG direct (raw results, no LLM synthesis, ~6-10s)
    searxng_result = await _search_searxng_direct(query, max_results)
    if searxng_result:
        # Embed in background (non-blocking)
        asyncio.ensure_future(_embed_web_results(query, searxng_result))
        return searxng_result

    # Method 3: duckduckgo-search package (ddgs 9.x)
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
                result_text = f"Search results for '{query}':\n\n" + "\n\n".join(lines)

                # Phase D: Embed DDG results
                await _embed_web_results(query, result_text)

                return result_text
            else:
                return f"No results found for '{query}'"
        except Exception as e:
            logger.warning("duckduckgo search failed, using curl fallback", error=str(e))

    # Method 4: curl via shell (Docker container has internet)
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
