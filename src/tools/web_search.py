# tools/web_search.py
"""
Web search using DuckDuckGo + page content fetch (primary).
Fallback chain: ddgs+pages → Perplexica/Vane → SearXNG direct → curl.

Supports two pipelines:
- **Quick** (factual queries): page_fetch + simple snippet formatting.
- **Deep** (product/review/research queries): page_fetch → Trafilatura
  extraction → BM25 relevance scoring → budget-allocated output.

Pipeline selection is automatic based on task hints (search_depth,
shopping_sub_intent, agent_type).
"""
import asyncio
import json
import os
import urllib.parse
from dataclasses import dataclass

import aiohttp

from src.infra.logging_config import get_logger
from src.tools import run_shell

logger = get_logger("tools.web_search")


# ---------------------------------------------------------------------------
# Intent inference — maps task hints to search parameters
# ---------------------------------------------------------------------------

@dataclass
class _SearchParams:
    max_results: int
    max_chars_per_page: int
    total_budget: int
    use_deep_pipeline: bool

_INTENT_PARAMS = {
    "factual":  _SearchParams(max_results=5,  max_chars_per_page=1500, total_budget=5000,  use_deep_pipeline=False),
    "product":  _SearchParams(max_results=7,  max_chars_per_page=2000, total_budget=10000, use_deep_pipeline=True),
    "reviews":  _SearchParams(max_results=8,  max_chars_per_page=2500, total_budget=15000, use_deep_pipeline=True),
    "market":   _SearchParams(max_results=10, max_chars_per_page=3000, total_budget=20000, use_deep_pipeline=True),
    "research": _SearchParams(max_results=10, max_chars_per_page=3000, total_budget=20000, use_deep_pipeline=True),
}


def _infer_search_intent(hints: dict) -> tuple[str, _SearchParams]:
    """Infer search intent from task context. Returns (intent_name, params)."""
    depth = hints.get("search_depth")
    if depth == "deep":
        return "research", _INTENT_PARAMS["research"]
    if depth == "standard":
        return "product", _INTENT_PARAMS["product"]
    if depth == "quick":
        return "factual", _INTENT_PARAMS["factual"]

    sub = hints.get("shopping_sub_intent")
    if sub in ("research", "exploration"):
        return "market", _INTENT_PARAMS["market"]
    if sub in ("compare", "price_check", "deal_hunt", "upgrade"):
        return "product", _INTENT_PARAMS["product"]
    if sub in ("purchase_advice", "complaint_return_help"):
        return "reviews", _INTENT_PARAMS["reviews"]

    agent = hints.get("agent_type", "")
    agent_map = {
        "researcher": "research",
        "analyst": "research",
        "deal_analyst": "market",
        "shopping_advisor": "product",
        "product_researcher": "product",
    }
    intent = agent_map.get(agent, "factual")
    return intent, _INTENT_PARAMS[intent]

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
                    "language": "en",
                    "engines": "duckduckgo,google,bing,brave,wikipedia",
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


# ---------------------------------------------------------------------------
# Quick and deep search pipelines
# ---------------------------------------------------------------------------

async def _quick_search_pipeline(query: str, ddgs_results: list, urls: list) -> str:
    """Existing fast path: page_fetch + simple format."""
    page_contents = {}
    if urls:
        try:
            from src.tools.page_fetch import fetch_pages
            page_contents = await fetch_pages(urls, max_pages=3, max_chars=1500)
            logger.debug("page_fetch: fetched pages", count=len(page_contents))
        except Exception as e:
            logger.debug("page_fetch: skipped", error=str(e)[:100])

    lines = []
    for i, r in enumerate(ddgs_results, 1):
        title = r.get("title", "No title")
        body = r.get("body", "")[:200]
        href = r.get("href", "")
        parts = [f"{i}. **{title}**\n   {body}\n   {href}"]
        if href in page_contents:
            parts.append(f"   ---\n   {page_contents[href]}")
        lines.append("\n".join(parts))

    return f"Search results for '{query}':\n\n" + "\n\n".join(lines)


_INTENT_TIER_MAP = {
    "factual": 0,   # HTTP only — fast, no escalation needed
    "product": 1,   # up to TLS fingerprinting
    "reviews": 1,   # up to TLS fingerprinting
    "market": 2,    # up to stealth (if available)
    "research": 2,  # up to stealth (if available)
}


async def _deep_search_pipeline(
    query: str, ddgs_results: list, urls: list, intent: str, params: _SearchParams
) -> str:
    """Deep path: fetch pages -> Trafilatura -> BM25 -> budget allocation."""
    from src.tools.page_fetch import fetch_pages
    from src.tools.content_extract import extract_content
    from src.tools.relevance import score_and_budget

    max_tier = _INTENT_TIER_MAP.get(intent, 1)

    # Fetch more pages with more content for deep search
    page_htmls = await fetch_pages(urls, max_pages=params.max_results, max_chars=50000, max_tier=max_tier)
    logger.debug("deep pipeline: fetched pages", count=len(page_htmls))

    if not page_htmls:
        logger.debug("deep pipeline: no pages fetched, falling back to quick")
        return await _quick_search_pipeline(query, ddgs_results, urls)

    # Extract content with Trafilatura
    contents = []
    for url, html in page_htmls.items():
        extracted = extract_content(html, url=url)
        if extracted.text and extracted.word_count > 10:
            contents.append(extracted)

    if not contents:
        logger.debug("deep pipeline: no content extracted, falling back to quick")
        return await _quick_search_pipeline(query, ddgs_results, urls)

    # Score relevance and allocate budgets
    budgeted = score_and_budget(contents, query, total_budget=params.total_budget, intent=intent)

    # Format output
    lines = []
    for i, r in enumerate(ddgs_results, 1):
        title = r.get("title", "No title")
        body = r.get("body", "")[:150]
        href = r.get("href", "")
        lines.append(f"{i}. **{title}** — {body} ({href})")

    lines.append("\n---\n**Detailed content (by relevance):**\n")
    for b in budgeted:
        if not b.truncated_text:
            continue
        title = b.content.title or b.content.url.split("/")[-1] or "Untitled"
        tags = []
        if b.content.has_prices:
            tags.append("prices")
        if b.content.has_reviews:
            tags.append("reviews")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"### {title}{tag_str}")
        lines.append(f"Source: {b.content.url}")
        lines.append(b.truncated_text)
        lines.append("")

    return "\n".join(lines)


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


async def web_search(query: str, max_results: int = 5, search_type: str = "web", _task_hints: dict | None = None) -> str:
    """
    Search the web using DuckDuckGo + page fetch (primary), with Perplexica and curl fallbacks.

    Fallback chain: ddgs+page_fetch (primary) → Perplexica/Vane (AI-synthesized)
    → SearXNG direct (raw results) → curl/DuckDuckGo API.

    Pipeline selection (quick vs deep) is automatic based on ``_task_hints``:
    - **Quick**: factual queries — snippet + page_fetch formatting.
    - **Deep**: product/review/research queries — Trafilatura extraction,
      BM25 relevance scoring, and budget-allocated output.

    Before issuing a live search, checks the web_knowledge vector store
    for recent relevant cached results (Phase D knowledge accumulation).

    search_type: "web", "academic", or "code"
    """
    hints = _task_hints or {}
    intent, params = _infer_search_intent(hints)
    effective_max = max(max_results, params.max_results)

    logger.info("web search query", query=query, max_results=effective_max, intent=intent, search_type=search_type)

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

    # Method 1 (primary): DuckDuckGo + page fetch (quick or deep)
    if _DDGS is not None:
        try:
            results = _DDGS().text(query, max_results=effective_max)
            if results:
                logger.debug("ddgs search ok", count=len(results), intent=intent)
                urls = [r.get("href", "") for r in results if r.get("href")]

                if params.use_deep_pipeline and urls:
                    result_text = await _deep_search_pipeline(query, results, urls, intent, params)
                else:
                    result_text = await _quick_search_pipeline(query, results, urls)

                await _embed_web_results(query, result_text)
                return result_text
        except Exception as e:
            logger.warning("ddgs primary search failed", error=str(e))

    # Method 2 (fallback): Perplexica/Vane AI synthesis
    if not is_degraded:
        perplexica_result = await _search_perplexica(query, max_results, search_type)
        if perplexica_result:
            logger.debug("using perplexica fallback for web search")
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
            await _embed_web_results(query, result_text)
            return result_text

    # Method 3 (fallback): SearXNG direct (raw results, no LLM)
    searxng_result = await _search_searxng_direct(query, max_results)
    if searxng_result and searxng_result.count("**") >= 6:
        asyncio.ensure_future(_embed_web_results(query, searxng_result))
        return searxng_result

    # Method 4 (last resort): curl DuckDuckGo
    try:
        safe_query = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={safe_query}&format=json&no_html=1&no_redirect=1"

        result = await run_shell(
            f'curl -s --max-time 10 "{url}"',
            timeout=15,
        )

        if result.startswith("\u2705"):
            result = result[1:].strip()

        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return f"Search returned non-JSON response for '{query}':\n{result[:1000]}"

        lines = []
        if data.get("Abstract"):
            lines.append(f"**Summary:** {data['Abstract']}")
            if data.get("AbstractURL"):
                lines.append(f"Source: {data['AbstractURL']}")

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                text = topic["Text"][:200]
                url = topic.get("FirstURL", "")
                lines.append(f"- {text}\n  {url}")

        if lines:
            return f"Search results for '{query}':\n\n" + "\n\n".join(lines)

        scrape_url = f"https://html.duckduckgo.com/html/?q={safe_query}"
        scrape_result = await run_shell(
            f'curl -s --max-time 10 "{scrape_url}" | grep -oP \'<a rel="nofollow" class="result__a" href="[^"]*">[^<]*</a>\' | head -5',
            timeout=15,
        )

        if scrape_result.startswith("\u2705"):
            scrape_result = scrape_result[1:].strip()

        if scrape_result and "\u274c" not in scrape_result:
            return f"Search results for '{query}':\n\n{scrape_result}"

        return f"No results found for '{query}'. All search backends failed."

    except Exception as e:
        logger.exception("web search all backends failed", error=str(e))
        return f"Search error: {e}"
