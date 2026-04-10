# tools/web_search.py
"""
Web search using DuckDuckGo + page content fetch (primary).
Fallback chain: ddgs+pages → Brave → Google CSE → Perplexica/Vane → SearXNG direct → curl.

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

import logging

logger = logging.getLogger(__name__)

# Injectable shell executor — lazy import from src.tools by default
_shell_fn = None


def _get_shell_fn():
    global _shell_fn
    if _shell_fn is None:
        from src.tools import run_shell
        _shell_fn = run_shell
    return _shell_fn


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



async def _search_brave(query: str, max_results: int = 5) -> list[dict] | None:
    """Search using Brave Search API (free tier: 2000 queries/month, 1 qps).

    Returns ddgs-compatible list of dicts with keys: title, body, href.
    Returns None if no API key or on any error (graceful skip).
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                },
                params={"q": query, "count": str(max_results)},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"brave search failed status={resp.status}")
                    return None
                data = await resp.json()

        web_results = data.get("web", {}).get("results", [])
        if not web_results:
            return None

        # Convert to ddgs-compatible format
        results = []
        for r in web_results[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "body": r.get("description", ""),
                "href": r.get("url", ""),
            })

        logger.info(f"brave search ok count={len(results)} query={query[:50]}")
        return results

    except asyncio.TimeoutError:
        logger.debug(f"brave search timeout (10s) query={query[:50]}")
    except Exception as e:
        logger.debug(f"brave search failed: {e}")
    return None


async def _search_google_cse(query: str, max_results: int = 10) -> list[dict] | None:
    """Search using Google Custom Search Engine.

    Free tier: 100 queries/day. Returns None if not configured.
    API: https://www.googleapis.com/customsearch/v1
    Params: key, cx, q, num
    """
    api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
    cx = os.getenv("GOOGLE_CSE_CX", "").strip()
    if not api_key or not cx:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": api_key,
                    "cx": cx,
                    "q": query,
                    "num": str(min(max_results, 10)),
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"google cse search failed status={resp.status}")
                    return None
                data = await resp.json()

        items = data.get("items", [])
        if not items:
            return None

        results = []
        for item in items[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "body": item.get("snippet", ""),
                "href": item.get("link", ""),
            })

        logger.info(f"google cse search ok count={len(results)} query={query[:50]}")
        return results

    except asyncio.TimeoutError:
        logger.debug(f"google cse search timeout (10s) query={query[:50]}")
    except Exception as e:
        logger.debug(f"google cse search failed: {e}")
    return None




# ---------------------------------------------------------------------------
# Source quality tracking helpers
# ---------------------------------------------------------------------------


async def _reorder_urls_by_quality(urls: list[str]) -> list[str]:
    """Reorder URLs putting known-good domains first."""
    from src.infra.db import get_source_quality

    domains = list({urllib.parse.urlparse(u).netloc for u in urls})
    quality = await get_source_quality(domains)

    def score(url):
        domain = urllib.parse.urlparse(url).netloc
        q = quality.get(domain)
        if not q:
            return 0.5  # unknown = neutral
        total = q["success_count"] + q["fail_count"] + q["block_count"]
        if total == 0:
            return 0.5
        success_rate = q["success_count"] / total
        return success_rate * 0.7 + min(q["avg_relevance"], 1.0) * 0.3

    return sorted(urls, key=score, reverse=True)


def _record_fetch_quality_fire_and_forget(
    fetched_urls: dict[str, str],
    all_urls: list[str],
    relevance_scores: dict[str, float] | None = None,
) -> None:
    """Record source quality for fetched pages (fire-and-forget).

    fetched_urls: {url: text} for successfully fetched pages
    all_urls: all URLs that were attempted
    relevance_scores: optional {url: bm25_score} from deep pipeline
    """
    async def _do_record():
        from src.infra.db import record_source_quality

        for url in all_urls:
            domain = urllib.parse.urlparse(url).netloc
            if not domain:
                continue
            if url in fetched_urls:
                text = fetched_urls[url]
                rel = (relevance_scores or {}).get(url, 0.0)
                await record_source_quality(domain, success=True, relevance=rel)
            else:
                # We don't know if it was blocked or just failed — record as fail
                await record_source_quality(domain, success=False)

    asyncio.ensure_future(_do_record())


# ---------------------------------------------------------------------------
# Quick and deep search pipelines
# ---------------------------------------------------------------------------

async def _quick_search_pipeline(query: str, ddgs_results: list, urls: list) -> str:
    """Existing fast path: page_fetch + simple format."""
    page_contents = {}
    if urls:
        try:
            urls = await _reorder_urls_by_quality(urls)
        except Exception:
            pass  # quality DB not available yet — use original order
        try:
            from src.tools.page_fetch import fetch_pages
            page_contents = await fetch_pages(urls, max_pages=3, max_chars=1500)
            logger.debug(f"page_fetch: fetched pages count={len(page_contents)}")
        except Exception as e:
            logger.debug(f"page_fetch: skipped error={str(e)[:100]}")

        # Record quality (fire-and-forget)
        _record_fetch_quality_fire_and_forget(page_contents, urls[:3])

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
    """Deep path: fetch pages -> BM25 relevance -> budget allocation."""
    from src.tools.page_fetch import fetch_pages
    from src.tools.relevance import score_and_budget

    max_tier = _INTENT_TIER_MAP.get(intent, 1)

    # Reorder URLs by domain quality before fetching
    try:
        urls = await _reorder_urls_by_quality(urls)
    except Exception:
        pass  # quality DB not available yet — use original order

    # Fetch more pages with more content for deep search
    page_htmls = await fetch_pages(urls, max_pages=params.max_results, max_chars=50000, max_tier=max_tier)
    logger.debug(f"deep pipeline: fetched pages count={len(page_htmls)}")

    if not page_htmls:
        logger.debug("deep pipeline: no pages fetched, falling back to quick")
        return await _quick_search_pipeline(query, ddgs_results, urls)

    # Build ExtractedContent from pre-extracted text.
    # fetch_pages already extracts via scraper → BeautifulSoup/extract_main_text,
    # so the values are plain text, NOT HTML. Running trafilatura on plain text
    # causes "parsed tree length: 0" errors.
    from src.tools.content_extract import ExtractedContent, _PRICE_PATTERNS, _REVIEW_PATTERNS
    contents = []
    for url, text in page_htmls.items():
        if not text:
            continue
        word_count = len(text.split())
        if word_count > 10:
            contents.append(ExtractedContent(
                text=text, url=url, word_count=word_count,
                has_prices=any(p.search(text) for p in _PRICE_PATTERNS),
                has_reviews=any(p.search(text) for p in _REVIEW_PATTERNS),
            ))

    if not contents:
        logger.debug("deep pipeline: no content extracted, falling back to quick")
        return await _quick_search_pipeline(query, ddgs_results, urls)

    # Score relevance and allocate budgets
    budgeted = score_and_budget(contents, query, total_budget=params.total_budget, intent=intent)

    # Record quality with relevance scores (fire-and-forget)
    relevance_scores = {b.content.url: b.relevance_score for b in budgeted}
    _record_fetch_quality_fire_and_forget(
        page_htmls, urls[:params.max_results], relevance_scores=relevance_scores
    )

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

    Fallback chain: ddgs+page_fetch (primary) → Brave Search → Google CSE
    → Perplexica/Vane (AI-synthesized) → SearXNG direct (raw results)
    → curl/DuckDuckGo API.

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

    logger.info(f"web search query={query} max_results={effective_max} intent={intent} search_type={search_type}")

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
            # Use cached results if recent (< 12h), close in embedding
            # space, AND topically related.  Embedding distance alone is
            # unreliable: "coffee machine price" matched "GDPR compliance"
            # at dist 0.4.  We strip stopwords, then require ≥30% of the
            # query's content words to appear in the cached query.  This
            # kills cross-topic pollution while still allowing paraphrase
            # matches like "KVKK uyum" ↔ "KVKK compliance requirements".
            _STOP = {
                "a","an","the","and","or","but","in","on","at","to","for",
                "of","with","by","from","is","are","was","were","be","been",
                "has","been","have","had","do","does","did","will","would",
                "can","could","may","might","shall","should","not","no",
                "this","that","these","those","it","its","i","we","you",
                "he","she","they","my","your","our","his","her","their",
                "what","which","who","whom","how","when","where","why",
                "bir","ve","ile","için","de","da","den","dan","bu","şu",
                "ne","nasıl","nerede","kim","mi","mı","mu","mü",
            }
            def _content_words(text):
                return {w for w in text.lower().split() if w not in _STOP and len(w) > 1}

            q_words = _content_words(query)
            fresh_results = []
            for r in cached:
                if r.get("distance", 1.0) >= 0.5:
                    continue
                if (_t.time() - r.get("metadata", {}).get("timestamp", 0)) >= 43200:
                    continue
                cached_query = r.get("metadata", {}).get("query", "")
                c_words = _content_words(cached_query)
                overlap = len(q_words & c_words)
                # Need ≥30% of query content words in cached query
                if not q_words or overlap / len(q_words) < 0.3:
                    continue
                fresh_results.append(r)
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
                logger.debug(f"ddgs search ok count={len(results)} intent={intent}")
                urls = [r.get("href", "") for r in results if r.get("href")]

                if params.use_deep_pipeline and urls:
                    result_text = await _deep_search_pipeline(query, results, urls, intent, params)
                else:
                    result_text = await _quick_search_pipeline(query, results, urls)

                await _embed_web_results(query, result_text)
                return result_text
        except Exception as e:
            logger.warning(f"ddgs primary search failed: {e}")

    # Method 1.5 (secondary): Brave Search API
    brave_results = await _search_brave(query, effective_max)
    if brave_results:
        logger.debug(f"brave search ok, using as fallback count={len(brave_results)}")
        urls = [r.get("href", "") for r in brave_results if r.get("href")]
        if params.use_deep_pipeline and urls:
            result_text = await _deep_search_pipeline(query, brave_results, urls, intent, params)
        else:
            result_text = await _quick_search_pipeline(query, brave_results, urls)
        await _embed_web_results(query, result_text)
        return result_text

    # Method 1.75 (tertiary): Google Custom Search Engine (100 queries/day free)
    gcse_results = await _search_google_cse(query, effective_max)
    if gcse_results:
        logger.debug(f"google cse search ok, using as fallback count={len(gcse_results)}")
        urls = [r.get("href", "") for r in gcse_results if r.get("href")]
        if params.use_deep_pipeline and urls:
            result_text = await _deep_search_pipeline(query, gcse_results, urls, intent, params)
        else:
            result_text = await _quick_search_pipeline(query, gcse_results, urls)
        await _embed_web_results(query, result_text)
        return result_text

    # Method 2 (last resort): curl DuckDuckGo
    try:
        safe_query = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={safe_query}&format=json&no_html=1&no_redirect=1"

        result = await _get_shell_fn()(
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
        scrape_result = await _get_shell_fn()(
            f'curl -s --max-time 10 "{scrape_url}" | grep -oP \'<a rel="nofollow" class="result__a" href="[^"]*">[^<]*</a>\' | head -5',
            timeout=15,
        )

        if scrape_result.startswith("\u2705"):
            scrape_result = scrape_result[1:].strip()

        if scrape_result and "\u274c" not in scrape_result:
            return f"Search results for '{query}':\n\n{scrape_result}"

        return f"No results found for '{query}'. All search backends failed."

    except Exception as e:
        logger.exception(f"web search all backends failed: {e}")
        return f"Search error: {e}"
