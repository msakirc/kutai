# Deep Search Integration — Design Spec

## Goal

Enhance `web_search` with tiered scraping, smart content extraction, and adaptive context budgeting so agents get depth-appropriate results without making search decisions themselves.

## Architecture Overview

Single `web_search` tool with auto-escalating fetch tiers, intent inferred from agent type + task context + classifier hint, and BM25-based adaptive content budgeting to fit results into the agent's context window.

```
User message
    │
    ▼
Task Classifier (adds search_depth: quick|standard|deep)
    │
    ▼
Agent calls web_search(query)  ← agent has NO depth/intent params
    │
    ▼
web_search orchestrator
    ├── 1. Infer intent from agent_type + shopping_sub_intent + search_depth
    ├── 2. Check ChromaDB cache
    ├── 3. Fetch via tiered scrapers (auto-escalate on failure)
    ├── 4. Extract content (Trafilatura)
    ├── 5. Score relevance (BM25) + allocate context budget
    ├── 6. Cache full content in ChromaDB (side effect)
    └── 7. Return budget-fitted results to agent
```

---

## Component Design

### 1. Intent Inference (`_infer_search_intent`)

**Location:** Inside `web_search()`, not agent-facing.

**Inputs:** `agent_type` (from task context), `shopping_sub_intent` (from task classifier), `search_depth` (from task classifier).

**Logic:**

```
search_depth (from classifier) takes priority when present:
  "deep"     → intent=research, min_pages=10, max_chars_per_page=3000
  "standard" → intent=product,  min_pages=5,  max_chars_per_page=2000
  "quick"    → intent=factual,  min_pages=3,  max_chars_per_page=1500

Otherwise, infer from agent_type + shopping_sub_intent:
  shopping_sub_intent in (research, exploration)        → market
  shopping_sub_intent in (compare, price_check, deal_hunt, upgrade) → product
  shopping_sub_intent in (purchase_advice, complaint)   → reviews
  agent_type in (researcher, analyst)                   → research
  agent_type in (shopping_advisor, product_researcher)  → product
  agent_type = deal_analyst                             → market
  everything else                                       → factual
```

**5 intents, each maps to extraction parameters:**

| Intent | Min Pages | Max Chars/Page | BM25 Bias | Depth Floor |
|--------|-----------|----------------|-----------|-------------|
| `factual` | 3 | 1500 | None | quick |
| `product` | 5 | 2000 | Price/spec patterns | standard |
| `reviews` | 8 | 2500 | Review/opinion content | deep |
| `market` | 10 | 3000 | Broad coverage | deep |
| `research` | 10 | 3000 | Competing perspectives | deep |

### 2. Task Classifier Enhancement

**File:** `src/core/task_classifier.py`

Add `search_depth` to the classification JSON schema:

```json
{
  "agent_type": "assistant",
  "difficulty": 4,
  "search_depth": "standard",
  ...
}
```

Classifier prompt addition (~3 lines):
```
search_depth: how much web research is needed?
  "quick" — simple fact, definition, date, status check
  "standard" — product info, comparison, how-to with examples
  "deep" — market analysis, multi-source research, review synthesis
  "none" — no web search needed
```

Keyword fallback heuristic:
- Query > 15 words or contains "analyze", "research", "in detail" → `deep`
- Contains "price", "fiyat", "vs", "compare", "review" → `standard`
- Default → `quick`

### 3. Tiered Scraper (`src/tools/scraper.py`)

**New standalone module.** Reusable by any tool or agent.

**Public API:**
```python
async def scrape_url(url: str, tier: str = "auto") -> ScrapeResult | None
async def scrape_urls(urls: list[str], tier: str = "auto", max_concurrent: int = 3) -> dict[str, ScrapeResult]
```

**`ScrapeResult` dataclass:**
```python
@dataclass
class ScrapeResult:
    url: str
    html: str          # raw HTML
    status: int        # HTTP status
    tier_used: str     # which tier succeeded
    fetch_time: float  # seconds
```

**Tiers (auto-escalate on failure):**

| Tier | Backend | RAM | When Used |
|------|---------|-----|-----------|
| `http` | aiohttp (existing `page_fetch.py`) | ~0 | Default first attempt |
| `tls` | curl_cffi | ~10-30MB | HTTP fails with 403/cloudflare |
| `stealth` | Scrapling StealthyFetcher | 300-500MB | TLS tier gets blocked |
| `browser` | Scrapling DynamicFetcher | 500-800MB | JS rendering needed |

**Auto-escalation logic:**
```
Try tier = max(requested_minimum, "http")
If 403/cloudflare-challenge/empty → try next tier up
If all tiers fail → return None
```

**On-demand lifecycle:** Browser tiers spin up per-request and shut down after. Zero idle RAM.

**Dependencies:** `curl_cffi`, `scrapling[all]` (installed on demand — start with curl_cffi only, add scrapling when needed).

### 4. Content Extractor (`src/tools/content_extract.py`)

**New standalone module.** Replaces the simple `extract_main_text()` in `page_fetch.py` for deep searches.

**Public API:**
```python
def extract_content(html: str, url: str = "") -> ExtractedContent
```

**`ExtractedContent` dataclass:**
```python
@dataclass
class ExtractedContent:
    text: str              # main body text (Trafilatura)
    title: str             # page title
    url: str               # source URL
    word_count: int        # for budget allocation
    has_prices: bool       # detected price patterns
    has_reviews: bool      # detected review patterns
    content_type: str      # "article" | "product" | "forum" | "listing" | "unknown"
```

**Implementation:**
- Primary: Trafilatura `extract()` with `include_tables=True`, `include_comments=True`
- Fallback: Existing BeautifulSoup extraction from `page_fetch.py`
- Content type detection: Heuristic based on HTML structure (product schema, review schema, forum patterns)
- Price detection: Regex patterns for TL/USD/EUR (reuse from `shopping/integrations/perplexica.py`)

**Dependencies:** `trafilatura` (new), `lxml` (existing).

### 5. Relevance Scorer + Budget Allocator (`src/tools/relevance.py`)

**New standalone module.**

**Public API:**
```python
def score_and_budget(
    contents: list[ExtractedContent],
    query: str,
    total_budget: int = 12000,  # chars
    intent: str = "factual",
) -> list[BudgetedContent]
```

**`BudgetedContent` dataclass:**
```python
@dataclass
class BudgetedContent:
    content: ExtractedContent
    relevance_score: float     # 0-1 from BM25
    allocated_chars: int       # budget for this page
    truncated_text: str        # text cut to budget, sentence-boundary
```

**Scoring logic:**
1. BM25 scores each document against the query
2. Intent biases adjust scores:
   - `product` intent: +0.2 boost for pages with `has_prices=True`
   - `reviews` intent: +0.2 boost for pages with `has_reviews=True`
   - `market`/`research`: no bias (want broad coverage)
3. Budget allocation: proportional to score, with minimum floor (200 chars) and maximum cap (40% of total budget per page)
4. Truncation: cut on sentence boundary, not mid-word

**Dependencies:** `bm25s` (new, lightweight — numpy/scipy only).

### 6. Web Search Orchestrator (`src/tools/web_search.py`)

**Modified existing module.** The `web_search()` function orchestrates the pipeline.

**New flow (replaces current Method 1 ddgs block):**

```python
async def web_search(query, max_results=5, search_type="web"):
    # 1. Infer intent from task context
    intent = _infer_search_intent()
    params = INTENT_PARAMS[intent]

    # 2. Check ChromaDB cache (existing, unchanged)
    cached = _check_cache(query)
    if cached: return cached

    # 3. ddgs search for URLs + snippets
    results = _DDGS().text(query, max_results=params.max_results)
    if not results:
        # fall through to Perplexica/SearXNG/curl (existing fallbacks)
        ...

    # 4. Fetch pages via tiered scraper
    urls = [r["href"] for r in results if r.get("href")]
    scrape_results = await scrape_urls(urls, max_concurrent=3)

    # 5. Extract content
    contents = [extract_content(sr.html, sr.url) for sr in scrape_results.values()]

    # 6. Score relevance + allocate budget
    budgeted = score_and_budget(contents, query, total_budget=params.total_budget, intent=intent)

    # 7. Format output
    result_text = _format_results(results, budgeted)

    # 8. Cache in ChromaDB (side effect, existing)
    await _embed_web_results(query, result_text)

    return result_text
```

**The existing fallback chain (Perplexica → SearXNG → curl) remains** as a last resort when ddgs fails entirely.

**Quick searches (intent=factual) use the existing fast path** — ddgs + page_fetch.py (no Trafilatura, no BM25). The new pipeline only activates for `standard` or `deep` intents.

### 7. Existing `page_fetch.py` — Unchanged

Stays as-is for Tier 0 (quick) fetches. The new `scraper.py` handles deeper tiers. `page_fetch.py` remains a standalone reusable tool.

---

## Agent-Facing API

**No changes to the tool signature or description.** Agents continue calling:

```json
{"action": "tool_call", "tool": "web_search", "args": {"query": "..."}}
```

The tool internally decides depth based on context. This is the key design decision — zero LLM choice = zero LLM error.

The tool description stays simple:
```
web_search: Search the web. Args: query (str)
```

---

## New Dependencies

| Package | Size | Purpose | Install |
|---------|------|---------|---------|
| `trafilatura` | ~5MB | Content extraction | `pip install trafilatura` |
| `bm25s` | ~50KB | Relevance scoring | `pip install bm25s` |
| `curl_cffi` | ~15MB | TLS fingerprint bypass (Tier 1) | `pip install curl_cffi` |
| `scrapling` | ~2MB + browsers | Stealth/browser scraping (Tier 2-3) | `pip install "scrapling[all]"` (deferred) |

**Phased installation:** Start with `trafilatura`, `bm25s`, `curl_cffi`. Add `scrapling` later when Tier 2-3 are implemented.

---

## File Structure

```
src/tools/
├── web_search.py         # MODIFIED: orchestrates pipeline, infer intent
├── page_fetch.py         # EXISTING: unchanged, Tier 0 quick fetch
├── scraper.py            # NEW: tiered scraping (curl_cffi → scrapling)
├── content_extract.py    # NEW: Trafilatura extraction + content typing
├── relevance.py          # NEW: BM25 scoring + budget allocation

src/core/
├── task_classifier.py    # MODIFIED: add search_depth field
```

---

## What Changes for Existing Functionality

| Component | Change | Risk |
|-----------|--------|------|
| `web_search()` | New orchestration logic, existing fallbacks preserved | Low — fallbacks unchanged |
| `page_fetch.py` | None | Zero |
| Task classifier | One new field in JSON output | Low — additive only |
| Agent prompts | None | Zero |
| Tool registry | None | Zero |
| Shopping integrations | None — `perplexica.py` unchanged | Zero |
| ChromaDB caching | Same `web_knowledge` collection, same embedding | Zero |

---

## Performance Estimates

| Intent | Fetch Time | Processing Time | Total | Pages | Output Size |
|--------|-----------|----------------|-------|-------|-------------|
| `factual` | 2-5s | ~100ms | ~3-5s | 3 | 2-5K chars |
| `product` | 5-10s | ~300ms | ~5-10s | 5 | 5-10K chars |
| `reviews` | 8-15s | ~400ms | ~9-16s | 8 | 8-15K chars |
| `market` | 10-20s | ~550ms | ~11-21s | 10 | 10-20K chars |
| `research` | 10-20s | ~550ms | ~11-21s | 10 | 10-20K chars |

**RAM impact:** Zero when idle. During deep search: ~50MB (Trafilatura + BM25). Tier 2-3 scraping adds 300-800MB for the duration of the fetch only.

---

## Implementation Phases

**Phase 1 — Core pipeline (no new scraper tiers):**
- `content_extract.py` (Trafilatura)
- `relevance.py` (BM25 + budgeting)
- Modify `web_search.py` to use the pipeline for standard/deep intents
- Modify `task_classifier.py` to add `search_depth`
- Dependencies: `trafilatura`, `bm25s`

**Phase 2 — Tiered scraper:**
- `scraper.py` with `http` and `tls` tiers
- Dependency: `curl_cffi`

**Phase 3 — Anti-bot tiers:**
- Add `stealth` and `browser` tiers to `scraper.py`
- Dependency: `scrapling[all]`

Each phase is independently deployable and testable. Phase 1 alone delivers 80% of the value.

---

## What This Does NOT Cover

- Perplexica on-demand Docker boot (deferred from earlier discussion — separate future work)
- Shopping scraper changes (existing scrapers in `src/shopping/scrapers/` are unchanged)
- Multi-query workflows (agent can call web_search multiple times; orchestration is the agent's job)
- CAPTCHA solving (out of scope — if CAPTCHA is hit, tier fails and escalates)
