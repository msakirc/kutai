# Web Search Architecture X-Ray

> Living document. Covers the web search system as implemented.
> Update as system evolves.

---

## Current Architecture (As-Is)

### Search Pipeline

```
Agent calls web_search(query)
    │
    ▼
Intent Inference (_infer_search_intent)
    │  Inputs: agent_type, shopping_sub_intent, search_depth (from classifier)
    │  Output: intent (factual|product|reviews|market|research) + params
    │
    ▼
ChromaDB Cache Check (web_knowledge collection)
    │  Hit if: distance < 0.5 AND age < 12 hours
    │  Returns cached result immediately if fresh match exists
    │
    ▼
DuckDuckGo Search (ddgs package)
    │  Returns: titles, snippets, URLs
    │
    ├── factual intent ──► Quick Pipeline
    │   │  page_fetch: top 3 URLs, 1500 chars each
    │   │  Output: snippets + page content (~5K chars)
    │   │
    └── product/reviews/market/research intent ──► Deep Pipeline
        │  page_fetch: up to 10 URLs, 50K chars each
        │  Tiered scraper: HTTP → TLS → stealth → browser (auto-escalation)
        │  Trafilatura: extract main content + metadata
        │  BM25: score relevance to query
        │  Budget allocator: distribute chars proportional to relevance
        │  Output: snippets + budgeted content (~8-22K chars)
        │
    ▼
Source Quality Recording (fire-and-forget)
    │  Record success/fail/block per domain in web_source_quality table
    │  URL reordering by domain success rate on next fetch
    │
    ▼
ChromaDB Embed (side effect, async)
    │  Store in web_knowledge for future cache hits
    │
    ▼
Fallback Chain (if ddgs fails)
    ├── Brave Search API (free tier, 2000 queries/month)
    ├── Perplexica/Vane (AI synthesis, 45s timeout)
    ├── SearXNG direct (raw results, 12s timeout)
    └── curl DuckDuckGo API (last resort)
```

### Tiered Scraper (`src/tools/scraper.py`)

The scraper provides four escalation tiers for fetching web pages. Lower
tiers are tried first; on detection of a block, the next tier is tried
automatically up to the configured `max_tier`.

```
HTTP (tier 0)          TLS (tier 1)           STEALTH (tier 2)        BROWSER (tier 3)
aiohttp + UA string    curl_cffi chrome131     Scrapling Camoufox      Scrapling Playwright
0 extra deps           ~10-30 MB RAM           ~300-500 MB on-demand   ~500-800 MB on-demand
timeout: 10s           timeout: 12s            timeout: 25s            timeout: 30s
```

**Auto-escalation logic** (`scrape_url()`):
1. Start at HTTP tier
2. If `_detect_block()` returns True, escalate to next tier
3. If error is NOT a block (timeout, connection error), stop immediately
4. Return first successful result, or last failure if all tiers exhausted

**Block detection** (`_detect_block()`):
- HTTP status 403, 429, 402, 451 → blocked
- HTTP 503 with `server: cloudflare` header → blocked
- HTTP 200 but HTML contains challenge markers (first 2000 chars):
  `"just a moment"`, `"checking your browser"`, `"cdn-cgi/challenge-platform"`,
  `"cf-browser-verification"`, `"attention required"`, `"ray id"`

**Intent → Max Tier mapping** (in `web_search.py`):
| Intent | Max Tier | Rationale |
|--------|----------|-----------|
| factual | HTTP (0) | Fast, no escalation needed |
| product | TLS (1) | Price sites often block plain HTTP |
| reviews | TLS (1) | Review sites often block plain HTTP |
| market | STEALTH (2) | Anti-bot sites need browser fingerprints |
| research | STEALTH (2) | Anti-bot sites need browser fingerprints |

**Integration with page_fetch**: `fetch_page_content()` calls `scrape_url()`
first; falls back to plain aiohttp if the scraper module is unavailable.

### Fallback Chain

| Step | Method | Timeout | Notes |
|------|--------|---------|-------|
| 1 | DuckDuckGo (ddgs) + page fetch | ~10-20s | Primary. Quick or deep pipeline based on intent |
| 1.5 | Brave Search API + page fetch | 10s | Requires `BRAVE_SEARCH_API_KEY` env var. Free tier: 2000 queries/month, 1 qps. Returns ddgs-compatible format |
| 2 | Perplexica/Vane | 45s | AI-synthesized answer. Quality gate rejects 0-source or "no data" responses. Skipped if model too slow (<5 tps) or using thinking mode. Circuit breaker: disabled after 3 consecutive failures, re-enabled after 5 min |
| 3 | SearXNG direct | 12s | Raw results, no LLM synthesis. Requires minimum 3 bold-formatted results |
| 4 | curl DuckDuckGo API | 10s | Last resort. Instant answer API, then HTML scrape |

### Source Quality Tracking

New `web_source_quality` table in SQLite tracks per-domain fetch outcomes:

```sql
CREATE TABLE web_source_quality (
    domain TEXT PRIMARY KEY,
    success_count INTEGER DEFAULT 0,
    fail_count INTEGER DEFAULT 0,
    block_count INTEGER DEFAULT 0,
    avg_relevance REAL DEFAULT 0.0,
    last_success TIMESTAMP,
    last_failure TIMESTAMP,
    last_block TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Recording**: After every page fetch (quick or deep pipeline),
`_record_fetch_quality_fire_and_forget()` records success/fail per URL
as an async fire-and-forget task. Deep pipeline also records BM25
relevance scores.

**URL reordering**: Before fetching, `_reorder_urls_by_quality()` queries
domain stats and sorts URLs by a composite score:
`score = success_rate * 0.7 + min(avg_relevance, 1.0) * 0.3`.
Unknown domains get a neutral score of 0.5.

### Search-Required Guard

In `src/agents/base.py`, a guard prevents LLMs from skipping web search
when the classifier determined search is needed.

**Trigger conditions** (all must be true):
- Agent is about to emit `final_answer`
- Agent has `web_search` in its allowed tools
- Task's `search_depth` is `quick`, `standard`, or `deep`
- `web_search` was NOT called (tracked via `tools_used_names` set)
- Current iteration is < 3

**Behavior**: Rejects the final_answer and injects a correction message
telling the LLM it must call `web_search` first. The `tools_used_names`
set is maintained across the entire agent loop (including checkpoint
save/restore) to track which tools have been called.

**Addresses**: Known Issue #1 (LLM Hallucination Before Search) where
small local models produce fabricated answers without searching.

### Time-Sensitive Query Detection

The task classifier (`src/core/task_classifier.py`) detects temporal and
live-event patterns in task text and upgrades `search_depth` accordingly.

**Two pattern tiers:**
- `_TIME_SENSITIVE_STANDARD` (upgrades to at least `standard`):
  predicted XI, lineup, kadro, live scores, stock prices, exchange rates
- `_TIME_SENSITIVE_QUICK` (upgrades to at least `quick`):
  today/tomorrow/this week (EN+TR), weather, latest, current, breaking,
  son dakika, guncel

**Logic** (`_apply_time_sensitivity()`): Only upgrades, never downgrades.
Applied after the initial keyword-based classification.

### Intent Inference Priority

1. `search_depth` from task classifier (highest priority)
2. `shopping_sub_intent` from task classifier
3. `agent_type` default mapping
4. Fallback: `factual`

### Intent → Parameters

| Intent | Max Results | Chars/Page | Total Budget | Pipeline |
|--------|-------------|------------|--------------|----------|
| factual | 5 | 1500 | 5,000 | Quick |
| product | 7 | 2000 | 10,000 | Deep |
| reviews | 8 | 2500 | 15,000 | Deep |
| market | 10 | 3000 | 20,000 | Deep |
| research | 10 | 3000 | 20,000 | Deep |

### Agent → Default Intent

| Agent | Default Intent |
|-------|---------------|
| assistant, coder, writer, planner, architect | factual |
| executor, summarizer, visual_reviewer | factual |
| researcher, analyst | research |
| shopping_advisor, product_researcher | product |
| deal_analyst | market |

Shopping sub-intent overrides agent default:
- compare, price_check, deal_hunt, upgrade → product
- research, exploration → market
- purchase_advice, complaint_return_help → reviews

### Task Classifier Enhancement

`search_depth` field added to classification output:
- `deep` — market analysis, multi-source research, review synthesis
- `standard` — product info, comparison, how-to with examples
- `quick` — simple fact, definition, date, status
- `none` — no web search needed (code tasks, file operations)

Keyword fallback heuristic when LLM classifier fails:
- "analyze", "in detail", "comprehensive" → deep
- "price", "compare", "vs", "review" → standard
- "write", "fix", "implement", "debug" → none
- Default → quick

### Notification Fallback Chain

`src/infra/notifications.py` provides a unified `notify()` function with
a three-tier delivery chain:

1. **ntfy** — if `NTFY_URL` env var is configured, POST to ntfy topic
   (errors → `orchestrator-errors`, logs → `orchestrator-logs`)
2. **Telegram DM** — if `TELEGRAM_ADMIN_CHAT_ID` is configured, send
   directly to admin via `telegram.Bot`. Scheduled as async
   fire-and-forget task.
3. **File log** — always writes JSON-line entries to
   `logs/notifications.log`. This sink runs unconditionally as a
   baseline.

Returns the delivery method used: `"ntfy"`, `"telegram"`, or `"file"`.
Never raises.

---

## File Structure

| File | Purpose | Lines |
|------|---------|-------|
| `src/tools/web_search.py` | Orchestrator: intent inference, pipeline routing, fallbacks, source quality | ~820 |
| `src/tools/scraper.py` | Tiered scraper: HTTP/TLS/stealth/browser with auto-escalation | ~270 |
| `src/tools/page_fetch.py` | Async page fetcher, integrates with scraper, BeautifulSoup extraction | ~170 |
| `src/tools/content_extract.py` | Trafilatura extraction + price/review detection | ~85 |
| `src/tools/relevance.py` | BM25 scoring + adaptive budget allocation | ~115 |
| `src/infra/notifications.py` | Notification fallback chain: ntfy → Telegram DM → file log | ~500 |
| `src/core/task_classifier.py` | Task classification with search_depth + time-sensitivity | ~370 |
| `src/tools/__init__.py` | execute_tool with task_hints plumbing, shopping_fetch_reviews tool | modified |
| `src/agents/base.py` | ReAct loop, search-required guard, tools_used_names tracking | modified |
| `src/infra/db.py` | web_source_quality table, record/query functions | modified |

### Test Coverage

| Test File | Tests | Covers |
|-----------|-------|--------|
| `tests/test_web_search_integration.py` | 24 | ddgs, Brave, Perplexica, fallback order, page fetch integration |
| `tests/test_page_fetch.py` | 15 | HTML extraction, fetch errors, parallel fetch, scraper integration |
| `tests/test_content_extract.py` | 8 | Trafilatura, price/review detection, fallback |
| `tests/test_relevance.py` | 10 | BM25 scoring, budget allocation, intent boosts |
| `tests/test_deep_search_integration.py` | 11 | Intent inference from all hint types |
| `tests/test_search_depth.py` | 18 | Classifier search_depth, time-sensitivity upgrades |
| `tests/test_scraper.py` | 25 | Tier escalation, block detection, concurrent scraping |
| `tests/test_source_quality.py` | 10 | Domain quality recording, URL reordering |
| **Total** | **121** | |

---

## Dependencies

| Package | Purpose | Installed In |
|---------|---------|-------------|
| `ddgs` | DuckDuckGo search API | requirements.txt |
| `aiohttp` | Async HTTP for page fetching | requirements.txt |
| `beautifulsoup4` | HTML parsing (quick path + fallback) | requirements.txt |
| `lxml` | Fast HTML parser backend | requirements.txt |
| `trafilatura` | Smart content extraction (deep path) | requirements.txt |
| `bm25s` | BM25 relevance scoring | requirements.txt |
| `curl_cffi` | TLS fingerprint bypass (scraper TLS tier) | requirements.txt |
| `scrapling[all]` | Stealth/browser tiers (Camoufox + Playwright) | optional |

---

## Known Issues

### 1. LLM Hallucination Before Search — MITIGATED

**Status:** Mitigated by the search-required guard (see above).

**Original problem:** Small local LLMs sometimes produce a `final_answer`
on iteration 1 without calling `web_search`, hallucinating facts.

**Fix applied:** Search-required guard in `base.py` rejects `final_answer`
when `search_depth != "none"` and `web_search` has not been called.
Uses `tools_used_names` set for accurate tracking. Guard only fires on
iterations < 3 to avoid infinite loops.

**Remaining risk:** The guard only applies when the classifier correctly
assigns a non-`none` search_depth. Tasks misclassified as `none` can
still hallucinate.

### 2. Page Fetch Blocked by Sites — MITIGATED

**Status:** Mitigated by tiered scraper with auto-escalation.

**Original problem:** Many content sites return 403/402.

**Fix applied:** `src/tools/scraper.py` implements 4-tier escalation
(HTTP → TLS → stealth → browser). `page_fetch.py` now uses the scraper
by default with TLS as the default max tier. Deep pipeline uses stealth
tier for market/research intents.

**Remaining risk:** Stealth and browser tiers require optional
dependencies (`scrapling[all]`). Without them, escalation stops at TLS.

### 3. BM25 Synonym Blindness

BM25 is lexical — it matches words, not meaning. "smartphone" won't match
"iPhone" or "Samsung Galaxy". For web search results this is usually fine
(search engines pre-filter for relevance), but can occasionally misrank.

**Impact:** ~15% of queries may have suboptimal budget allocation.

### 4. Perplexica/Vane Container

Currently the Vane Docker container must be manually started. Planned future
work: on-demand Docker boot when Perplexica fallback is needed.

---

## Shopping Fixes (This Session)

Several shopping-related bugs were fixed:

### Product Serialization
Product objects were not JSON-serializable when passed between tools,
causing `TypeError` in comparison and search planner flows.

### Compare Type Mismatch
`shopping_compare` received mixed types (dicts vs Product objects),
causing attribute access failures. Fixed to normalize input types.

### Search Planner Source Names
Search planner was using internal source identifiers that didn't match
the actual tool names, causing failed lookups.

### Review Fetching Tool
Added `shopping_fetch_reviews` tool to fetch product reviews from a
URL. Available to `shopping_advisor`, `product_researcher`, and
`deal_analyst` agents.

### Todo Accept Button
Fixed the suggestion Accept button in the todo inline menu, including
datetime format issues and safety guards.

---

## Performance Characteristics

### Quick Pipeline (factual intent)

| Metric | Value |
|--------|-------|
| Total time | 3-5s |
| Pages fetched | 3 |
| Output size | ~5K chars |
| RAM delta | ~0 |
| GPU impact | None |

### Deep Pipeline (product/reviews/market/research)

| Metric | Value |
|--------|-------|
| Total time | 10-20s |
| Pages fetched | 5-10 |
| Output size | 8-22K chars |
| RAM delta | ~50MB (Trafilatura + BM25) |
| GPU impact | None |
| BM25 scoring time | <1ms |
| Trafilatura extraction | ~50ms/page |

---

## Future Phases

### Phase 4: Perplexica On-Demand Docker

**Goal:** Boot Vane container only when fallback is needed.

**Approach:** `web_search.py` calls `docker start vane` before Perplexica
request, waits for health check, proceeds. Container stops after idle timeout
(5 min).

### Research Completed (Not Yet Planned)

**27 scraping tools evaluated** — full comparison in agent output from this session.
Top recommendations by use case:
- Quick scraping: curl_cffi + Trafilatura
- Deep research: Crawl4AI or Scrapling
- Anti-bot: Camoufox (best) or Patchright (lighter)
- Price monitoring: curl_cffi (lightweight, scheduled)
- Review synthesis: Crawl4AI (LLM-ready markdown output)

---

## Infrastructure Fixes (Previous Sessions)

These fixes were implemented alongside the web search work:

### Cold-Start Race Condition
**File:** `src/core/llm_dispatcher.py`
**Problem:** OVERHEAD calls (classifier) failed on cold start because no
model was loaded and no cloud keys configured.
**Fix:** `_wait_for_model_load()` polls up to 15s for proactive loader to complete.

### Stuck Task Recovery
**File:** `src/core/orchestrator.py`, `src/infra/db.py`
**Problems:**
1. `claim_task()` used `isoformat()` — watchdog comparison with SQLite `datetime()` never matched
2. Watchdog didn't enforce retry limits
3. Dedup blocked retries of stuck tasks
4. `_handle_complete` crashed on malformed JSON context
**Fixes:** strftime format, retry limit enforcement, stuck-task reset in dedup, JSON safety.

### Watchdog/Idle-Unloader Race
**File:** `src/models/local_model_manager.py`
**Problem:** Idle unloader stopping server was detected as a "crash" by health watchdog.
**Fix:** `_idle_unload_in_progress` flag. Also reduced idle timeout from 600s to 60s.
