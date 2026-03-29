# Web Search Architecture X-Ray

> Living document. Covers the web search system as implemented on branch
> `feat/web-search-v2-ddgs-pagefetch`. Update as system evolves.

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
        │  Trafilatura: extract main content + metadata
        │  BM25: score relevance to query
        │  Budget allocator: distribute chars proportional to relevance
        │  Output: snippets + budgeted content (~8-22K chars)
        │
    ▼
ChromaDB Embed (side effect, async)
    │  Store in web_knowledge for future cache hits
    │
    ▼
Fallback Chain (if ddgs fails)
    ├── Perplexica/Vane (AI synthesis, 45s timeout)
    ├── SearXNG direct (raw results, 12s timeout)
    └── curl DuckDuckGo API (last resort)
```

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

---

## File Structure

| File | Purpose | Lines |
|------|---------|-------|
| `src/tools/web_search.py` | Orchestrator: intent inference, pipeline routing, fallbacks | ~550 |
| `src/tools/page_fetch.py` | Async HTTP page fetcher with BeautifulSoup extraction | ~100 |
| `src/tools/content_extract.py` | Trafilatura extraction + price/review detection | ~90 |
| `src/tools/relevance.py` | BM25 scoring + adaptive budget allocation | ~120 |
| `src/core/task_classifier.py` | Task classification with search_depth field | modified |
| `src/tools/__init__.py` | execute_tool with task_hints plumbing | modified |
| `src/agents/base.py` | Builds task_hints dict for execute_tool | modified |

### Test Coverage

| Test File | Tests | Covers |
|-----------|-------|--------|
| `tests/test_web_search_integration.py` | 16 | ddgs, Perplexica, fallback order, page fetch integration |
| `tests/test_page_fetch.py` | 15 | HTML extraction, fetch errors, parallel fetch |
| `tests/test_content_extract.py` | 8 | Trafilatura, price/review detection, fallback |
| `tests/test_relevance.py` | 10 | BM25 scoring, budget allocation, intent boosts |
| `tests/test_deep_search_integration.py` | 11 | Intent inference from all hint types |
| `tests/test_search_depth.py` | 5 | Classifier search_depth field |
| **Total** | **65** | |

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

---

## Known Issues

### 1. LLM Hallucination Before Search

**Problem:** Small local LLMs sometimes produce a `final_answer` on iteration 1
without calling `web_search`, hallucinating facts. On iteration 2 they search,
then iteration 3 gives another answer — but may still not incorporate the
search results properly.

**Observed:** Task #1684 ("predicted xi for Turkey") — LLM answered with a
fabricated lineup before searching, then searched, then answered again.

**Root cause:** Agent prompt says "use web_search for factual questions that
might be outdated" but the LLM decides it "knows" the answer. Small models
are poor at recognizing when they need to search.

**Potential fixes:**
- Force web_search on first iteration for researcher/assistant agents
- Add "ALWAYS search before answering factual questions" to agent prompts
- Use the `needs_tools` classifier field to require tool use before final_answer

### 2. Page Fetch Blocked by Sites

Many content sites (Serious Eats, Food Network, etc.) return 403/402. The
system degrades gracefully to snippets-only, but deep searches on these sites
produce less content than expected.

**Mitigation:** Phase 2 (curl_cffi TLS fingerprinting) and Phase 3
(Scrapling stealth browser) will address this.

### 3. BM25 Synonym Blindness

BM25 is lexical — it matches words, not meaning. "smartphone" won't match
"iPhone" or "Samsung Galaxy". For web search results this is usually fine
(search engines pre-filter for relevance), but can occasionally misrank.

**Impact:** ~15% of queries may have suboptimal budget allocation.

### 4. Perplexica/Vane Container

Currently the Vane Docker container must be manually started. Planned future
work: on-demand Docker boot when Perplexica fallback is needed.

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

### Phase 2: Tiered Scraper (curl_cffi)

**Goal:** Add TLS fingerprint bypass for sites that block standard HTTP clients.

**New file:** `src/tools/scraper.py`

**Tiers:**
- `http` — existing aiohttp (page_fetch.py)
- `tls` — curl_cffi with browser TLS fingerprints (~10-30MB RAM)

**Auto-escalation:** If HTTP returns 403/Cloudflare challenge, retry with TLS tier.

**Dependency:** `curl_cffi` (already researched, prebuilt Windows wheels available)

### Phase 3: Anti-Bot Tiers (Scrapling)

**Goal:** Handle Cloudflare-protected and JS-heavy sites.

**Additional tiers in scraper.py:**
- `stealth` — Scrapling StealthyFetcher with Camoufox (300-500MB on-demand)
- `browser` — Scrapling DynamicFetcher with Playwright (500-800MB on-demand)

**On-demand lifecycle:** Browsers spin up per-request, shut down after. Zero idle RAM.

**Dependency:** `scrapling[all]` (includes camoufox + playwright)

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

## Infrastructure Fixes (This Session)

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
