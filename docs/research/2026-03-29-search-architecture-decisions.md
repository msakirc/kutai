# Search Architecture Decision Log (March 2026)

> Key design decisions made during the web search v2 session, with reasoning.

---

## Decision 1: Drop Perplexica as Primary, Keep as Fallback

**Context:** Perplexica/Vane bundles SearXNG for search + local LLM for synthesis.
SearXNG's engines (Google, Bing, Brave) were getting blocked/CAPTCHAd. Synthesis
used the same GPU as agent tasks (contention). 50-70s response time even with fast models.

**Explored:** 4get-hijacked connector (replaces SearXNG scrapers with 4get's PHP scrapers),
fixing SearXNG engines, bypassing SearXNG entirely.

**Decision:** Replace with ddgs (DuckDuckGo API) + page fetch. Zero GPU contention,
no ban risk, no container dependency. Perplexica kept as fallback for when ddgs fails.

**Reasoning:** Vane synthesis uses the local LLM — it's just a prompt + search results.
The agent already pays for an LLM call; having it reason over raw results costs nothing extra.

## Decision 2: Single Tool with Auto-Escalation (Not Two Separate Tools)

**Options considered:**
- (a) Two tools: `web_search` (quick) + `deep_search` (heavy)
- (b) Single `web_search` with depth parameter
- (c) Smart auto-escalation with agent hints

**Decision:** Combined b+c — single `web_search`, auto-escalation, agents hint minimum depth.

**Reasoning:** Two similar tools confuse small LLMs. The agent knows task context
(shopping vs research) but shouldn't decide scraping strategy. One tool, zero LLM
choice, zero LLM error.

## Decision 3: Intent Inferred from Context, Not LLM Choice

**Options considered:**
- (a) Agent passes `intent="price_comparison"` explicitly
- (b) Hardcode intent per agent type
- (c) Infer from agent_type + task context + classifier hint

**Decision:** Option (c) with classifier providing `search_depth` as an additional signal.

**Reasoning:** Small LLMs will pick wrong intents ~30% of the time. Agent type alone
misses cases (assistant asked a shopping question). The classifier already reasons about
the task — adding `search_depth` to its output is one more JSON field, zero extra LLM calls.

**Gap analysis:**
- Assistant getting `factual` for "compare iPhone vs Samsung" — acceptable, quick path
  still gives a decent answer. User can use `/shop` for proper shopping flow.
- Researcher in quick_search workflow getting `research` — overkill but auto-escalation
  handles it (quick tier runs first anyway).

## Decision 4: Adaptive Budget Allocation (Strategy C) over ChromaDB Store (Strategy B)

**Benchmarked three strategies:**

| Strategy | Time | Quality | Cross-doc comparison |
|----------|------|---------|---------------------|
| A: Intent-aware extraction | 1-2s | 6/10 | Partial |
| B: ChromaDB store + digest | 2-4s | 5/10 | Poor |
| C: BM25 + budget allocation | **0.5s** | **7/10** | **Good** |

**Decision:** Strategy C as primary.

**Reasoning:** The agent needs all data in its context window to compare across
documents ("which product is cheapest?"). ChromaDB chunks isolate documents,
making comparison impossible without multiple round-trips. BM25 is fast (<1ms),
needs no LLM, and fits everything into the context window with intelligent allocation.

## Decision 5: Five Intents (Not 16)

**Analysis found 16 distinct search categories** across 13 agents and 11 workflows.
Collapsed to 5 based on shared extraction behavior:

| Intent | Covers | Depth |
|--------|--------|-------|
| factual | Knowledge, docs, events, API refs | Quick |
| product | Prices, specs, comparison, alternatives | Standard |
| reviews | Opinions, forums, complaints, reputation | Deep |
| market | Analysis, competitor, landscape, timing, deals | Deep |
| research | Deep research, technology comparison, synthesis | Deep |

**Reasoning:** Small LLMs need a small, clear set. Many categories share the same
extraction strategy (e.g., "price lookup" and "product comparison" both need
price-pattern extraction). Five intents, clearly differentiated by extraction behavior.

## Decision 6: Phased Implementation (Not Big Bang)

**Three phases:**
1. Core pipeline (Trafilatura + BM25 + intent inference) — **done**
2. Tiered scraper (curl_cffi TLS bypass) — planned
3. Anti-bot (Scrapling stealth/browser) — planned

**Reasoning:** Phase 1 delivers 80% of value. Each phase adds capability without
breaking what exists. Phase 2 and 3 are additive — the `scraper.py` module slots
into the existing pipeline at the fetch step.

---

## Agent/Workflow Search Needs Analysis

Full analysis of all 13 agents with web_search and 11 workflows was conducted.
Key findings:

### Agents Using web_search (13 total)

| Agent | Primary Use | Default Intent |
|-------|-------------|----------------|
| researcher | Deep multi-source research | research |
| shopping_advisor | Fallback to dedicated shopping tools | product |
| product_researcher | Niche sources, price gaps, international | product |
| deal_analyst | Price history, timing, fake discount detection | market |
| assistant | General Q&A, current events | factual |
| analyst | Market data, feasibility, benchmarks | research |
| coder | Error messages, library APIs | factual |
| planner | Framework comparisons | factual |
| architect | Library docs, API references | factual |
| writer | Documentation conventions | factual |
| executor | Anything (fallback agent) | factual |
| summarizer | Rarely (URL content) | factual |
| visual_reviewer | UI/UX references | factual |

### Shopping Sub-Intents (10 types)

| Sub-Intent | Intent Override | Example |
|------------|----------------|---------|
| price_check | product | "iPhone fiyat" |
| compare | product | "iPhone vs Samsung" |
| deal_hunt | product | "en ucuz RTX 4070" |
| upgrade | product | "switch from iPhone 14" |
| purchase_advice | reviews | "should I buy?" |
| complaint_return_help | reviews | "Trendyol iade" |
| research | market | "araştır robot süpürge" |
| exploration | market | "almak istiyorum" |
| gift | (default product) | "hediye fikri" |

### Workflows with Heavy Search Usage
- **shopping** (10+ phases) — product_researcher + deal_analyst
- **research** (5 phases) — two researcher steps + analyst
- **combo_research** — per-component search + compatibility check
- **exploration** — landscape mapping + progressive search
- **quick_search** — single researcher search (should be fast)

## SearXNG / 4get / Perplexica Relationship

Clarified during initial discussion:

| Layer | Role | Example |
|-------|------|---------|
| **Search engines** (Google, Bing) | Actual web indexes | google.com |
| **Meta-search** (SearXNG, 4get) | Scrape multiple engines, aggregate | SearXNG returns links+snippets |
| **AI synthesizer** (Perplexica/Vane) | Feed results to LLM, produce answer | "Based on 5 sources..." |

- 4get and SearXNG are at the same layer (meta-search scrapers)
- 4get-hijacked connector replaces SearXNG's broken scrapers with 4get's working ones
- SearXNG remains the aggregation API — 4get just swaps the engine
- Perplexica sits on top and synthesizes — it's "just an LLM with a prompt"
- GitHub discussion #5651: SearXNG maintainers now open to browser-based solutions
