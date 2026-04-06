# KutAI Shopping Intelligence X-Ray

> Architecture deep-dive and risk assessment for the shopping subsystem.
> Living document -- update as system evolves.

---

## System Overview

The shopping intelligence system enables product discovery, comparison, and deal analysis for the Turkish e-commerce market. It combines LLM-powered analysis with rule-based heuristics, web scrapers for live product data, and a resilience layer with multi-level fallback.

### Component Map

```
Telegram Bot                     Orchestrator
  /shop, /price, /compare  -->  task queue  -->  Agent dispatch
  /watch, /research_product                      |
                                                 v
                                    +------------------------+
                                    |   Shopping Agents (4)  |
                                    +------------------------+
                                    | shopping_advisor       |  (main, 8 iterations)
                                    | product_researcher     |  (search executor, 6 iter)
                                    | deal_analyst           |  (value/fake analysis, 4 iter)
                                    | shopping_clarifier     |  (single-shot, questions)
                                    +------------------------+
                                              |
                                    uses tools (8)
                                              |
                    +-------------------------+-------------------------+
                    |                         |                         |
          Intelligence Layer          Scraper Layer             Memory Layer
          (query analysis,            (trendyol, hepsiburada,   (user profiles,
           search planning,            akakce, amazon_tr,        price watches,
           scoring, reviews)           google_cse, perplexica)   purchase history)
                    |                         |
              _llm.py helper          Resilience Layer
              (LLMDispatcher)         (fallback chain, cache)
```

### Data Flow: `/shop coffee machine`

```
1. telegram_bot.py:cmd_shop()
   -> add_task(agent_type="shopping_advisor", priority=8)

2. orchestrator.py:process_task()
   -> claims task, skips classifier (agent_type already set)
   -> get_agent("shopping_advisor")

3. base.py:execute() -- ReAct loop (max 8 iterations)
   -> _build_model_requirements() looks up AGENT_REQUIREMENTS["shopping_advisor"]
   -> LLMDispatcher.request(MAIN_WORK, reqs, messages)

4. Agent calls shopping_search tool (iteration 1)
   -> tools/__init__.py:_tool_shopping_search()
   -> query_analyzer.analyze_query("coffee machine")        [LLM + keyword fallback]
   -> search_planner.generate_search_plan(analyzed)          [LLM + rule fallback]
   -> fallback_chain.get_product_with_fallback(query, sources)
      -> try: source-specific scraper (e.g. akakce)
      -> try: Perplexica
      -> try: Google CSE
      -> try: stale cache (72h)
   -> returns JSON: {analysis, search_plan, products, product_count}

5. Agent calls shopping_compare, shopping_reviews, etc. (iterations 2-7)

6. Agent returns final_answer with product recommendations

7. orchestrator.py:_handle_complete() -> telegram.send_result()
```

---

## Agents

### Agent Registry

| Agent | Class | Tier | Max Iter | Execution | Tools |
|-------|-------|------|----------|-----------|-------|
| `shopping_advisor` | ShoppingAdvisorAgent | medium | 8 | ReAct loop | 10 shopping + web_search + blackboard |
| `product_researcher` | ProductResearcherAgent | cheap | 6 | ReAct loop | 6 shopping + web_search + blackboard |
| `deal_analyst` | DealAnalystAgent | medium | 4 | ReAct loop | 6 shopping + web_search + blackboard |
| `shopping_clarifier` | ShoppingClarifierAgent | cheap | 3 | single_shot | shopping_user_profile only |

### Agent Collaboration (Blackboard)

Agents share state via blackboard keys:

| Key | Writer | Reader | Content |
|-----|--------|--------|---------|
| `shopping_intent` | shopping_advisor | product_researcher, deal_analyst | Parsed intent + constraints |
| `shopping_constraints` | shopping_advisor | product_researcher | Budget, dimensions, brand |
| `shopping_top_products` | product_researcher | deal_analyst, shopping_advisor | Candidate products |
| `shopping_price_comparisons` | product_researcher | deal_analyst | Cross-retailer prices |
| `shopping_deal_verdicts` | deal_analyst | shopping_advisor | Value scores, red flags |

**Status**: Blackboard tools (`read_blackboard`, `write_blackboard`) are registered in allowed_tools for all shopping agents. Collaboration depends on `shopping_advisor` creating subtasks for `product_researcher` and `deal_analyst` -- the agent has `can_create_subtasks=True` and instructions to delegate. However, subtask creation is **LLM-driven** (the agent must decide to create subtasks), so collaboration depth varies by model quality.

### AGENT_REQUIREMENTS (Router Integration)

| Agent | difficulty | output_tokens | function_calling | prefer_local | prefer_speed |
|-------|-----------|---------------|-----------------|-------------|-------------|
| `shopping_advisor` | 5 | 2500 | yes | yes | yes |
| `product_researcher` | 4 | 2000 | yes | yes | yes |
| `deal_analyst` | 5 | 2000 | yes | yes | no |
| `shopping_clarifier` | 3 | 1000 | no | yes | yes |

**Task profile**: `shopping_advisor` in `capabilities.py` TASK_PROFILES (domain_knowledge=1.0, analysis=0.9, instruction_adherence=0.8, tool_use=0.6).

---

## Tools (8 Shopping-Specific)

| Tool | Function | Dependencies |
|------|----------|-------------|
| `shopping_search` | Query analysis + search plan + scraper execution | query_analyzer, search_planner, fallback_chain |
| `shopping_compare` | Value scoring + delivery comparison | value_scorer, delivery_compare |
| `shopping_reviews` | Review aggregation with temporal weighting | review_synthesizer |
| `shopping_constraints` | Filter by budget/dimensions/brand | constraints |
| `shopping_timing` | Buy-now vs wait advice | timing (seasonal_advisor) |
| `shopping_alternatives` | Generate alternative product suggestions | alternatives |
| `shopping_user_profile` | Get/update user preferences and owned items | user_profile (SQLite) |
| `shopping_price_watch` | Add/list/remove price watches | price_watch (SQLite) |

All tools return JSON strings. Errors are caught and returned as `{"error": "..."}` -- never raise to the agent loop.

---

## Intelligence Layer

### Query Analyzer (`intelligence/query_analyzer.py`)

**Dual-path**: LLM first, keyword-based fallback.

LLM extracts: intent, category, products_mentioned, constraints, budget, urgency, experience_level, language, preferred_sources.

Keyword fallback has:
- 5 intent categories (find_cheapest, find_best, compare, explore, specific_product)
- 7 product categories (electronics, appliances, furniture, grocery, clothing, automotive, beauty)
- Budget extraction via regex (`\d+[\.,]?\d*\s*(tl|lira|₺)`)

| Strength | Weakness |
|----------|----------|
| Good Turkish keyword coverage (46 intent, 61+ category keywords) | LLM path fails silently -- falls back to keywords with no user indication |
| Handles Turkish normalization (İ/ı, Ö/ö, etc.) | Budget regex misses informal patterns ("around 5k", "max beş bin") |
| Markdown fence cleanup for malformed LLM JSON | Keyword matching is naive substring (false positives: "phone" in "telephone") |
| | Fallback missing `language`, `source`, `experience_level` fields |
| | No multi-product query support ("laptop OR tablet") |
| | No query rewriting if initial parse is low-confidence |

### Search Planner (`intelligence/search_planner.py`)

**Dual-path**: LLM first, rule-based fallback.

Returns flat `list[dict]` where each dict has: `query`, `sources`, `purpose`, `phase` (1 or 2).

Budget: 10 phase-1 tasks (immediate), 10 phase-2 tasks (dependent on phase-1 results).

Sources per category:
- electronics: akakce, trendyol, hepsiburada, amazon_tr, n11
- appliances: akakce, trendyol, hepsiburada, teknosa
- furniture: trendyol, hepsiburada, ikea_tr, koctas
- grocery: trendyol, migros, a101, getir
- default: akakce, trendyol, hepsiburada

| Strength | Weakness |
|----------|----------|
| Phase budget prevents runaway searches | LLM plan rarely succeeds (strict JSON validation) -- rule-based used ~80%+ |
| Category-aware source selection | Phase-2 templates (`{top_product}`) are **never filled** by any caller |
| Search variant generation via text_utils | Rule-based plan always takes first 3 sources -- no priority/reliability weighting |
| | Sources hardcoded -- no runtime configuration |
| | No plan cost estimation (some sources have lower daily budgets) |

### Product Matcher (`intelligence/product_matcher.py`)

Hierarchical cross-source matching: EAN (0.99) > MPN (0.95) > fuzzy name (0.70) > spec fingerprint (0.60). Overall threshold: 0.55.

| Strength | Weakness |
|----------|----------|
| Sound multi-signal hierarchy | Greedy algorithm (first-match wins, no global optimization) |
| Variant detection (color, size, bundle) | Merged specs may contradict (color: red AND blue in same group) |
| Handles missing fields gracefully | Cross-source price variance not flagged (20% difference silently merged) |
| | Canonical name = longest cleaned name (longest != best) |

### Value Scorer (`intelligence/value_scorer.py`)

Composite 0-100 score: price, seller rating, shipping, warranty, review rating, availability, review volume. Category-aware weights.

| Strength | Weakness |
|----------|----------|
| Category-specific weighting (electronics vs grocery) | Installment scoring uses hardcoded 1000 TL divisor (arbitrary) |
| Bayesian prior for low-review products (pulls to 3.5) | Warranty weight (0.15) penalizes gray imports heavily |
| Multiple perspectives (best_price, best_tco, best_installment) | Seller rating scale assumed 0-5 or 0-10 by magnitude -- no source awareness |
| | No user preference weighting overlay |
| | Availability only maps 4 states (no "last 3 in stock") |

### Review Synthesizer (`intelligence/review_synthesizer.py`)

Aggregates reviews with temporal weighting (2x for <6 months) and volume-adjusted confidence (0.3-1.0). LLM extracts themes, defects, Turkey-specific issues.

| Strength | Weakness |
|----------|----------|
| Temporal weighting is sensible | Cliff at 6 months (should be gradual decay) |
| Cross-source divergence warning (>1.5 stars) | No review authenticity/spam detection |
| LLM-powered theme extraction | Helpful count semantics differ across platforms |
| Review quality assessment (verified %, text length) | Sample selection greedy (top 30 by recency/helpful, excludes diverse opinions) |
| | Quality thresholds arbitrary (50-char minimum excludes "Great product!") |

### Delivery Compare (`intelligence/delivery_compare.py`)

Knows 12+ Turkish retailers with default carrier, delivery days, free shipping thresholds.

| Strength | Weakness |
|----------|----------|
| Turkish carrier awareness (HepsiJet, Trendyol Express, etc.) | Defaults are static -- no API for real-time delivery estimates |
| Free shipping threshold per retailer | Doesn't account for user location (Istanbul vs rural) |
| Effective price calculation (product + shipping) | Carrier speed estimates may be outdated |

### Constraints (`intelligence/constraints.py`)

Filters products by budget, dimensions, brand inclusion/exclusion, compatibility.

### Timing Advisor (`intelligence/timing.py`)

Category-aware buy-now vs wait advice. Knows Turkish sales calendar: Ramadan, 11.11, Black Friday, Yaz Indirimleri.

**Risk**: Ramadan dates hardcoded for 2025-2027 (needs yearly update).

### Alternatives (`intelligence/alternatives.py`)

Generates alternative product suggestions from different brands, previous-gen models, adjacent categories.

### Special Intelligence (`intelligence/special/`)

| Module | Status | Quality |
|--------|--------|---------|
| `fake_discount_detector.py` | Real | Good -- cross-store consistency, evidence-based |
| `unit_price_calculator.py` | Real | Good -- Turkish units (kg, L, adet, m2, tablet, kapsul) |
| `seasonal_advisor.py` | Real | Hardcoded calendar, needs yearly Ramadan update |
| `seller_trust.py` | Real | Seller reputation heuristics |
| `tco_calculator.py` | Real | Total cost of ownership (energy, consumables) |
| `exchange_rate.py` | Real | Import price with exchange rate |
| `fraud_detector.py` | Real | Fraud pattern detection |
| `used_market.py` | Partial | Used/refurbished evaluation |
| `bulk_detector.py` | Partial | Bulk deal detection |
| `bundle_detector.py` | Partial | Bundle deal detection |

---

## Scraper Layer

### Base Scraper (`scrapers/base.py`)

Abstract base with: exponential backoff (5s/15s/45s), daily budget guard, rate limiting with jitter, User-Agent rotation (4 browsers), request logging, structured data extraction (JSON-LD, OpenGraph, Schema.org), block page detection.

### Source Scrapers

| Scraper | Method | Daily Budget | Delay | Status | Risk |
|---------|--------|-------------|-------|--------|------|
| `trendyol.py` | Public API + `__NEXT_DATA__` | 100/day | 5s | **BROKEN** | `public.trendyol.com` and `public-mdc.trendyol.com` DNS dead — Trendyol retired these subdomains. `apigw.trendyol.com` returns "Service Unavailable". Needs rewrite to scrape `www.trendyol.com` via scrapling. |
| `hepsiburada.py` | `__NEXT_DATA__` + HTML fallback | 50/day | 15s | **High-risk** | Aggressive bot detection (Akamai/Datadome) |
| `akakce.py` | HTML parsing (BeautifulSoup) | 200/day | 10s | **Partial** | CSS selector breakage |
| `amazon_tr.py` | Standard scraping | 500/day | 3s | **Working** | Amazon bot detection |
| `n11.py` | Standard scraping | -- | -- | **Partial** | Untested |
| `google_cse.py` | Google Custom Search API | 100/day | 1s | **Working** | Requires API key |
| `perplexica.py` | Perplexica bridge | -- | -- | **Partial** | External dependency |

### Trendyol (Primary Source)

- Uses legitimate public API for search (`/discovery/v2/search`) and reviews
- `__NEXT_DATA__` extraction for product pages (reliable but brittle)
- Smart price extraction (tries sellingPrice > discountedPrice > originalPrice)
- Installment parsing

**Risks**: `scoringAlgorithmId` and `productStampType` params hardcoded; content ID regex (`-p-(\d+)`) may break on URL restructure; max 20 results hardcoded.

### Hepsiburada (Highest Risk)

- Marked `HIGH_RISK = True` in code
- Akamai/Datadome bot detection means standard httpx requests are likely blocked
- HTML selectors are from early 2024, likely outdated
- **Effectively non-functional without proxy rotation or browser emulation**

### Resilience: Fallback Chain (`resilience/fallback_chain.py`)

```
get_product_with_fallback(query, sources=["akakce", "trendyol", "hepsiburada"])
                                              |
                                    uses sources[0] ONLY  <-- BUG: ignores sources[1:]
                                              |
                                              v
                             build_fallback_chain("akakce")
                                              |
                    [akakce_scraper, perplexica, google_cse, stale_cache]
                              |          |          |          |
                           try each until one returns results
```

| Strength | Weakness |
|----------|----------|
| 4-level degradation prevents total failure | Only uses **first source** -- ignores rest of source list |
| Stale cache (72h) as last resort | No circuit breaker (waits for timeout on every failing source) |
| Async-aware fallback execution | No per-fallback timeout (all inherit same timeout) |
| | No metrics on which fallback succeeded |
| | 72-hour stale cache is risky for fast-changing prices |

---

## Memory Layer

### User Profile (`memory/user_profile.py`)

Tables: `user_profiles`, `owned_items`, `preferences`, `behaviors`.

Stores: dietary restrictions, location, stated/inferred preferences, purchase behavior patterns, owned items. Supports vector store embedding for semantic recall.

**Weakness**: No privacy controls; embedding failure silently ignored; only dietary_restrictions and location are updateable (constraints reserved for future).

### Price Watch (`memory/price_watch.py`)

Tables: `price_watches`, `price_watch_history`.

Tracks: product watches with target price, historical prices per watch, triggered/expired status. Auto-expires after 90 days.

**Weakness**: No notification mechanism -- watches are stored and tracked but **alerts are never sent to user**. Hardcoded 90-day expiry. No cross-product recommendations.

### Purchase History (`memory/purchase_history.py`)

Records past purchases for preference inference.

### Session Tracking (`memory/session.py`)

Shopping session lifecycle management.

---

## Internal LLM Calls (`intelligence/_llm.py`)

Shopping intelligence modules make LLM calls via a shared helper that correctly routes through `LLMDispatcher`:

```python
from src.core.llm_dispatcher import get_dispatcher, CallCategory
response = await get_dispatcher().request(CallCategory.MAIN_WORK, reqs, messages)
```

Uses `CallCategory.MAIN_WORK` with `difficulty=3`, `prefer_speed=True`. Falls back to empty string on any error.

| Strength | Weakness |
|----------|----------|
| Correctly uses LLMDispatcher (no direct call_model) | Uses MAIN_WORK instead of OVERHEAD (can trigger model swaps for analysis) |
| Graceful fallback on failure | No retry logic -- any transient error fails immediately |
| Lazy imports for test isolation | Token estimation: `len(prompt) // 4` (unreliable for Turkish) |
| | Silent failure returns empty string -- caller can't distinguish "no result" from "error" |

---

## Configuration (`config.py`)

YAML-overridable with sensible defaults. Singleton pattern.

### Rate Limits (Default)

| Source | Delay | Daily Budget | Effective Searches/Session |
|--------|-------|-------------|--------------------------|
| akakce | 10s | 200 | ~20 |
| trendyol | 5s | 100 | ~10 |
| hepsiburada | 15s | 50 | ~5 |
| amazon_tr | 3s | 500 | ~50 |
| google_cse | 1s | 100 | ~50 |

**Risk**: Trendyol at 100/day and hepsiburada at 50/day are production-hostile. A single shopping session with multi-source search can exhaust 20-30 requests.

### Cache TTLs

| Type | TTL | Risk |
|------|-----|------|
| Specs | 30 days | Safe |
| Prices | 24 hours | Stale during flash sales |
| Reviews | 7 days | Acceptable |
| Search results | 12 hours | May miss new listings |

---

## Text Utilities (`text_utils.py`)

Turkish-specific text processing: normalization (İ/ı, Ö/ö, Ç/ç, Ş/ş, Ğ/ğ, Ü/ü), Turkish-English term translation, material keyword extraction, filler phrase removal, search variant generation.

---

## Known Weaknesses

### Critical (Blocks Production Use)

| # | Component | Issue | Impact |
|---|-----------|-------|--------|
| ~~W1~~ | ~~`fallback_chain.py`~~ | ~~Only uses `sources[0]`~~ | **FIXED**: Now tries all source scrapers, then shared fallbacks (Perplexica, Google CSE) |
| W2 | Hepsiburada scraper | Bot detection (Akamai/Datadome) blocks standard requests | Second-largest Turkish retailer is inaccessible |
| W3 | Rate limits | Trendyol 100/day, Hepsiburada 50/day | Real sessions exhaust budgets quickly |
| W4 | Price watch | Watches stored but no notification mechanism | Users set watches that never alert |
| ~~W16~~ | ~~`shopping_search` tool~~ | ~~No timeouts on LLM analysis or scraper calls~~ | **FIXED**: 15s cap on LLM analysis, 30s cap on scraper chain |

### High (Degraded Quality)

| # | Component | Issue | Impact |
|---|-----------|-------|--------|
| W5 | `_llm.py` | Uses `MAIN_WORK` for analysis overhead | Can trigger unnecessary model swaps during tool execution |
| W6 | Search planner | Phase-2 templates (`{top_product}`) never filled | Phase-2 searches are unusable |
| W7 | Query analyzer fallback | Missing fields vs LLM path (language, source, experience_level) | Downstream code may get KeyError or wrong defaults |
| W8 | Scraper User-Agents | Dated (Chrome 120, Firefox 121 -- early 2024) | Increasing bot detection risk |

### Medium (Edge Cases / Technical Debt)

| # | Component | Issue |
|---|-----------|-------|
| W9 | Product matcher | Greedy matching (no global optimization) |
| W10 | Value scorer | Installment scoring uses hardcoded 1000 TL divisor |
| W11 | Review synthesizer | 6-month temporal weight cliff (should be gradual decay) |
| W12 | Seasonal advisor | Ramadan dates hardcoded for 2025-2027 |
| W13 | Base scraper | No proxy support, no CloudFlare bypass |
| W14 | Trendyol scraper | `__NEXT_DATA__` format changes break silently |
| W15 | Config singleton | No per-user rate limiting -- all users share global budget |

---

## Risks

### Scraper Fragility

All scrapers parse HTML or undocumented APIs. Any of these can break without warning:
- **Trendyol**: API parameter changes, `__NEXT_DATA__` restructure, URL scheme change
- **Hepsiburada**: Already likely blocked; any changes make it worse
- **Akakce**: CSS selector changes during A/B tests

**Mitigation**: Fallback chain + stale cache prevent total failure, but degraded results are silent.

### Silent Degradation

The system is designed for graceful degradation, but this means failures are invisible:
- LLM analysis fails -> keyword fallback (lower quality, no user notification)
- Scraper blocked -> falls through to Google CSE or cache (stale data)
- Model without tool calling selected -> agent can't use shopping tools (hard fail)

**Recommendation**: Add a quality indicator to shopping results ("Based on live data from 3 sources" vs "Based on cached data from 12 hours ago").

### Rate Budget Exhaustion

With current limits, a power user doing 3-4 shopping searches/day can exhaust Trendyol's daily budget. No inter-session budget tracking means morning searches can starve afternoon ones.

### Data Freshness

The 24-hour price cache means flash sales, limited-time offers, and competitor price wars are invisible until cache expires. The 72-hour stale cache fallback makes this worse.

---

## Improvement Opportunities

### P0 -- Fix for Production

1. **Fix fallback chain to try all sources** -- iterate `sources` list, not just `sources[0]`
2. **Add price watch notifications** -- wire watch triggers to Telegram alerts
3. **Increase Trendyol/Hepsiburada daily budgets** or implement user-level metering
4. **Disable Hepsiburada scraper** until proxy/browser solution is in place (saves error budget)

### P1 -- High Value

5. **Change `_llm.py` to use `OVERHEAD` category** -- prevents model swaps for analysis
6. **Implement Phase-2 template filling** -- or remove Phase-2 from search planner
7. **Add result quality indicator** -- tell user if results are fresh vs cached vs degraded
8. **Update User-Agent pool** -- rotate through current browser versions
9. **Add circuit breaker** to fallback chain -- skip known-down sources for 30 minutes

### P2 -- Polish

10. **Gradual temporal decay** for review weighting (exponential instead of cliff)
11. **Global-optimal product matching** (Hungarian algorithm or similar)
12. **Adaptive rate limiting** -- slow down on 429, speed up when budget available
13. **Per-source fallback timeout** -- Perplexica gets 5s, Google CSE gets 10s
14. **Query rewriting** -- if initial search yields no results, suggest reformulation
15. **Ramadan date calculation** -- compute from Hijri calendar instead of hardcoding

### P3 -- Future Features

16. **Proxy rotation** for scraper resilience
17. **Browser emulation** (Playwright) for Hepsiburada and advanced bot detection
18. **ML-based product matching** -- train similarity model on historical matches
19. **Aspect-based review sentiment** -- "battery good, design poor"
20. **Real-time delivery estimates** -- API integration with carriers (Yurtici, Aras, etc.)

---

## Testing Gaps

| Area | Current Coverage | Needed |
|------|-----------------|--------|
| Shopping tools | Import test only | Integration test: query -> scraper -> result |
| Fallback chain | None visible | Test each degradation level |
| Scraper parsing | None visible | Test with saved HTML/JSON fixtures |
| Query analyzer | None visible | Test Turkish text, mixed language, edge cases |
| Product matcher | None visible | Test cross-source matching with known products |
| Value scorer | None visible | Test category weights, edge cases (no reviews, no price) |
| Agent collaboration | None visible | Test blackboard read/write across subtasks |
| Rate limiting | None visible | Test budget enforcement, inter-request delays |
| Price watch triggers | None visible | Test watch creation, price check, notification |

---

## File Reference

| File | Purpose |
|------|---------|
| `src/agents/shopping_advisor.py` | Main shopping agent (8 iterations, 10 tools) |
| `src/agents/product_researcher.py` | Search executor (6 iterations, 6 tools) |
| `src/agents/deal_analyst.py` | Value analysis (4 iterations, 6 tools) |
| `src/agents/shopping_clarifier.py` | Clarifying questions (single-shot) |
| `src/tools/__init__.py:334-449` | All 8 shopping tool implementations |
| `src/shopping/intelligence/_llm.py` | Shared LLM helper (LLMDispatcher integration) |
| `src/shopping/intelligence/query_analyzer.py` | Query -> structured intent |
| `src/shopping/intelligence/search_planner.py` | Intent -> search task list |
| `src/shopping/intelligence/product_matcher.py` | Cross-source product matching |
| `src/shopping/intelligence/value_scorer.py` | Composite 0-100 value scoring |
| `src/shopping/intelligence/review_synthesizer.py` | Review aggregation + themes |
| `src/shopping/intelligence/delivery_compare.py` | Delivery options + effective price |
| `src/shopping/intelligence/constraints.py` | Product filtering |
| `src/shopping/intelligence/timing.py` | Buy-now vs wait advice |
| `src/shopping/intelligence/alternatives.py` | Alternative suggestions |
| `src/shopping/intelligence/special/` | Fake discount, fraud, TCO, unit price, etc. |
| `src/shopping/scrapers/base.py` | Abstract scraper base (rate limit, retry, cache) |
| `src/shopping/scrapers/trendyol.py` | Trendyol API + __NEXT_DATA__ |
| `src/shopping/scrapers/hepsiburada.py` | Hepsiburada (HIGH_RISK, bot detection) |
| `src/shopping/scrapers/akakce.py` | Akakce HTML parsing |
| `src/shopping/scrapers/amazon_tr.py` | Amazon Turkey |
| `src/shopping/scrapers/google_cse.py` | Google Custom Search fallback |
| `src/shopping/scrapers/perplexica.py` | Perplexica bridge |
| `src/shopping/resilience/fallback_chain.py` | 4-level degradation chain |
| `src/shopping/resilience/cache_fallback.py` | Stale cache retrieval |
| `src/shopping/memory/user_profile.py` | User profiles, preferences, behaviors |
| `src/shopping/memory/price_watch.py` | Price watches + history |
| `src/shopping/memory/purchase_history.py` | Purchase records |
| `src/shopping/memory/session.py` | Shopping session management |
| `src/shopping/config.py` | Rate limits, cache TTLs, feature flags |
| `src/shopping/models.py` | Product, Review, ShoppingQuery dataclasses |
| `src/shopping/text_utils.py` | Turkish text normalization + variants |
| `src/models/capabilities.py:336-351` | shopping_advisor task profile |
| `src/core/router.py:1492-1496` | Shopping AGENT_REQUIREMENTS |
