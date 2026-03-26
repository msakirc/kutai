# Shopping Intelligence -- Issues & Fixes

## Critical Issues

### C1. Shopping DB schemas never initialised at startup
**Files:** `src/shopping/cache.py:61`, `src/shopping/request_tracker.py:17`, `src/shopping/memory/__init__.py:48`
**Problem:** `init_cache_db()`, `init_request_db()`, and `init_memory_db()` are defined but **never called** from the orchestrator startup path (`src/core/orchestrator.py`). The first time any shopping tool runs, the cache/memory SQLite tables may not exist, causing `aiosqlite.OperationalError: no such table` on every query, price watch, and user profile operation.
**Fix:** Add an `await init_shopping_dbs()` call to the orchestrator's `start()` method (around line 140 of `orchestrator.py`) that initialises all three DB schemas. Wrap in try/except so shopping failures don't block general startup.

### C2. `shopping_search` tool only plans -- never executes actual searches
**File:** `src/tools/__init__.py:334-341`
**Problem:** The `shopping_search` tool calls `analyze_query()` + `generate_search_plan()` and returns a JSON plan. It does **not** call any scraper, `web_search`, or `get_product_with_fallback()`. The shopping_advisor agent receives a plan but has no tool to *execute* it. The agent's only option is `web_search` (generic DuckDuckGo/Perplexica), which cannot use the dedicated scrapers (Akakce, Trendyol, Hepsiburada, Amazon TR).
**Impact:** All the scraper infrastructure (15+ scrapers, cache, rate limiting, circuit breakers) is **dead code** from the agent's perspective. Shopping results come entirely from generic web search.
**Fix:** Either (a) add a `shopping_execute_search` tool that takes a plan and dispatches to scrapers via `get_product_with_fallback()`, or (b) make `shopping_search` execute the phase-1 plan automatically and return actual product results instead of just a plan.

### C3. `chat_id` not passed for natural-language shopping classification
**File:** `src/app/telegram_bot.py:1636-1649`
**Problem:** When a message is classified as `"shopping"` via the NLP classifier (line 1636), the task is created with `add_task(...)` but no `context={"chat_id": chat_id}` is passed. The `/shop` command (line 2283) also omits `chat_id` from context. Only `/research_product`, `/price`, `/watch`, and `/compare` pass `chat_id` in context. This means shopping tools like `shopping_user_profile` and `shopping_price_watch` cannot identify which user they're serving.
**Fix:** Add `context={"chat_id": chat_id}` to both the natural-language shopping path (line 1637) and `cmd_shop` (line 2283).

### C4. Perplexica search timeout at 180 seconds blocks the entire agent
**File:** `src/tools/web_search.py:187`
**Problem:** The Perplexica search timeout is 180 seconds. The `shopping_advisor` agent timeout is 600 seconds (`orchestrator.py:57`). A single slow Perplexica call consumes 30% of the agent's time budget. If the agent makes 2-3 web searches (typical for shopping), it can exhaust its timeout with Perplexica hanging before getting any results.
**Fix:** Reduce the Perplexica timeout to 30-45 seconds for shopping queries. The DuckDuckGo fallback is fast and sufficient for getting results when Perplexica is slow.

### C5. `_llm_call` silently returns empty string on all errors
**File:** `src/shopping/intelligence/_llm.py:53`
**Problem:** The shared LLM helper catches ALL exceptions and returns `""`. This means every intelligence module (`query_analyzer`, `search_planner`, `review_synthesizer`, etc.) silently degrades to rule-based fallbacks with zero logging of the actual error. Debugging shopping intelligence failures is extremely difficult.
**Fix:** Log the exception before returning empty: `logger.warning("_llm_call failed", error=str(e), prompt_len=len(prompt))`.

---

## Integration Issues

### I1. Scraper infrastructure entirely disconnected from agent tools
**Files:** `src/shopping/scrapers/` (all), `src/tools/__init__.py:318-519`
**Problem:** The tools exposed to agents (`shopping_search`, `shopping_compare`, etc.) call intelligence modules (query_analyzer, value_scorer, etc.) but never invoke any scraper. The scrapers (Akakce, Trendyol, Hepsiburada, Amazon TR, forums, grocery, etc.) with their caching, rate limiting, and circuit breaker infrastructure exist in isolation. The only way products get found is via `web_search` (generic Perplexica/DuckDuckGo).
**Impact:** No structured product data (prices, ratings, reviews, seller info, installment data) reaches the agent. All shopping recommendations are based on free-text web search results.

### I2. Blackboard usage described but potentially not functional
**Files:** `src/agents/shopping_advisor.py:59-68`, `src/agents/product_researcher.py:46-53`, `src/agents/deal_analyst.py:46-53`
**Problem:** The agent system prompts extensively describe blackboard keys (`shopping_intent`, `shopping_constraints`, `shopping_top_products`, `shopping_price_comparisons`, `shopping_deal_verdicts`) for cross-agent communication. However, the shopping_advisor does not create subtasks for product_researcher or deal_analyst -- it runs as a single agent. The blackboard coordination pattern requires the shopping_advisor to spawn subtasks, which requires `can_create_subtasks = True` (set correctly) but the system prompt never instructs the agent *how* to create subtasks.
**Fix:** Either (a) add subtask creation instructions to the shopping_advisor system prompt, or (b) remove blackboard references and make the shopping_advisor self-contained.

### I3. Fallback chain references non-existent `perplexica_search` import
**File:** `src/shopping/resilience/fallback_chain.py:137`
**Problem:** `from src.shopping.integrations import perplexica_search` -- but `src/shopping/integrations/__init__.py` exports `search_perplexica` (not `perplexica_search`). The fallback chain's Perplexica step will always fail with `ImportError`.
**Fix:** Change `perplexica_search` to `search_perplexica`.

### I4. `get_scraper()` returns class, not instance -- fallback chain calls `.search()` on class
**File:** `src/shopping/resilience/fallback_chain.py:129-131`
**Problem:** `get_scraper(source)` returns the scraper *class*, but the fallback chain immediately calls `scraper.search(query)` without instantiation. This will fail because `search()` is an instance method expecting `self`.
**Fix:** Change to `scraper = get_scraper(source)(); return await scraper.search(query)` (add `()` to instantiate).

### I5. Deprecated Perplexica module still exported and may confuse code paths
**File:** `src/shopping/integrations/perplexica.py:8-19`
**Problem:** `search_perplexica()` is marked as deprecated in favor of `src/tools/web_search.web_search`, but is still exported from `src/shopping/integrations/__init__.py` and referenced in `fallback_chain.py`. It lacks the circuit-breaker, model-discovery, and DuckDuckGo fallback logic from `web_search.py`. Two parallel Perplexica code paths exist with different error handling.
**Fix:** Replace all internal callers of `search_perplexica` with `web_search`, then remove the deprecated function.

### I6. `shopping_compare` tool requires pre-structured product data the agent doesn't have
**File:** `src/tools/__init__.py:343-351`
**Problem:** `shopping_compare` expects a JSON string of product dicts with specific fields (`name`, `price`, etc.). But since `shopping_search` only returns a plan (not products), the agent has no structured product data to pass. The agent would need to manually parse web search results into the expected format, which is unreliable.

### I7. `shopping_reviews` tool requires pre-collected reviews the agent can't get
**File:** `src/tools/__init__.py:353-360`
**Problem:** `shopping_reviews` expects a JSON list of review dicts. There's no tool to fetch reviews -- the scraper review methods exist but are not exposed to agents. The agent has no way to collect the input this tool needs.

---

## Missing Features

### M1. No tool to actually fetch/scrape product data
**Impact:** Critical gap. The entire scraper infrastructure is unreachable.
**Fix:** Add a `shopping_fetch_products` tool that takes a query string, dispatches to scrapers via `get_product_with_fallback()`, and returns structured product JSON.

### M2. No tool to fetch product reviews
**Fix:** Add a `shopping_fetch_reviews` tool that takes a product URL and source, calls the scraper's `get_reviews()`, and returns structured review data.

### M3. No scheduled price-watch checker
**File:** `src/shopping/memory/price_watch.py`
**Problem:** The price watch system can store watches and record history, but there's no scheduled task that periodically checks current prices against watches and triggers alerts. `/watch` creates a one-time task but doesn't set up recurring monitoring.
**Fix:** Add a scheduled job (cron or orchestrator periodic task) that runs every 4-6 hours, iterates active watches, fetches current prices via scrapers, updates `price_watch_history`, and sends Telegram alerts when target prices are reached.

### M4. No cache warmup integration
**File:** `src/shopping/resilience/cache_fallback.py:156-182`
**Problem:** `warmup_cache()` exists but is never called. Popular product queries could be pre-fetched during quiet hours.
**Fix:** Integrate with the orchestrator's startup or a scheduled task.

### M5. Phase 2 search plan templates never filled
**File:** `src/shopping/intelligence/search_planner.py:101-121`
**Problem:** Phase 2 search tasks use `{top_product}` and `{runner_up}` placeholders. There's no code that fills these templates after Phase 1 completes. The plan generator returns them as literal strings.
**Fix:** Add a `fill_phase2_templates()` function that takes Phase 1 results and substitutes the placeholders.

### M6. Output formatters never used in the delivery pipeline
**Files:** `src/shopping/output/formatters.py`, `src/shopping/output/product_cards.py`, `src/shopping/output/summary.py`
**Problem:** Comprehensive Telegram-formatted output modules exist (`format_comparison_table`, `format_top_pick`, `format_budget_option`, product cards with inline buttons). None are connected to the agent result pipeline. Agent results are delivered as raw text via `send_result()`.
**Fix:** Post-process shopping_advisor results through the output formatters before sending to Telegram.

### M7. No shopping session continuity
**File:** `src/shopping/memory/session.py`
**Problem:** Session management exists (create, update, add products/questions) but is never used by the agent flow. Each shopping query is treated as independent -- no memory of "we were just looking at laptops."
**Fix:** Thread session IDs through the task context and use them in the shopping_advisor agent.

### M8. `shopping_user_profile` tool has no user_id context
**Problem:** The tool requires a `user_id` string parameter, but the agent has no way to know the current user's Telegram chat_id. The task context may contain `chat_id` (when passed correctly -- see C3), but the agent prompt doesn't explain how to access it.
**Fix:** Auto-inject `user_id` from the task context into shopping tool calls, or provide a `get_current_user_id` tool.

---

## UX Issues

### U1. Generic "task queued" response for all shopping queries
**File:** `src/app/telegram_bot.py:1645-1648`, `2291-2294`
**Problem:** Every shopping query gets the same response: "Shopping task #N queued. I'll search prices and compare options for you." No immediate feedback on what was understood (category, budget, intent). The user waits with no indication of what's happening.
**Fix:** After creating the task, run a quick keyword analysis (`_fallback_analyze`) to provide immediate feedback: "Got it -- looking for laptops under 5000 TL. Comparing Trendyol, Hepsiburada, Amazon..."

### U2. Shopping results delivered as a wall of text
**Problem:** Results come through `send_result()` which truncates to 500 chars + file attachment for long results. Shopping recommendations should use the product card and comparison table formatters that already exist in `src/shopping/output/`.
**Fix:** Detect shopping results and format them using `format_comparison_table(format="telegram")` before sending.

### U3. No inline buttons for buy/watch/compare actions
**File:** `src/shopping/output/product_cards.py`
**Problem:** Product card formatters generate `reply_markup` dicts with "Buy" and "Compare" buttons, but these are never sent to Telegram because the output formatters aren't connected to the delivery pipeline.
**Fix:** Wire product cards through to `send_result()` using `reply_markup` parameter on `send_message`.

### U4. `/deals` command fails if memory DB not initialised
**File:** `src/app/telegram_bot.py:2377`
**Problem:** `/deals` imports `get_all_active_watches` which requires the memory DB to be initialised. If `init_memory_db()` was never called (see C1), this will throw `OperationalError`.
**Fix:** Addressed by C1.

### U5. `/mystuff` command fails if memory DB not initialised
**File:** `src/app/telegram_bot.py:2424`
**Problem:** Same as U4 -- `get_user_profile` needs memory DB tables.

### U6. No progress updates during long shopping searches
**Problem:** Shopping advisor has a 600-second timeout. The user sees "task queued" and then waits up to 10 minutes with no feedback. For shopping queries that involve multiple web searches, the user should get intermediate updates ("Found 5 products on Trendyol, checking Hepsiburada...").
**Fix:** Add streaming progress notifications via the Telegram interface during agent iterations.

### U7. `/compare` splitting fragile
**File:** `src/app/telegram_bot.py:2455`
**Problem:** Split on `\s+vs\.?\s+` only. "iPhone 15 ve Samsung S24" (Turkish "and") won't work. "iPhone 15 mi Samsung S24 mi" (Turkish comparison) won't work.
**Fix:** Add Turkish comparison patterns: `\s+(?:vs\.?|ve|mi\s+.*?mi|ile)\s+`.

---

## Recommendations

### Priority 1: Make shopping actually work (fix C1, C2, I1, M1)
1. **Initialise shopping DBs on startup** (C1) -- 15 min fix
2. **Add `shopping_fetch_products` tool** (M1) that calls scrapers via `get_product_with_fallback()` -- 1-2 hours
3. **Fix `shopping_search` to return products, not just a plan** (C2) -- rework the tool to execute Phase 1 searches and return structured results -- 2-3 hours
4. **Fix fallback chain bugs** (I3, I4) -- import name and instantiation -- 15 min

### Priority 2: Fix integration plumbing (C3, C5, I2, M8)
5. **Pass `chat_id` in all shopping task creation paths** (C3) -- 15 min
6. **Add error logging to `_llm_call`** (C5) -- 5 min
7. **Auto-inject `user_id` from task context into shopping tools** (M8) -- 30 min
8. **Fix or remove blackboard subtask pattern** (I2) -- 1-2 hours

### Priority 3: Improve result quality (C4, M5, M2)
9. **Reduce Perplexica timeout to 30-45s for shopping** (C4) -- 10 min
10. **Add `shopping_fetch_reviews` tool** (M2) -- 1 hour
11. **Implement Phase 2 template filling** (M5) -- 1 hour
12. **Remove deprecated Perplexica module** (I5) -- 30 min

### Priority 4: UX polish (U1, U2, U3, U6, U7, M6)
13. **Immediate intent feedback on shopping query** (U1) -- 1 hour
14. **Wire output formatters to Telegram delivery** (M6, U2, U3) -- 2-3 hours
15. **Turkish comparison patterns in `/compare`** (U7) -- 15 min
16. **Streaming progress for shopping** (U6) -- 2-3 hours

### Priority 5: Completeness features (M3, M4, M7)
17. **Scheduled price-watch checker** (M3) -- 3-4 hours
18. **Session continuity** (M7) -- 2-3 hours
19. **Cache warmup on startup** (M4) -- 1 hour

### Estimated total: ~20-25 hours of work
**Critical path:** Items 1-4 (Priority 1) unblock all shopping functionality. Without them, the entire scraper/intelligence/resilience stack is dead code and all shopping relies on generic web search.
