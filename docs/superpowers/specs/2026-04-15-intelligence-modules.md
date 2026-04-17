# Shopping Intelligence Modules — Integration Spec

**Date**: 2026-04-15  
**Status**: Survey complete, integration pending  
**2026-04-16 Update**: Three-workflow split landed (`quick_search` unchanged, `product_research` new, `shopping` refocused on category). Community relevance filter + deterministic fake-discount flag wired into pipeline. Review-synthesis / delivery-compare / timing modules remain intentionally *disabled* — see the "Disabled Until Scrapers Improve" section at the bottom. The Priority Order section above is still the roadmap for re-enabling each.  
**Goal**: Wire existing intelligence modules into the shopping pipeline

---

## Current State

The shopping pipeline (`src/workflows/shopping/pipeline.py`) does:
1. Search all scrapers → collect products
2. Relevance filter → remove off-topic products
3. Product matcher → deduplicate across sources
4. Format → pick winner by site ranking, show prices

**Missing**: 17 intelligence modules exist in `src/shopping/intelligence/` but only `product_matcher` is wired. The pipeline returns raw scraper data with no scoring, no review synthesis, no delivery comparison, no timing advice.

---

## Module Inventory

### Pure Python (no LLM, safe to call in quick_search pipeline)

| Module | File | Public API | What It Does |
|--------|------|-----------|--------------|
| **value_scorer** | `value_scorer.py` | `score_products(products: list[Product], category="") -> list[dict]` | 0-100 composite score (price, seller, shipping, warranty, rating, availability, reviews). Category-aware weights for electronics/appliances/etc. |
| **product_matcher** | `product_matcher.py` | `match_products(products: list[Product]) -> list[dict]` | Cross-source dedup via EAN > MPN > fuzzy name > spec fingerprint. **Already wired.** |
| **sentiment** | `sentiment.py` | `analyze_sentiment(text) -> dict`, `analyze_reviews_batch(reviews) -> dict` | Keyword-based sentiment (Turkish + English, ~70 keywords, negation handling). Returns score -1 to +1, positive/negative word lists. |
| **constraints** | `constraints.py` | `check_constraints(products, constraints) -> list[dict]` | Validates dimensional, budget, electrical, compatibility, availability constraints. |
| **installment_calculator** | `installment_calculator.py` | `calculate_installments(price: float, store: str) -> list[dict]` | Taksit options from store-bank knowledge base. Knows faizsiz tiers for Trendyol/Hepsiburada/etc. |

### LLM-Required (use in full shopping workflow, NOT quick_search)

| Module | File | Public API | What It Does |
|--------|------|-----------|--------------|
| **review_synthesizer** | `review_synthesizer.py` | `synthesize_reviews(reviews, product_name) -> dict` | Theme extraction, defect patterns, temporal weighting (recent reviews 2x), confidence-adjusted rating. Embeds to vector store. |
| **delivery_compare** | `delivery_compare.py` | `compare_delivery(products: list[dict]) -> list[dict]` | Effective price (product + shipping), estimated delivery dates, international shipping flags. Knows 9 Turkish stores' carriers/thresholds. |
| **timing** | `timing.py` | `advise_timing(category, product_name="") -> dict` | Buy-now vs wait using Turkish sale calendar (11.11, Efsane Cuma, etc.), seasonal windows, price trends. |
| **alternatives** | `alternatives.py` | `generate_alternatives(query, category="", constraints=None) -> list[dict]` | Suggests alternatives (iPhone → Galaxy, etc.) via rules + LLM. |
| **substitution** | `substitution.py` | `suggest_substitutions(product, intent, category, budget=None) -> list[dict]` | Different products solving same need (robot vacuum → cordless stick). |
| **combo_builder** | `combo_builder.py` | `build_combos(components, budget, constraints) -> list[dict]` | Multi-component builds (PC, kitchen) with compatibility checking. Budget/mid/premium tiers. |
| **return_analyzer** | `return_analyzer.py` | `analyze_return_policy(product, store) -> dict` | Return ease score (0-1) per store. Knows return windows, free return, marketplace caveats for 10 stores. |
| **query_analyzer** | `query_analyzer.py` | `analyze_query(raw_query) -> dict` | Structured query parsing: intent, category, budget, constraints, complexity. **Used in pipeline for full workflow step 0.1.** |
| **search_planner** | `search_planner.py` | `generate_search_plan(analyzed_query) -> list[dict]` | Ordered search task list with rate budgets. Two-phase execution. |

### Support Modules

| Module | File | Purpose |
|--------|------|---------|
| **_llm** | `_llm.py` | Centralized LLM call routing for all intelligence modules. Uses `CallCategory.OVERHEAD`. |
| **vector_bridge** | `vector_bridge.py` | Embeds products, reviews, sessions, purchases to ChromaDB "shopping" collection. |

### Output Formatters (`src/shopping/output/`)

| Module | File | Used? | What It Does |
|--------|------|-------|--------------|
| **formatters** | `formatters.py` | Yes (pipeline) | `format_price()`, `format_comparison_table()`, `format_installment_options()` |
| **summary** | `summary.py` | Imported but unused | `format_recommendation_summary()` — top pick, budget option, alternatives, warnings, timing. Adapts complexity. |
| **product_cards** | `product_cards.py` | No | Telegram inline button product cards. Orphaned. |

---

## Integration Points

### Quick Search Pipeline (zero LLM, must stay fast)

Current steps:
```
0.1: execute_product_search → search + filter + match
1.1: format_and_deliver → format for Telegram
```

**Can wire in (pure Python, fast):**
- `value_scorer.score_products()` → after matching, before format. Adds value_score to each product.
- `sentiment.analyze_reviews_batch()` → if reviews available, add sentiment to output.
- `installment_calculator.calculate_installments()` → for the winner product.
- Format step should use `value_score` for ranking instead of price-only logic.

### Full Shopping Workflow (LLM allowed, multi-minute)

Current steps:
```
0.1: shopping_pipeline — analyze query
1.1: shopping_clarifier — clarification (LLM)
2.1: shopping_pipeline — search + reviews
3.1: deal_analyst — value analysis (LLM)
4.1: shopping_advisor — recommendation (LLM)
5.1: shopping_pipeline — format
```

**Can wire in:**
- Step 2.1: After search, call `review_synthesizer`, `delivery_compare`, `timing`
- Step 3.1: Pass `value_scorer` output + review synthesis + delivery data to deal_analyst
- Step 5.1: Use `summary.format_recommendation_summary()` with full data

### Community Data Quality

Şikayetvar currently returns irrelevant complaints (prison phone issues for a coffee machine search). The community search uses the same raw query without product-specific filtering. Community data scraping might need its own relevance filter similar to product filtering.

---

## Priority Order for Integration

1. **value_scorer** in quick_search — biggest impact, pure Python, replaces dumb price sorting
2. **Format step upgrade** — use `summary.format_recommendation_summary()` or at least score-based ranking
3. **installment_calculator** in format — Turkish users care about taksit
4. **delivery_compare** in full workflow — effective price matters
5. **review_synthesizer** in full workflow — review quality signal
6. **timing** in full workflow — "wait for 11.11" advice
7. **Community relevance filter** — filter Şikayetvar/Technopat results same way as products

---

## Data Reality Table (as of 2026-04-16)

What scrapers actually populate today — the signal behind the "disabled" status of several modules.

| Field | Populated by scrapers? | Notes |
|-------|------------------------|-------|
| `name` | ✅ Reliable | All 15 scrapers. |
| `url` | ✅ Reliable | All 15 scrapers. |
| `source` | ✅ Reliable | Domain key. |
| `original_price` | ✅ Usually | Missing when site doesn't show strikethrough. |
| `discounted_price` | ✅ Usually | Same. |
| `free_shipping` | 🟡 Partial | Trendyol/Hepsiburada: yes. Others: often false-default. |
| `shipping_cost` | ❌ ~0% | Rarely visible on listing pages — only on cart/checkout. |
| `shipping_time_days` | 🟡 <30% | Trendyol sometimes. Most scrapers leave this None. |
| `rating` | 🟡 Partial | Listing JSON when available. |
| `review_count` | 🟡 Partial | Same. |
| `review text` | ❌ ~0% | `get_reviews()` exists for 3 scrapers (Trendyol, Hepsiburada, Amazon TR); returns short snippets, not enough for theme extraction yet. |
| `warranty_months` | ❌ ~0% | Almost never on listing pages. |
| `specs` | 🟡 Partial | Varies wildly — no consistent key set. |
| Price history | ❌ None | No DB table wired yet. |
| Community post body | ❌ None | Only `name` (title) and `url`. |

**Implication:** Any intelligence module that depends on review text, shipping cost, warranty, or price history can't run meaningfully today. We surface what we do have (cross-store price spread, per-store discount ratios) and scaffold the rest.

---

## Three-Workflow Structure (2026-04-16)

| Workflow | Entry Point | Triggers | Steps | LLM? |
|----------|-------------|----------|-------|------|
| **quick_search** | `/shop <short query>` (simple heuristic) | Fast price comparison | 2 (search, format) | No |
| **product_research** | `/shop` → "🎯 Belirli ürün" button | Named brand+model | 3 live (search, enrich, deliver) + 3 stubs (reviews, delivery, timing) | No (stubs are neutral placeholders) |
| **shopping** | `/shop` → "🏷 Kategori" button, or complex-heuristic `/shop foo` | Category/discovery queries | 6 (parse, clarify, search+reviews, analyze, recommend, format) | Yes (clarify, analyze, recommend) |

`combo_research`, `gift_recommendation`, `price_watch`, and `exploration` remain unchanged — they serve distinct entry points.

---

## Disabled Until Scrapers Improve

Modules that EXIST in `src/shopping/intelligence/` but are NOT wired into the live pipeline because their input data isn't populated at useful scale yet.

| Module | Needs | Becomes useful when |
|--------|-------|---------------------|
| `review_synthesizer` | 10+ review texts per product | `get_reviews()` expanded from 3 → 10+ scrapers and returns full bodies |
| `delivery_compare` | `shipping_cost`, `shipping_time_days` per product | Listing-page scrapers extract these OR a cart-probe pass is added |
| `sentiment.analyze_reviews_batch` | Review text (not just counts) | Same condition as review_synthesizer |
| `return_analyzer` | Per-product return policy or at least a store-level lookup | Either a static store-policy table or scraper extraction |

**Stub behavior:** `product_research.json` includes these as steps 1.2 / 1.3 so the JSON stays valid and the mission completes. Their handler returns `{"status": "disabled", "reason": "..."}` — downstream steps (and the human reading logs) can see which modules are inert.

---

## Dead Until Price History

`timing` (buy-now vs wait) is a special case. It's not just waiting on scrapers — it needs a `price_history` table plus a daily re-scrape job to populate it. Until then:
- `timing.advise_timing` stays wired to the stub.
- `check_price_inflation` in `fake_discount_detector` similarly falls back to the "no history" branch, meaning we can detect *cross-store* inflation (which we now do) but not *temporal* inflation.

---

## Migration Roadmap

Tied to scraper improvements, not calendar dates:

1. **When `get_reviews()` supports 10+ scrapers returning full bodies** → replace `synthesize_product_reviews` stub with real `review_synthesizer` + `sentiment` pipeline.
2. **When listing scrapers populate `shipping_time_days` ≥70% of the time** → replace `compare_delivery_options` stub with real `delivery_compare`.
3. **When a `price_history` table exists with ≥14 days of data** → replace `advise_buy_timing` stub with real `timing.advise_timing`, and extend `fake_discount_detector` to use temporal inflation on top of cross-store.
4. **When community scrapers return post body, not just title** → upgrade community relevance filter to score on body text, not just title.

Each migration is a one-line handler swap in `_STEP_HANDLERS` plus the underlying wiring — the JSON workflow itself doesn't change.
