# Shopping Workflows Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split shopping workflows into three purposes — `quick_search` (unchanged), new `product_research` (specific product, lean but scaffolded for future review/delivery/timing modules), `shopping` (refocused on category/discovery). Wire relevance filter for community results, deterministic fake-discount detection, and a Telegram intent fork so the user picks "specific product" vs "category" when starting `/shop`.

**Architecture:**
- New workflow `src/workflows/shopping/product_research.json` calls new `shopping_pipeline` step handlers (all deterministic Python).
- Community data (Şikayetvar, Technopat) is filtered with the existing `_filter_relevant` (strict mode) before being returned — same token-match logic already used for products.
- Fake-discount detection runs inside `_match_and_flatten`: for each matched-product group, computes per-store `original/discounted` ratio, flags stores whose ratio is >1.5× the group median and >1.2 absolute.
- Telegram `/shop` with no args sends inline buttons "🎯 Belirli ürün" / "🏷 Kategori". Callback stores a `sub_intent` in `_pending_action`; the next user message creates a mission with that sub_intent, routing to `product_research` or `shopping`.
- LLM-based review synthesis, delivery comparison, and buy-timing steps are intentionally *not wired* — those depend on richer scraper data (review text, shipping fields, price history) that doesn't exist yet. Stubs are registered so the JSON stays valid; they return a neutral "not enough data" result until scrapers catch up.

**Tech Stack:** Python 3.10, async, `python-telegram-bot` v20, `pytest` (unittest-style), existing shopping pipeline modules under `src/shopping/` and `src/workflows/shopping/`.

---

## File Structure

**Create:**
- `src/workflows/shopping/product_research.json` — 2-phase specific-product workflow
- `tests/shopping/test_product_research.py` — workflow/step-handler tests

**Modify:**
- `src/workflows/shopping/pipeline.py`
  - Add `strict` kwarg to `_filter_relevant` (return empty instead of original when nothing passes)
  - Apply relevance filter to community data in `_step_search` and `_step_search_and_reviews`
  - Add `_annotate_fake_discounts(groups)` helper; call it inside `_match_and_flatten` before flattening
  - Add new handlers: `_step_search_for_product`, `_step_enrich_product`, `_step_deliver_product_research`
  - Add stub handlers: `_step_synthesize_product_reviews`, `_step_compare_delivery_options`, `_step_advise_buy_timing`
  - Register all new handlers in `_STEP_HANDLERS`
- `src/workflows/shopping/shopping.json`
  - Metadata description and step instructions reworded for category/discovery (drop model-specific framing in 1.1 and 3.1)
- `src/app/telegram_bot.py`
  - `cmd_shop()` (around line 4495): when no args, send inline keyboard instead of text prompt
  - `handle_callback()` (around line 4821): handle `shop:specific` and `shop:category` prefixes
  - `_create_shopping_mission()` wf_map (around line 4474): add `"product_research": "product_research"` entry
- `docs/superpowers/specs/2026-04-15-intelligence-modules.md`
  - Add sections: Data Reality Table, Three-Workflow Structure, Disabled-Until-Scrapers-Improve list, Dead-Until-Price-History list, Migration Roadmap

**Test:**
- `tests/shopping/test_pipeline.py` — extend with new cases for community filter, fake-discount flag, new handlers

---

## Task 1: Community relevance filter (strict mode)

**Files:**
- Modify: `src/workflows/shopping/pipeline.py` (function `_filter_relevant` at lines 48–77, step handlers at lines 173+, 253+)
- Test: `tests/shopping/test_pipeline.py`

- [ ] **Step 1.1: Write the failing test — strict mode drops all when nothing passes**

Append to `tests/shopping/test_pipeline.py`:

```python
class TestCommunityRelevanceFilter(unittest.TestCase):
    def test_strict_mode_returns_empty_when_no_match(self):
        from src.workflows.shopping.pipeline import _filter_relevant
        items = [
            {"name": "Ahmet'in cezaevi telefon sikayeti", "source": "sikayetvar"},
            {"name": "Cepte fatura sorunu", "source": "sikayetvar"},
        ]
        # Query has zero overlap with these complaint titles
        out = _filter_relevant(items, "siemens s100 kahve makinesi", strict=True)
        self.assertEqual(out, [])

    def test_strict_mode_keeps_matches(self):
        from src.workflows.shopping.pipeline import _filter_relevant
        items = [
            {"name": "Siemens S100 kahve makinesi arıza", "source": "sikayetvar"},
            {"name": "Tamamen alakasız bir post", "source": "teknopat"},
        ]
        out = _filter_relevant(items, "siemens s100 kahve", strict=True)
        names = [i["name"] for i in out]
        self.assertIn("Siemens S100 kahve makinesi arıza", names)
        self.assertNotIn("Tamamen alakasız bir post", names)

    def test_default_mode_still_falls_back(self):
        """Non-strict mode keeps existing behavior — return original if nothing passes."""
        from src.workflows.shopping.pipeline import _filter_relevant
        items = [{"name": "Cezaevi telefon", "source": "sikayetvar"}]
        out = _filter_relevant(items, "coffee machine", strict=False)
        self.assertEqual(out, items)  # fallback to original
```

- [ ] **Step 1.2: Run test to confirm failure**

Run: `timeout 30 pytest tests/shopping/test_pipeline.py::TestCommunityRelevanceFilter -v`
Expected: FAIL — `TypeError: _filter_relevant() got an unexpected keyword argument 'strict'`

- [ ] **Step 1.3: Add the `strict` parameter to `_filter_relevant`**

Replace the function at `src/workflows/shopping/pipeline.py:48-77` with:

```python
def _filter_relevant(products: list, query: str, strict: bool = False) -> list:
    """Keep only products whose names are relevant to *query*.

    Strategy: score every product, then keep those within 20% of the best
    score, with a hard floor of 0.5 (at least half the query tokens must
    appear).  If nothing passes the floor:
      - strict=False (default): return the original list so the user still
        gets *something* (safe for product results).
      - strict=True: return an empty list (used for community results
        where irrelevant complaints are worse than no data).
    """
    if not products or not query:
        return products

    scored = []
    for p in products:
        name = p.name if hasattr(p, "name") else (p.get("name", "") if isinstance(p, dict) else "")
        s = _relevance_score(name, query)
        scored.append((p, s))

    max_score = max(s for _, s in scored)
    threshold = max(max_score * 0.8, 0.5)

    filtered = [p for p, s in scored if s >= threshold]

    dropped = len(products) - len(filtered)
    if dropped:
        logger.info(
            "relevance filter: %d/%d kept (threshold=%.2f, strict=%s)",
            len(filtered), len(products), threshold, strict,
        )

    if filtered:
        return filtered
    return [] if strict else products
```

- [ ] **Step 1.4: Run test to confirm pass**

Run: `timeout 30 pytest tests/shopping/test_pipeline.py::TestCommunityRelevanceFilter -v`
Expected: PASS (3 tests)

- [ ] **Step 1.5: Apply strict community filter in both search handlers**

In `_step_search` at `src/workflows/shopping/pipeline.py:208-209`, replace:

```python
    # ── Relevance filtering ──
    products = _filter_relevant(products, query)
```

with:

```python
    # ── Relevance filtering ──
    products = _filter_relevant(products, query)
    community = _filter_relevant(community, query, strict=True)
```

In `_step_search_and_reviews` at `src/workflows/shopping/pipeline.py:285-286`, replace:

```python
    # ── Relevance filtering ──
    products = _filter_relevant(products, query)
```

with:

```python
    # ── Relevance filtering ──
    products = _filter_relevant(products, query)
    community = _filter_relevant(community, query, strict=True)
```

- [ ] **Step 1.6: Full pipeline tests still pass**

Run: `timeout 60 pytest tests/shopping/test_pipeline.py -v`
Expected: all PASS (old tests + 3 new ones).

- [ ] **Step 1.7: Commit**

```bash
git add src/workflows/shopping/pipeline.py tests/shopping/test_pipeline.py
git commit -m "feat(shopping): strict relevance filter for community data

Şikayetvar/Technopat results share the same scrape query as product
search, so unrelated complaint titles were leaking through. Re-use
_filter_relevant with a new strict=True flag — empties the list when
nothing passes the token-match floor instead of falling back to the
original noisy list."
```

---

## Task 2: Fake-discount detection (cross-store ratio outlier)

**Files:**
- Modify: `src/workflows/shopping/pipeline.py` (`_match_and_flatten` at lines 80–126)
- Test: `tests/shopping/test_pipeline.py`

**Design note:** For each matched-product group, compute `original_price / discounted_price` per store. If the group has ≥2 entries with both fields populated, take the median ratio. Any entry whose ratio is >1.5× the median AND >1.2 absolute gets `is_suspicious_discount=True` with a human-readable `discount_flag_reason`. Runs inside `_match_and_flatten` while group info is still available.

- [ ] **Step 2.1: Write failing test — outlier ratio gets flagged**

Append to `tests/shopping/test_pipeline.py`:

```python
class TestFakeDiscountAnnotation(unittest.TestCase):
    def test_outlier_ratio_flagged(self):
        from src.workflows.shopping.pipeline import _annotate_fake_discounts
        # Store A claims 50% off, B and C show ~10% off → A's "original" is inflated
        group = {
            "products": [
                {"name": "X", "source": "trendyol",
                 "original_price": 10000, "discounted_price": 5000, "url": "a"},
                {"name": "X", "source": "hepsiburada",
                 "original_price": 5500, "discounted_price": 5000, "url": "b"},
                {"name": "X", "source": "amazon_tr",
                 "original_price": 5200, "discounted_price": 4800, "url": "c"},
            ],
        }
        flags = _annotate_fake_discounts([group])
        # flags is dict keyed by (name, source, url) -> {is_suspicious_discount, reason}
        self.assertTrue(flags[("X", "trendyol", "a")]["is_suspicious_discount"])
        self.assertFalse(flags.get(("X", "hepsiburada", "b"), {}).get("is_suspicious_discount", False))

    def test_no_flag_when_consistent(self):
        from src.workflows.shopping.pipeline import _annotate_fake_discounts
        group = {
            "products": [
                {"name": "Y", "source": "trendyol",
                 "original_price": 5200, "discounted_price": 5000, "url": "a"},
                {"name": "Y", "source": "hepsiburada",
                 "original_price": 5500, "discounted_price": 5000, "url": "b"},
            ],
        }
        flags = _annotate_fake_discounts([group])
        for key, f in flags.items():
            self.assertFalse(f.get("is_suspicious_discount", False))

    def test_single_entry_group_skipped(self):
        from src.workflows.shopping.pipeline import _annotate_fake_discounts
        group = {"products": [
            {"name": "Z", "source": "trendyol",
             "original_price": 10000, "discounted_price": 1000, "url": "a"},
        ]}
        flags = _annotate_fake_discounts([group])
        # Not enough data to compare — no flag
        self.assertEqual(flags, {})
```

- [ ] **Step 2.2: Run to confirm failure**

Run: `timeout 30 pytest tests/shopping/test_pipeline.py::TestFakeDiscountAnnotation -v`
Expected: FAIL — `ImportError: cannot import name '_annotate_fake_discounts'`

- [ ] **Step 2.3: Add `_annotate_fake_discounts`**

Add after `_filter_relevant` (before `_match_and_flatten`) in `src/workflows/shopping/pipeline.py`:

```python
def _annotate_fake_discounts(groups: list[dict]) -> dict[tuple, dict]:
    """Flag stores whose discount ratio is way off from the group median.

    For each matched-product group, compute ``original_price /
    discounted_price`` per store. If the group has ≥2 entries with both
    fields populated, flag any entry whose ratio is >1.5× the median AND
    >1.2 absolute (i.e. claiming at least 20% off when peers aren't).

    Returns a dict keyed by ``(name, source, url)`` tuple → flag payload.
    Empty dict when no flags apply.
    """
    import statistics

    flags: dict[tuple, dict] = {}
    for group in groups:
        entries = group.get("products", [])
        if len(entries) < 2:
            continue

        pairs: list[tuple[dict, float]] = []
        for e in entries:
            orig = e.get("original_price")
            disc = e.get("discounted_price")
            if orig and disc and disc > 0 and orig > disc:
                pairs.append((e, orig / disc))

        if len(pairs) < 2:
            continue

        ratios = [r for _, r in pairs]
        median = statistics.median(ratios)
        for entry, ratio in pairs:
            if ratio > median * 1.5 and ratio > 1.2:
                key = (entry.get("name", ""), entry.get("source", ""), entry.get("url", ""))
                flags[key] = {
                    "is_suspicious_discount": True,
                    "discount_flag_reason": (
                        f"Bu mağazada indirim oranı ({(ratio - 1) * 100:.0f}%) "
                        f"diğer mağazalardaki medyana ({(median - 1) * 100:.0f}%) "
                        f"göre çok yüksek — 'orijinal fiyat' şişirilmiş olabilir"
                    ),
                }
    return flags
```

- [ ] **Step 2.4: Wire flags into `_match_and_flatten`**

Modify `_match_and_flatten` in `src/workflows/shopping/pipeline.py:80-126`. Replace the block starting at `try:\n        from src.shopping.intelligence.product_matcher import match_products` through the end of the function with:

```python
    try:
        from src.shopping.intelligence.product_matcher import match_products
        groups = await match_products(product_objs)
    except Exception as exc:
        logger.warning("product_matcher failed, returning unmatched: %s", exc)
        return list(orig_lookup.values()) + plain_dicts

    # ── Fake-discount flags from cross-store ratio analysis ──
    flags = _annotate_fake_discounts(groups)

    # Flatten: for each group, look up the original full dict for every
    # product entry.  Fall back to the matcher's stripped dict if lookup
    # misses (shouldn't happen, but be safe).
    flat: list[dict] = []
    for group in groups:
        for prod in group.get("products", []):
            key = (prod.get("name", ""), prod.get("source", ""), prod.get("url", ""))
            full = orig_lookup.get(key, prod)
            flag = flags.get(key)
            if flag:
                full = {**full, **flag}
            flat.append(full)

    flat.extend(plain_dicts)
    return flat
```

- [ ] **Step 2.5: Run tests to confirm pass**

Run: `timeout 30 pytest tests/shopping/test_pipeline.py::TestFakeDiscountAnnotation -v`
Expected: PASS (3 tests).

- [ ] **Step 2.6: Commit**

```bash
git add src/workflows/shopping/pipeline.py tests/shopping/test_pipeline.py
git commit -m "feat(shopping): flag outlier discounts via cross-store median

Detects inflated 'original price' claims by comparing each store's
original/discounted ratio to the median across all matched stores for
the same product. Pure Python, no LLM. Flag attaches as
is_suspicious_discount + discount_flag_reason on the product dict, ready
for the formatter to render."
```

---

## Task 3: New step handlers for `product_research`

**Files:**
- Modify: `src/workflows/shopping/pipeline.py` (add handlers, extend `_STEP_HANDLERS`)
- Test: `tests/shopping/test_product_research.py` (new file)

**Design note:** We reuse the existing `_step_search_and_reviews` search/filter/score/match logic via a wrapper that renames the step for clarity. The enrich step is where new, deterministic enrichments (delivery-neutral, installment, fake-discount counts) are aggregated. The deliver step formats without any LLM. Stubs for future LLM steps return a neutral placeholder artifact.

- [ ] **Step 3.1: Write failing test — new handlers dispatchable**

Create `tests/shopping/test_product_research.py`:

```python
"""Tests for product_research workflow step handlers (deterministic)."""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from src.shopping.models import Product


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _fake_products():
    return [
        Product(
            name="Siemens S100 Inverter",
            url="https://www.trendyol.com/siemens-s100",
            source="trendyol",
            original_price=4500.0,
            discounted_price=3999.0,
            rating=4.5,
            review_count=42,
        ),
        Product(
            name="Siemens S100 AC Drive",
            url="https://www.hepsiburada.com/siemens-s100",
            source="hepsiburada",
            original_price=4800.0,
            discounted_price=4200.0,
        ),
    ]


class TestProductResearchHandlers(unittest.TestCase):

    def test_search_for_product_returns_products(self):
        from src.workflows.shopping.pipeline import ShoppingPipeline

        pipeline = ShoppingPipeline()
        task = {
            "mission_id": 77,
            "context": {
                "step_name": "search_for_product",
                "input_artifacts": ["user_query"],
            },
        }

        with patch(
            "src.workflows.shopping.pipeline._read_artifacts",
            new=AsyncMock(return_value={"user_query": "siemens s100"}),
        ), patch(
            "src.shopping.resilience.fallback_chain.get_product_with_fallback",
            new=AsyncMock(return_value=_fake_products()),
        ), patch(
            "src.shopping.resilience.fallback_chain.get_community_data",
            new=AsyncMock(return_value=[]),
        ):
            out = run_async(pipeline.run(task))

        self.assertEqual(out["status"], "completed")
        data = json.loads(out["result"])
        self.assertGreaterEqual(data["product_count"], 1)

    def test_enrich_product_returns_enrichment_dict(self):
        from src.workflows.shopping.pipeline import ShoppingPipeline

        pipeline = ShoppingPipeline()
        search_artifact = json.dumps({
            "products": [
                {"name": "X", "source": "trendyol",
                 "original_price": 5000, "discounted_price": 4500, "url": "a"},
                {"name": "X", "source": "hepsiburada",
                 "original_price": 5200, "discounted_price": 4800, "url": "b"},
            ],
            "community": [],
            "product_count": 2,
        })
        task = {
            "mission_id": 77,
            "context": {
                "step_name": "enrich_product_results",
                "input_artifacts": ["search_results", "user_query"],
            },
        }
        with patch(
            "src.workflows.shopping.pipeline._read_artifacts",
            new=AsyncMock(return_value={
                "search_results": search_artifact,
                "user_query": "X",
            }),
        ):
            out = run_async(pipeline.run(task))

        self.assertEqual(out["status"], "completed")
        data = json.loads(out["result"])
        self.assertIn("products", data)
        self.assertIn("cross_store_summary", data)
        self.assertIn("suspicious_discount_count", data["cross_store_summary"])

    def test_deliver_product_research_formats_for_telegram(self):
        from src.workflows.shopping.pipeline import ShoppingPipeline

        pipeline = ShoppingPipeline()
        enriched = json.dumps({
            "products": [
                {"name": "Siemens S100", "source": "trendyol",
                 "discounted_price": 3999, "url": "https://t.co/x",
                 "rating": 4.5, "review_count": 42},
            ],
            "community": [],
            "cross_store_summary": {
                "store_count": 1, "suspicious_discount_count": 0,
            },
        })
        task = {
            "mission_id": 77,
            "context": {
                "step_name": "deliver_product_research",
                "input_artifacts": ["enriched_product_data", "user_query"],
            },
        }
        with patch(
            "src.workflows.shopping.pipeline._read_artifacts",
            new=AsyncMock(return_value={
                "enriched_product_data": enriched,
                "user_query": "Siemens S100",
            }),
        ):
            out = run_async(pipeline.run(task))

        self.assertEqual(out["status"], "completed")
        # Format step returns a string (not JSON)
        self.assertIsInstance(out["result"], str)
        self.assertIn("Siemens S100", out["result"])
        self.assertIn("3999", out["result"].replace(".", "").replace(",", ""))

    def test_stub_review_synthesis_returns_placeholder(self):
        from src.workflows.shopping.pipeline import ShoppingPipeline

        pipeline = ShoppingPipeline()
        task = {
            "mission_id": 77,
            "context": {
                "step_name": "synthesize_product_reviews",
                "input_artifacts": ["enriched_product_data"],
            },
        }
        with patch(
            "src.workflows.shopping.pipeline._read_artifacts",
            new=AsyncMock(return_value={"enriched_product_data": "{}"}),
        ):
            out = run_async(pipeline.run(task))
        self.assertEqual(out["status"], "completed")
        data = json.loads(out["result"])
        self.assertEqual(data["status"], "disabled")
        self.assertIn("scraper", data["reason"].lower())
```

- [ ] **Step 3.2: Run to confirm failure**

Run: `timeout 30 pytest tests/shopping/test_product_research.py -v`
Expected: FAIL on all 4 tests — unknown step names.

- [ ] **Step 3.3: Add the search wrapper handler**

After `_step_search_and_reviews` (around `src/workflows/shopping/pipeline.py:358`), add:

```python
async def _step_search_for_product(task: dict, artifacts: dict) -> str:
    """Search step for product_research workflow.

    Same search/filter/score/match logic as _step_search_and_reviews but
    named distinctly so the workflow JSON can be read for its intent.
    Reviews are fetched when a matching scraper exists; if the scraper
    doesn't expose detailed reviews yet, the reviews field is an empty
    list — the enrich step treats that as 'no data' gracefully.
    """
    return await _step_search_and_reviews(task, artifacts)
```

- [ ] **Step 3.4: Add the enrich handler**

After `_step_search_for_product`, add:

```python
async def _step_enrich_product(task: dict, artifacts: dict) -> str:
    """Deterministic enrichment for specific-product research.

    Reads ``search_results``, attaches a ``cross_store_summary`` section
    (store count, how many flagged as suspicious discounts, price spread),
    and passes the product list through unchanged.  No LLM. No review
    synthesis, delivery calculation, or timing — those are stubs until
    scrapers provide the underlying data.
    """
    from src.shopping.intelligence.special.fake_discount_detector import (
        check_cross_store_consistency,
    )

    raw = artifacts.get("search_results", "{}")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            data = {}
    else:
        data = raw if isinstance(raw, dict) else {}

    products = data.get("products", [])
    community = data.get("community", [])
    reviews = data.get("reviews", [])

    # Price spread across stores (uses first product group's prices;
    # post-match each source appears once with its best match).
    prices_by_source: dict[str, float] = {}
    for p in products:
        src = p.get("source", "")
        price = p.get("discounted_price") or p.get("original_price")
        if src and price and src not in prices_by_source:
            prices_by_source[src] = float(price)

    consistency = check_cross_store_consistency(prices_by_source)
    suspicious_count = sum(
        1 for p in products if p.get("is_suspicious_discount")
    )

    cross_store_summary = {
        "store_count": len(prices_by_source),
        "suspicious_discount_count": suspicious_count,
        "price_spread_pct": consistency.get("spread_pct", 0.0),
        "cheapest_store": consistency.get("cheapest"),
        "most_expensive_store": consistency.get("most_expensive"),
        "notes": consistency.get("notes", []),
    }

    enriched = {
        "products": products,
        "community": community,
        "reviews": reviews,
        "cross_store_summary": cross_store_summary,
        "product_count": data.get("product_count", len(products)),
        "community_count": data.get("community_count", len(community)),
    }
    return json.dumps(enriched, default=str, ensure_ascii=False)
```

- [ ] **Step 3.5: Add the delivery (formatting) handler**

After `_step_enrich_product`, add:

```python
async def _step_deliver_product_research(task: dict, artifacts: dict) -> str:
    """Format enriched product research data for Telegram delivery.

    Reuses _step_format but passes the enriched artifact in under the
    ``search_results`` key so the existing formatter logic (winner,
    others, community, reviews) runs unchanged. Adds a "fake discount"
    callout when any products carry is_suspicious_discount=True.
    """
    enriched_raw = artifacts.get("enriched_product_data", "{}")
    # Adapt shape: formatter reads artifacts["search_results"]
    adapted = {
        "search_results": enriched_raw,
        "user_query": artifacts.get("user_query", ""),
    }
    base = await _step_format(task, adapted)

    # Append fake-discount callout if any flags present
    try:
        data = json.loads(enriched_raw) if isinstance(enriched_raw, str) else enriched_raw
    except (json.JSONDecodeError, ValueError):
        data = {}
    suspicious = [
        p for p in data.get("products", [])
        if p.get("is_suspicious_discount")
    ]
    if suspicious:
        base += "\n\n⚠️ *Şüpheli İndirim:*\n"
        for p in suspicious[:3]:
            src = p.get("source", "?")
            reason = p.get("discount_flag_reason", "")
            base += f"  • {src}: {reason}\n"

    return base
```

- [ ] **Step 3.6: Add stub handlers for LLM-dependent steps**

After `_step_deliver_product_research`, add:

```python
async def _step_stub_disabled(task: dict, artifacts: dict) -> str:
    """Placeholder for steps that depend on data scrapers don't provide yet.

    Returns a neutral status=disabled artifact so downstream steps can
    check and skip. When scrapers improve (review text, price history,
    shipping fields), replace this handler with the real implementation.
    """
    context = task.get("context", {}) if isinstance(task, dict) else {}
    step_name = context.get("step_name", "unknown") if isinstance(context, dict) else "unknown"
    return json.dumps({
        "status": "disabled",
        "step": step_name,
        "reason": "Scraper data insufficient — this module activates when "
                  "scrapers populate review text / shipping fields / price history.",
    }, ensure_ascii=False)
```

- [ ] **Step 3.7: Register handlers in `_STEP_HANDLERS`**

Replace `_STEP_HANDLERS` at `src/workflows/shopping/pipeline.py:634-639`:

```python
_STEP_HANDLERS = {
    # quick_search
    "execute_product_search": _step_search,
    "format_and_deliver": _step_format,
    # shopping (full category/discovery workflow)
    "search_and_collect_reviews": _step_search_and_reviews,
    "understand_query_check_clarity": _step_analyze_query,
    # product_research (specific product workflow — NEW)
    "search_for_product": _step_search_for_product,
    "enrich_product_results": _step_enrich_product,
    "deliver_product_research": _step_deliver_product_research,
    # product_research stubs — scaffolded, activate when scrapers improve
    "synthesize_product_reviews": _step_stub_disabled,
    "compare_delivery_options": _step_stub_disabled,
    "advise_buy_timing": _step_stub_disabled,
}
```

- [ ] **Step 3.8: Run tests to confirm pass**

Run: `timeout 30 pytest tests/shopping/test_product_research.py -v`
Expected: PASS (4 tests).

Also run: `timeout 30 pytest tests/shopping/test_pipeline.py -v`
Expected: still PASS (no regression).

- [ ] **Step 3.9: Commit**

```bash
git add src/workflows/shopping/pipeline.py tests/shopping/test_product_research.py
git commit -m "feat(shopping): product_research step handlers (deterministic)

Adds search_for_product, enrich_product_results, and
deliver_product_research — pure Python, no LLM. Plus three stubs
(synthesize_product_reviews, compare_delivery_options, advise_buy_timing)
that return a neutral 'disabled' artifact until the underlying scraper
data exists. Cross-store price consistency and is_suspicious_discount
flags now bubble through the delivery output."
```

---

## Task 4: Write `product_research.json` workflow

**Files:**
- Create: `src/workflows/shopping/product_research.json`
- Test: extend `tests/shopping/test_product_research.py`

- [ ] **Step 4.1: Write failing test — workflow loads cleanly**

Append to `tests/shopping/test_product_research.py`:

```python
class TestProductResearchWorkflow(unittest.TestCase):
    def test_workflow_loads(self):
        from src.workflows.engine.loader import load_workflow
        wf = run_async(load_workflow("product_research"))
        self.assertEqual(wf["plan_id"], "product_research")
        step_names = [s["name"] for s in wf["steps"]]
        # Live steps
        self.assertIn("search_for_product", step_names)
        self.assertIn("enrich_product_results", step_names)
        self.assertIn("deliver_product_research", step_names)
        # Stub steps (scaffolded)
        self.assertIn("synthesize_product_reviews", step_names)
        self.assertIn("compare_delivery_options", step_names)
        self.assertIn("advise_buy_timing", step_names)

    def test_every_step_has_a_registered_handler(self):
        from src.workflows.engine.loader import load_workflow
        from src.workflows.shopping.pipeline import _STEP_HANDLERS

        wf = run_async(load_workflow("product_research"))
        for step in wf["steps"]:
            name = step["name"]
            self.assertIn(
                name, _STEP_HANDLERS,
                f"Step {name!r} has no handler in _STEP_HANDLERS",
            )
```

- [ ] **Step 4.2: Confirm failure**

Run: `timeout 30 pytest tests/shopping/test_product_research.py::TestProductResearchWorkflow -v`
Expected: FAIL — workflow file missing.

- [ ] **Step 4.3: Create `product_research.json`**

Write `src/workflows/shopping/product_research.json`:

```json
{
  "plan_id": "product_research",
  "version": "1.0",
  "metadata": {
    "description": "Specific-product research workflow. User already named a product (brand+model or exact item); pipeline searches all scrapers, enriches with deterministic signals (cross-store price consistency, fake-discount flag, installment calc), and delivers a Telegram-ready summary. Review-synthesis, delivery-comparison, and buy-timing steps are scaffolded but disabled until scrapers populate review text, shipping fields, and price history. All live steps are pure Python — no LLM cost.",
    "agents_required": ["shopping_pipeline"],
    "performance_target": {
      "max_duration_seconds": 90,
      "escalation_trigger": "no_products_found"
    },
    "timeout_hours": 1
  },
  "phases": [
    {
      "id": "phase_0",
      "name": "Search",
      "goal": "Run scrapers + community and capture structured results.",
      "depends_on_phases": []
    },
    {
      "id": "phase_1",
      "name": "Enrich",
      "goal": "Deterministic enrichment — cross-store consistency, fake-discount flags, installment. Plus three scaffolded stubs that will activate when scraper data improves.",
      "depends_on_phases": ["phase_0"]
    },
    {
      "id": "phase_2",
      "name": "Deliver",
      "goal": "Format enriched data for Telegram.",
      "depends_on_phases": ["phase_1"]
    }
  ],
  "steps": [
    {
      "id": "0.1",
      "phase": "phase_0",
      "name": "search_for_product",
      "title": "Search Product",
      "agent": "shopping_pipeline",
      "difficulty": "easy",
      "tools_hint": [],
      "depends_on": [],
      "input_artifacts": ["user_query"],
      "output_artifacts": ["search_results"],
      "instruction": "Run the shopping search pipeline for the specific product named in user_query. The pipeline handles 16 e-commerce scrapers + 4 community sources, applies product-name relevance filter, scores with value_scorer, matches cross-source duplicates, and flags discount outliers via cross-store median comparison. No LLM involvement.",
      "done_when": "search_results contains products[] with prices, community[] filtered to this product, and per-entry is_suspicious_discount flags where applicable."
    },
    {
      "id": "1.1",
      "phase": "phase_1",
      "name": "enrich_product_results",
      "title": "Enrich with Cross-Store Signals",
      "agent": "shopping_pipeline",
      "difficulty": "easy",
      "tools_hint": [],
      "depends_on": ["0.1"],
      "input_artifacts": ["search_results", "user_query"],
      "output_artifacts": ["enriched_product_data"],
      "instruction": "Read search_results. Compute cross_store_summary: store_count, suspicious_discount_count, price_spread_pct, cheapest_store, most_expensive_store. Pass products/community/reviews through unchanged. Pure Python.",
      "done_when": "enriched_product_data has products, community, cross_store_summary fields."
    },
    {
      "id": "1.2",
      "phase": "phase_1",
      "name": "synthesize_product_reviews",
      "title": "(Stub) Synthesize Reviews",
      "agent": "shopping_pipeline",
      "difficulty": "easy",
      "tools_hint": [],
      "depends_on": ["1.1"],
      "input_artifacts": ["enriched_product_data"],
      "output_artifacts": ["review_synthesis"],
      "instruction": "SCAFFOLD ONLY — returns status=disabled until scrapers expose review text at volume. When activated, will call src.shopping.intelligence.review_synthesizer.synthesize_reviews for theme extraction, defect patterns, and confidence-adjusted rating.",
      "done_when": "review_synthesis artifact written (placeholder allowed)."
    },
    {
      "id": "1.3",
      "phase": "phase_1",
      "name": "compare_delivery_options",
      "title": "(Stub) Compare Delivery",
      "agent": "shopping_pipeline",
      "difficulty": "easy",
      "tools_hint": [],
      "depends_on": ["1.1"],
      "input_artifacts": ["enriched_product_data"],
      "output_artifacts": ["delivery_comparison"],
      "instruction": "SCAFFOLD ONLY — returns status=disabled until scrapers populate shipping_cost and shipping_time_days consistently. When activated, will call src.shopping.intelligence.delivery_compare.compare_delivery for effective price + ETA per store.",
      "done_when": "delivery_comparison artifact written (placeholder allowed)."
    },
    {
      "id": "1.4",
      "phase": "phase_1",
      "name": "advise_buy_timing",
      "title": "(Stub) Buy Timing",
      "agent": "shopping_pipeline",
      "difficulty": "easy",
      "tools_hint": [],
      "depends_on": ["1.1"],
      "input_artifacts": ["enriched_product_data"],
      "output_artifacts": ["timing_advice"],
      "instruction": "SCAFFOLD ONLY — returns status=disabled until price history exists in DB. When activated, will call src.shopping.intelligence.timing.advise_timing with Turkish sale-calendar and historical price trend.",
      "done_when": "timing_advice artifact written (placeholder allowed)."
    },
    {
      "id": "2.1",
      "phase": "phase_2",
      "name": "deliver_product_research",
      "title": "Deliver Research Summary",
      "agent": "shopping_pipeline",
      "difficulty": "easy",
      "tools_hint": [],
      "depends_on": ["1.1"],
      "input_artifacts": ["enriched_product_data", "user_query"],
      "output_artifacts": ["shopping_response"],
      "instruction": "Format enriched_product_data for Telegram. Reuses the standard product formatter (winner, others, community, reviews) and appends a fake-discount warning block if any products carry is_suspicious_discount=True. No LLM.",
      "done_when": "shopping_response is a user-ready Markdown message string."
    }
  ]
}
```

- [ ] **Step 4.4: Run workflow-load tests**

Run: `timeout 30 pytest tests/shopping/test_product_research.py::TestProductResearchWorkflow -v`
Expected: PASS (2 tests).

Also run: `timeout 30 pytest tests/shopping/test_product_research.py -v`
Expected: all PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/workflows/shopping/product_research.json tests/shopping/test_product_research.py
git commit -m "feat(shopping): product_research workflow (lean + scaffolded)

Three live steps (search → enrich → deliver, all deterministic Python)
and three scaffold-stubs (synthesize_product_reviews,
compare_delivery_options, advise_buy_timing) that return
status=disabled until scrapers populate review text, shipping fields,
and price history."
```

---

## Task 5: Refocus `shopping.json` on category/discovery

**Files:**
- Modify: `src/workflows/shopping/shopping.json`

**Design note:** The current `shopping.json` mixes specific-product and category query framing. We're narrowing it to category/discovery — the specific-product path now has its own workflow (product_research). We drop the "siemens s100" / "dyson v15" brand+model examples from the clarifier and analyst instructions, and reword 0.1 / 1.1 / 3.1 / 4.1 to frame decisions in terms of category exploration (budget, use case, category segment).

- [ ] **Step 5.1: Update metadata description**

In `src/workflows/shopping/shopping.json` at line 5, replace:

```
    "description": "Full deep-research shopping workflow. Takes a user shopping query through intent parsing, optional clarification, multi-source search (16 e-commerce sites + 4 community forums via shopping_search), review collection, deal analysis, recommendation synthesis, and Telegram-optimized delivery. shopping_search handles scraping, matching, and community data automatically — LLM steps focus on analysis, judgment, and presentation.",
```

with:

```
    "description": "Category/discovery shopping workflow. Handles broad exploratory queries like 'good coffee machine' or 'gaming laptop under 30k' — parses category + constraints, clarifies when too vague, searches the category across 16 e-commerce sites + 4 community forums, analyzes value/risks across options, and synthesizes a tiered recommendation (best overall / budget pick / alternatives). For named specific products (brand+model), the product_research workflow is used instead.",
```

- [ ] **Step 5.2: Reword step 0.1 instruction (drop model-specific framing)**

In `src/workflows/shopping/shopping.json` at line 75, replace the `"instruction"` value for step 0.1 with:

```
    "instruction": "Read the user's query — this workflow handles category/discovery queries (specific product lookups are routed to product_research). Extract: category or product-type, budget if mentioned, key use case, brand preferences if narrow. Check user profile for past preferences. If the query has no category AND no use case AND no constraint, set needs_clarification=true and list 1-3 specific questions (budget? use case? must-have feature?). Otherwise set needs_clarification=false.",
```

- [ ] **Step 5.3: Reword step 1.1 instruction (category-focused clarification)**

In `src/workflows/shopping/shopping.json` at lines 95-98, replace the 1.1 step's `instruction` value with:

```
    "instruction": "Read parsed_intent and user_query. This workflow handles CATEGORY/DISCOVERY queries — the user named a category or broad need, not a specific brand+model. Decide: does the query have enough constraints (budget, use case, key feature) to shortlist candidates, or should we ask 2-3 questions first?\n\nHAS ENOUGH: e.g. 'coffee machine under 5000 TL for a household of 2', 'gaming laptop for League of Legends on 1080p'. Return the query as clarified_query immediately.\n\nNEEDS CLARIFICATION: generic category with no constraints ('coffee machine', '3d printers', 'laptop'). Ask 2-3 tight questions about budget, use case, and one must-have constraint. Use the needs_clarification response to pause.",
```

- [ ] **Step 5.4: Reword step 3.1 instruction (category value analysis)**

In `src/workflows/shopping/shopping.json` at line 131, replace the 3.1 step's `instruction` value with:

```
    "instruction": "Read search_results and review_data. This is a category-discovery analysis: compare candidates against each other on value (price-to-performance), coverage of the user's use case, and risk (recurring complaints on Şikayetvar, thin review counts). Do NOT frame as buy-or-wait for a specific SKU — frame as which segment of the category matches the user best. Use shopping_compare for value scoring across the shortlist. Produce deal_analysis with: per-candidate value verdict, segment recommendations, and red flags to avoid.",
```

- [ ] **Step 5.5: Reword step 4.1 instruction (category recommendation)**

In `src/workflows/shopping/shopping.json` at line 151, replace the 4.1 step's `instruction` value with:

```
    "instruction": "Read all artifacts. Build a tiered CATEGORY recommendation: Best Overall (best value across the shortlist matching the use case), Budget Pick (cheapest that still meets the key requirement), and 1-2 alternatives from adjacent segments. Use shopping_alternatives if a different brand family or previous-gen meets the need at lower cost. Include warnings from deal_analysis. Structure as clear readable text with category rationale, product names, prices, sources, and when each pick fits best.",
```

- [ ] **Step 5.6: Smoke-test workflow still loads and has all handlers**

Run: `timeout 30 pytest tests/shopping/test_workflow_dispatch.py -v`
Expected: PASS (no regression — only textual changes in JSON).

Also try a manual import check:

```bash
python -c "import asyncio; from src.workflows.engine.loader import load_workflow; wf = asyncio.run(load_workflow('shopping')); print('steps:', [s['name'] for s in wf['steps']])"
```

Expected: prints all 6 step names (0.1 through 5.1) matching the original structure.

- [ ] **Step 5.7: Commit**

```bash
git add src/workflows/shopping/shopping.json
git commit -m "refactor(shopping): refocus shopping workflow on category/discovery

Drops specific-brand+model framing from clarifier and analyst steps
(that path is now product_research). Description and instructions
explicitly call this the category/discovery workflow. No structural
changes — same steps, same handlers."
```

---

## Task 6: Telegram intent fork (inline buttons on /shop)

**Files:**
- Modify: `src/app/telegram_bot.py` (cmd_shop ~4495, handle_callback ~4821, _create_shopping_mission wf_map ~4474)
- Test: `tests/test_shopping_mission.py` (extend)

**Design note:** When `/shop` is called with no args, send inline buttons instead of a text prompt. Callback sets `_pending_action[chat_id] = {"command": "shop", "sub_intent": "...", "ts": ...}`. The existing pending-action flow at line 3628 calls `_resolve_cmd_handler("shop")` with `context.args = text.split()` — we extend `cmd_shop` to read `self._pending_action_followup.get(chat_id, {}).get("sub_intent")` OR stash the sub_intent on `self` before dispatch. Simpler: stash it on the `pending_action` dict and let the pending-action dispatcher pass it via a new attribute.

We'll use a per-chat `self._pending_shop_subintent: dict[int, str]` wiped on use.

- [ ] **Step 6.1: Write failing test — cmd_shop no-args sends inline keyboard**

Append to `tests/test_shopping_mission.py` (or create `tests/test_shopping_intent_fork.py` if you prefer — keep consistency with existing test layout):

```python
class TestShopIntentFork(unittest.TestCase):
    def _fresh_interface(self):
        """Build a minimal TelegramInterface shell for testing."""
        from src.app.telegram_bot import TelegramInterface
        # __init__ wires up Telegram; bypass via __new__ + manual init of fields
        iface = TelegramInterface.__new__(TelegramInterface)
        iface._pending_action = {}
        iface._pending_shop_subintent = {}
        iface._kb_state = {}
        return iface

    def test_cmd_shop_no_args_sends_inline_buttons(self):
        from unittest.mock import AsyncMock, MagicMock
        iface = self._fresh_interface()
        update = MagicMock()
        update.effective_chat.id = 42
        msg = MagicMock()
        msg.reply_text = AsyncMock()
        update.message = msg
        context = MagicMock()
        context.args = []

        run_async(iface.cmd_shop(update, context))

        # Should have called reply_text with reply_markup containing inline buttons
        _args, kwargs = msg.reply_text.call_args
        self.assertIn("reply_markup", kwargs)
        markup = kwargs["reply_markup"]
        # Flatten inline keyboard and assert callback_data values
        callback_values = []
        for row in markup.inline_keyboard:
            for btn in row:
                callback_values.append(btn.callback_data)
        self.assertIn("shop:specific", callback_values)
        self.assertIn("shop:category", callback_values)

    def test_callback_shop_specific_sets_pending_subintent(self):
        from unittest.mock import AsyncMock, MagicMock
        iface = self._fresh_interface()
        query = MagicMock()
        query.answer = AsyncMock()
        query.data = "shop:specific"
        query.message = MagicMock()
        query.message.reply_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        update.effective_chat.id = 42
        context = MagicMock()

        run_async(iface.handle_callback(update, context))

        self.assertIn(42, iface._pending_action)
        self.assertEqual(iface._pending_action[42]["command"], "shop")
        self.assertEqual(iface._pending_shop_subintent.get(42), "specific")

    def test_callback_shop_category_sets_pending_subintent(self):
        from unittest.mock import AsyncMock, MagicMock
        iface = self._fresh_interface()
        query = MagicMock()
        query.answer = AsyncMock()
        query.data = "shop:category"
        query.message = MagicMock()
        query.message.reply_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        update.effective_chat.id = 42
        context = MagicMock()

        run_async(iface.handle_callback(update, context))

        self.assertEqual(iface._pending_shop_subintent.get(42), "category")
```

Also ensure the file has `import unittest`, `import asyncio`, and the `run_async` helper near the top (copy from `tests/shopping/test_pipeline.py` if not already present).

- [ ] **Step 6.2: Confirm failure**

Run: `timeout 30 pytest tests/test_shopping_mission.py::TestShopIntentFork -v`
Expected: FAIL — attributes or handlers missing.

- [ ] **Step 6.3: Add `_pending_shop_subintent` to TelegramInterface init**

Find the `__init__` (search for `_pending_action = {}` in telegram_bot.py) and add next to it:

```python
        self._pending_shop_subintent: dict[int, str] = {}
```

Run: `grep -n "_pending_action: " src/app/telegram_bot.py` to find the exact line, then add `self._pending_shop_subintent = {}` right below it.

- [ ] **Step 6.4: Modify `cmd_shop` no-args path to send inline keyboard**

Replace lines 4497–4501 in `src/app/telegram_bot.py`:

```python
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "shop"}
            await self._reply(update, "🛒 What are you looking for?")
            return
```

with:

```python
        if not context.args:
            chat_id = update.effective_chat.id
            # Intent fork: specific product vs category/discovery
            buttons = [[
                InlineKeyboardButton("🎯 Belirli ürün", callback_data="shop:specific"),
                InlineKeyboardButton("🏷 Kategori", callback_data="shop:category"),
            ]]
            await update.message.reply_text(
                "🛒 Ne arıyorsunuz?",
                reply_markup=InlineKeyboardMarkup(buttons),
            )
            return
```

(`InlineKeyboardButton` and `InlineKeyboardMarkup` are already imported — verify with `grep -n "InlineKeyboardButton\|InlineKeyboardMarkup" src/app/telegram_bot.py | head -5`.)

- [ ] **Step 6.5: Add callback branch in `handle_callback`**

Find `handle_callback` (line 4821) and insert right after `data = query.data` (around line 4824), BEFORE the first `if data.startswith(...)`:

```python
        # ── Shopping Intent Fork ──────────────────────────────────
        if data.startswith("shop:"):
            sub = data.split(":", 1)[1]
            chat_id = update.effective_chat.id
            import time as _time
            self._pending_action[chat_id] = {
                "command": "shop",
                "ts": _time.time(),
            }
            self._pending_shop_subintent[chat_id] = sub
            prompt = (
                "🎯 Hangi ürün? (marka + model yazın)"
                if sub == "specific" else
                "🏷 Hangi kategori? (örn. 'kahve makinesi 5000 TL altı')"
            )
            await query.message.reply_text(prompt)
            return
```

- [ ] **Step 6.6: Make `cmd_shop` with args respect `_pending_shop_subintent`**

Replace lines 4502–4523 in `src/app/telegram_bot.py`:

```python
        query = " ".join(context.args)
        chat_id = update.effective_chat.id

        # Two-tier routing: complex queries get a full research mission
        if self._is_complex_shopping_query(query):
            mission_id = await self._create_shopping_mission(query, chat_id)
            await self._reply(
                update,
                f"🔬 Shopping research mission #{mission_id} started.\n"
                "I'll research products, analyze deals, and send you a "
                "recommendation. This may take a few minutes.",
            )
        else:
            # Simple query → quick_search workflow
            mission_id = await self._create_shopping_mission(
                query, chat_id, sub_intent="quick_search"
            )
            await self._reply(
                update,
                f"🛒 Searching for *{query}*... (mission #{mission_id})",
                parse_mode="Markdown",
            )
```

with:

```python
        query = " ".join(context.args)
        chat_id = update.effective_chat.id

        # If the user just picked an intent button, honor it explicitly
        sub = self._pending_shop_subintent.pop(chat_id, None)
        if sub == "specific":
            mission_id = await self._create_shopping_mission(
                query, chat_id, sub_intent="product_research"
            )
            await self._reply(
                update,
                f"🔬 Ürün araştırması başladı: *{query}* (mission #{mission_id})",
                parse_mode="Markdown",
            )
            return
        if sub == "category":
            mission_id = await self._create_shopping_mission(
                query, chat_id, sub_intent="deep_research"
            )
            await self._reply(
                update,
                f"🏷 Kategori araştırması başladı: *{query}* (mission #{mission_id})",
                parse_mode="Markdown",
            )
            return

        # No intent chosen — fall back to existing two-tier heuristic
        if self._is_complex_shopping_query(query):
            mission_id = await self._create_shopping_mission(query, chat_id)
            await self._reply(
                update,
                f"🔬 Shopping research mission #{mission_id} started.\n"
                "I'll research products, analyze deals, and send you a "
                "recommendation. This may take a few minutes.",
            )
        else:
            mission_id = await self._create_shopping_mission(
                query, chat_id, sub_intent="quick_search"
            )
            await self._reply(
                update,
                f"🛒 Searching for *{query}*... (mission #{mission_id})",
                parse_mode="Markdown",
            )
```

- [ ] **Step 6.7: Map `product_research` sub_intent to the new workflow**

In `_create_shopping_mission` at `src/app/telegram_bot.py:4474-4481`, replace:

```python
        wf_map = {
            "deep_research": "shopping",
            "research": "shopping",
            "compare": "combo_research",
            "gift": "gift_recommendation",
            "deals": "exploration",
            "quick_search": "quick_search",
        }
```

with:

```python
        wf_map = {
            "deep_research": "shopping",
            "research": "shopping",
            "compare": "combo_research",
            "gift": "gift_recommendation",
            "deals": "exploration",
            "quick_search": "quick_search",
            "product_research": "product_research",
        }
```

- [ ] **Step 6.8: Run intent-fork tests**

Run: `timeout 30 pytest tests/test_shopping_mission.py::TestShopIntentFork -v`
Expected: PASS (3 tests).

- [ ] **Step 6.9: Full shopping test suite sanity**

Run: `timeout 120 pytest tests/test_shopping_mission.py tests/shopping/ -v`
Expected: PASS. Fix any regressions immediately — intent-fork changes must not break existing flows.

- [ ] **Step 6.10: Commit**

```bash
git add src/app/telegram_bot.py tests/test_shopping_mission.py
git commit -m "feat(telegram): intent fork on /shop — specific product vs category

/shop with no args now asks 'Belirli ürün mü, kategori mi?' with inline
buttons. 'Specific' routes to the new product_research workflow;
'Category' routes to the refocused shopping workflow. /shop with args
keeps the existing two-tier heuristic so power users aren't slowed down."
```

---

## Task 7: Update intelligence-modules spec doc

**Files:**
- Modify: `docs/superpowers/specs/2026-04-15-intelligence-modules.md`

**Design note:** Add four new sections at the end (Data Reality Table, Three-Workflow Structure, Disabled-Until-Scrapers-Improve, Migration Roadmap) plus a short "2026-04-16 Update" preamble. Do NOT rewrite existing sections — they're still valid history; we're layering.

- [ ] **Step 7.1: Append update preamble after current Status line**

In `docs/superpowers/specs/2026-04-15-intelligence-modules.md`, find the line:

```
**Status**: Survey complete, integration pending  
```

Insert immediately after:

```
**2026-04-16 Update**: Three-workflow split landed (`quick_search` unchanged, `product_research` new, `shopping` refocused on category). Community relevance filter + deterministic fake-discount flag wired into pipeline. Review-synthesis / delivery-compare / timing modules remain intentionally *disabled* — see the "Disabled Until Scrapers Improve" section at the bottom. The Priority Order section above is still the roadmap for re-enabling each.
```

- [ ] **Step 7.2: Append new sections at the end of the file**

At the end of `docs/superpowers/specs/2026-04-15-intelligence-modules.md` (after "7. **Community relevance filter**"), append:

```markdown

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
```

- [ ] **Step 7.3: Verify no Markdown syntax errors**

Open the file and eyeball the appended sections. Then:

```bash
python -c "import pathlib; print(pathlib.Path('docs/superpowers/specs/2026-04-15-intelligence-modules.md').read_text(encoding='utf-8').count('```'))"
```

Expected: an even number (every fenced block opens and closes).

- [ ] **Step 7.4: Commit**

```bash
git add docs/superpowers/specs/2026-04-15-intelligence-modules.md
git commit -m "docs: 2026-04-16 update — three-workflow structure + data reality

Adds data reality table, three-workflow structure, disabled-until-
scrapers-improve list, dead-until-price-history list, and migration
roadmap tied to scraper improvements. Existing priority order and
module inventory stay — this is a layer, not a rewrite."
```

---

## Task 8: Full-suite sanity pass

**Files:** None modified — verification only.

- [ ] **Step 8.1: Run all touched tests**

```bash
timeout 120 pytest tests/shopping/ tests/test_shopping_mission.py tests/test_shopping_tools.py tests/test_shopping_scorer.py tests/integration/test_shopping_flow.py -v
```

Expected: all PASS. If any fail, investigate before proceeding — do not mark the plan complete on red tests.

- [ ] **Step 8.2: Verify import hygiene**

```bash
python -c "from src.workflows.shopping.pipeline import _STEP_HANDLERS, _annotate_fake_discounts, _step_search_for_product, _step_enrich_product, _step_deliver_product_research; print('handlers:', sorted(_STEP_HANDLERS.keys()))"
```

Expected: lists all 10 handler keys including the three new ones and three stubs.

- [ ] **Step 8.3: Verify workflow discoverability**

```bash
python -c "import asyncio; from src.workflows.engine.loader import load_workflow; [print(n, '→', len(asyncio.run(load_workflow(n))['steps']), 'steps') for n in ['quick_search', 'product_research', 'shopping']]"
```

Expected: prints `quick_search → 2 steps`, `product_research → 6 steps`, `shopping → 6 steps`.

- [ ] **Step 8.4: Optional — manual Telegram smoke via dev bot**

If the KutAI instance is running against a dev chat:
1. Send `/shop`. Should see two inline buttons "🎯 Belirli ürün" / "🏷 Kategori".
2. Tap "🎯 Belirli ürün". Should see "Hangi ürün?" prompt.
3. Reply `siemens s100`. Should see "Ürün araştırması başladı... (mission #N)".
4. Monitor mission via `/wfstatus N` — all 6 steps (3 live + 3 stubs) should complete. Stub steps return `status=disabled`; the delivery step produces a user-facing message.

Skip this step if no dev Telegram is available — say so explicitly rather than claiming success.

---

## Self-Review Notes

- **Spec coverage:** All five design-summary bullets implemented — product_research.json (Task 4), scaffolded stubs (Task 3+4), shopping.json refocus (Task 5), intent fork (Task 6), community filter (Task 1), fake discount (Task 2), docs update (Task 7).
- **Types/names:** `_annotate_fake_discounts`, `is_suspicious_discount`, `discount_flag_reason`, `cross_store_summary`, `_pending_shop_subintent`, `sub_intent="product_research"` — used consistently across files.
- **No placeholders:** Every code step shows the exact code. Every test has full assertions. Commit messages are pre-written.
- **YAGNI check:** Stubs are a real YAGNI risk — we chose to include them because the spec explicitly asks for scaffolding ("Step handlers registered in _STEP_HANDLERS, stubs in place for review/delivery/timing that will light up when data exists"). That makes them scope, not speculation.

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-16-shopping-workflows-refactor.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
