# Shopping Variant Disambiguation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pipeline v2 detects ambiguous queries post-scrape, separates SKUs structurally, drops accessories/parts/knockoffs/refurb via LLM labels, and asks the user to pick a variant through a Telegram clarify flow when multiple authentic variants survive filtering.

**Architecture:** Extend `Product`, `Candidate`, `ProductGroup` with SKU + label fields. Rewrite `step_group` to bucket by SKU first; add `step_label` (LLM-taxonomy), `step_filter`, `step_variant_gate`, `step_compare_all`. Branch `shopping_v2.json` into synth-one vs clarify paths. Telegram reuses `_pending_action` + inline buttons. Scrapers audited in 6 batches to populate `sku` and `category_path`.

**Tech Stack:** Python 3.10, asyncio, pytest, aiosqlite, python-telegram-bot 20+, litellm via HaLLederiz Kadir, Salako mechanical dispatcher, BeautifulSoup/lxml for scraper parsing.

---

## File Structure

**New files**
- `src/workflows/shopping/variant_gate.py` — pure-code filter + gate logic (tested in isolation).
- `src/workflows/shopping/labels.py` — label taxonomy constants + LLM call wrapper (`step_label`).
- `tests/shopping/test_variant_gate.py`
- `tests/shopping/test_labels.py`
- `tests/shopping/fixtures/` — per-scraper HTML/API response fixtures for SKU/category tests.
- `tests/shopping/verify_variant_flow_live.py`

**Modified files**
- `src/shopping/models.py` — add `Product.sku`.
- `src/workflows/shopping/pipeline_v2.py` — extend `Candidate`, `ProductGroup`; rewrite `step_group`; add step orchestration; add `step_compare_all`.
- `src/workflows/shopping/prompts_v2.py` — new `LABEL_PROMPT`; `GROUPING_PROMPT` gets sku/category hints.
- `src/workflows/shopping/shopping_v2.json`, `quick_search_v2.json`, `product_research_v2.json` — new steps + branch.
- `src/app/telegram_bot.py` — variant-choice callback + `_pending_action` kind.
- `packages/salako/` — ensure `clarify` executor supports variant-choice payload (likely already does).
- 15+ `src/shopping/scrapers/*.py` — populate `sku` + `category_path`.

---

## Task 1: Extend `Product` schema with `sku`

**Files:**
- Modify: `src/shopping/models.py:6-30`
- Test: `tests/shopping/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/shopping/test_models.py
from src.shopping.models import Product

def test_product_has_sku_field_default_none():
    p = Product(name="x", url="u", source="trendyol")
    assert p.sku is None

def test_product_sku_accepts_string():
    p = Product(name="x", url="u", source="trendyol", sku="HBCV00004X9ZCH")
    assert p.sku == "HBCV00004X9ZCH"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_models.py -v`
Expected: FAIL with "got an unexpected keyword argument 'sku'".

- [ ] **Step 3: Implement — add `sku` field**

```python
# src/shopping/models.py — inside Product dataclass, near category_path
    sku: str | None = None
    category_path: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_models.py -v`
Expected: PASS 2/2.

- [ ] **Step 5: Commit**

```bash
rtk git add src/shopping/models.py tests/shopping/test_models.py
rtk git commit -m "feat(shopping/models): add Product.sku field"
```

---

## Task 2: Extend `Candidate` to carry `sku` + `category_path`

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py:18-29`
- Test: `tests/shopping/test_pipeline_v2.py` (new test added)

- [ ] **Step 1: Write the failing test**

```python
# tests/shopping/test_pipeline_v2.py — append new test
async def test_candidate_passes_sku_and_category_path():
    from src.workflows.shopping.pipeline_v2 import step_resolve
    p = Product(name="Galaxy S25", url="https://trendyol.com/p-1", source="trendyol",
                sku="TY-123", category_path="Elektronik > Telefon")
    with patch("src.workflows.shopping.pipeline_v2._fetch_products",
               new=AsyncMock(return_value=[p])):
        cands = await step_resolve("s25", per_site_n=3)
    assert cands[0].sku == "TY-123"
    assert cands[0].category_path == "Elektronik > Telefon"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_candidate_passes_sku_and_category_path -v`
Expected: FAIL with AttributeError or TypeError on Candidate construction.

- [ ] **Step 3: Implement — add fields + passthrough**

```python
# src/workflows/shopping/pipeline_v2.py — Candidate dataclass
@dataclass
class Candidate:
    title: str
    site: str
    site_rank: int
    price: float | None
    original_price: float | None
    url: str
    rating: float | None
    review_count: int | None
    review_snippets: list[str] = field(default_factory=list)
    sku: str | None = None
    category_path: str | None = None
```

In `step_resolve`, inside the Candidate construction loop, add:
```python
                sku=_attr(p, "sku"),
                category_path=_attr(p, "category_path"),
```

Also update `_candidates_to_json` / `_candidates_from_json` to round-trip the two new fields.

- [ ] **Step 4: Run tests**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -x -q`
Expected: PASS all (existing + new).

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
rtk git commit -m "feat(pipeline_v2): Candidate carries sku + category_path"
```

---

## Task 3: Extend `ProductGroup` with label fields

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py:32-39`
- Test: `tests/shopping/test_pipeline_v2.py`

- [ ] **Step 1: Write the failing test**

```python
def test_product_group_accepts_label_fields():
    from src.workflows.shopping.pipeline_v2 import ProductGroup
    g = ProductGroup(
        representative_title="Samsung Galaxy S25",
        member_indices=[0, 1],
        is_accessory_or_part=False,
        prominence=2.0,
        product_type="authentic_product",
        base_model="Samsung Galaxy S25",
        variant=None,
        authenticity_confidence=0.95,
        matches_user_intent=True,
    )
    assert g.product_type == "authentic_product"
    assert g.variant is None
    assert g.authenticity_confidence == 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_product_group_accepts_label_fields -v`
Expected: FAIL on unexpected keyword argument.

- [ ] **Step 3: Implement**

```python
@dataclass
class ProductGroup:
    representative_title: str
    member_indices: list[int]
    is_accessory_or_part: bool
    prominence: float
    product_type: str = "unknown"           # authentic_product | accessory | replacement_part | knockoff | refurbished | unknown
    base_model: str = ""
    variant: str | None = None
    authenticity_confidence: float = 0.5
    matches_user_intent: bool = True
```

- [ ] **Step 4: Run tests**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -x -q`
Expected: PASS all.

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
rtk git commit -m "feat(pipeline_v2): ProductGroup carries label fields"
```

---

## Task 4: Rewrite `step_group` — SKU-first bucketing

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py` (step_group function)
- Test: `tests/shopping/test_pipeline_v2.py`

- [ ] **Step 1: Write the failing test**

```python
async def test_step_group_buckets_by_sku_across_sites():
    from src.workflows.shopping.pipeline_v2 import Candidate, step_group
    # Same SKU across two sites = one group, no LLM involved.
    cands = [
        Candidate(title="Galaxy S25", site="trendyol", site_rank=1,
                  price=30000, original_price=None, url="u1",
                  rating=None, review_count=None, sku="SAMSUNG-S25-128"),
        Candidate(title="S. Galaxy S25", site="hepsiburada", site_rank=1,
                  price=30200, original_price=None, url="u2",
                  rating=None, review_count=None, sku="SAMSUNG-S25-128"),
        Candidate(title="Some other phone", site="amazon_tr", site_rank=1,
                  price=99000, original_price=None, url="u3",
                  rating=None, review_count=None, sku=None),
    ]
    # Don't let the LLM layer run — verify SKU bucketing alone, without
    # touching the residual grouping path.
    with patch("src.workflows.shopping.pipeline_v2._grouping_llm_call",
               new=AsyncMock(return_value={"content": '{"groups": []}'})):
        groups = await step_group(cands, query="samsung s25")
    # SKU bucket collapses indices 0+1 into a single group.
    sku_groups = [g for g in groups if 0 in g.member_indices and 1 in g.member_indices]
    assert len(sku_groups) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_step_group_buckets_by_sku_across_sites -v`
Expected: FAIL — either `step_group` doesn't accept `query=` kwarg, or SKU bucketing isn't implemented.

- [ ] **Step 3: Implement SKU-first bucketing**

Rewrite `step_group(candidates)` → `step_group(candidates, query: str = "")`:

```python
async def step_group(candidates: list[Candidate], query: str = "") -> list[ProductGroup]:
    """SKU-first deterministic bucket, then LLM-group the residuals."""
    if not candidates:
        return []

    # Bucket by non-null sku — same sku across sites == same product.
    sku_buckets: dict[str, list[int]] = {}
    unbucketed: list[int] = []
    for i, c in enumerate(candidates):
        if c.sku:
            sku_buckets.setdefault(c.sku, []).append(i)
        else:
            unbucketed.append(i)

    groups: list[ProductGroup] = []
    for sku, indices in sku_buckets.items():
        first = candidates[indices[0]]
        prominence = sum(1.0 / candidates[i].site_rank for i in indices)
        groups.append(ProductGroup(
            representative_title=first.title,
            member_indices=indices,
            is_accessory_or_part=False,
            prominence=prominence,
        ))

    # LLM group the residuals only, using the existing prompt path.
    if unbucketed:
        residual_cands = [candidates[i] for i in unbucketed]
        residual_groups = await _llm_group_residuals(residual_cands, query)
        # Map residual indices back to original candidate indices.
        for g in residual_groups:
            g.member_indices = [unbucketed[i] for i in g.member_indices]
            groups.append(g)

    return groups
```

Extract the existing LLM-grouping body into `_llm_group_residuals(cands, query) -> list[ProductGroup]`. Feed `query` plus per-candidate `sku` and `category_path` hints into the prompt (extend the view dict). Keep the fallback to `_per_site_top1_fallback` on LLM/parse error.

Update `prompts_v2.GROUPING_PROMPT` to accept and optionally use `sku` and `category_path` hints.

- [ ] **Step 4: Run tests**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -x -q`
Expected: PASS all (existing + new).

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/shopping/pipeline_v2.py src/workflows/shopping/prompts_v2.py tests/shopping/test_pipeline_v2.py
rtk git commit -m "feat(pipeline_v2): step_group buckets by sku first, LLM-groups residuals"
```

---

## Task 5: Add `step_label` — LLM taxonomy per group

**Files:**
- Create: `src/workflows/shopping/labels.py`
- Create: `tests/shopping/test_labels.py`
- Modify: `src/workflows/shopping/prompts_v2.py` (add `LABEL_PROMPT`)

- [ ] **Step 1: Write the failing test**

```python
# tests/shopping/test_labels.py
import pytest
from unittest.mock import AsyncMock, patch
from src.workflows.shopping.pipeline_v2 import ProductGroup, Candidate

@pytest.mark.asyncio
async def test_step_label_parses_taxonomy():
    from src.workflows.shopping.labels import step_label
    groups = [
        ProductGroup(representative_title="Samsung Galaxy S25",
                     member_indices=[0], is_accessory_or_part=False, prominence=1.0),
        ProductGroup(representative_title="Galaxy S25 Silicone Case",
                     member_indices=[1], is_accessory_or_part=False, prominence=0.5),
    ]
    cands = [
        Candidate(title="Samsung Galaxy S25", site="trendyol", site_rank=1,
                  price=30000, original_price=None, url="u1",
                  rating=None, review_count=None, category_path="Telefon > Akıllı Telefon"),
        Candidate(title="Galaxy S25 Silicone Case", site="trendyol", site_rank=2,
                  price=200, original_price=None, url="u2",
                  rating=None, review_count=None, category_path="Aksesuar > Kılıf"),
    ]
    fake_resp = {"content": '{"groups": [\
        {"group_id": 0, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": null, "authenticity_confidence": 0.95, "matches_user_intent": true},\
        {"group_id": 1, "product_type": "accessory", "base_model": "Samsung Galaxy S25", "variant": null, "authenticity_confidence": 0.8, "matches_user_intent": false}\
    ]}'}
    with patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(return_value=fake_resp)):
        labelled = await step_label(groups, cands, query="samsung s25")
    assert labelled[0].product_type == "authentic_product"
    assert labelled[0].matches_user_intent is True
    assert labelled[1].product_type == "accessory"
    assert labelled[1].matches_user_intent is False

@pytest.mark.asyncio
async def test_step_label_falls_back_on_llm_error():
    from src.workflows.shopping.labels import step_label
    groups = [ProductGroup(representative_title="x", member_indices=[0],
                           is_accessory_or_part=False, prominence=1.0)]
    cands = [Candidate(title="x", site="s", site_rank=1, price=None,
                       original_price=None, url="u", rating=None, review_count=None)]
    with patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        labelled = await step_label(groups, cands, query="x")
    # Fallback: treat as authentic_product with medium confidence.
    assert labelled[0].product_type == "authentic_product"
    assert labelled[0].matches_user_intent is True
    assert labelled[0].authenticity_confidence == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_labels.py -v`
Expected: FAIL with ModuleNotFoundError on `src.workflows.shopping.labels`.

- [ ] **Step 3: Implement `labels.py` and `LABEL_PROMPT`**

```python
# src/workflows/shopping/labels.py
"""Label step — LLM taxonomy per ProductGroup."""
from __future__ import annotations

import json
import re

from src.infra.logging_config import get_logger
from src.workflows.shopping.pipeline_v2 import Candidate, ProductGroup, _strip_json_fences
from src.workflows.shopping.prompts_v2 import LABEL_PROMPT

logger = get_logger("workflows.shopping.labels")

VALID_PRODUCT_TYPES = {
    "authentic_product", "accessory", "replacement_part",
    "knockoff", "refurbished", "unknown",
}


async def _label_llm_call(prompt: str) -> dict:
    """Dispatch the label prompt. Split so tests patch this one function."""
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    dispatcher = get_dispatcher()
    return await dispatcher.request(
        category=CallCategory.MAIN_WORK,
        task="shopping_labeler",
        agent_type="shopping_pipeline_v2",
        difficulty=4,
        messages=[
            {"role": "system", "content": "You output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )


def _fallback_labels(groups: list[ProductGroup]) -> list[ProductGroup]:
    """On LLM/parse failure: mark all as authentic, medium confidence."""
    for g in groups:
        g.product_type = "authentic_product"
        g.base_model = g.representative_title
        g.variant = None
        g.authenticity_confidence = 0.5
        g.matches_user_intent = True
    return groups


async def step_label(
    groups: list[ProductGroup],
    candidates: list[Candidate],
    query: str,
) -> list[ProductGroup]:
    """Label every group with taxonomy via one LLM call. Mutates groups in place."""
    if not groups:
        return groups

    view = []
    for i, g in enumerate(groups):
        member_cands = [candidates[m] for m in g.member_indices if 0 <= m < len(candidates)]
        category = next((c.category_path for c in member_cands if c.category_path), "")
        view.append({
            "group_id": i,
            "title": g.representative_title,
            "category_path": category,
            "member_count": len(g.member_indices),
        })

    prompt = LABEL_PROMPT.format(
        query=query,
        groups_json=json.dumps(view, ensure_ascii=False),
    )

    try:
        resp = await _label_llm_call(prompt)
    except Exception as exc:
        logger.warning("label LLM failed, using fallback: %s", exc)
        return _fallback_labels(groups)

    content = _strip_json_fences(str(resp.get("content", "")).strip())
    try:
        parsed = json.loads(content)
        entries = parsed.get("groups", [])
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("label LLM parse failed: %s", exc)
        return _fallback_labels(groups)

    by_id: dict[int, dict] = {e["group_id"]: e for e in entries if isinstance(e, dict) and "group_id" in e}
    for i, g in enumerate(groups):
        e = by_id.get(i)
        if not e:
            # Partial coverage — fill with fallback for this group only.
            g.product_type = "authentic_product"
            g.matches_user_intent = True
            g.authenticity_confidence = 0.5
            g.base_model = g.representative_title
            continue
        pt = str(e.get("product_type", "unknown"))
        g.product_type = pt if pt in VALID_PRODUCT_TYPES else "unknown"
        g.base_model = str(e.get("base_model", g.representative_title))
        variant = e.get("variant")
        g.variant = str(variant) if variant else None
        try:
            g.authenticity_confidence = float(e.get("authenticity_confidence", 0.5))
        except (TypeError, ValueError):
            g.authenticity_confidence = 0.5
        g.matches_user_intent = bool(e.get("matches_user_intent", True))

    return groups
```

Add to `src/workflows/shopping/prompts_v2.py`:

```python
LABEL_PROMPT = """You classify product-search result groups for a Turkish shopping bot.

User query: {query}

Groups (each is one or more scraped listings that we already think refer to the same product):
{groups_json}

For EVERY group, return a JSON object:
- group_id: copy the input id verbatim
- product_type: one of "authentic_product", "accessory", "replacement_part", "knockoff", "refurbished", "unknown"
  * authentic_product = a real, new, first-party product that matches the query
  * accessory = a case, charger, cable, holder, screen protector, bag, strap, etc.
  * replacement_part = a screen panel, battery, motherboard, button — a part of a product, not the product
  * knockoff = counterfeit / non-branded clone / suspicious listing
  * refurbished = used / refurbished / grade-B / open-box
  * unknown = cannot tell from title + category
- base_model: the canonical product line, e.g. "Samsung Galaxy S25" (strip variant suffix)
- variant: the variant suffix if any (e.g. "FE", "Plus", "Ultra", "Pro", "Mini", color code, storage); null when base model has no variant
- authenticity_confidence: 0.0–1.0 — how sure you are the listing is the authentic product
- matches_user_intent: true if answering this group tells the user what they asked; false if they'd consider it the wrong thing (accessory for a phone query, part for a product query, knockoff, etc.)

Return only the JSON object:
{{
  "groups": [ {{...}}, {{...}}, ... ]
}}
"""
```

- [ ] **Step 4: Run tests**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/shopping/test_labels.py -v`
Expected: PASS 2/2.

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/shopping/labels.py src/workflows/shopping/prompts_v2.py tests/shopping/test_labels.py
rtk git commit -m "feat(shopping): step_label tags groups with taxonomy"
```

---

## Task 6: Add `step_filter` — drop non-authentic / intent-mismatched groups

**Files:**
- Create: `src/workflows/shopping/variant_gate.py`
- Create: `tests/shopping/test_variant_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/shopping/test_variant_gate.py
from src.workflows.shopping.pipeline_v2 import ProductGroup

def _g(title, product_type="authentic_product", conf=0.9, intent=True,
       variant=None, base_model=None, prom=1.0):
    return ProductGroup(
        representative_title=title,
        member_indices=[0],
        is_accessory_or_part=False,
        prominence=prom,
        product_type=product_type,
        base_model=base_model or title,
        variant=variant,
        authenticity_confidence=conf,
        matches_user_intent=intent,
    )


def test_step_filter_drops_accessories_parts_knockoffs_refurbished():
    from src.workflows.shopping.variant_gate import step_filter
    groups = [
        _g("Samsung S25", product_type="authentic_product"),
        _g("S25 Case", product_type="accessory"),
        _g("S25 Battery", product_type="replacement_part"),
        _g("Galaxy S25 Clone", product_type="knockoff"),
        _g("Galaxy S25 Refurb", product_type="refurbished"),
    ]
    survivors = step_filter(groups)
    assert [g.representative_title for g in survivors] == ["Samsung S25"]


def test_step_filter_drops_intent_mismatch():
    from src.workflows.shopping.variant_gate import step_filter
    groups = [
        _g("Samsung S25", intent=True),
        _g("Samsung S25 Case", intent=False),
    ]
    survivors = step_filter(groups)
    assert [g.representative_title for g in survivors] == ["Samsung S25"]


def test_step_filter_drops_low_confidence():
    from src.workflows.shopping.variant_gate import step_filter, FILTER_AUTHENTICITY_MIN
    groups = [
        _g("Samsung S25", conf=0.95),
        _g("Samsung S25 Fake", conf=0.4),
    ]
    survivors = step_filter(groups)
    assert len(survivors) == 1
    assert survivors[0].authenticity_confidence >= FILTER_AUTHENTICITY_MIN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_variant_gate.py -v`
Expected: FAIL with ModuleNotFoundError on `variant_gate`.

- [ ] **Step 3: Implement `step_filter`**

```python
# src/workflows/shopping/variant_gate.py
"""Pure-code filter + gate logic for variant disambiguation."""
from __future__ import annotations

from src.workflows.shopping.pipeline_v2 import ProductGroup

FILTER_AUTHENTICITY_MIN = 0.7


def step_filter(groups: list[ProductGroup]) -> list[ProductGroup]:
    """Drop groups that aren't authentic, intent-matched products."""
    return [
        g for g in groups
        if g.product_type == "authentic_product"
        and g.matches_user_intent
        and g.authenticity_confidence >= FILTER_AUTHENTICITY_MIN
    ]
```

- [ ] **Step 4: Run tests**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_variant_gate.py -v`
Expected: PASS 3/3.

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/shopping/variant_gate.py tests/shopping/test_variant_gate.py
rtk git commit -m "feat(shopping): step_filter drops non-authentic / intent-mismatched groups"
```

---

## Task 7: Add `step_variant_gate` — pick-or-clarify decision

**Files:**
- Modify: `src/workflows/shopping/variant_gate.py`
- Modify: `tests/shopping/test_variant_gate.py`

- [ ] **Step 1: Write the failing test**

```python
def test_variant_gate_zero_survivors_signals_escalation():
    from src.workflows.shopping.variant_gate import step_variant_gate
    out = step_variant_gate(survivors=[], all_groups=[])
    assert out["kind"] == "escalation"
    assert out["reason"] == "all_filtered"


def test_variant_gate_single_variant_returns_chosen():
    from src.workflows.shopping.variant_gate import step_variant_gate
    g = _g("Samsung S25", base_model="Samsung Galaxy S25", variant=None, prom=3.0)
    out = step_variant_gate(survivors=[g], all_groups=[g])
    assert out["kind"] == "chosen"
    assert out["group"] is g


def test_variant_gate_multiple_variants_returns_clarify():
    from src.workflows.shopping.variant_gate import step_variant_gate
    vanilla = _g("Galaxy S25", base_model="Samsung Galaxy S25", variant=None, prom=2.0)
    fe = _g("Galaxy S25 FE", base_model="Samsung Galaxy S25", variant="FE", prom=3.0)
    ultra = _g("Galaxy S25 Ultra", base_model="Samsung Galaxy S25", variant="Ultra", prom=2.5)
    out = step_variant_gate(survivors=[vanilla, fe, ultra], all_groups=[vanilla, fe, ultra])
    assert out["kind"] == "clarify"
    # Options sorted by prominence desc, capped at 5.
    labels = [opt["label"] for opt in out["options"]]
    assert labels[0] == "Galaxy S25 FE"        # prominence 3.0
    assert len(out["options"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_variant_gate.py -v`
Expected: FAIL on `step_variant_gate` missing.

- [ ] **Step 3: Implement**

Append to `src/workflows/shopping/variant_gate.py`:

```python
MAX_CLARIFY_OPTIONS = 5


def step_variant_gate(
    survivors: list[ProductGroup],
    all_groups: list[ProductGroup],
) -> dict:
    """Decide how to proceed after filtering.

    Returns a dict shaped for downstream handlers:
      {"kind": "chosen", "group": ProductGroup}
      {"kind": "clarify", "options": [{"label", "group_id", "prominence"}], "payloads": {gid: group}}
      {"kind": "escalation", "reason": "all_filtered"}
    """
    if not survivors:
        return {"kind": "escalation", "reason": "all_filtered"}

    # Count distinct (base_model, variant) among survivors.
    by_variant: dict[tuple[str, str | None], list[ProductGroup]] = {}
    for g in survivors:
        by_variant.setdefault((g.base_model, g.variant), []).append(g)

    if len(by_variant) == 1:
        # Single variant — pick the highest-prominence group within it.
        only_bucket = next(iter(by_variant.values()))
        chosen = max(only_bucket, key=lambda g: g.prominence)
        return {"kind": "chosen", "group": chosen}

    # Multiple variants — emit clarify payload.
    # One option per variant, highest-prominence group represents the variant.
    options: list[dict] = []
    payloads: dict[int, ProductGroup] = {}
    sorted_buckets = sorted(
        by_variant.items(),
        key=lambda kv: max(g.prominence for g in kv[1]),
        reverse=True,
    )
    # Use index in all_groups as stable id.
    gid_for = {id(g): i for i, g in enumerate(all_groups)}
    for (_, _), bucket in sorted_buckets[:MAX_CLARIFY_OPTIONS]:
        rep = max(bucket, key=lambda g: g.prominence)
        gid = gid_for.get(id(rep), -1)
        options.append({
            "label": rep.representative_title,
            "group_id": gid,
            "prominence": rep.prominence,
        })
        payloads[gid] = rep

    return {"kind": "clarify", "options": options, "payloads": payloads}
```

- [ ] **Step 4: Run tests**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_variant_gate.py -v`
Expected: PASS 6/6.

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/shopping/variant_gate.py tests/shopping/test_variant_gate.py
rtk git commit -m "feat(shopping): step_variant_gate decides chosen vs clarify vs escalation"
```

---

## Task 8: Add `step_compare_all` — compact comparison table

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py` (append new function + tests)

- [ ] **Step 1: Write the failing test**

```python
def test_compare_all_table_format():
    from src.workflows.shopping.pipeline_v2 import step_compare_all, Candidate, ProductGroup
    cands = [
        Candidate(title="Galaxy S25", site="trendyol", site_rank=1,
                  price=32500, original_price=None, url="u",
                  rating=4.7, review_count=1200),
        Candidate(title="Galaxy S25", site="hepsiburada", site_rank=1,
                  price=34800, original_price=None, url="u",
                  rating=4.7, review_count=800),
        Candidate(title="Galaxy S25 Ultra", site="trendyol", site_rank=1,
                  price=48000, original_price=None, url="u",
                  rating=4.9, review_count=2100),
    ]
    vanilla = ProductGroup(representative_title="Galaxy S25",
                           member_indices=[0, 1], is_accessory_or_part=False,
                           prominence=2.0, base_model="Samsung Galaxy S25",
                           variant=None)
    ultra = ProductGroup(representative_title="Galaxy S25 Ultra",
                         member_indices=[2], is_accessory_or_part=False,
                         prominence=1.0, base_model="Samsung Galaxy S25",
                         variant="Ultra")
    md = step_compare_all([vanilla, ultra], cands, base_label="Samsung Galaxy S25")
    assert "Galaxy S25" in md
    assert "Ultra" in md
    assert "32.500" in md      # price min formatted TR-style
    assert "34.800" in md      # price max
    assert "48.000" in md
    assert "⭐" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_compare_all_table_format -v`
Expected: FAIL with ImportError or AttributeError.

- [ ] **Step 3: Implement**

Append to `src/workflows/shopping/pipeline_v2.py`:

```python
def step_compare_all(
    groups: list[ProductGroup],
    candidates: list[Candidate],
    base_label: str,
) -> str:
    """Render a compact variant-comparison markdown table."""
    lines: list[str] = [f"*{base_label} — Karşılaştırma*", "─" * 20]
    for g in groups:
        members = [candidates[i] for i in g.member_indices if 0 <= i < len(candidates)]
        prices = [m.price for m in members if m.price is not None]
        pmin = min(prices) if prices else None
        pmax = max(prices) if prices else None
        rating = next((m.rating for m in members if m.rating is not None), None)
        review_total = sum(m.review_count or 0 for m in members)
        variant_label = g.variant or "Vanilla"

        price_str = (
            f"{_fmt_price_tr(pmin)}–{_fmt_price_tr(pmax)} TL"
            if pmin is not None else "fiyat yok"
        )
        rating_str = (
            f" ⭐ {rating:.1f} ({review_total})" if rating is not None else ""
        )
        lines.append(f"• *{variant_label}* — {price_str}{rating_str}")
    lines.append("─" * 20)
    lines.append("Seçmek için sorunuzu daraltın.")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_compare_all_table_format -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
rtk git commit -m "feat(shopping): step_compare_all renders variant comparison table"
```

---

## Task 9: Merge handler — `group_label_filter_gate`

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py` (handler + step registration)

- [ ] **Step 1: Write the failing test**

```python
async def test_group_label_filter_gate_single_variant_path():
    from src.workflows.shopping.pipeline_v2 import (
        _handler_group_label_filter_gate, Candidate,
    )
    cands = [
        Candidate(title="Samsung Galaxy S25", site="trendyol", site_rank=1,
                  price=30000, original_price=None, url="u1",
                  rating=4.7, review_count=100, sku="S25-128",
                  category_path="Telefon"),
    ]
    task = {"id": 1, "context": {}}
    artifacts = {"search_results": json.dumps({
        "candidates": [{
            "title": c.title, "site": c.site, "site_rank": c.site_rank,
            "price": c.price, "original_price": c.original_price,
            "url": c.url, "rating": c.rating, "review_count": c.review_count,
            "review_snippets": [], "sku": c.sku, "category_path": c.category_path,
        } for c in cands]),
        "query": "samsung s25",
    }}
    # Mock LLM label so it returns authentic product matching intent.
    with patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(return_value={"content": '{"groups": [\
                   {"group_id": 0, "product_type": "authentic_product",\
                    "base_model": "Samsung Galaxy S25", "variant": null,\
                    "authenticity_confidence": 0.95, "matches_user_intent": true}]}'})):
        result = await _handler_group_label_filter_gate(task, artifacts, {})
    assert result["gate"]["kind"] == "chosen"
    assert result["chosen_group"]["representative_title"] == "Samsung Galaxy S25"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_group_label_filter_gate_single_variant_path -v`
Expected: FAIL on missing handler.

- [ ] **Step 3: Implement the merged handler**

Append to `src/workflows/shopping/pipeline_v2.py`:

```python
def _group_to_dict(g: ProductGroup) -> dict:
    return {
        "representative_title": g.representative_title,
        "member_indices": g.member_indices,
        "is_accessory_or_part": g.is_accessory_or_part,
        "prominence": g.prominence,
        "product_type": g.product_type,
        "base_model": g.base_model,
        "variant": g.variant,
        "authenticity_confidence": g.authenticity_confidence,
        "matches_user_intent": g.matches_user_intent,
    }


def _group_from_dict(d: dict) -> ProductGroup:
    return ProductGroup(
        representative_title=d["representative_title"],
        member_indices=list(d["member_indices"]),
        is_accessory_or_part=bool(d.get("is_accessory_or_part", False)),
        prominence=float(d.get("prominence", 0.0)),
        product_type=str(d.get("product_type", "unknown")),
        base_model=str(d.get("base_model", "")),
        variant=d.get("variant"),
        authenticity_confidence=float(d.get("authenticity_confidence", 0.5)),
        matches_user_intent=bool(d.get("matches_user_intent", True)),
    )


async def _handler_group_label_filter_gate(
    task: dict, artifacts: dict, ctx: dict,
) -> dict:
    from src.workflows.shopping.labels import step_label
    from src.workflows.shopping.variant_gate import step_filter, step_variant_gate

    raw = artifacts.get("search_results", "{}")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    cands = _candidates_from_json(payload.get("candidates", []))
    query = payload.get("query", "")
    if not cands:
        return {"gate": {"kind": "escalation", "reason": "no_candidates"}, "candidates": []}

    groups = await step_group(cands, query=query)
    groups = await step_label(groups, cands, query=query)
    survivors = step_filter(groups)
    gate = step_variant_gate(survivors, groups)

    out: dict = {
        "gate": {"kind": gate["kind"]},
        "candidates": _candidates_to_json(cands),
        "query": query,
    }
    if gate["kind"] == "chosen":
        out["chosen_group"] = _group_to_dict(gate["group"])
    elif gate["kind"] == "clarify":
        out["clarify_options"] = gate["options"]
        out["clarify_payloads"] = {str(gid): _group_to_dict(g) for gid, g in gate["payloads"].items()}
        out["base_label"] = survivors[0].base_model if survivors else ""
    elif gate["kind"] == "escalation":
        out["gate"]["reason"] = gate.get("reason", "unknown")
    return out
```

Register the handler in `_STEP_HANDLERS_V2`:

```python
_STEP_HANDLERS_V2 = {
    "resolve_candidates": _handler_resolve_candidates,
    "group_label_filter_gate": _handler_group_label_filter_gate,
    "group_and_synthesize": _handler_group_and_synthesize,   # kept for back-compat
    "format_response": _handler_format_response,
}
```

- [ ] **Step 4: Run tests**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -x -q`
Expected: PASS all.

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
rtk git commit -m "feat(shopping): group_label_filter_gate merged handler"
```

---

## Task 10: Synth-one + format-compare handlers

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py`

- [ ] **Step 1: Write the failing test**

```python
async def test_synth_one_handler_uses_chosen_group():
    from src.workflows.shopping.pipeline_v2 import _handler_synth_one
    gate = {
        "gate": {"kind": "chosen"},
        "chosen_group": {
            "representative_title": "Samsung Galaxy S25",
            "member_indices": [0], "is_accessory_or_part": False,
            "prominence": 1.0, "product_type": "authentic_product",
            "base_model": "Samsung Galaxy S25", "variant": None,
            "authenticity_confidence": 0.9, "matches_user_intent": True,
        },
        "candidates": [{
            "title": "Samsung Galaxy S25", "site": "trendyol", "site_rank": 1,
            "price": 30000, "original_price": None, "url": "u",
            "rating": 4.7, "review_count": 100,
            "review_snippets": ["köpük iyi", "hızlı kargo"],
        }],
        "query": "samsung s25",
    }
    task = {"id": 2, "context": {}}
    artifacts = {"gate_result": json.dumps(gate)}
    with patch("src.workflows.shopping.pipeline_v2._synthesis_llm_call",
               new=AsyncMock(return_value={"content": '{"praise": ["iyi"], "complaints": [], "red_flags": [], "insufficient_data": false}'})):
        out = await _handler_synth_one(task, artifacts, {})
    assert "Samsung Galaxy S25" in out["cards"][0]


async def test_format_compare_handler_renders_table():
    from src.workflows.shopping.pipeline_v2 import _handler_format_compare
    gate = {
        "gate": {"kind": "clarify"},
        "clarify_options": [
            {"label": "Galaxy S25", "group_id": 0, "prominence": 2.0},
            {"label": "Galaxy S25 Ultra", "group_id": 1, "prominence": 1.0},
        ],
        "clarify_payloads": {
            "0": {"representative_title": "Galaxy S25", "member_indices": [0],
                  "is_accessory_or_part": False, "prominence": 2.0,
                  "product_type": "authentic_product",
                  "base_model": "Samsung Galaxy S25", "variant": None,
                  "authenticity_confidence": 0.9, "matches_user_intent": True},
            "1": {"representative_title": "Galaxy S25 Ultra", "member_indices": [1],
                  "is_accessory_or_part": False, "prominence": 1.0,
                  "product_type": "authentic_product",
                  "base_model": "Samsung Galaxy S25", "variant": "Ultra",
                  "authenticity_confidence": 0.9, "matches_user_intent": True},
        },
        "candidates": [
            {"title": "Galaxy S25", "site": "trendyol", "site_rank": 1,
             "price": 32500, "original_price": None, "url": "u1",
             "rating": 4.7, "review_count": 100, "review_snippets": []},
            {"title": "Galaxy S25 Ultra", "site": "trendyol", "site_rank": 1,
             "price": 48000, "original_price": None, "url": "u2",
             "rating": 4.9, "review_count": 200, "review_snippets": []},
        ],
        "base_label": "Samsung Galaxy S25",
    }
    task = {"id": 3, "context": {}}
    artifacts = {"gate_result": json.dumps(gate)}
    out = await _handler_format_compare(task, artifacts, {})
    assert "Karşılaştırma" in out["formatted_text"]
    assert "Ultra" in out["formatted_text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -k "synth_one or format_compare" -v`
Expected: FAIL — handlers missing.

- [ ] **Step 3: Implement both handlers**

Append to `src/workflows/shopping/pipeline_v2.py`:

```python
async def _handler_synth_one(task: dict, artifacts: dict, ctx: dict) -> dict:
    raw = artifacts.get("gate_result", "{}")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if payload.get("gate", {}).get("kind") != "chosen":
        return {"cards": [], "escalation_needed": True}
    group = _group_from_dict(payload["chosen_group"])
    cands = _candidates_from_json(payload.get("candidates", []))
    syn = await step_synthesize_reviews(group, cands)
    cards = [format_group_card(group, syn, cands)]
    return {"cards": cards, "escalation_needed": False}


async def _handler_format_compare(task: dict, artifacts: dict, ctx: dict) -> dict:
    raw = artifacts.get("gate_result", "{}")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    payloads = payload.get("clarify_payloads", {}) or {}
    base_label = payload.get("base_label") or "Ürün"
    cands = _candidates_from_json(payload.get("candidates", []))
    groups = [_group_from_dict(v) for v in payloads.values()]
    text = step_compare_all(groups, cands, base_label=base_label)
    return {"formatted_text": text, "escalation": False}
```

Register both:

```python
_STEP_HANDLERS_V2.update({
    "synth_one": _handler_synth_one,
    "format_compare": _handler_format_compare,
})
```

- [ ] **Step 4: Run tests**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
rtk git commit -m "feat(shopping): synth_one + format_compare handlers"
```

---

## Task 11: Workflow JSON — branch shopping_v2

**Files:**
- Modify: `src/workflows/shopping/shopping_v2.json`
- Modify: `src/workflows/shopping/quick_search_v2.json`
- Modify: `src/workflows/shopping/product_research_v2.json`

- [ ] **Step 1: Write the failing test**

```python
# tests/shopping/test_workflow_json.py
import json
from pathlib import Path

WF_PATHS = [
    "src/workflows/shopping/shopping_v2.json",
    "src/workflows/shopping/quick_search_v2.json",
    "src/workflows/shopping/product_research_v2.json",
]

def test_workflows_reference_new_steps():
    for p in WF_PATHS:
        wf = json.loads(Path(p).read_text(encoding="utf-8"))
        names = [s.get("name") for s in wf.get("steps", [])]
        assert "group_label_filter_gate" in names, f"{p} missing group_label_filter_gate"
        # After the gate, must have both branches wired via skip_when.
        assert "synth_one" in names or "synth_one" in [s.get("name") for s in wf["steps"]]
        assert "clarify_variant" in names
        assert "format_compare" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_workflow_json.py -v`
Expected: FAIL.

- [ ] **Step 3: Edit each workflow**

Replace the `group_and_synthesize` + `format_response` pair with the new branch in each of the three workflow JSONs. Full edit for `shopping_v2.json` (apply same shape to quick_search_v2 / product_research_v2, adjusting `per_site_n` as they already do):

```json
{
  "name": "shopping_v2",
  "steps": [
    {
      "id": "1.0", "phase": "phase_1", "name": "resolve_candidates",
      "title": "Resolve Candidates", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": [],
      "input_artifacts": ["clarified_query", "user_query"],
      "output_artifacts": ["search_results"],
      "instruction": "Step name: resolve_candidates.",
      "done_when": "search_results.candidates populated.",
      "context": {"per_site_n": 3, "requires_grading": false}
    },
    {
      "id": "1.1", "phase": "phase_1", "name": "group_label_filter_gate",
      "title": "Group, Label, Filter, Variant Gate", "agent": "shopping_pipeline_v2",
      "difficulty": "medium", "tools_hint": [], "depends_on": ["1.0"],
      "input_artifacts": ["search_results"],
      "output_artifacts": ["gate_result"],
      "instruction": "Step name: group_label_filter_gate.",
      "done_when": "gate_result.gate.kind is chosen|clarify|escalation."
    },
    {
      "id": "2.0", "phase": "phase_2", "name": "synth_one",
      "title": "Synthesize Reviews (single variant)", "agent": "shopping_pipeline_v2",
      "difficulty": "medium", "tools_hint": [], "depends_on": ["1.1"],
      "input_artifacts": ["gate_result"],
      "output_artifacts": ["synth_result"],
      "instruction": "Step name: synth_one.",
      "done_when": "synth_result.cards populated.",
      "skip_when": "gate_result.gate.kind != 'chosen'"
    },
    {
      "id": "2.1", "phase": "phase_2", "name": "clarify_variant",
      "title": "Ask user which variant", "agent": "mechanical",
      "difficulty": "easy", "tools_hint": [], "depends_on": ["1.1"],
      "input_artifacts": ["gate_result"],
      "output_artifacts": ["clarify_choice"],
      "instruction": "Mechanical: clarify (variant_choice).",
      "done_when": "clarify_choice populated by user tap.",
      "skip_when": "gate_result.gate.kind != 'clarify'",
      "context": {
        "executor": "clarify",
        "payload_from": "gate_result",
        "kind": "variant_choice",
        "requires_grading": false
      }
    },
    {
      "id": "2.2", "phase": "phase_2", "name": "synth_after_choice",
      "title": "Synthesize chosen variant", "agent": "shopping_pipeline_v2",
      "difficulty": "medium", "tools_hint": [], "depends_on": ["2.1"],
      "input_artifacts": ["clarify_choice", "gate_result"],
      "output_artifacts": ["synth_result"],
      "instruction": "Step name: synth_one.",
      "done_when": "synth_result.cards populated or skipped.",
      "skip_when": "clarify_choice.kind != 'variant'"
    },
    {
      "id": "2.3", "phase": "phase_2", "name": "format_compare",
      "title": "Compare all variants", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": ["2.1"],
      "input_artifacts": ["clarify_choice", "gate_result"],
      "output_artifacts": ["shopping_response"],
      "instruction": "Step name: format_compare.",
      "done_when": "shopping_response.formatted_text populated.",
      "skip_when": "clarify_choice.kind != 'compare_all'",
      "context": {"requires_grading": false}
    },
    {
      "id": "3.0", "phase": "phase_3", "name": "format_response",
      "title": "Format & Deliver Response", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": ["2.0", "2.2", "2.3"],
      "input_artifacts": ["synth_result"],
      "output_artifacts": ["shopping_response"],
      "instruction": "Step name: format_response.",
      "done_when": "shopping_response.formatted_text populated.",
      "skip_when": "synth_result.cards is empty",
      "context": {"requires_grading": false}
    }
  ]
}
```

- [ ] **Step 4: Run tests**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_workflow_json.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/shopping/*.json tests/shopping/test_workflow_json.py
rtk git commit -m "feat(shopping_v2): workflow branches on variant gate outcome"
```

---

## Task 12: Salako `clarify` executor supports variant_choice payload

**Files:**
- Modify: `packages/salako/src/salako/clarify.py` (find exact path by grepping)
- Modify: `packages/salako/tests/test_clarify.py` (or wherever salako tests live)

- [ ] **Step 1: Locate clarify executor**

```bash
rtk grep -r "def.*clarify" packages/salako/
```

Expected: one or two definitions — `clarify_step` handler that emits a pending question and writes a `notify_user` Telegram message with a keyboard.

- [ ] **Step 2: Write the failing test**

```python
# packages/salako/tests/test_clarify_variant.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_clarify_variant_choice_sends_keyboard():
    from salako.clarify import run_clarify
    task = {"id": 9, "mission_id": 1, "context": {
        "executor": "clarify",
        "kind": "variant_choice",
        "payload_from": "gate_result",
    }}
    artifacts = {"gate_result": {
        "gate": {"kind": "clarify"},
        "clarify_options": [
            {"label": "Galaxy S25", "group_id": 0, "prominence": 2.0},
            {"label": "Galaxy S25 Ultra", "group_id": 1, "prominence": 1.0},
        ],
        "base_label": "Samsung Galaxy S25",
    }}
    with patch("salako.clarify.send_variant_keyboard",
               new=AsyncMock(return_value=None)) as sent:
        await run_clarify(task, artifacts)
    sent.assert_awaited_once()
    kwargs = sent.await_args.kwargs or sent.await_args.args
    # Keyboard must include two variant buttons + "Compare all"
    assert any("Galaxy S25 Ultra" in str(kwargs)), kwargs
    assert any("Compare" in str(kwargs) or "Karşılaştır" in str(kwargs)), kwargs
```

- [ ] **Step 3: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest packages/salako/tests/test_clarify_variant.py -v`
Expected: FAIL — no variant-specific path.

- [ ] **Step 4: Implement**

In `packages/salako/src/salako/clarify.py`, add a branch on `context.kind == "variant_choice"`:

```python
# Pseudo-patch — adapt to the file's actual structure.
async def run_clarify(task: dict, artifacts: dict):
    ctx = task.get("context", {}) or {}
    kind = ctx.get("kind")
    if kind == "variant_choice":
        payload_key = ctx.get("payload_from", "gate_result")
        gate = artifacts.get(payload_key, {})
        options = gate.get("clarify_options", [])
        base_label = gate.get("base_label", "Ürün")
        await send_variant_keyboard(
            mission_id=task["mission_id"],
            task_id=task["id"],
            base_label=base_label,
            options=options,
        )
        return {"status": "needs_clarification", "kind": "variant_choice",
                "prompt": f"{base_label} için hangi model?"}
    # ... existing default clarify logic unchanged ...
```

Add a stub `send_variant_keyboard` that calls the Telegram-side `_pending_action` setter (implemented in Task 13). For now emit a log + raise NotImplementedError so Task 13 wires it.

- [ ] **Step 5: Run tests + commit**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest packages/salako/tests/test_clarify_variant.py -v`
Expected: PASS.

```bash
rtk git add packages/salako/
rtk git commit -m "feat(salako): clarify executor supports variant_choice payload"
```

---

## Task 13: Telegram variant-choice callback

**Files:**
- Modify: `src/app/telegram_bot.py` — add `_handle_variant_choice` callback + `send_variant_keyboard`
- Test: inline test against the callback handler via mocked Update/Context.

- [ ] **Step 1: Write the failing test**

```python
# tests/app/test_variant_choice_callback.py
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_variant_choice_variant_resumes_mission():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._pending_action = {}
    iface._resume_mission_at_step = AsyncMock()
    iface._reply_text = AsyncMock()
    chat_id = 42
    iface._pending_action[chat_id] = {
        "kind": "variant_choice",
        "mission_id": 7,
        "task_id": 2,
        "options": [{"label": "Galaxy S25", "group_id": 0}],
        "base_label": "Samsung Galaxy S25",
    }
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.data = "variant_choice:0"
    update.callback_query.answer = AsyncMock()
    context = MagicMock()
    await iface._handle_variant_choice(update, context)
    iface._resume_mission_at_step.assert_awaited_once()
    # After resume, pending_action cleared.
    assert chat_id not in iface._pending_action


@pytest.mark.asyncio
async def test_variant_choice_compare_all_runs_format_compare():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._pending_action = {}
    iface._run_compare_all_and_reply = AsyncMock()
    chat_id = 42
    iface._pending_action[chat_id] = {
        "kind": "variant_choice", "mission_id": 7, "task_id": 2,
        "options": [{"label": "A", "group_id": 0}], "base_label": "A",
    }
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.data = "variant_choice:compare_all"
    update.callback_query.answer = AsyncMock()
    context = MagicMock()
    await iface._handle_variant_choice(update, context)
    iface._run_compare_all_and_reply.assert_awaited_once()
    assert chat_id not in iface._pending_action
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/app/test_variant_choice_callback.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `src/app/telegram_bot.py`:

```python
# Inside class TelegramInterface
async def send_variant_keyboard(self, chat_id: int, mission_id: int,
                                task_id: int, base_label: str,
                                options: list[dict]) -> None:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    buttons = [
        [InlineKeyboardButton(opt["label"],
                              callback_data=f"variant_choice:{opt['group_id']}")]
        for opt in options
    ]
    buttons.append([InlineKeyboardButton(
        "📊 Hepsini karşılaştır", callback_data="variant_choice:compare_all"
    )])
    markup = InlineKeyboardMarkup(buttons)
    await self.app.bot.send_message(
        chat_id=chat_id,
        text=f"*{base_label}* için hangi model?",
        reply_markup=markup,
        parse_mode="Markdown",
    )
    self._pending_action[chat_id] = {
        "kind": "variant_choice",
        "mission_id": mission_id,
        "task_id": task_id,
        "options": options,
        "base_label": base_label,
    }


async def _handle_variant_choice(self, update, context):
    chat_id = update.effective_chat.id
    pending = self._pending_action.get(chat_id)
    if not pending or pending.get("kind") != "variant_choice":
        await update.callback_query.answer("Bu seçenek artık geçerli değil.")
        return
    data = update.callback_query.data or ""
    if not data.startswith("variant_choice:"):
        return
    choice = data.split(":", 1)[1]
    await update.callback_query.answer()
    mission_id = pending["mission_id"]
    task_id = pending["task_id"]
    self._pending_action.pop(chat_id, None)
    if choice == "compare_all":
        await self._run_compare_all_and_reply(chat_id, mission_id, task_id)
    else:
        try:
            gid = int(choice)
        except ValueError:
            return
        await self._resume_mission_at_step(
            mission_id=mission_id, after_task_id=task_id,
            clarify_choice={"kind": "variant", "group_id": gid},
        )
```

Register the callback in `_setup_handlers`:

```python
self.app.add_handler(CallbackQueryHandler(
    self._handle_variant_choice, pattern=r"^variant_choice:"
))
```

Implement `_resume_mission_at_step` to write `clarify_choice` artifact + mark the `clarify_variant` task completed, and `_run_compare_all_and_reply` to invoke the `format_compare` step directly and deliver the result via `_reply`.

Wire `salako.send_variant_keyboard` to call `TelegramInterface.send_variant_keyboard` via the existing notify bridge.

- [ ] **Step 4: Run tests**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/app/test_variant_choice_callback.py tests/shopping/ -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/app/telegram_bot.py packages/salako/ tests/app/test_variant_choice_callback.py
rtk git commit -m "feat(telegram): variant-choice inline keyboard + callback handler"
```

---

## Task 14: Integration test — variant gate end-to-end

**Files:**
- Create: `tests/shopping/test_variant_flow_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/shopping/test_variant_flow_integration.py
import json
import pytest
from unittest.mock import AsyncMock, patch
from src.shopping.models import Product


def _p(name, source, sku=None, cat=None, price=30000, rank=0):
    p = Product(name=name, url=f"https://{source}/{sku or name}", source=source,
                original_price=price, sku=sku, category_path=cat)
    p.site_rank = rank
    return p


@pytest.mark.asyncio
async def test_end_to_end_phone_ambiguous_triggers_clarify():
    from src.workflows.shopping.pipeline_v2 import (
        _handler_resolve_candidates, _handler_group_label_filter_gate,
    )
    products = [
        _p("Samsung Galaxy S25 128GB",     "trendyol",    sku="TY-1", cat="Telefon"),
        _p("Samsung Galaxy S25 128GB",     "hepsiburada", sku="HB-1", cat="Telefon"),
        _p("Samsung Galaxy S25 FE 128GB",  "trendyol",    sku="TY-2", cat="Telefon"),
        _p("Samsung Galaxy S25 Ultra 256", "amazon_tr",   sku="AM-3", cat="Telefon"),
        _p("Samsung S25 Silicone Case",    "trendyol",    sku="TY-9", cat="Aksesuar"),
    ]
    label_resp = json.dumps({"groups": [
        {"group_id": 0, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": None, "authenticity_confidence": 0.95, "matches_user_intent": True},
        {"group_id": 1, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": None, "authenticity_confidence": 0.95, "matches_user_intent": True},
        {"group_id": 2, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": "FE", "authenticity_confidence": 0.95, "matches_user_intent": True},
        {"group_id": 3, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": "Ultra", "authenticity_confidence": 0.95, "matches_user_intent": True},
        {"group_id": 4, "product_type": "accessory",         "base_model": "Samsung Galaxy S25", "variant": None, "authenticity_confidence": 0.95, "matches_user_intent": False},
    ]})
    with patch("src.workflows.shopping.pipeline_v2._fetch_products",
               new=AsyncMock(return_value=products)), \
         patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(return_value={"content": label_resp})):
        r1 = await _handler_resolve_candidates(
            task={"id": 1}, artifacts={"user_query": "samsung s25"},
            ctx={"per_site_n": 3},
        )
        r2 = await _handler_group_label_filter_gate(
            task={"id": 2}, artifacts={"search_results": json.dumps(r1)}, ctx={},
        )
    assert r2["gate"]["kind"] == "clarify"
    labels = [opt["label"] for opt in r2["clarify_options"]]
    assert "FE" in " ".join(labels) or "Ultra" in " ".join(labels)
    # Accessory was filtered out — never appears as a clarify option.
    assert not any("Case" in lbl for lbl in labels)


@pytest.mark.asyncio
async def test_end_to_end_single_variant_skips_clarify():
    from src.workflows.shopping.pipeline_v2 import (
        _handler_resolve_candidates, _handler_group_label_filter_gate,
    )
    products = [_p("Samsung Galaxy S25 Ultra 256GB", "trendyol", sku="TY-3", cat="Telefon")]
    label_resp = json.dumps({"groups": [
        {"group_id": 0, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": "Ultra", "authenticity_confidence": 0.95, "matches_user_intent": True},
    ]})
    with patch("src.workflows.shopping.pipeline_v2._fetch_products",
               new=AsyncMock(return_value=products)), \
         patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(return_value={"content": label_resp})):
        r1 = await _handler_resolve_candidates(
            task={"id": 1}, artifacts={"user_query": "samsung s25 ultra 256gb"},
            ctx={"per_site_n": 3},
        )
        r2 = await _handler_group_label_filter_gate(
            task={"id": 2}, artifacts={"search_results": json.dumps(r1)}, ctx={},
        )
    assert r2["gate"]["kind"] == "chosen"
    assert "Ultra" in r2["chosen_group"]["representative_title"]


@pytest.mark.asyncio
async def test_end_to_end_all_filtered_escalates():
    from src.workflows.shopping.pipeline_v2 import (
        _handler_resolve_candidates, _handler_group_label_filter_gate,
    )
    products = [_p("S25 Knockoff",  "aliexpress", sku="AE-1", cat="Telefon", rank=0)]
    label_resp = json.dumps({"groups": [
        {"group_id": 0, "product_type": "knockoff", "base_model": "Samsung Galaxy S25", "variant": None, "authenticity_confidence": 0.3, "matches_user_intent": False},
    ]})
    with patch("src.workflows.shopping.pipeline_v2._fetch_products",
               new=AsyncMock(return_value=products)), \
         patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(return_value={"content": label_resp})):
        r1 = await _handler_resolve_candidates(
            task={"id": 1}, artifacts={"user_query": "samsung s25"},
            ctx={"per_site_n": 3},
        )
        r2 = await _handler_group_label_filter_gate(
            task={"id": 2}, artifacts={"search_results": json.dumps(r1)}, ctx={},
        )
    assert r2["gate"]["kind"] == "escalation"
    assert r2["gate"]["reason"] == "all_filtered"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/shopping/test_variant_flow_integration.py -v`
Expected: At least one FAIL or ImportError if earlier tasks aren't applied in order.

- [ ] **Step 3: Fix any integration issues**

Read failure; usually something trivial — missing `query` propagation, wrong json shape, missing `sku` extraction. Patch as needed, keeping scope to wiring only.

- [ ] **Step 4: Run tests**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/shopping/ -x -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
rtk git add tests/shopping/test_variant_flow_integration.py
rtk git commit -m "test(shopping): variant flow integration tests (3 scenarios)"
```

---

## Task 15: Live smoke harness

**Files:**
- Create: `tests/shopping/verify_variant_flow_live.py`

- [ ] **Step 1: Write the harness (no failing-test cycle — this hits live sites)**

```python
"""Live smoke for variant disambiguation — NOT a pytest.

Run manually:
    timeout 180 .venv/Scripts/python.exe -m tests.shopping.verify_variant_flow_live
"""
from __future__ import annotations
import asyncio, json
from dotenv import load_dotenv
load_dotenv()


async def main():
    from src.workflows.shopping.pipeline_v2 import (
        _handler_resolve_candidates, _handler_group_label_filter_gate,
    )
    query = "samsung s25"
    r1 = await _handler_resolve_candidates(
        task={"id": 1}, artifacts={"user_query": query}, ctx={"per_site_n": 3},
    )
    print(f"candidates={len(r1['candidates'])}")
    r2 = await _handler_group_label_filter_gate(
        task={"id": 2}, artifacts={"search_results": json.dumps(r1)}, ctx={},
    )
    print(f"gate={r2['gate']['kind']}")
    if r2["gate"]["kind"] == "clarify":
        for opt in r2["clarify_options"]:
            print(f"  option: {opt['label']} (prominence {opt['prominence']:.2f})")
    elif r2["gate"]["kind"] == "chosen":
        print(f"  chosen: {r2['chosen_group']['representative_title']}")
    else:
        print(f"  reason: {r2['gate'].get('reason')}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run it**

Run: `timeout 180 .venv/Scripts/python.exe -m tests.shopping.verify_variant_flow_live`
Expected: `gate=clarify` with ≥2 options for "samsung s25". Log for record.

- [ ] **Step 3: Commit**

```bash
rtk git add tests/shopping/verify_variant_flow_live.py
rtk git commit -m "test(shopping): live variant-flow smoke harness"
```

---

## Task 16: Scraper audit — trendyol, hepsiburada, amazon_tr (batch 1)

**Files:**
- Modify: `src/shopping/scrapers/trendyol.py`, `hepsiburada.py`, `amazon_tr.py`
- Create: `tests/shopping/fixtures/trendyol_search.html`, `hepsiburada_search.html`, `amazon_tr_search.html` (minimal fixtures cut from a real search response, enough to exercise the parser)
- Create: `tests/shopping/test_scraper_sku_batch1.py`

- [ ] **Step 1: Write the failing tests**

Each scraper gets a parse test using a fixture file:

```python
# tests/shopping/test_scraper_sku_batch1.py
from pathlib import Path
from src.shopping.scrapers.trendyol import TrendyolScraper
from src.shopping.scrapers.hepsiburada import HepsiburadaScraper
from src.shopping.scrapers.amazon_tr import AmazonTrScraper

FIX = Path(__file__).parent / "fixtures"

def test_trendyol_search_populates_sku_and_category():
    html = (FIX / "trendyol_search.html").read_text(encoding="utf-8")
    s = TrendyolScraper()
    products = s._parse_search_html(html)   # or whatever internal parser is named
    assert any(p.sku for p in products), "at least one product has sku"
    # Specific assertions tied to the fixture:
    iphone = next(p for p in products if "Apple" in p.name)
    assert iphone.sku  # e.g. content id
    assert iphone.category_path  # e.g. "Elektronik > Cep Telefonu"

def test_hepsiburada_search_populates_sku_and_category():
    html = (FIX / "hepsiburada_search.html").read_text(encoding="utf-8")
    s = HepsiburadaScraper()
    products = s._parse_search_html(html)
    hit = next((p for p in products if p.sku), None)
    assert hit is not None
    assert hit.sku.startswith("HB")  # Hepsiburada SKUs start with HB...
    assert hit.category_path

def test_amazon_tr_search_populates_sku_and_category():
    html = (FIX / "amazon_tr_search.html").read_text(encoding="utf-8")
    s = AmazonTrScraper()
    products = s._parse_search_html(html)
    hit = next((p for p in products if p.sku), None)
    assert hit is not None
    assert len(hit.sku) == 10  # Amazon ASINs are 10 chars
```

- [ ] **Step 2: Record fixtures**

One-time operation: call each scraper's `search("iphone 15")` live, save the raw HTML response under `tests/shopping/fixtures/<scraper>_search.html`. Keep fixtures small (trim irrelevant ads/navs if needed to stay <200KB each).

- [ ] **Step 3: Run tests to verify they fail**

Run: `timeout 30 .venv/Scripts/python.exe -m pytest tests/shopping/test_scraper_sku_batch1.py -v`
Expected: FAIL — parser doesn't populate sku/category yet.

- [ ] **Step 4: Implement extractors in each scraper**

For each scraper, update the `_parse_search_*` and `_dict_to_product` (or equivalent) paths to populate `Product.sku` and `Product.category_path`:

- **trendyol**: extract content-id from URL (`/p-(\d+)` regex); category from breadcrumb list in page JSON-LD (`categoryName` fields).
- **hepsiburada**: SKU from URL (`-p-(HBC[A-Z0-9]+|HBV[A-Z0-9]+)`); category from `productCategory`/breadcrumb JSON.
- **amazon_tr**: ASIN from `data-asin` attribute or URL (`/dp/([A-Z0-9]{10})`); category from breadcrumb (if present in search result; set None otherwise).

Make the smallest change per scraper. Keep existing behaviour identical.

- [ ] **Step 5: Run tests**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/shopping/test_scraper_sku_batch1.py -v`
Expected: PASS 3/3.

- [ ] **Step 6: Commit**

```bash
rtk git add src/shopping/scrapers/trendyol.py src/shopping/scrapers/hepsiburada.py src/shopping/scrapers/amazon_tr.py tests/shopping/fixtures/ tests/shopping/test_scraper_sku_batch1.py
rtk git commit -m "feat(scrapers): populate sku + category_path for trendyol, hepsiburada, amazon_tr"
```

---

## Task 17: Scraper audit — akakce, epey, kitapyurdu (batch 2)

Same pattern as Task 16. Fixtures and tests for:

- `akakce`: SKU-ish ID from URL path last segment; category from breadcrumb (Akakçe exposes its category tree prominently).
- `epey`: product-slug from URL (`akilli-telefonlar/apple-iphone-15-pro.html` → `apple-iphone-15-pro`); category from breadcrumb (`Akıllı Telefonlar`).
- `kitapyurdu`: numeric book id from URL (`/kitap/<slug>/<id>.html`); category (Kitap, genre).

Files:
- Modify: `src/shopping/scrapers/{akakce,epey,kitapyurdu}.py`
- Create: 3 fixtures, 1 test module `tests/shopping/test_scraper_sku_batch2.py`

TDD cycle identical to Task 16. Commit at end:

```bash
rtk git commit -m "feat(scrapers): populate sku + category_path for akakce, epey, kitapyurdu"
```

---

## Task 18: Scraper audit — dr_com_tr, decathlon_tr, home_improvement (batch 3)

Same pattern. `home_improvement` contains both `koctas` and `ikea` parsers — test both.

Files:
- Modify: `src/shopping/scrapers/{dr_com_tr,decathlon_tr,home_improvement}.py`
- Create: 4 fixtures (dr, decathlon, koctas, ikea), 1 test module `tests/shopping/test_scraper_sku_batch3.py`

TDD cycle identical. Commit:

```bash
rtk git commit -m "feat(scrapers): populate sku + category_path for dr, decathlon, koctas, ikea"
```

---

## Task 19: Scraper audit — grocery (migros + getir), batch 4

Files:
- Modify: `src/shopping/scrapers/grocery.py`
- Create: 2 fixtures (migros, getir), 1 test module `tests/shopping/test_scraper_sku_batch4.py`

TDD cycle identical. Migros/Getir are API-JSON — record one `.json` fixture each and parse. Commit:

```bash
rtk git commit -m "feat(scrapers): populate sku + category_path for migros, getir"
```

---

## Task 20: Scraper audit — sahibinden, arabam, direnc_net (batch 5)

Files:
- Modify: `src/shopping/scrapers/{sahibinden,arabam,direnc_net}.py`
- Create: 3 fixtures, 1 test module `tests/shopping/test_scraper_sku_batch5.py`

Note: sahibinden/arabam are listing-style; `sku` = listing id, `category_path` = listing category (vehicle model/category tree).

TDD cycle identical. Commit:

```bash
rtk git commit -m "feat(scrapers): populate sku + category_path for sahibinden, arabam, direnc_net"
```

---

## Task 21: Scraper audit — google_cse, forums (batch 6)

Files:
- Modify: `src/shopping/scrapers/google_cse.py`, `src/shopping/scrapers/forums.py`
- Create: 2-3 fixtures, 1 test module `tests/shopping/test_scraper_sku_batch6.py`

google_cse: derive sku from destination URL when parseable; category from CSE metadata if available.
forums (technopat, donanimhaber): no meaningful SKU — leave `None`. Category = "community". Test that both parse without error and leave `sku=None`.

Skip SKU extraction for `eksisozluk`, `sikayetvar` (community, no real SKU). Document in test comments.

TDD cycle identical. Commit:

```bash
rtk git commit -m "feat(scrapers): populate sku + category_path for google_cse, forums"
```

---

## Task 22: Full-suite regression run

**Files:** none — just verification.

- [ ] **Step 1: Run full targeted suite**

Run: `timeout 300 .venv/Scripts/python.exe -m pytest tests/shopping/ -x -q`
Expected: all green.

- [ ] **Step 2: Run whole repo test suite (non-LLM)**

Run: `timeout 300 .venv/Scripts/python.exe -m pytest tests/ -m "not llm" -x -q --ignore=tests/integration`
Expected: all green. Fix any regressions introduced; scope fixes to variant-disambiguation code only.

- [ ] **Step 3: Live end-to-end via KutAI**

Restart KutAI via Telegram `/restart`. From Telegram:
- `/shop samsung s25` → expect clarify keyboard with ≥3 variant buttons + "Hepsini karşılaştır".
- Tap "Galaxy S25" → expect single-variant card with reviews.
- `/shop samsung s25 ultra 256gb` → expect single card, no clarify.
- `/shop iphone 15 case` → expect either a clarify across case variants OR escalation (both acceptable; document which).

Record observations in `docs/superpowers/plans/2026-04-22-shopping-variant-disambiguation.md` under a new "Live results" appendix.

- [ ] **Step 4: Final commit**

```bash
rtk git add docs/superpowers/plans/2026-04-22-shopping-variant-disambiguation.md
rtk git commit -m "docs(plan): variant disambiguation live results"
```

---

## Post-plan notes

- The spec's follow-up ("quantitative review intelligence") starts only after this plan lands and the live behaviour has stabilised.
- If scraper audit reveals that most sites gate SKU behind JS rendering (not raw HTML), escalate the scraper tier to STEALTH for SKU extraction only, or defer that scraper to a later batch. Don't block pipeline tasks on scraper completeness — SKU-less products still flow through the LLM grouping path.
- Benchmark: time `step_label` on real queries. If P95 > 3s, split into groups-of-10 batches with parallel dispatch.

---

## Live results (2026-04-23)

Branch `feat/shopping-variant-disambig`, 22 tasks executed via subagent-driven TDD. 25 commits from `765d6fa` (Task 1) through the UTF-8 harness fix at the end.

**Regression**: `tests/shopping/` 732/732 green (excluding one pre-existing flaky timing test in `test_performance.py`).

**Live smoke `python -m tests.shopping.verify_variant_flow_live`** — query `"samsung s25"`:

- 20 candidates returned from live scrapers (most successful: epey, trendyol, hepsiburada, amazon_tr, akakce).
- Gate: **clarify**. Five variant options surfaced with real product titles (S25 vanilla, S25 FE 256GB in multiple storage variants, S25 Ultra).
- One accessory ("Samsung Galaxy S25 Ultra Ekran Koruyucu") leaked into the clarify options because the label LLM path fell back (see below). With the label LLM working, accessories would be filtered out structurally.

**Investigation — "label LLM failed" during smoke**:

- Initial finding: `shopping_labeler`, `shopping_grouper`, `shopping_review_synthesizer` all erred at the model-selection layer with `"No model candidates available"`, triggering `_fallback_labels`.
- Root cause: the smoke harness did not call `fatih_hoca.init(catalog_path=..., models_dir=..., available_providers=...)` before invoking handlers. Without init, Fatih Hoca's registry is empty and `select()` returns None for every task.
- Production is unaffected — `src/app/run.py` initializes Fatih Hoca at boot. `model_pick_log` in the prod DB confirms `shopping_grouper` and `shopping_review_synthesizer` successfully pick local GGUFs (gemma-4-26B, Qwen3.5-9B/35B) and complete. The newly-added `shopping_labeler` profile will select the same way.
- Fix committed to the harness: `_bootstrap_fatih_hoca()` mirrors `src/app/run.py`'s init so the smoke runs against a real registry. Live re-run of the harness is risky while KutAI is running (fights over the shared llama-server), so final validation is via the Telegram manual steps below.

**Follow-ups queued**:

1. Diagnose why shopping-family task profiles can't resolve a model in Fatih Hoca. Likely a missing hook in `select()` or the eligibility filter — needs a dedicated debugging session.
2. Quantitative review intelligence spec (originally Option B-after-A, now truly deferred).
3. Category-search UX — spec mentions "evolve Compare all into category search later". When the bot has a sharper idea of what matches the user's intent (after quantitative reviews land), this turns into a structured comparison page rather than a flat list.

Manual Telegram validation step (required before declaring done):

- Restart KutAI via Telegram `/restart`.
- `/shop Samsung s25` → expect clarify keyboard with ≥3 variant buttons + "📊 Hepsini karşılaştır".
- Tap a variant → expect single-variant card with reviews.
- `/shop Samsung s25 ultra 256gb` → expect single card, no clarify.
- Tap "Hepsini karşılaştır" on an ambiguous query → expect compact comparison table.

