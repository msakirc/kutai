# Shopping Pipeline v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the heuristic product matcher + relevance filter with an LLM-based grouping step over each site's native ordering, and add LLM review synthesis. Reviews-first Telegram output.

**Architecture:** New `pipeline_v2.py` module with five pure-function steps (`resolve → group → synthesize → format`). Two LLM calls go through `LLMDispatcher` (`main_work` category, grouping difficulty=3, synthesis difficulty=6). Three parallel `_v2.json` workflows reuse the same pipeline with different breadth. Cutover by switching `wf_map` in `telegram_bot.py` — no feature flag.

**Tech Stack:** Python 3.10 async, `LLMDispatcher` (`src.core.llm_dispatcher`), existing `shopping_search` / scraper fallback (`src.shopping.resilience.fallback_chain`), pytest with `pytest-asyncio`. LLM tests use fixtures, no live calls.

**Spec:** `docs/superpowers/specs/2026-04-21-shopping-trust-sites-synthesize-reviews-design.md`

---

## File Structure

**Create:**
- `src/workflows/shopping/pipeline_v2.py` — dataclasses + step functions + `ShoppingPipelineV2` dispatch class
- `src/workflows/shopping/prompts_v2.py` — `GROUPING_PROMPT`, `SYNTHESIS_PROMPT`, frozen strings
- `src/workflows/shopping/product_research_v2.json`
- `src/workflows/shopping/quick_search_v2.json`
- `src/workflows/shopping/shopping_v2.json`
- `tests/shopping/test_pipeline_v2.py`
- `tests/shopping/test_pipeline_v2_integration.py`

**Modify:**
- `src/core/orchestrator.py:25-32` — add `shopping_pipeline_v2` to `AGENT_TIMEOUTS`
- `src/core/orchestrator.py:82-84` — add agent-type branch for `shopping_pipeline_v2`
- `src/app/telegram_bot.py:4503-4511` — `wf_map` entries point to `_v2` workflow names

**Untouched (deleted in follow-up PR):**
- `src/workflows/shopping/pipeline.py`
- `src/workflows/shopping/product_matcher.py`
- `src/workflows/shopping/shopping.json`, `product_research.json`, `quick_search.json`

---

## Task 1: Scaffold pipeline_v2 module with dataclasses

**Files:**
- Create: `src/workflows/shopping/pipeline_v2.py`
- Test: `tests/shopping/test_pipeline_v2.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/shopping/test_pipeline_v2.py
"""Tests for shopping pipeline v2."""
from __future__ import annotations

import pytest

from src.workflows.shopping.pipeline_v2 import (
    Candidate,
    ProductGroup,
    ReviewSynthesis,
)


def test_dataclass_shapes():
    c = Candidate(
        title="Siemens EQ.6 Plus",
        site="hepsiburada",
        site_rank=1,
        price=24745.0,
        original_price=None,
        url="https://example.com",
        rating=4.5,
        review_count=312,
        review_snippets=["Köpük güzel"],
    )
    assert c.site_rank == 1

    g = ProductGroup(
        representative_title="Siemens EQ.6 Plus",
        member_indices=[0, 2],
        is_accessory_or_part=False,
        prominence=1.5,
    )
    assert g.member_indices == [0, 2]

    s = ReviewSynthesis(
        praise=["Köpük güzel"],
        complaints=[],
        red_flags=[],
        insufficient_data=False,
    )
    assert s.praise == ["Köpük güzel"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_dataclass_shapes -v`
Expected: `ModuleNotFoundError: No module named 'src.workflows.shopping.pipeline_v2'`

- [ ] **Step 3: Create the module with dataclasses**

```python
# src/workflows/shopping/pipeline_v2.py
"""Shopping pipeline v2 — trust site ordering, LLM-based grouping + review synthesis.

See docs/superpowers/specs/2026-04-21-shopping-trust-sites-synthesize-reviews-design.md
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.infra.logging_config import get_logger

logger = get_logger("workflows.shopping.pipeline_v2")


@dataclass
class Candidate:
    """One search result from one site, in that site's original order."""
    title: str
    site: str
    site_rank: int                 # 1-based position in site's result list
    price: float | None
    original_price: float | None
    url: str
    rating: float | None
    review_count: int | None
    review_snippets: list[str] = field(default_factory=list)


@dataclass
class ProductGroup:
    """A cluster of Candidates judged to refer to the same product."""
    representative_title: str
    member_indices: list[int]      # indices into the Candidate list
    is_accessory_or_part: bool
    prominence: float              # sum(1 / site_rank) across members


@dataclass
class ReviewSynthesis:
    """Synthesised pros/cons/red-flags for one ProductGroup."""
    praise: list[str]
    complaints: list[str]
    red_flags: list[str]
    insufficient_data: bool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_dataclass_shapes -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
git commit -m "feat(shopping_v2): scaffold pipeline_v2 module with dataclasses"
```

---

## Task 2: step_resolve — trust site ordering, no filtering

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py`
- Test: `tests/shopping/test_pipeline_v2.py`

The scraper tool is `get_product_with_fallback` in `src/shopping/resilience/fallback_chain.py` (already used by v1 at line 237 of pipeline.py). Results are dataclasses; convert to `Candidate` preserving per-site order. Per-site N cap is the only filtering.

- [ ] **Step 1: Write the failing test**

Append to `tests/shopping/test_pipeline_v2.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_step_resolve_preserves_site_order_and_caps_per_site_n():
    """Per-site top-N only; no token-overlap filter; site_rank is 1-based."""
    from src.workflows.shopping.pipeline_v2 import step_resolve

    # Fake scraper returns 4 products from site A, 2 from site B
    def _fake_product(name: str, site: str, url: str, price: float):
        # Mimic the scraper dataclass shape (see src/shopping/models.py)
        from types import SimpleNamespace
        return SimpleNamespace(
            name=name, site=site, url=url, price=price,
            original_price=None, rating=None, review_count=None,
            review_snippets=[],
        )

    fake_products = [
        _fake_product("A1", "trendyol", "u1", 100),
        _fake_product("A2", "trendyol", "u2", 110),
        _fake_product("A3", "trendyol", "u3", 120),
        _fake_product("A4", "trendyol", "u4", 130),
        _fake_product("B1", "hepsiburada", "u5", 140),
        _fake_product("B2", "hepsiburada", "u6", 150),
    ]

    with patch(
        "src.workflows.shopping.pipeline_v2._fetch_products",
        new=AsyncMock(return_value=fake_products),
    ):
        cands = await step_resolve("test query", per_site_n=2)

    # Top-2 from each site kept, site_rank is 1-based per site
    trendyol = [c for c in cands if c.site == "trendyol"]
    hepsi = [c for c in cands if c.site == "hepsiburada"]
    assert [c.title for c in trendyol] == ["A1", "A2"]
    assert [c.site_rank for c in trendyol] == [1, 2]
    assert [c.title for c in hepsi] == ["B1", "B2"]
    assert [c.site_rank for c in hepsi] == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_step_resolve_preserves_site_order_and_caps_per_site_n -v`
Expected: `ImportError: cannot import name 'step_resolve'`

- [ ] **Step 3: Implement step_resolve**

Append to `src/workflows/shopping/pipeline_v2.py`:

```python
import asyncio
from collections import OrderedDict


async def _fetch_products(query: str) -> list:
    """Thin wrapper around the shopping scraper fleet — mocked in tests."""
    from src.shopping.resilience.fallback_chain import get_product_with_fallback
    return await asyncio.wait_for(get_product_with_fallback(query), timeout=45)


def _attr(obj, name: str, default=None):
    """Read attribute from dataclass OR dict — scrapers mix both."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


async def step_resolve(query: str, per_site_n: int) -> list[Candidate]:
    """Fetch scraper results, keep top-N per site in site order, no filtering."""
    logger.info("step_resolve start", query=query[:80], per_site_n=per_site_n)
    try:
        raw = await _fetch_products(query)
    except Exception as exc:
        logger.warning("resolve fetch failed: %s", exc)
        raw = []

    # Group by site preserving order of first appearance
    by_site: "OrderedDict[str, list]" = OrderedDict()
    for p in raw or []:
        site = _attr(p, "site") or "unknown"
        by_site.setdefault(site, []).append(p)

    cands: list[Candidate] = []
    for site, products in by_site.items():
        for rank, p in enumerate(products[:per_site_n], start=1):
            cands.append(
                Candidate(
                    title=str(_attr(p, "name") or ""),
                    site=site,
                    site_rank=rank,
                    price=_attr(p, "price"),
                    original_price=_attr(p, "original_price"),
                    url=str(_attr(p, "url") or ""),
                    rating=_attr(p, "rating"),
                    review_count=_attr(p, "review_count"),
                    review_snippets=list(_attr(p, "review_snippets") or []),
                )
            )
    logger.info("step_resolve done", candidate_count=len(cands))
    return cands
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
git commit -m "feat(shopping_v2): step_resolve — trust site ordering"
```

---

## Task 3: Prompts module with frozen templates

**Files:**
- Create: `src/workflows/shopping/prompts_v2.py`
- Test: `tests/shopping/test_pipeline_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/shopping/test_pipeline_v2.py`:

```python
def test_prompt_templates_exist_and_have_required_placeholders():
    from src.workflows.shopping import prompts_v2

    # GROUPING_PROMPT must accept the candidates JSON block
    assert "{candidates_json}" in prompts_v2.GROUPING_PROMPT
    # It must instruct the model to flag accessories
    assert "accessor" in prompts_v2.GROUPING_PROMPT.lower() or "part" in prompts_v2.GROUPING_PROMPT.lower()

    # SYNTHESIS_PROMPT must accept the representative title and snippets
    assert "{representative_title}" in prompts_v2.SYNTHESIS_PROMPT
    assert "{review_snippets_json}" in prompts_v2.SYNTHESIS_PROMPT
    # Must instruct output JSON schema
    assert "praise" in prompts_v2.SYNTHESIS_PROMPT
    assert "complaints" in prompts_v2.SYNTHESIS_PROMPT
    assert "insufficient_data" in prompts_v2.SYNTHESIS_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_prompt_templates_exist_and_have_required_placeholders -v`
Expected: `ModuleNotFoundError: No module named 'src.workflows.shopping.prompts_v2'`

- [ ] **Step 3: Create prompts_v2.py**

```python
# src/workflows/shopping/prompts_v2.py
"""Frozen LLM prompt templates for shopping pipeline v2.

Keep prompts versioned in this module. Any change that affects output shape
must bump the version comment and update the parser tests.
"""
from __future__ import annotations

# ─── Grouping ──────────────────────────────────────────────────────────────
# Version: v2.0.0
# Output contract: JSON list of {representative_title, member_indices, is_accessory_or_part}
GROUPING_PROMPT = """You are grouping shopping search results across multiple e-commerce sites.

Input candidates (each has an integer `index`):
{candidates_json}

Task:
1. Cluster candidates that refer to the SAME product (same brand + model + variant).
   Different colours or storage tiers of the same model are the same group.
   Different models from the same product line are DIFFERENT groups (e.g. Siemens EQ.3 vs EQ.6).
2. Flag accessories, replacement parts, filters, covers, or spare components as `is_accessory_or_part: true`.
   A full coffee machine is NOT a part. A brewing unit / demleme ünitesi sold separately IS a part.
3. Pick a clean representative_title for each group (shortest member title is usually best).

Return ONLY valid JSON in this exact shape, no prose, no markdown fences:
{{
  "groups": [
    {{
      "representative_title": "string",
      "member_indices": [int, ...],
      "is_accessory_or_part": bool
    }}
  ]
}}
"""

# ─── Synthesis ─────────────────────────────────────────────────────────────
# Version: v2.0.0
# Output contract: JSON {praise, complaints, red_flags, insufficient_data}
SYNTHESIS_PROMPT = """You are summarising user reviews for a product across multiple sources.

Product: {representative_title}

Review snippets (Turkish and/or English, from multiple sources):
{review_snippets_json}

Task:
- Extract recurring PRAISE points (what users like). Max 3 bullets, short phrases.
- Extract recurring COMPLAINTS (what users dislike). Max 3 bullets.
- Extract RED FLAGS (safety, reliability, fraud concerns, complaint-site mentions). Max 3 bullets.
- If the snippets are too few, too short, or irrelevant to judge this product, set insufficient_data=true and leave lists empty.
- Do NOT fabricate points that aren't supported by the snippets. Better to output insufficient_data=true than to guess.
- Output in the dominant language of the snippets (Turkish if Turkish dominates).

Return ONLY valid JSON, no prose, no markdown fences:
{{
  "praise": ["string", ...],
  "complaints": ["string", ...],
  "red_flags": ["string", ...],
  "insufficient_data": bool
}}
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/workflows/shopping/prompts_v2.py tests/shopping/test_pipeline_v2.py
git commit -m "feat(shopping_v2): frozen grouping + synthesis prompts"
```

---

## Task 4: step_group — LLM call, parser, and fallback

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py`
- Test: `tests/shopping/test_pipeline_v2.py`

`LLMDispatcher.request(...)` returns a dict; the `content` string field holds the LLM text (see `src/core/grading.py:241-260` for a working example). We use `CallCategory.MAIN_WORK`, `difficulty=3`, `task="shopping_grouper"`.

- [ ] **Step 1: Write failing test — happy path parse**

Append to `tests/shopping/test_pipeline_v2.py`:

```python
@pytest.mark.asyncio
async def test_step_group_parses_llm_response_and_filters_accessories():
    """Grouping LLM output is parsed; accessory-flagged groups kept with flag set."""
    from src.workflows.shopping.pipeline_v2 import Candidate, step_group

    cands = [
        Candidate(title="Siemens EQ.6 S100 coffee machine", site="hepsiburada",
                  site_rank=1, price=24745.0, original_price=None,
                  url="h1", rating=4.5, review_count=100, review_snippets=[]),
        Candidate(title="DL-Pro demleme ünitesi Siemens EQ.3 S100",
                  site="amazon_tr", site_rank=1, price=4800.0,
                  original_price=None, url="a1", rating=5.0,
                  review_count=3, review_snippets=[]),
    ]
    fake_llm_response = {
        "content": (
            '{"groups": ['
            '  {"representative_title": "Siemens EQ.6 S100", '
            '   "member_indices": [0], "is_accessory_or_part": false},'
            '  {"representative_title": "Siemens EQ.3 brewing unit", '
            '   "member_indices": [1], "is_accessory_or_part": true}'
            ']}'
        ),
        "model": "fake-model",
        "cost": 0.0,
    }
    with patch(
        "src.workflows.shopping.pipeline_v2._grouping_llm_call",
        new=AsyncMock(return_value=fake_llm_response),
    ):
        groups = await step_group(cands)

    assert len(groups) == 2
    assert groups[0].representative_title.startswith("Siemens EQ.6")
    assert groups[0].is_accessory_or_part is False
    assert groups[1].is_accessory_or_part is True


@pytest.mark.asyncio
async def test_step_group_fallback_on_llm_failure():
    """LLM failure → each site's top-1 becomes its own group (trust-sites fallback)."""
    from src.workflows.shopping.pipeline_v2 import Candidate, step_group

    cands = [
        Candidate(title="A1", site="trendyol", site_rank=1, price=100,
                  original_price=None, url="u1", rating=None,
                  review_count=None, review_snippets=[]),
        Candidate(title="A2", site="trendyol", site_rank=2, price=110,
                  original_price=None, url="u2", rating=None,
                  review_count=None, review_snippets=[]),
        Candidate(title="B1", site="hepsiburada", site_rank=1, price=140,
                  original_price=None, url="u3", rating=None,
                  review_count=None, review_snippets=[]),
    ]
    with patch(
        "src.workflows.shopping.pipeline_v2._grouping_llm_call",
        new=AsyncMock(side_effect=RuntimeError("LLM boom")),
    ):
        groups = await step_group(cands)

    # One group per site, containing that site's rank-1 candidate only
    assert len(groups) == 2
    titles = {g.representative_title for g in groups}
    assert titles == {"A1", "B1"}


@pytest.mark.asyncio
async def test_step_group_fallback_on_malformed_json():
    from src.workflows.shopping.pipeline_v2 import Candidate, step_group

    cands = [
        Candidate(title="X1", site="s1", site_rank=1, price=1, original_price=None,
                  url="u", rating=None, review_count=None, review_snippets=[]),
    ]
    bad = {"content": "not json at all", "model": "m", "cost": 0}
    with patch(
        "src.workflows.shopping.pipeline_v2._grouping_llm_call",
        new=AsyncMock(return_value=bad),
    ):
        groups = await step_group(cands)

    assert len(groups) == 1
    assert groups[0].representative_title == "X1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v -k step_group`
Expected: `ImportError: cannot import name 'step_group'`

- [ ] **Step 3: Implement step_group + LLM helper + parser + fallback**

Append to `src/workflows/shopping/pipeline_v2.py`:

```python
import json
import re


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` fences some models emit."""
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL)
    return m.group(1) if m else text


def _per_site_top1_fallback(candidates: list[Candidate]) -> list[ProductGroup]:
    """Grouping fallback: one group per site's rank-1 candidate."""
    seen_sites: set[str] = set()
    groups: list[ProductGroup] = []
    for idx, c in enumerate(candidates):
        if c.site_rank != 1:
            continue
        if c.site in seen_sites:
            continue
        seen_sites.add(c.site)
        groups.append(
            ProductGroup(
                representative_title=c.title,
                member_indices=[idx],
                is_accessory_or_part=False,
                prominence=1.0,
            )
        )
    return groups


async def _grouping_llm_call(prompt: str) -> dict:
    """Dispatch the grouping prompt. Returns the dispatcher response dict.

    Split out so tests can patch this one function instead of the dispatcher.
    """
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    dispatcher = get_dispatcher()
    return await dispatcher.request(
        category=CallCategory.MAIN_WORK,
        task="shopping_grouper",
        agent_type="shopping_pipeline_v2",
        difficulty=3,
        messages=[
            {"role": "system", "content": "You output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )


async def step_group(candidates: list[Candidate]) -> list[ProductGroup]:
    """LLM-based grouping of candidates into product groups.

    Falls back to one group per site's rank-1 candidate on any LLM or parse error.
    """
    if not candidates:
        return []

    from src.workflows.shopping.prompts_v2 import GROUPING_PROMPT

    # Compact JSON view for the LLM — titles + site + price only
    view = [
        {"index": i, "title": c.title, "site": c.site, "price": c.price}
        for i, c in enumerate(candidates)
    ]
    prompt = GROUPING_PROMPT.format(candidates_json=json.dumps(view, ensure_ascii=False))

    try:
        resp = await _grouping_llm_call(prompt)
    except Exception as exc:
        logger.warning("grouping LLM failed, using per-site fallback: %s", exc)
        return _per_site_top1_fallback(candidates)

    content = _strip_json_fences(str(resp.get("content", "")).strip())
    try:
        parsed = json.loads(content)
        raw_groups = parsed.get("groups", [])
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning("grouping LLM output not parseable, using fallback: %s", exc)
        return _per_site_top1_fallback(candidates)

    groups: list[ProductGroup] = []
    n = len(candidates)
    for g in raw_groups:
        members = [i for i in (g.get("member_indices") or []) if isinstance(i, int) and 0 <= i < n]
        if not members:
            continue
        title = str(g.get("representative_title") or candidates[members[0]].title)
        is_acc = bool(g.get("is_accessory_or_part"))
        prominence = sum(1.0 / candidates[i].site_rank for i in members)
        groups.append(
            ProductGroup(
                representative_title=title,
                member_indices=members,
                is_accessory_or_part=is_acc,
                prominence=prominence,
            )
        )

    if not groups:
        logger.warning("grouping returned no valid groups, using fallback")
        return _per_site_top1_fallback(candidates)

    logger.info(
        "step_group done",
        group_count=len(groups),
        accessory_drop_count=sum(1 for g in groups if g.is_accessory_or_part),
    )
    return groups
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
git commit -m "feat(shopping_v2): step_group LLM call with per-site fallback"
```

---

## Task 5: step_synthesize_reviews — LLM call, parser, insufficient_data branch

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py`
- Test: `tests/shopping/test_pipeline_v2.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/shopping/test_pipeline_v2.py`:

```python
@pytest.mark.asyncio
async def test_step_synthesize_parses_full_response():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, step_synthesize_reviews,
    )

    cands = [
        Candidate(title="T", site="s", site_rank=1, price=100, original_price=None,
                  url="u", rating=None, review_count=None,
                  review_snippets=["köpük harika", "sessiz"]),
    ]
    group = ProductGroup(
        representative_title="T",
        member_indices=[0],
        is_accessory_or_part=False,
        prominence=1.0,
    )
    fake_llm = {
        "content": (
            '{"praise": ["köpük iyi"], '
            ' "complaints": ["pahalı"], '
            ' "red_flags": [], '
            ' "insufficient_data": false}'
        ),
        "model": "m", "cost": 0,
    }
    with patch(
        "src.workflows.shopping.pipeline_v2._synthesis_llm_call",
        new=AsyncMock(return_value=fake_llm),
    ):
        syn = await step_synthesize_reviews(group, cands)

    assert syn.insufficient_data is False
    assert syn.praise == ["köpük iyi"]
    assert syn.complaints == ["pahalı"]
    assert syn.red_flags == []


@pytest.mark.asyncio
async def test_step_synthesize_short_circuits_when_no_snippets():
    """No snippets at all → return insufficient_data without calling LLM."""
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, step_synthesize_reviews,
    )

    cands = [
        Candidate(title="T", site="s", site_rank=1, price=1, original_price=None,
                  url="u", rating=None, review_count=None, review_snippets=[]),
    ]
    group = ProductGroup("T", [0], False, 1.0)

    called = AsyncMock()
    with patch("src.workflows.shopping.pipeline_v2._synthesis_llm_call", new=called):
        syn = await step_synthesize_reviews(group, cands)

    assert syn.insufficient_data is True
    assert syn.praise == [] and syn.complaints == [] and syn.red_flags == []
    called.assert_not_called()


@pytest.mark.asyncio
async def test_step_synthesize_insufficient_data_flag_from_llm():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, step_synthesize_reviews,
    )

    cands = [
        Candidate(title="T", site="s", site_rank=1, price=1, original_price=None,
                  url="u", rating=None, review_count=None,
                  review_snippets=["tek yorum"]),
    ]
    group = ProductGroup("T", [0], False, 1.0)
    fake_llm = {
        "content": ('{"praise":[],"complaints":[],"red_flags":[],"insufficient_data":true}'),
        "model": "m", "cost": 0,
    }
    with patch(
        "src.workflows.shopping.pipeline_v2._synthesis_llm_call",
        new=AsyncMock(return_value=fake_llm),
    ):
        syn = await step_synthesize_reviews(group, cands)

    assert syn.insufficient_data is True


@pytest.mark.asyncio
async def test_step_synthesize_failure_returns_insufficient_data():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, step_synthesize_reviews,
    )

    cands = [
        Candidate(title="T", site="s", site_rank=1, price=1, original_price=None,
                  url="u", rating=None, review_count=None,
                  review_snippets=["x"]),
    ]
    group = ProductGroup("T", [0], False, 1.0)
    with patch(
        "src.workflows.shopping.pipeline_v2._synthesis_llm_call",
        new=AsyncMock(side_effect=RuntimeError("LLM boom")),
    ):
        syn = await step_synthesize_reviews(group, cands)
    assert syn.insufficient_data is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v -k synthesize`
Expected: `ImportError: cannot import name 'step_synthesize_reviews'`

- [ ] **Step 3: Implement step_synthesize_reviews**

Append to `src/workflows/shopping/pipeline_v2.py`:

```python
async def _synthesis_llm_call(prompt: str) -> dict:
    """Dispatch the synthesis prompt. Returns the dispatcher response dict."""
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    dispatcher = get_dispatcher()
    return await dispatcher.request(
        category=CallCategory.MAIN_WORK,
        task="shopping_review_synthesizer",
        agent_type="shopping_pipeline_v2",
        difficulty=6,
        messages=[
            {"role": "system", "content": "You output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )


def _insufficient() -> ReviewSynthesis:
    return ReviewSynthesis(praise=[], complaints=[], red_flags=[], insufficient_data=True)


async def step_synthesize_reviews(
    group: ProductGroup, candidates: list[Candidate],
) -> ReviewSynthesis:
    """LLM-based review synthesis for one group. Returns insufficient_data on failure."""
    from src.workflows.shopping.prompts_v2 import SYNTHESIS_PROMPT

    # Gather snippets from all group members
    snippets: list[str] = []
    for idx in group.member_indices:
        if 0 <= idx < len(candidates):
            snippets.extend(s for s in candidates[idx].review_snippets if s and s.strip())

    if not snippets:
        logger.info(
            "synthesize short-circuit (no snippets)",
            representative_title=group.representative_title,
        )
        return _insufficient()

    prompt = SYNTHESIS_PROMPT.format(
        representative_title=group.representative_title,
        review_snippets_json=json.dumps(snippets, ensure_ascii=False),
    )

    try:
        resp = await _synthesis_llm_call(prompt)
    except Exception as exc:
        logger.warning("synthesis LLM failed: %s", exc)
        return _insufficient()

    content = _strip_json_fences(str(resp.get("content", "")).strip())
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("synthesis LLM output not parseable: %s", exc)
        return _insufficient()

    def _take_list(key: str) -> list[str]:
        v = parsed.get(key) or []
        return [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()][:3]

    syn = ReviewSynthesis(
        praise=_take_list("praise"),
        complaints=_take_list("complaints"),
        red_flags=_take_list("red_flags"),
        insufficient_data=bool(parsed.get("insufficient_data", False)),
    )
    logger.info(
        "step_synthesize done",
        representative_title=group.representative_title,
        snippet_count=len(snippets),
        insufficient=syn.insufficient_data,
    )
    return syn
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
git commit -m "feat(shopping_v2): step_synthesize_reviews with insufficient_data fallbacks"
```

---

## Task 6: Group selection — rank and cap per workflow breadth

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py`
- Test: `tests/shopping/test_pipeline_v2.py`

Per spec: drop accessories; sort by prominence desc; keep first, keep second/third only if prominence ≥ 50% of top. Named-product caps at 2, category at 3.

- [ ] **Step 1: Write failing test**

Append to `tests/shopping/test_pipeline_v2.py`:

```python
def test_select_groups_drops_accessories_and_applies_50pct_rule():
    from src.workflows.shopping.pipeline_v2 import ProductGroup, select_groups

    groups = [
        ProductGroup("Dominant",  [0, 1], False, prominence=2.0),
        ProductGroup("Close runner", [2], False, prominence=1.2),   # 60% of top → keep
        ProductGroup("Weak runner", [3], False, prominence=0.8),    # 40% of top → drop
        ProductGroup("Accessory", [4], True, prominence=99.0),      # always drop
    ]

    kept_named = select_groups(groups, max_groups=2)
    assert [g.representative_title for g in kept_named] == ["Dominant", "Close runner"]

    # At max_groups=3 same behaviour (weak runner still under 50%)
    kept_cat = select_groups(groups, max_groups=3)
    assert [g.representative_title for g in kept_cat] == ["Dominant", "Close runner"]


def test_select_groups_empty_input():
    from src.workflows.shopping.pipeline_v2 import select_groups
    assert select_groups([], max_groups=2) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v -k select_groups`
Expected: `ImportError: cannot import name 'select_groups'`

- [ ] **Step 3: Implement select_groups**

Append to `src/workflows/shopping/pipeline_v2.py`:

```python
def select_groups(
    groups: list[ProductGroup], max_groups: int,
) -> list[ProductGroup]:
    """Filter accessories, sort by prominence desc, apply the 50%-of-top rule."""
    non_acc = [g for g in groups if not g.is_accessory_or_part]
    if not non_acc:
        return []
    non_acc.sort(key=lambda g: g.prominence, reverse=True)
    top = non_acc[0]
    kept: list[ProductGroup] = [top]
    for g in non_acc[1:max_groups]:
        if g.prominence >= top.prominence * 0.5:
            kept.append(g)
        else:
            break
    return kept
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
git commit -m "feat(shopping_v2): select_groups — accessory drop + 50pct rule"
```

---

## Task 7: format_group_card and format_response

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py`
- Test: `tests/shopping/test_pipeline_v2.py`

Output shape per spec Section "Output Format": reviews-first; omit empty sections; footer when `insufficient_data`; no community line when no community data. We accept a `community_counts: dict[str, int]` side-input for the community line; first iteration passes `{}` and leaves the line off (community scraping hookup is a follow-up task if needed).

- [ ] **Step 1: Write failing tests**

Append to `tests/shopping/test_pipeline_v2.py`:

```python
def test_format_group_card_full_output():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, ReviewSynthesis, format_group_card,
    )
    cands = [
        Candidate(title="Siemens EQ.6 Plus S100", site="hepsiburada",
                  site_rank=1, price=24745.0, original_price=None,
                  url="https://h.com/1", rating=4.5, review_count=312,
                  review_snippets=[]),
        Candidate(title="Siemens EQ.6 Plus S100", site="akakce",
                  site_rank=1, price=25499.0, original_price=None,
                  url="https://a.com/2", rating=None, review_count=None,
                  review_snippets=[]),
    ]
    group = ProductGroup("Siemens EQ.6 Plus S100", [0, 1], False, prominence=2.0)
    syn = ReviewSynthesis(
        praise=["Köpük kalitesi iyi", "Sessiz çalışıyor"],
        complaints=["Fiyat yüksek"],
        red_flags=["Şikayetvar'da 47 şikayet"],
        insufficient_data=False,
    )
    card = format_group_card(group, syn, cands, community_counts={})
    assert "Siemens EQ.6 Plus S100" in card
    assert "⭐ 4.5" in card and "312" in card
    assert "👍" in card and "Köpük kalitesi iyi" in card
    assert "👎" in card and "Fiyat yüksek" in card
    assert "⚠️" in card and "Şikayetvar" in card
    assert "Hepsiburada" in card.title() or "hepsiburada" in card
    assert "24.745" in card or "24745" in card
    assert "Yeterli inceleme verisi yok" not in card


def test_format_group_card_insufficient_data_shows_footer():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, ReviewSynthesis, format_group_card,
    )
    cands = [
        Candidate(title="X", site="trendyol", site_rank=1, price=100.0,
                  original_price=None, url="u", rating=None, review_count=None,
                  review_snippets=[]),
    ]
    group = ProductGroup("X", [0], False, 1.0)
    syn = ReviewSynthesis([], [], [], insufficient_data=True)
    card = format_group_card(group, syn, cands, community_counts={})
    assert "Yeterli inceleme verisi yok" in card
    assert "👍" not in card
    assert "👎" not in card


def test_format_group_card_community_line_when_present():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, ReviewSynthesis, format_group_card,
    )
    cands = [Candidate(title="X", site="t", site_rank=1, price=1,
                       original_price=None, url="u", rating=None,
                       review_count=None, review_snippets=[])]
    group = ProductGroup("X", [0], False, 1.0)
    syn = ReviewSynthesis([], [], [], insufficient_data=True)
    card = format_group_card(group, syn, cands,
                             community_counts={"Technopat": 12, "Ekşi": 8})
    assert "💬" in card and "Technopat (12" in card and "Ekşi (8" in card


def test_format_response_joins_cards():
    from src.workflows.shopping.pipeline_v2 import format_response
    out = format_response(["CARD1", "CARD2"])
    assert "CARD1" in out and "CARD2" in out
    assert out.index("CARD1") < out.index("CARD2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v -k format_`
Expected: `ImportError: cannot import name 'format_group_card'`

- [ ] **Step 3: Implement formatters**

Append to `src/workflows/shopping/pipeline_v2.py`:

```python
def _fmt_price_tr(value: float | None) -> str:
    if value is None:
        return "—"
    # Turkish format: thousands dot, decimal comma
    s = f"{value:,.0f}"
    return s.replace(",", ".")


def _site_label(site: str) -> str:
    mapping = {
        "trendyol": "Trendyol", "hepsiburada": "Hepsiburada",
        "amazon_tr": "Amazon.tr", "akakce": "Akakçe", "n11": "n11",
        "gittigidiyor": "GittiGidiyor", "epey": "Epey",
        "teknosa": "Teknosa", "vatan": "Vatan",
    }
    return mapping.get(site, site.title())


def format_group_card(
    group: ProductGroup,
    synthesis: ReviewSynthesis,
    candidates: list[Candidate],
    community_counts: dict[str, int] | None = None,
) -> str:
    """Reviews-first Telegram markdown for one product group."""
    members = [candidates[i] for i in group.member_indices if 0 <= i < len(candidates)]

    # Pick best rating / review_count across members for the headline
    rating = next((m.rating for m in members if m.rating is not None), None)
    review_count = next((m.review_count for m in members if m.review_count), None)

    lines: list[str] = []
    head = f"*{group.representative_title}*"
    if rating is not None:
        rc = f" ({review_count} değerlendirme)" if review_count else ""
        head += f" ⭐ {rating:.1f}/5{rc}"
    lines.append(head)
    lines.append("")

    if not synthesis.insufficient_data:
        if synthesis.praise:
            lines.append("👍 Kullanıcılar beğeniyor:")
            lines.extend(f"• {p}" for p in synthesis.praise)
            lines.append("")
        if synthesis.complaints:
            lines.append("👎 Şikayetler:")
            lines.extend(f"• {c}" for c in synthesis.complaints)
            lines.append("")
        if synthesis.red_flags:
            lines.append("⚠️ Dikkat:")
            lines.extend(f"• {r}" for r in synthesis.red_flags)
            lines.append("")

    # Price block — one line per site, deduped, sorted by price asc
    seen_sites: set[str] = set()
    price_rows: list[tuple[str, float | None, str]] = []  # (site_label, price, url)
    for m in members:
        if m.site in seen_sites:
            continue
        seen_sites.add(m.site)
        price_rows.append((_site_label(m.site), m.price, m.url))
    price_rows.sort(key=lambda r: (r[1] is None, r[1] or 0))
    if price_rows:
        lines.append("💰 *Fiyatlar:*")
        for label, price, url in price_rows:
            if price is None:
                lines.append(f"• {label} — stokta yok")
            else:
                lines.append(f"• {label} — {_fmt_price_tr(price)} TL")
        lines.append("")

    community_counts = community_counts or {}
    if community_counts:
        bits = ", ".join(f"{k} ({v} konu)" for k, v in community_counts.items())
        lines.append(f"💬 Topluluk: {bits}")
        lines.append("")

    if synthesis.insufficient_data:
        lines.append("⚠️ Yeterli inceleme verisi yok")

    return "\n".join(lines).rstrip() + "\n"


def format_response(cards: list[str]) -> str:
    """Join per-group cards with a blank-line separator."""
    return "\n".join(c.rstrip() for c in cards if c).strip() + "\n"
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v`
Expected: `16 passed`

- [ ] **Step 5: Commit**

```bash
git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
git commit -m "feat(shopping_v2): format_group_card + format_response"
```

---

## Task 8: ShoppingPipelineV2 dispatch class with step handlers

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py`
- Test: `tests/shopping/test_pipeline_v2.py`

The pipeline dispatch class mirrors v1 `ShoppingPipeline.run(task)` in `src/workflows/shopping/pipeline.py:837-929`. It reads `step_name` from the task context and routes to one of: `resolve_candidates`, `group_and_synthesize`, `format_response`. Also provides `_step_clarify` for the category workflow — we reuse v1's clarify logic via import (it's an LLM call through the dispatcher, stable).

- [ ] **Step 1: Write failing test for full end-to-end (all LLMs patched)**

Append to `tests/shopping/test_pipeline_v2.py`:

```python
@pytest.mark.asyncio
async def test_pipeline_v2_full_run_siemens_style():
    """End-to-end with patched scraper + patched LLMs.

    Emulates the 2026-04-20 Siemens S100 bug: ensures the accessory is dropped
    and the output contains the real machine, not the brewing unit part.
    """
    from types import SimpleNamespace
    from src.workflows.shopping.pipeline_v2 import (
        step_resolve, step_group, step_synthesize_reviews,
        select_groups, format_group_card, format_response,
    )

    fake_products = [
        SimpleNamespace(
            name="Siemens EQ.6 Plus S100", site="hepsiburada",
            url="https://h.com/1", price=24745.0, original_price=None,
            rating=4.5, review_count=312,
            review_snippets=["köpük iyi", "sessiz", "fiyat yüksek"],
        ),
        SimpleNamespace(
            name="Siemens EQ.6 Plus S100", site="akakce",
            url="https://a.com/2", price=25499.0, original_price=None,
            rating=None, review_count=None, review_snippets=[],
        ),
        SimpleNamespace(
            name="DL-Pro demleme ünitesi Siemens EQ.3 S100",
            site="amazon_tr", url="https://am.tr/3", price=4800.0,
            original_price=None, rating=5.0, review_count=3,
            review_snippets=[],
        ),
    ]

    grouping_resp = {
        "content": (
            '{"groups": ['
            '  {"representative_title": "Siemens EQ.6 Plus S100", '
            '   "member_indices": [0, 1], "is_accessory_or_part": false},'
            '  {"representative_title": "Siemens EQ.3 brewing unit", '
            '   "member_indices": [2], "is_accessory_or_part": true}'
            ']}'
        ),
        "model": "m", "cost": 0.0,
    }
    synth_resp = {
        "content": (
            '{"praise":["köpük iyi","sessiz"],'
            ' "complaints":["fiyat yüksek"],'
            ' "red_flags":[], "insufficient_data": false}'
        ),
        "model": "m", "cost": 0.0,
    }

    with patch(
        "src.workflows.shopping.pipeline_v2._fetch_products",
        new=AsyncMock(return_value=fake_products),
    ), patch(
        "src.workflows.shopping.pipeline_v2._grouping_llm_call",
        new=AsyncMock(return_value=grouping_resp),
    ), patch(
        "src.workflows.shopping.pipeline_v2._synthesis_llm_call",
        new=AsyncMock(return_value=synth_resp),
    ):
        cands = await step_resolve("Siemens s100", per_site_n=3)
        groups = await step_group(cands)
        kept = select_groups(groups, max_groups=2)
        cards: list[str] = []
        for g in kept:
            syn = await step_synthesize_reviews(g, cands)
            cards.append(format_group_card(g, syn, cands, community_counts={}))
        response = format_response(cards)

    # Accessory is dropped
    assert "demleme ünitesi" not in response
    assert "EQ.3" not in response
    # Real machine is present
    assert "EQ.6" in response
    assert "24.745" in response
    assert "köpük" in response
    assert "fiyat yüksek" in response
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py::test_pipeline_v2_full_run_siemens_style -v`
Expected: passes IF Tasks 1-7 complete. (This is a regression-style assertion; no new code required to make it pass.) If it fails, fix the relevant step from the task that introduced the bug.

- [ ] **Step 3: Write failing test for ShoppingPipelineV2 dispatch class**

Append to `tests/shopping/test_pipeline_v2.py`:

```python
@pytest.mark.asyncio
async def test_shopping_pipeline_v2_unknown_step_returns_failed():
    from src.workflows.shopping.pipeline_v2 import ShoppingPipelineV2
    task = {
        "id": 1,
        "title": "[0.1] nonsense_step",
        "context": {"step_name": "nonsense_step"},
    }
    result = await ShoppingPipelineV2().run(task)
    assert result["status"] == "failed"
    assert "nonsense_step" in result["result"].lower() or "unknown" in result["result"].lower()


@pytest.mark.asyncio
async def test_shopping_pipeline_v2_resolve_candidates_step():
    from types import SimpleNamespace
    from src.workflows.shopping.pipeline_v2 import ShoppingPipelineV2

    fake = [SimpleNamespace(name="X", site="trendyol", url="u", price=100,
                             original_price=None, rating=None, review_count=None,
                             review_snippets=[])]
    task = {
        "id": 2,
        "title": "[0.1] resolve_candidates",
        "context": {
            "step_name": "resolve_candidates",
            "input_artifacts": [],
            "per_site_n": 3,
        },
        "description": "X",
    }
    with patch(
        "src.workflows.shopping.pipeline_v2._fetch_products",
        new=AsyncMock(return_value=fake),
    ):
        result = await ShoppingPipelineV2().run(task)
    assert result["status"] == "completed"
    payload = json.loads(result["result"])
    assert payload["candidates"][0]["title"] == "X"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v -k shopping_pipeline_v2`
Expected: `ImportError: cannot import name 'ShoppingPipelineV2'`

- [ ] **Step 5: Implement dispatch class + step handlers**

Append to `src/workflows/shopping/pipeline_v2.py`:

```python
# ── Workflow step handlers (task-shaped I/O) ────────────────────────────────

def _parse_context(task: dict) -> dict:
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            return json.loads(ctx)
        except (json.JSONDecodeError, ValueError):
            return {}
    return ctx or {}


def _candidates_to_json(cands: list[Candidate]) -> list[dict]:
    return [
        {
            "title": c.title, "site": c.site, "site_rank": c.site_rank,
            "price": c.price, "original_price": c.original_price,
            "url": c.url, "rating": c.rating, "review_count": c.review_count,
            "review_snippets": c.review_snippets,
        }
        for c in cands
    ]


def _candidates_from_json(items: list[dict]) -> list[Candidate]:
    return [
        Candidate(
            title=i.get("title", ""), site=i.get("site", ""),
            site_rank=int(i.get("site_rank", 1)),
            price=i.get("price"), original_price=i.get("original_price"),
            url=i.get("url", ""), rating=i.get("rating"),
            review_count=i.get("review_count"),
            review_snippets=list(i.get("review_snippets") or []),
        )
        for i in items
    ]


async def _read_artifacts(mission_id: int, keys: list[str]) -> dict:
    """Reuse v1's artifact reader — same table, same semantics."""
    from src.workflows.shopping.pipeline import _read_artifacts as _v1_read
    return await _v1_read(mission_id, keys)


async def _handler_resolve_candidates(task: dict, artifacts: dict, ctx: dict) -> dict:
    # user_query may be a raw string OR a JSON dict with clarified_query
    query = ""
    for key in ("clarified_query", "user_query"):
        raw = artifacts.get(key, "")
        if not raw:
            continue
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                parsed = json.loads(raw)
                query = parsed.get("clarified_query") or parsed.get("query") or parsed.get("user_query", "")
                if query:
                    break
            except (json.JSONDecodeError, ValueError):
                pass
        if isinstance(raw, str) and raw.strip():
            query = raw.strip()
            break
    if not query:
        query = task.get("description", "")
    per_site_n = int(ctx.get("per_site_n", 3))
    cands = await step_resolve(query, per_site_n=per_site_n)
    return {
        "query": query,
        "candidates": _candidates_to_json(cands),
        "escalation_needed": len(cands) == 0,
    }


async def _handler_group_and_synthesize(task: dict, artifacts: dict, ctx: dict) -> dict:
    payload_raw = artifacts.get("search_results", "{}")
    payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
    cands = _candidates_from_json(payload.get("candidates", []))
    if not cands:
        return {"cards": [], "escalation_needed": True}
    groups = await step_group(cands)
    max_groups = int(ctx.get("max_groups", 2))
    kept = select_groups(groups, max_groups=max_groups)
    community_counts = payload.get("community_counts") or {}
    cards: list[str] = []
    for g in kept:
        syn = await step_synthesize_reviews(g, cands)
        cards.append(format_group_card(g, syn, cands, community_counts=community_counts))
    return {"cards": cards, "escalation_needed": False}


async def _handler_format_response(task: dict, artifacts: dict, ctx: dict) -> dict:
    raw = artifacts.get("grouped_synth", "{}")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    cards = payload.get("cards", [])
    if not cards:
        return {"formatted_text": "🔍 Sonuç bulunamadı.", "escalation": True}
    return {"formatted_text": format_response(cards), "escalation": False}


def _alias_v1(step_name: str):
    """Delegate to v1's clarify-related handlers. Late-bound to avoid import cycles."""
    async def _run(task: dict, artifacts: dict, ctx: dict):
        from src.workflows.shopping import pipeline as _v1
        handler = _v1._STEP_HANDLERS.get(step_name)
        if not handler:
            raise RuntimeError(f"v1 handler not found: {step_name}")
        return await handler(task, artifacts)
    return _run


_STEP_HANDLERS_V2 = {
    "resolve_candidates": _handler_resolve_candidates,
    "group_and_synthesize": _handler_group_and_synthesize,
    "format_response": _handler_format_response,
    # Clarifier steps reuse v1's handlers — same clarify LLM, no duplication
    "analyze_query": _alias_v1("analyze_query"),
    "clarify_if_vague": _alias_v1("clarify_if_vague"),
}


class ShoppingPipelineV2:
    """Dispatch class — same contract as v1 ShoppingPipeline."""

    async def run(self, task: dict) -> dict:
        ctx = _parse_context(task)
        step_name = ctx.get("step_name", "")
        if not step_name:
            title = task.get("title", "")
            if "] " in title:
                step_name = title.split("] ", 1)[1]
        if not step_name:
            step_name = ctx.get("workflow_step_id", "")

        logger.info("pipeline_v2 dispatch", step_name=step_name, task_id=task.get("id"))

        handler = _STEP_HANDLERS_V2.get(step_name)
        if not handler:
            return {
                "status": "failed",
                "result": f"Unknown step: {step_name!r}",
                "model": "shopping_pipeline_v2",
                "cost": 0.0,
                "iterations": 1,
            }

        mission_id = task.get("mission_id")
        input_artifacts = ctx.get("input_artifacts", [])
        artifacts = (
            await _read_artifacts(mission_id, input_artifacts)
            if mission_id
            else {}
        )

        try:
            result = await handler(task, artifacts, ctx)
            if isinstance(result, dict) and result.get("_needs_clarification"):
                return {
                    "status": "needs_clarification",
                    "clarification": result.get("clarification", "More info needed"),
                    "result": json.dumps(result, ensure_ascii=False, default=str),
                    "model": "shopping_pipeline_v2",
                    "cost": 0.0,
                    "iterations": 1,
                }
            return {
                "status": "completed",
                "result": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str),
                "model": "shopping_pipeline_v2",
                "cost": 0.0,
                "iterations": 1,
            }
        except Exception as exc:
            logger.exception("pipeline_v2 step %r failed", step_name)
            return {
                "status": "failed",
                "result": f"Pipeline v2 error: {exc}",
                "model": "shopping_pipeline_v2",
                "cost": 0.0,
                "iterations": 1,
            }
```

- [ ] **Step 6: Run tests to verify all pass**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v`
Expected: all tests pass (count = 19).

- [ ] **Step 7: Commit**

```bash
git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_pipeline_v2.py
git commit -m "feat(shopping_v2): ShoppingPipelineV2 dispatch + step handlers"
```

---

## Task 9: Wire shopping_pipeline_v2 into orchestrator

**Files:**
- Modify: `src/core/orchestrator.py:25-32` and `:82-84`

- [ ] **Step 1: Read current orchestrator registration**

Already confirmed at lines 25-32 (`AGENT_TIMEOUTS` dict) and 82-84 (agent-type branch).

- [ ] **Step 2: Add shopping_pipeline_v2 timeout**

Edit `src/core/orchestrator.py`, inside the `AGENT_TIMEOUTS` dict near line 31. Change:

```python
    "shopping_pipeline": 60, "shopping_clarifier": 120,
```

to:

```python
    "shopping_pipeline": 60, "shopping_clarifier": 120,
    "shopping_pipeline_v2": 120,
```

- [ ] **Step 3: Add dispatch branch for shopping_pipeline_v2**

Edit `src/core/orchestrator.py`, just after lines 82-84 (the existing `shopping_pipeline` branch). Change:

```python
            if agent_type == "shopping_pipeline":
                from src.workflows.shopping.pipeline import ShoppingPipeline
                return await ShoppingPipeline().run(task)
```

to:

```python
            if agent_type == "shopping_pipeline":
                from src.workflows.shopping.pipeline import ShoppingPipeline
                return await ShoppingPipeline().run(task)
            if agent_type == "shopping_pipeline_v2":
                from src.workflows.shopping.pipeline_v2 import ShoppingPipelineV2
                return await ShoppingPipelineV2().run(task)
```

- [ ] **Step 4: Verify orchestrator imports cleanly**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -c "from src.core import orchestrator; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py
git commit -m "feat(shopping_v2): register shopping_pipeline_v2 agent in orchestrator"
```

---

## Task 10: Write the three v2 workflow JSONs

**Files:**
- Create: `src/workflows/shopping/product_research_v2.json`
- Create: `src/workflows/shopping/quick_search_v2.json`
- Create: `src/workflows/shopping/shopping_v2.json`

- [ ] **Step 1: Create `quick_search_v2.json`**

```json
{
  "plan_id": "quick_search_v2",
  "version": "2.0",
  "metadata": {
    "description": "v2 quick shopping search. Trusts each site's native ordering (top-3 per site), groups cross-site results with a cheap LLM call, synthesises reviews per group, renders reviews-first Telegram cards. Escalates to shopping_v2 if no products.",
    "agents_required": ["shopping_pipeline_v2"],
    "performance_target": {"max_duration_seconds": 60, "escalation_trigger": "insufficient_results"},
    "escalation_target": "shopping_v2",
    "timeout_hours": 1
  },
  "phases": [
    {"id": "phase_0", "name": "Resolve candidates", "goal": "Fetch top-N per site.", "depends_on_phases": []},
    {"id": "phase_1", "name": "Group & synthesise", "goal": "LLM group + review synthesis.", "depends_on_phases": ["phase_0"]},
    {"id": "phase_2", "name": "Format & deliver", "goal": "Render Telegram cards.", "depends_on_phases": ["phase_1"]}
  ],
  "steps": [
    {
      "id": "0.1", "phase": "phase_0", "name": "resolve_candidates",
      "title": "Resolve Candidates", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": [],
      "input_artifacts": ["user_query"],
      "output_artifacts": ["search_results"],
      "instruction": "Step name: resolve_candidates. Calls the shopping_search fleet, keeps top-3 per site, no filtering. Stores candidates in search_results.",
      "done_when": "search_results.candidates is populated, or escalation_needed=true.",
      "context": {"per_site_n": 3}
    },
    {
      "id": "1.1", "phase": "phase_1", "name": "group_and_synthesize",
      "title": "Group & Synthesize Reviews", "agent": "shopping_pipeline_v2",
      "difficulty": "medium", "tools_hint": [], "depends_on": ["0.1"],
      "input_artifacts": ["search_results"],
      "output_artifacts": ["grouped_synth"],
      "instruction": "Step name: group_and_synthesize. LLM groups candidates across sites, drops accessories, synthesises reviews per kept group (max 2).",
      "done_when": "grouped_synth.cards contains one Telegram card per kept group.",
      "context": {"max_groups": 2}
    },
    {
      "id": "2.1", "phase": "phase_2", "name": "format_response",
      "title": "Format & Deliver Response", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": ["1.1"],
      "input_artifacts": ["grouped_synth"],
      "output_artifacts": ["shopping_response"],
      "instruction": "Step name: format_response. Joins grouped_synth.cards into a single Telegram message.",
      "done_when": "shopping_response.formatted_text is ready or escalation=true."
    }
  ]
}
```

- [ ] **Step 2: Create `product_research_v2.json`**

Identical to `quick_search_v2.json` except:
- `plan_id`: `"product_research_v2"`
- `metadata.description`: "v2 specific-product research. Same pipeline as quick_search_v2 but with deeper per-site N (4) for more review material."
- Remove `escalation_target`.
- `0.1` step `context.per_site_n`: `4`
- `1.1` step `context.max_groups`: `2`

```json
{
  "plan_id": "product_research_v2",
  "version": "2.0",
  "metadata": {
    "description": "v2 specific-product research. Trusts site ordering (top-4 per site), LLM grouping, review synthesis, reviews-first Telegram output.",
    "agents_required": ["shopping_pipeline_v2"],
    "performance_target": {"max_duration_seconds": 120, "escalation_trigger": "no_products_found"},
    "timeout_hours": 1
  },
  "phases": [
    {"id": "phase_0", "name": "Resolve candidates", "goal": "Fetch top-N per site.", "depends_on_phases": []},
    {"id": "phase_1", "name": "Group & synthesise", "goal": "LLM group + review synthesis.", "depends_on_phases": ["phase_0"]},
    {"id": "phase_2", "name": "Format & deliver", "goal": "Render Telegram cards.", "depends_on_phases": ["phase_1"]}
  ],
  "steps": [
    {
      "id": "0.1", "phase": "phase_0", "name": "resolve_candidates",
      "title": "Resolve Candidates", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": [],
      "input_artifacts": ["user_query"],
      "output_artifacts": ["search_results"],
      "instruction": "Step name: resolve_candidates. Fetches top-4 per site.",
      "done_when": "search_results.candidates populated.",
      "context": {"per_site_n": 4}
    },
    {
      "id": "1.1", "phase": "phase_1", "name": "group_and_synthesize",
      "title": "Group & Synthesize Reviews", "agent": "shopping_pipeline_v2",
      "difficulty": "medium", "tools_hint": [], "depends_on": ["0.1"],
      "input_artifacts": ["search_results"],
      "output_artifacts": ["grouped_synth"],
      "instruction": "Step name: group_and_synthesize. Keeps up to 2 non-accessory groups.",
      "done_when": "grouped_synth.cards populated.",
      "context": {"max_groups": 2}
    },
    {
      "id": "2.1", "phase": "phase_2", "name": "format_response",
      "title": "Format & Deliver Response", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": ["1.1"],
      "input_artifacts": ["grouped_synth"],
      "output_artifacts": ["shopping_response"],
      "instruction": "Step name: format_response.",
      "done_when": "shopping_response.formatted_text populated."
    }
  ]
}
```

- [ ] **Step 3: Create `shopping_v2.json` (category flow with clarifier)**

```json
{
  "plan_id": "shopping_v2",
  "version": "2.0",
  "metadata": {
    "description": "v2 category/discovery shopping. Clarifies vague queries, fetches top-8 per site, LLM groups into product tiers, synthesises reviews for top 3 tiers, renders reviews-first cards.",
    "agents_required": ["shopping_pipeline_v2"],
    "performance_target": {"max_duration_seconds": 180, "escalation_trigger": "no_products_found"},
    "timeout_hours": 2
  },
  "phases": [
    {"id": "phase_0", "name": "Understand query", "goal": "Parse intent, clarify if vague.", "depends_on_phases": []},
    {"id": "phase_1", "name": "Resolve candidates", "goal": "Fetch top-N per site.", "depends_on_phases": ["phase_0"]},
    {"id": "phase_2", "name": "Group & synthesise", "goal": "LLM group + review synthesis.", "depends_on_phases": ["phase_1"]},
    {"id": "phase_3", "name": "Format & deliver", "goal": "Render cards.", "depends_on_phases": ["phase_2"]}
  ],
  "steps": [
    {
      "id": "0.1", "phase": "phase_0", "name": "analyze_query",
      "title": "Analyze Query", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": [],
      "input_artifacts": ["user_query"],
      "output_artifacts": ["parsed_intent"],
      "instruction": "Step name: analyze_query. Delegates to v1 clarify analyser — parses constraints and flags vagueness.",
      "done_when": "parsed_intent artifact is populated."
    },
    {
      "id": "0.2", "phase": "phase_0", "name": "clarify_if_vague",
      "title": "Clarify If Vague", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": ["0.1"],
      "input_artifacts": ["parsed_intent", "user_query"],
      "output_artifacts": ["clarified_query"],
      "instruction": "Step name: clarify_if_vague. Delegates to v1 clarify — asks user a question if the query is too vague, else passes through.",
      "done_when": "clarified_query artifact contains either the original query or a refined version."
    },
    {
      "id": "1.1", "phase": "phase_1", "name": "resolve_candidates",
      "title": "Resolve Candidates", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": ["0.2"],
      "input_artifacts": ["clarified_query"],
      "output_artifacts": ["search_results"],
      "instruction": "Step name: resolve_candidates. Top-8 per site for wider category sweep.",
      "done_when": "search_results.candidates populated.",
      "context": {"per_site_n": 8}
    },
    {
      "id": "2.1", "phase": "phase_2", "name": "group_and_synthesize",
      "title": "Group & Synthesize Reviews", "agent": "shopping_pipeline_v2",
      "difficulty": "medium", "tools_hint": [], "depends_on": ["1.1"],
      "input_artifacts": ["search_results"],
      "output_artifacts": ["grouped_synth"],
      "instruction": "Step name: group_and_synthesize. Keeps up to 3 non-accessory groups for category tiers.",
      "done_when": "grouped_synth.cards populated.",
      "context": {"max_groups": 3}
    },
    {
      "id": "3.1", "phase": "phase_3", "name": "format_response",
      "title": "Format & Deliver Response", "agent": "shopping_pipeline_v2",
      "difficulty": "easy", "tools_hint": [], "depends_on": ["2.1"],
      "input_artifacts": ["grouped_synth"],
      "output_artifacts": ["shopping_response"],
      "instruction": "Step name: format_response.",
      "done_when": "shopping_response.formatted_text populated."
    }
  ]
}
```

- [ ] **Step 4: Verify workflows load**

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
.venv/Scripts/python.exe -c "from src.workflows.engine.loader import load_workflow; \
    [print(load_workflow(w).metadata['description'][:60]) \
     for w in ('quick_search_v2','product_research_v2','shopping_v2')]"
```
Expected: three description snippets printed without exceptions.

- [ ] **Step 5: Commit**

```bash
git add src/workflows/shopping/quick_search_v2.json \
        src/workflows/shopping/product_research_v2.json \
        src/workflows/shopping/shopping_v2.json
git commit -m "feat(shopping_v2): workflow JSONs for quick_search_v2 / product_research_v2 / shopping_v2"
```

---

## Task 11: Cutover wf_map in telegram_bot

**Files:**
- Modify: `src/app/telegram_bot.py:4503-4511`

- [ ] **Step 1: Update wf_map**

Find the `wf_map` dict inside `_create_shopping_mission` at line 4503. Change:

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
        workflow_name = wf_map.get(sub_intent or "shopping", "shopping")
```

to:

```python
        wf_map = {
            "deep_research": "shopping_v2",
            "research": "shopping_v2",
            "compare": "combo_research",
            "gift": "gift_recommendation",
            "deals": "exploration",
            "quick_search": "quick_search_v2",
            "product_research": "product_research_v2",
        }
        workflow_name = wf_map.get(sub_intent or "shopping", "shopping_v2")
```

- [ ] **Step 2: Verify import**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -c "from src.app import telegram_bot; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/app/telegram_bot.py
git commit -m "feat(shopping_v2): point wf_map at v2 workflows (cutover)"
```

---

## Task 12: Full test sweep and manual validation checklist

**Files:**
- Test: `tests/shopping/test_pipeline_v2.py` (full run)

- [ ] **Step 1: Run all shopping_v2 tests**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && .venv/Scripts/python.exe -m pytest tests/shopping/test_pipeline_v2.py -v`
Expected: all tests pass. Investigate any failure before proceeding.

- [ ] **Step 2: Run broader regression — v1 tests still pass**

Run: `cd C:/Users/sakir/Dropbox/Workspaces/kutay && timeout 120 .venv/Scripts/python.exe -m pytest tests/shopping/ -v`
Expected: both v1 and v2 test files green. v1 tests must not regress (v1 pipeline still exists, untouched).

- [ ] **Step 3: Manual validation — run KutAI and test real queries**

1. Via Telegram on the dev bot:
   - `/shop Siemens EQ6 S100` — confirm accessory is not the winner, reviews section populated when scrapers return snippets
   - `🔬 Detaylı Araştır` → `🎯 Belirli ürün` → "Philips 3200 LatteGo" — confirm v2 path runs, reviews-first card
   - `🔬 Detaylı Araştır` → `🏷 Kategori` → "kahve makinesi 5000 TL altı" — confirm clarifier still runs, category output shows up to 3 cards
2. Log the `mission_id` and `task_id` from each run; check `model_pick_log` for two LLM calls per run (grouping + synthesis).
3. Record observations in `docs/research/2026-04-21-shopping-v2-validation.md` (create if needed) — good, bad, regressions, follow-ups.

No code changes in this step — the deliverable is the validation doc.

- [ ] **Step 4: Commit the validation doc**

```bash
git add docs/research/2026-04-21-shopping-v2-validation.md
git commit -m "docs(shopping_v2): manual validation notes"
```

---

## Out of Scope (follow-up PRs)

1. **Delete v1 code** after ≥1 week of v2 running clean: `src/workflows/shopping/pipeline.py`, `product_matcher.py`, v1 JSONs, dead helpers.
2. **Deep scrape review text** from product detail pages — separate project, unblocks richer synthesis.
3. **Community counts wiring** — the `community_counts` field is currently passed empty from handlers; adding a lightweight scraper query for Technopat/Ekşi thread counts belongs in a follow-up.
4. Migrate `combo_research.json`, `gift_recommendation.json`, `exploration.json`, `price_watch.json` to v2 primitives once the main three are proven.
