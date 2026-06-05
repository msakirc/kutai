# Shopping Workflow-Step Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `src/workflows/shopping/` so every LLM call is an admitted `agent:<type>` workflow step (or CPS chain), retiring the `request()`/`await_inline` shim and unblocking CPS SP5.

**Architecture:** Each inline LLM call splits into a `prep (deterministic) → agent:<shopping_producer> (LLM) → apply (deterministic)` triad of workflow steps, reusing the existing agent-step + artifact-passing machinery (no engine feature added). Compare-all's ≤5-line sequential synth uses a CPS continuation chain (`enqueue(on_complete=…)`). `shopping_clarifier` drops `single_shot` → react, killing the last non-pipeline `request()` caller.

**Tech Stack:** Python 3.10, async, pytest (120s timeout), General Beckman (admission/CPS), coulson (react worker), workflow engine (`src/workflows/engine/`), SQLite.

**Spec:** `docs/superpowers/specs/2026-06-05-shopping-workflow-step-migration-design.md` (rev2).

**Standing rules:** `timeout 120 pytest` always; never concurrent pytest; `tests/` and `packages/*/tests/` in SEPARATE invocations (colliding conftest). Live-test via Telegram after each path migrates. Commit per task. Push to `main` directly.

---

## File Structure

**Create:**
- `src/agents/shopping_grouper.py` — grouping producer agent (prompt-only, react one-pass)
- `src/agents/shopping_labeler.py` — labeling producer agent
- `src/agents/shopping_synthesizer.py` — review-synthesis producer agent
- `src/workflows/shopping/shopping_v3.json` — new plan (`plan_id: shopping_v3`)
- `src/workflows/shopping/compare_continuations.py` — CPS chain for sequential compare-all
- `tests/agents/test_shopping_producers.py` — prompt-quality + golden-shape tests
- `tests/shopping/test_v3_handlers.py` — prep/apply handler units
- `tests/shopping/test_v3_wiring.py` — agent-step → artifact wiring + deadlock regression

**Modify:**
- `src/workflows/shopping/pipeline_v2.py` — add prep/apply handlers; register in `_STEP_HANDLERS_V2`
- `src/workflows/shopping/labels.py` — extract label parse into an apply-callable
- `src/agents/shopping_clarifier.py:21` — drop `execution_pattern = "single_shot"`
- `src/agents/__init__.py` (or the agent registry) — register 3 new producers
- `src/agents/classifier.py` (or classifier profile) — cover 3 new producers
- `src/app/telegram_bot.py:8490-8498` — flip launch to `shopping_v3`; `:10960-11005` route compare-all through workflow resume
- `src/workflows/shopping/quick_search_v2.json:8` — `escalation_target` → `shopping_v3`
- `general_beckman/continuations._HANDLER_MODULES` — add `compare_continuations`
- `src/core/llm_dispatcher.py` — delete `request()` + `_request_kwargs_to_spec` (+ `_task_result_to_request_response` after re-grep)
- `packages/coulson/src/coulson/single_shot.py` — delete (after clarifier flips)
- `packages/coulson/src/coulson/reflection.py:89`, `src/workflows/engine/constrained_emit.py:147` — delete dead `request()` callers

---

## Task 1: `shopping_grouper` producer agent

**Files:**
- Create: `src/agents/shopping_grouper.py`
- Test: `tests/agents/test_shopping_producers.py`

The prompt is lifted from `prompts_v2.py::GROUPING_PROMPT`, reshaped to pass the 3 invariants (`tests/agents/test_prompt_quality.py`): first line `You are …`; body has must/always + don't/never; body has `final_answer` + fenced ` ```json `. The candidate data (the `{candidates_json}` view) arrives as the step's input artifact (the agent's context), so the system prompt holds only the rules + schema.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_shopping_producers.py
import json
import pytest
from src.agents.shopping_grouper import ShoppingGrouperAgent

def test_grouper_prompt_invariants():
    p = ShoppingGrouperAgent().get_system_prompt({})
    assert p.lstrip().startswith("You are ")
    low = p.lower()
    assert ("must" in low or "always" in low) and ("don't" in low or "never" in low)
    assert "final_answer" in low
    assert "```json" in p

def test_grouper_is_prompt_only_react():
    a = ShoppingGrouperAgent()
    assert a.name == "shopping_grouper"
    assert a.allowed_tools == []
    # defaults to react_loop (no single_shot)
    assert getattr(a, "execution_pattern", "react_loop") == "react_loop"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 120 pytest tests/agents/test_shopping_producers.py -v`
Expected: FAIL — `ModuleNotFoundError: src.agents.shopping_grouper`

- [ ] **Step 3: Write the agent**

```python
# src/agents/shopping_grouper.py
"""Shopping grouping producer — clusters search candidates into product groups.
Prompt-only react agent; the candidate JSON arrives as the step's input artifact."""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.shopping_grouper")


class ShoppingGrouperAgent(BaseAgent):
    name = "shopping_grouper"
    description = "Clusters shopping search results into same-product groups"
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 1
    allowed_tools: list[str] = []

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a shopping search-result grouping engine. The user message "
            "contains a JSON array of candidate listings, each with an integer "
            "`index`.\n\n"
            "You MUST cluster candidates that refer to the SAME product (same brand "
            "+ model + variant). Different colours or storage tiers of the same model "
            "are the same group. Different models from the same product line are "
            "DIFFERENT groups (e.g. Siemens EQ.3 vs EQ.6). You MUST flag accessories, "
            "replacement parts, filters, covers, or spare components as "
            "`is_accessory_or_part: true` (a full machine is NOT a part; a brewing "
            "unit sold separately IS). Always pick a clean `representative_title` "
            "(shortest member title is usually best).\n\n"
            "Do NOT invent candidates or indices that are not in the input. Never emit "
            "prose or markdown fences. Output ONLY valid JSON.\n\n"
            "Return your final_answer as JSON in this exact shape:\n"
            "```json\n"
            '{"groups": [{"representative_title": "string", '
            '"member_indices": [0], "is_accessory_or_part": false}]}\n'
            "```"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 120 pytest tests/agents/test_shopping_producers.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Register + classifier coverage**

Add `ShoppingGrouperAgent` to the agent registry (mirror how `ShoppingClarifierAgent` is registered — grep `shopping_clarifier` in `src/agents/__init__.py` and the classifier, add a sibling entry). Run the prompt-quality + classifier-coverage suites:

Run: `timeout 120 pytest tests/agents/test_prompt_quality.py -v`
Expected: PASS (new agent satisfies invariants)

- [ ] **Step 6: Commit**

```bash
git add src/agents/shopping_grouper.py src/agents/__init__.py tests/agents/test_shopping_producers.py
git commit -m "feat(shopping): shopping_grouper producer agent"
```

---

## Task 2: `shopping_labeler` producer agent

**Files:**
- Create: `src/agents/shopping_labeler.py`
- Test: `tests/agents/test_shopping_producers.py` (append)

Prompt lifted from `LABEL_PROMPT` (the taxonomy rules + `line_id`/`base_model`/`variant` schema). The group `view` + query arrive as the input artifact.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/agents/test_shopping_producers.py (append)
from src.agents.shopping_labeler import ShoppingLabelerAgent

def test_labeler_prompt_invariants():
    p = ShoppingLabelerAgent().get_system_prompt({})
    assert p.lstrip().startswith("You are ")
    low = p.lower()
    assert ("must" in low or "always" in low) and ("don't" in low or "never" in low)
    assert "final_answer" in low and "```json" in p
    # taxonomy contract preserved
    assert "line_id" in low and "base_model" in low and "product_type" in low
```

- [ ] **Step 2: Run to verify it fails**

Run: `timeout 120 pytest tests/agents/test_shopping_producers.py::test_labeler_prompt_invariants -v`
Expected: FAIL — `ModuleNotFoundError: src.agents.shopping_labeler`

- [ ] **Step 3: Write the agent**

```python
# src/agents/shopping_labeler.py
"""Shopping labeling producer — tags product groups with taxonomy (line_id,
base_model, variant, product_type). Prompt-only react agent."""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.shopping_labeler")


class ShoppingLabelerAgent(BaseAgent):
    name = "shopping_labeler"
    description = "Tags shopping product groups with line_id/base_model/variant taxonomy"
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 1
    allowed_tools: list[str] = []

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a product-taxonomy classifier for a Turkish shopping bot. The "
            "user message contains the user query and a JSON array of product groups "
            "(each one or more listings believed to be the same product).\n\n"
            "For EVERY group you MUST return: `group_id` (copied verbatim); `line_id` "
            "(a canonical lowercase ASCII slug [a-z0-9-] identifying the product LINE "
            "— NOT the SKU, NOT the variant; same line across colour/storage/seller "
            "shares the slug, e.g. 'Samsung Galaxy S25 256GB Buz Mavisi' and 'Galaxy "
            "S25 FE' are 'samsung-galaxy-s25' vs 'samsung-galaxy-s25-fe'); "
            "`product_type` (one of authentic_product|accessory|replacement_part|"
            "knockoff|refurbished|unknown); `base_model` (the product LINE including "
            "line-extension qualifiers FE/Pro/Max/Plus/Ultra/Lite/Mini/SE/Air — these "
            "stay; strip only colour/storage/RAM/seller tags); `variant` (the sub-axis "
            "suffix or null); `authenticity_confidence` (0.0-1.0); `matches_user_intent` "
            "(bool).\n\n"
            "You MUST keep line-extension qualifiers in base_model (S25, S25 FE, S25 "
            "Ultra are THREE distinct base_models). Do NOT invent groups not in the "
            "input. Never emit prose or fences. Output ONLY valid JSON.\n\n"
            "Return your final_answer as JSON in this exact shape:\n"
            "```json\n"
            '{"groups": [{"group_id": 0, "line_id": "lowercase-ascii-slug", '
            '"product_type": "authentic_product", "base_model": "string", '
            '"variant": null, "authenticity_confidence": 0.9, '
            '"matches_user_intent": true}]}\n'
            "```"
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `timeout 120 pytest tests/agents/test_shopping_producers.py -v`
Expected: PASS

- [ ] **Step 5: Register + classifier (as Task 1 Step 5), then commit**

```bash
git add src/agents/shopping_labeler.py src/agents/__init__.py tests/agents/test_shopping_producers.py
git commit -m "feat(shopping): shopping_labeler producer agent"
```

---

## Task 3: `shopping_synthesizer` producer agent

**Files:**
- Create: `src/agents/shopping_synthesizer.py`
- Test: `tests/agents/test_shopping_producers.py` (append)

Prompt lifted from `SYNTHESIS_PROMPT` (aspects/praise/complaints/red_flags + insufficient_data contract). The representative_title + snippet pile arrive as the input artifact.

- [ ] **Step 1: Failing test (append)**

```python
# tests/agents/test_shopping_producers.py (append)
from src.agents.shopping_synthesizer import ShoppingSynthesizerAgent

def test_synthesizer_prompt_invariants():
    p = ShoppingSynthesizerAgent().get_system_prompt({})
    assert p.lstrip().startswith("You are ")
    low = p.lower()
    assert ("must" in low or "always" in low) and ("don't" in low or "never" in low)
    assert "final_answer" in low and "```json" in p
    assert "aspects" in low and "insufficient_data" in low
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 120 pytest tests/agents/test_shopping_producers.py::test_synthesizer_prompt_invariants -v`
Expected: FAIL — module missing

- [ ] **Step 3: Write the agent**

```python
# src/agents/shopping_synthesizer.py
"""Shopping review-synthesis producer — mines aspects/praise/complaints/red-flags
from a review snippet pile for one product line. Prompt-only react agent."""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.shopping_synthesizer")


class ShoppingSynthesizerAgent(BaseAgent):
    name = "shopping_synthesizer"
    description = "Synthesises user reviews into aspect-level insights for one product"
    default_tier = "balanced"
    min_tier = "cheap"
    max_iterations = 1
    allowed_tools: list[str] = []

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are an INTELLIGENCE module summarising user reviews for one product. "
            "The user message contains the product title and a JSON array of review "
            "snippets (Turkish and/or English, multi-source).\n\n"
            "You MUST mine substance and ignore boilerplate ('teşekkürler', 'kargo "
            "hızlı'). For each aspect ACTUALLY discussed, emit one entry with: `aspect` "
            "(one of kamera|pil|ekran|performans|yapım_kalitesi|yazılım|fiyat|satıcı|"
            "kargo|ses|şarj|güncellemeler|oyun|boyut|ergonomi|ısınma — only those that "
            "appear); `sentiment` float [-1,1]; `mention_count` int; `summary` (one "
            "short Turkish line); `quote` (ONE verbatim snippet ≤140 chars). Always "
            "sort aspects by mention_count desc, up to 8. Use the dominant language of "
            "the snippets.\n\n"
            "Do NOT fabricate — if a snippet doesn't say it, don't write it. Never emit "
            "prose or fences. Set `insufficient_data` true only if <3 substantive "
            "snippets; when true leave lists empty. Output ONLY valid JSON.\n\n"
            "Return your final_answer as JSON in this exact shape:\n"
            "```json\n"
            '{"aspects": [{"aspect": "kamera", "sentiment": 0.5, "mention_count": 3, '
            '"summary": "string", "quote": "string"}], "comparative_mentions": [], '
            '"notable_quote": "string", "overall_sentiment": 0.0, "praise": [], '
            '"complaints": [], "red_flags": [], "insufficient_data": false}\n'
            "```"
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `timeout 120 pytest tests/agents/test_shopping_producers.py -v`
Expected: PASS (all producer prompt tests)

- [ ] **Step 5: Register + classifier, then commit**

```bash
git add src/agents/shopping_synthesizer.py src/agents/__init__.py tests/agents/test_shopping_producers.py
git commit -m "feat(shopping): shopping_synthesizer producer agent"
```

---

## Task 4: Group/label prep + apply handlers

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py` (add handlers + register)
- Modify: `src/workflows/shopping/labels.py` (extract `apply_labels(groups, parsed)` pure fn)
- Test: `tests/shopping/test_v3_handlers.py`

The current `step_group` does SKU-bucket + LLM-group; `step_label` does view-build + LLM + parse. Split each: prep builds the LLM input artifact; the producer agent (Tasks 1-2) makes the call; apply parses + applies. Reuse the verbatim parse bodies from `_llm_group_residuals` and `step_label`.

- [ ] **Step 1: Write failing tests**

```python
# tests/shopping/test_v3_handlers.py
import json
import pytest
from src.workflows.shopping.pipeline_v2 import (
    _candidates_to_json, Candidate,
    handler_group_prep, handler_group_apply_label_prep, handler_label_apply_filter_gate,
)

def _cands():
    return [
        Candidate(title="Galaxy S25 256GB", site="a", site_rank=1, price=50000.0,
                  original_price=None, url="u1", rating=4.5, review_count=10,
                  review_snippets=["iyi"], sku=None, category_path="telefon"),
        Candidate(title="Galaxy S25 FE", site="b", site_rank=1, price=40000.0,
                  original_price=None, url="u2", rating=4.0, review_count=5,
                  review_snippets=["fena değil"], sku=None, category_path="telefon"),
    ]

@pytest.mark.asyncio
async def test_group_prep_emits_input_and_residual_flag():
    cands = _cands()
    art = {"search_results": json.dumps({"candidates": _candidates_to_json(cands), "query": "s25"})}
    out = await handler_group_prep(task={}, artifacts=art, ctx={})
    # sku-less candidates => residuals => group_input built, has_residuals true
    gi = json.loads(out["group_input"])
    assert gi["has_residuals"] == "true"
    assert any("index" in v for v in gi["view"])
    assert out["groups_state"]  # bucketed (empty here) carried forward

@pytest.mark.asyncio
async def test_group_apply_parses_producer_raw():
    cands = _cands()
    groups_state = json.dumps({"groups": [], "candidates": _candidates_to_json(cands), "query": "s25"})
    raw = json.dumps({"groups": [
        {"representative_title": "Galaxy S25", "member_indices": [0], "is_accessory_or_part": False},
        {"representative_title": "Galaxy S25 FE", "member_indices": [1], "is_accessory_or_part": False},
    ]})
    out = await handler_group_apply_label_prep(
        task={}, artifacts={"group_raw": raw, "groups_state": groups_state}, ctx={})
    li = json.loads(out["label_input"])
    assert len(li["view"]) == 2
    assert li["query"] == "s25"
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 120 pytest tests/shopping/test_v3_handlers.py -v`
Expected: FAIL — `ImportError: cannot import name 'handler_group_prep'`

- [ ] **Step 3: Implement the handlers**

In `src/workflows/shopping/pipeline_v2.py`, add (reusing the verbatim SKU-bucket from `step_group` and parse from `_llm_group_residuals`):

```python
async def handler_group_prep(task: dict, artifacts: dict, ctx: dict) -> dict:
    raw = artifacts.get("search_results", "{}")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    cands = _candidates_from_json(payload.get("candidates", []))
    query = payload.get("query", "")
    # deterministic SKU-first bucketing (verbatim from step_group)
    sku_buckets: dict[str, list[int]] = {}
    unbucketed: list[int] = []
    for i, c in enumerate(cands):
        (sku_buckets.setdefault(c.sku, []).append(i) if c.sku else unbucketed.append(i))
    bucketed: list[dict] = []
    for _sku, indices in sku_buckets.items():
        first = cands[indices[0]]
        bucketed.append(_group_to_dict(ProductGroup(
            representative_title=first.title, member_indices=indices,
            is_accessory_or_part=False,
            prominence=sum(1.0 / cands[i].site_rank for i in indices))))
    groups_state = {"groups": bucketed,
                    "candidates": _candidates_to_json(cands), "query": query,
                    "unbucketed": unbucketed}
    if unbucketed:
        view = [{"index": j, "title": cands[i].title, "site": cands[i].site,
                 "price": cands[i].price, "sku": cands[i].sku,
                 "category_path": cands[i].category_path}
                for j, i in enumerate(unbucketed)]
        group_input = {"view": view, "has_residuals": "true"}
    else:
        group_input = {"view": [], "has_residuals": "false"}
    return {"group_input": json.dumps(group_input, ensure_ascii=False),
            "groups_state": json.dumps(groups_state, ensure_ascii=False)}


async def handler_group_apply_label_prep(task: dict, artifacts: dict, ctx: dict) -> dict:
    gs = json.loads(artifacts.get("groups_state", "{}"))
    cands = _candidates_from_json(gs.get("candidates", []))
    query = gs.get("query", "")
    unbucketed = gs.get("unbucketed", [])
    groups = [_group_from_dict(g) for g in gs.get("groups", [])]
    raw = artifacts.get("group_raw")
    if raw and unbucketed:
        residual_cands = [cands[i] for i in unbucketed]
        residual = _parse_grouping_raw(raw, residual_cands)  # verbatim parse + fallback
        for g in residual:
            g.member_indices = [unbucketed[j] for j in g.member_indices]
            groups.append(g)
    # build label view (verbatim from step_label)
    view = []
    for i, g in enumerate(groups):
        member_cands = [cands[m] for m in g.member_indices if 0 <= m < len(cands)]
        category = next((c.category_path for c in member_cands if c.category_path), "")
        view.append({"group_id": i, "title": g.representative_title,
                     "category_path": category, "member_count": len(g.member_indices)})
    label_input = {"view": view, "query": query}
    new_state = {"groups": [_group_to_dict(g) for g in groups],
                 "candidates": _candidates_to_json(cands), "query": query}
    return {"label_input": json.dumps(label_input, ensure_ascii=False),
            "groups_state": json.dumps(new_state, ensure_ascii=False)}
```

Add `_parse_grouping_raw(raw_text, candidates)` — the verbatim parse body from `_llm_group_residuals` (strip fences, `json.loads`, validate member indices, build `ProductGroup`, `_per_site_top1_fallback` on error). Add `handler_label_apply_filter_gate` reusing `step_label`'s parse loop (extract it into `labels.apply_labels(groups, parsed_entries)`), then `step_filter` + `step_variant_gate`, emitting `gate_result` in the same shape as the current `_handler_group_label_filter_gate`. Register all three in `_STEP_HANDLERS_V2`.

- [ ] **Step 4: Run to verify pass**

Run: `timeout 120 pytest tests/shopping/test_v3_handlers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/workflows/shopping/pipeline_v2.py src/workflows/shopping/labels.py tests/shopping/test_v3_handlers.py
git commit -m "feat(shopping): group/label prep+apply handlers for v3 triads"
```

---

## Task 5: Synth prep + apply handlers

**Files:**
- Modify: `src/workflows/shopping/pipeline_v2.py`
- Test: `tests/shopping/test_v3_handlers.py` (append)

Split `step_synthesize_reviews`: prep gathers snippets (incl. `deep_scrape` community fetch) + builds `synth_input` (title + snippets + `est_input_tokens`); apply parses producer `synth_raw` → `ReviewSynthesis` → `format_group_card` → `synth_result`.

- [ ] **Step 1: Failing tests (append)**

```python
# tests/shopping/test_v3_handlers.py (append)
from src.workflows.shopping.pipeline_v2 import handler_synth_prep, handler_synth_apply

@pytest.mark.asyncio
async def test_synth_prep_builds_input_with_snippets():
    cands = _cands()
    gate = {"gate": {"kind": "chosen"},
            "chosen_group": {"representative_title": "Galaxy S25", "member_indices": [0],
                             "is_accessory_or_part": False, "prominence": 1.0,
                             "product_type": "authentic_product", "base_model": "Galaxy S25",
                             "variant": None, "authenticity_confidence": 0.9,
                             "matches_user_intent": True, "line_id": "galaxy-s25"},
            "candidates": _candidates_to_json(cands), "query": "s25"}
    out = await handler_synth_prep(task={}, artifacts={"gate_result": json.dumps(gate)}, ctx={})
    si = json.loads(out["synth_input"])
    assert si["representative_title"] == "Galaxy S25"
    assert si["snippets"] and si["est_input_tokens"] > 0

@pytest.mark.asyncio
async def test_synth_apply_parses_aspects_into_card():
    si = {"representative_title": "Galaxy S25", "group": {
        "representative_title": "Galaxy S25", "member_indices": [0],
        "is_accessory_or_part": False, "prominence": 1.0, "product_type": "authentic_product",
        "base_model": "Galaxy S25", "variant": None, "authenticity_confidence": 0.9,
        "matches_user_intent": True, "line_id": "galaxy-s25"},
        "candidates": _candidates_to_json(_cands())}
    raw = json.dumps({"aspects": [{"aspect": "kamera", "sentiment": 0.8, "mention_count": 3,
                                   "summary": "net", "quote": "kamera çok net"}],
                      "praise": ["hızlı"], "complaints": [], "red_flags": [],
                      "insufficient_data": False, "overall_sentiment": 0.7,
                      "comparative_mentions": [], "notable_quote": "iyi telefon"})
    out = await handler_synth_apply(task={}, artifacts={"synth_raw": raw, "synth_input": json.dumps(si)}, ctx={})
    assert out["synth_result"]
    res = json.loads(out["synth_result"]) if isinstance(out["synth_result"], str) else out["synth_result"]
    assert res["cards"]
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 120 pytest tests/shopping/test_v3_handlers.py -k synth -v`
Expected: FAIL — handlers missing

- [ ] **Step 3: Implement**

Add `handler_synth_prep` (resolve group from `gate_result` chosen/variant payload; gather snippets via the verbatim snippet-collection + optional `_fetch_community_reviews` when `deep_scrape`; cap at `_MAX_SNIPPETS_PER_PRODUCT*(3 if deep else 2)`; compute `est=len(prompt)//3`; emit `synth_input={representative_title, snippets, group, candidates, est_input_tokens, deep}`) and `handler_synth_apply` (parse `synth_raw` with the verbatim `ReviewSynthesis` parse body incl. `_insufficient()` fallback; `format_group_card`; emit `synth_result={cards, escalation_needed:False}`). Factor a `_parse_synthesis_raw(raw, snippet_count) -> ReviewSynthesis` from the current `step_synthesize_reviews` parse block. Register both in `_STEP_HANDLERS_V2`.

- [ ] **Step 4: Run to verify pass**

Run: `timeout 120 pytest tests/shopping/test_v3_handlers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/workflows/shopping/pipeline_v2.py tests/shopping/test_v3_handlers.py
git commit -m "feat(shopping): synth prep+apply handlers for v3 triad"
```

---

## Task 6: `shopping_v3.json` (chosen/variant/group/label paths)

**Files:**
- Create: `src/workflows/shopping/shopping_v3.json`
- Test: `tests/shopping/test_v3_wiring.py`

Wire steps 0.1, 0.2, 1.0, the group/label triad (1.1a–e), the synth triads (2.0/2.2), clarify (2.1), format (3.0). Compare-all (2.3) lands in Task 8. Producer steps use `agent: shopping_grouper|shopping_labeler|shopping_synthesizer`. `skip_when` uses the string-flag form (`group_input.has_residuals == 'false'`).

- [ ] **Step 1: Write the wiring test**

```python
# tests/shopping/test_v3_wiring.py
import json, pathlib, pytest
from src.workflows.engine.loader import load_workflow

def test_v3_loads_and_has_producer_steps():
    wf = load_workflow("shopping_v3")
    assert wf["plan_id"] == "shopping_v3"
    agents = {s["id"]: s["agent"] for s in wf["steps"]}
    assert agents["1.1b"] == "shopping_grouper"
    assert agents["1.1d"] == "shopping_labeler"
    assert agents["2.0b"] == "shopping_synthesizer"
    # group residual producer skips when no residuals
    s11b = next(s for s in wf["steps"] if s["id"] == "1.1b")
    assert s11b.get("skip_when") == "group_input.has_residuals == 'false'"
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 120 pytest tests/shopping/test_v3_wiring.py::test_v3_loads_and_has_producer_steps -v`
Expected: FAIL — `shopping_v3` not found

- [ ] **Step 3: Write `shopping_v3.json`**

Create the plan with `plan_id: "shopping_v3"`, `agents_required` including the 3 producers + `shopping_pipeline_v2` + `shopping_clarifier` + `mechanical`, and the steps per spec §4 (Phase 0/1/2 chosen+variant+group/label, Phase 3). Each producer step: `input_artifacts` = its `<x>_input`, `output_artifacts` = `<x>_raw`, `requires_grading: false`. Each prep/apply step: `agent: "shopping_pipeline_v2"`, `context: {"step_name": "<handler>"}`. (Use `shopping_v2.json` as the structural template.)

- [ ] **Step 4: Run to verify pass**

Run: `timeout 120 pytest tests/shopping/test_v3_wiring.py -v`
Expected: PASS

Also run the existing JSON validator:
Run: `timeout 120 pytest tests/shopping/test_workflow_json.py -v`
Expected: PASS (add `shopping_v3.json` to its validated-files list)

- [ ] **Step 5: Commit**

```bash
git add src/workflows/shopping/shopping_v3.json tests/shopping/test_v3_wiring.py tests/shopping/test_workflow_json.py
git commit -m "feat(shopping): shopping_v3.json — producer-agent step triads"
```

---

## Task 7: Agent-step → artifact wiring + deadlock regression

**Files:**
- Test: `tests/shopping/test_v3_wiring.py` (append)

Prove a producer step's `final_answer` lands as its `<x>_raw` artifact and the apply step consumes it, and that a producer admits without holding a slot (deadlock-free).

- [ ] **Step 1: Write the wiring + deadlock test**

```python
# tests/shopping/test_v3_wiring.py (append)
import pytest

@pytest.mark.asyncio
async def test_group_producer_output_feeds_apply(monkeypatch, tmp_path):
    """prep -> stub producer (final_answer JSON) -> apply, through the real
    artifact store: assert apply parses the producer's raw output."""
    # Drive handler_group_prep, store group_input, simulate the agent step by
    # writing the producer's final_answer JSON to the group_raw artifact, then
    # run handler_group_apply_label_prep and assert groups_state has 2 groups.
    # (Use ArtifactStore(use_db=False) in-memory; see test_v3_handlers for shapes.)
    ...

@pytest.mark.asyncio
async def test_producer_admits_without_holding_slot():
    """Saturate the oneshot lane reservation, enqueue a producer-shaped raw_dispatch
    task, assert next_task can still admit it (no parent holds a slot mid-call) —
    the inline request() path would have blocked here."""
    ...
```

Fill the `...` with concrete drivers: for the first, instantiate `ArtifactStore(use_db=False)`, call the handlers in sequence storing/reading the named artifacts. For the second, mirror the SP3b deadlock-regression test (drive real `enqueue → next_task` with the lane reservation count maxed; assert the producer task is still selectable because no task is blocked awaiting it).

- [ ] **Step 2: Run to verify fail → implement drivers → Step 3: Run to verify pass**

Run: `timeout 120 pytest tests/shopping/test_v3_wiring.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/shopping/test_v3_wiring.py
git commit -m "test(shopping): v3 producer->artifact wiring + deadlock regression"
```

---

## Task 8: Compare-all sequential CPS chain + telegram resume

**Files:**
- Create: `src/workflows/shopping/compare_continuations.py`
- Modify: `src/workflows/shopping/pipeline_v2.py` (compare_init + compare_assemble handlers)
- Modify: `src/workflows/shopping/shopping_v3.json` (2.3a/2.3e steps)
- Modify: `src/app/telegram_bot.py:10960-11005` (route compare-all through workflow resume; delete direct `_handler_format_compare` call)
- Modify: `general_beckman.continuations._HANDLER_MODULES` (add `compare_continuations`)
- Test: `tests/shopping/test_compare_chain.py`

`compare_init` (workflow step) lists ≤5 lines, renders the header, then enqueues the synth producer for line 0 with `on_complete="shopping.compare_next"` and `cont_state={lines, cursor:0, cards:[], header, mission_id}`. The continuation appends the card, and either enqueues the next line's producer (cursor+1) or writes `shopping_response` + advances the mission via `compare_assemble`. Each producer is admitted (no `await_inline`). Lane = `oneshot` only. Use the dual-shape `_extract_content` (copy from `posthook_continuations.py`).

- [ ] **Step 1: Write the chain test**

```python
# tests/shopping/test_compare_chain.py
import json, pytest
from src.workflows.shopping import compare_continuations as cc

@pytest.mark.asyncio
async def test_compare_next_appends_and_advances(monkeypatch):
    enq = []
    async def fake_enqueue(spec, **kw):
        enq.append((spec, kw)); return 999
    monkeypatch.setattr(cc, "_enqueue_synth_producer", fake_enqueue)
    state = {"lines": [{"line_id": "a"}, {"line_id": "b"}], "cursor": 0,
             "cards": [], "header": "H", "mission_id": 1}
    raw = {"result": {"content": json.dumps({"aspects": [], "praise": ["x"],
            "complaints": [], "red_flags": [], "insufficient_data": False})}}
    await cc.compare_next(child_task_id=5, result=raw, state=state)
    # cursor 0 of 2 -> enqueues next producer (cursor 1), no finalize yet
    assert len(enq) == 1

@pytest.mark.asyncio
async def test_compare_next_finalizes_on_last_line(monkeypatch):
    finalized = {}
    async def fake_final(mission_id, header, cards):
        finalized["done"] = (mission_id, len(cards))
    monkeypatch.setattr(cc, "_finalize_compare", fake_final)
    state = {"lines": [{"line_id": "a"}], "cursor": 0, "cards": [], "header": "H", "mission_id": 1}
    raw = {"content": json.dumps({"aspects": [], "praise": [], "complaints": [],
           "red_flags": [], "insufficient_data": True})}  # restart-reconcile top-level shape
    await cc.compare_next(child_task_id=5, result=raw, state=state)
    assert finalized["done"][0] == 1
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 120 pytest tests/shopping/test_compare_chain.py -v`
Expected: FAIL — module missing

- [ ] **Step 3: Implement `compare_continuations.py`**

```python
# src/workflows/shopping/compare_continuations.py
"""Sequential compare-all via CPS continuation chain. Each line's review synth is
an admitted shopping_synthesizer producer (no await_inline); compare_next appends
its card and enqueues the next line or finalizes shopping_response."""
import json
from src.infra.logging_config import get_logger
logger = get_logger("shopping.compare_chain")


def _extract_content(result: dict) -> str:
    # dual-shape: normal result['result']['content'] vs reconcile top-level result['content']
    if isinstance(result, dict):
        inner = result.get("result")
        if isinstance(inner, dict) and "content" in inner:
            return str(inner.get("content") or "")
        if "content" in result:
            return str(result.get("content") or "")
    return ""


async def _enqueue_synth_producer(spec, **kw):
    import general_beckman
    return await general_beckman.enqueue(spec, **kw)


def _producer_spec(line: dict, mission_id: int) -> dict:
    # build the synth_input + a raw_dispatch/agent producer spec for one line
    ...  # mirror handler_synth_prep's synth_input + an agent:shopping_synthesizer spec


async def _finalize_compare(mission_id: int, header: str, cards: list[str]) -> None:
    from src.workflows.engine.artifacts import get_artifact_store
    store = get_artifact_store()
    body = ("\n" + ("─" * 20) + "\n").join(cards)
    await store.store(mission_id, "shopping_response",
                      json.dumps({"formatted_text": f"{header}\n{body}", "escalation": False}))
    # advance the mission past compare so 3.0/delivery proceeds (mirror _resume_mission_at_step)
    ...


async def compare_next(child_task_id: int, result: dict, state: dict) -> None:
    from src.workflows.shopping.pipeline_v2 import _parse_synthesis_raw, format_group_card, _group_from_dict
    content = _extract_content(result)
    cards = list(state.get("cards", []))
    line = state["lines"][state["cursor"]]
    try:
        syn = _parse_synthesis_raw(content, snippet_count=line.get("snippet_count", 0))
        cards.append(format_group_card(_group_from_dict(line["group"]), syn,
                                       []))  # candidates carried in line if needed
    except Exception as exc:
        logger.warning("compare line synth failed: %s", exc)
    nxt = state["cursor"] + 1
    if nxt < len(state["lines"]):
        new_state = {**state, "cursor": nxt, "cards": cards}
        spec = _producer_spec(state["lines"][nxt], state["mission_id"])
        await _enqueue_synth_producer(spec, lane="oneshot",
                                      on_complete="shopping.compare_next", cont_state=new_state)
    else:
        await _finalize_compare(state["mission_id"], state["header"], cards)


def register_continuations() -> None:
    from general_beckman.continuations import register_resume
    register_resume("shopping.compare_next", compare_next)
```

Fill the `...` blocks (the `_producer_spec` synth_input/agent spec mirroring `handler_synth_prep`; `_finalize_compare` mission-advance mirroring `_resume_mission_at_step`). Add `"src.workflows.shopping.compare_continuations"` to `general_beckman.continuations._HANDLER_MODULES`. Implement `compare_init` (workflow step: build `lines` from `clarify_payloads`, render header via `step_compare_all`, enqueue line-0 producer with `on_complete`) and a thin `compare_assemble` (only needed if not finalizing in the continuation).

- [ ] **Step 4: Run to verify pass**

Run: `timeout 120 pytest tests/shopping/test_compare_chain.py -v`
Expected: PASS

- [ ] **Step 5: Telegram resume re-route**

In `_run_compare_all_and_reply` (`telegram_bot.py:~10955`), delete the direct `_handler_format_compare` call (lines ~10978-10985) and instead resume the mission at the compare path (`_resume_mission_at_step(mission_id, after_task_id=task_id, clarify_choice={"kind": "compare_all"})`), letting steps 2.3a→chain run in the pump. Keep the keyboard re-attach (already `vc:` after E1) and final-render path.

- [ ] **Step 6: Commit**

```bash
git add src/workflows/shopping/compare_continuations.py src/workflows/shopping/pipeline_v2.py src/workflows/shopping/shopping_v3.json src/app/telegram_bot.py tests/shopping/test_compare_chain.py
# (also the _HANDLER_MODULES edit)
git commit -m "feat(shopping): compare-all sequential CPS chain + telegram workflow resume"
```

---

## Task 9: `shopping_clarifier` → react

**Files:**
- Modify: `src/agents/shopping_clarifier.py:21`
- Test: `tests/agents/test_shopping_producers.py` (append)

- [ ] **Step 1: Failing test**

```python
# tests/agents/test_shopping_producers.py (append)
from src.agents.shopping_clarifier import ShoppingClarifierAgent

def test_clarifier_is_react_not_single_shot():
    a = ShoppingClarifierAgent()
    assert getattr(a, "execution_pattern", "react_loop") == "react_loop"
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 120 pytest tests/agents/test_shopping_producers.py::test_clarifier_is_react_not_single_shot -v`
Expected: FAIL — `execution_pattern == 'single_shot'`

- [ ] **Step 3: Drop the line**

Delete `execution_pattern = "single_shot"` from `src/agents/shopping_clarifier.py:21`.

- [ ] **Step 4: Run to verify pass + clarifier suite**

Run: `timeout 120 pytest tests/agents/test_shopping_producers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/shopping_clarifier.py tests/agents/test_shopping_producers.py
git commit -m "refactor(shopping): shopping_clarifier single_shot -> react (kills single_shot request caller)"
```

---

## Task 10: Switch launch to v3 + validation window

**Files:**
- Modify: `src/app/telegram_bot.py:8490-8498`
- Modify: `src/workflows/shopping/quick_search_v2.json:8`
- Modify: `tests/integration/test_e2e_llm_pipeline.py:142`

- [ ] **Step 1: Flip the launch map**

In `_create_shopping_mission` (`telegram_bot.py:8490-8498`), change `"deep_research": "shopping_v2"` and `"research": "shopping_v2"` and the default `wf_map.get(sub_intent or "shopping", "shopping_v2")` → `shopping_v3`. Repoint `quick_search_v2.json:8` `escalation_target` → `shopping_v3`. Update `test_e2e_llm_pipeline.py:142`.

- [ ] **Step 2: Verify import + targeted tests**

Run: `python -c "import json; json.load(open('src/workflows/shopping/shopping_v3.json'))"` → no error
Run: `timeout 120 pytest tests/shopping/ -v`
Expected: PASS

- [ ] **Step 3: LIVE Telegram validation (founder, manual)**

After `/restart`: run a vague query (clarifier react), a specific query (chosen path), a multi-line query → pick a variant (variant path), and "compare all" (CPS chain). Confirm cards render and `SELECT lane,status,COUNT(*) FROM tasks WHERE mission_id=<m> GROUP BY 1,2` shows producer children completing on `oneshot`, none orphaned.

- [ ] **Step 4: Commit**

```bash
git add src/app/telegram_bot.py src/workflows/shopping/quick_search_v2.json tests/integration/test_e2e_llm_pipeline.py
git commit -m "feat(shopping): route shopping missions to shopping_v3"
```

---

## Task 11: Delete the shim + dead callers (SP5 unblock)

**Files:**
- Delete: `packages/coulson/src/coulson/single_shot.py`
- Modify: `packages/coulson/src/coulson/__init__.py` (remove `single_shot` branch), `reflection.py:89`, `src/workflows/engine/constrained_emit.py:147`
- Modify: `src/core/llm_dispatcher.py` (delete `request()` + `_request_kwargs_to_spec`; `_task_result_to_request_response` after re-grep)
- Modify: retire `shopping_v2.json` after validation window

**Do this ONLY after Task 10's live validation passes and no `shopping_v2` missions are in flight.**

- [ ] **Step 1: Re-grep to confirm zero live callers**

Run: `rg -n "\.request\(|_request_kwargs_to_spec|_task_result_to_request_response|execution_pattern\s*=\s*.single_shot|single_shot\.run|_single_shot_run" src packages --glob '!*/tests/*'`
Expected: only the definitions themselves + the dead `reflection.py:89`/`constrained_emit.py:147`. If any other live caller appears, STOP — handle it first.

- [ ] **Step 2: Delete dead `request()` callers**

Remove the `await_inline`/`request()` blocks at `coulson/reflection.py:89` and `constrained_emit.py:147` (the functions are dead per the dispatcher docstring — delete the dead functions or their request bodies; confirm no import breaks).

Run: `timeout 120 pytest packages/coulson/tests/ -v` (separate invocation from `tests/`)
Expected: PASS

- [ ] **Step 3: Delete `single_shot`**

Remove the `if profile.execution_pattern == "single_shot": _result = await _single_shot_run(...)` branch in `coulson/__init__.py:88-93` (leave only the react branch) and delete `single_shot.py`.

Run: `timeout 120 pytest packages/coulson/tests/ -v`
Expected: PASS (fix any test referencing single_shot)

- [ ] **Step 4: Delete the shim**

Delete `LLMDispatcher.request()` + `_request_kwargs_to_spec` from `llm_dispatcher.py`. Re-grep `_task_result_to_request_response`; if `request()` was its only caller, delete it too.

Run: `timeout 120 pytest tests/core/ -v` and `timeout 120 pytest packages/general_beckman/tests/ -v` (separate invocations)
Expected: PASS

- [ ] **Step 5: Retire `shopping_v2.json`**

After the validation window with no v2 missions, delete `shopping_v2.json` + its handlers-only-used-by-v2 dead code in `pipeline_v2.py` (`_handler_group_label_filter_gate`, `_handler_synth_one`, `_handler_format_compare`, `_grouping_llm_call`, `_synthesis_llm_call`, and `labels._label_llm_call`). Keep the deterministic functions reused by v3 (`step_filter`, `step_variant_gate`, `format_group_card`, `step_compare_all`, `_fetch_community_reviews`, parse helpers).

Run: `timeout 120 pytest tests/shopping/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore(shopping): delete request() shim + single_shot + dead callers — SP5 unblocked (src/workflows/shopping)"
```

---

## Self-Review notes

- **`src/shopping/` (site #5) remains** — SP5 cannot delete `await_inline` until that subsystem's own spec lands (spec §7). This plan unblocks only the `src/workflows/shopping/` half.
- **Materializer interaction:** producer steps emit plain JSON artifacts with no declared file `produces`, so `materialize_produces` is a no-op for them (spec §8 risk 6) — confirm during Task 6 that no v3 step declares a `.md`/`.json` file produces.
- **Live-error context:** a large restart-gated batch (schema-gate 240-tightening, SP4a, materializer, S7/S6) went live 2026-06-05 and is under triage (`docs/handoff/2026-06-05-deterministic-materializer-handoff.md` §0). Do NOT start live validation (Task 10 Step 3) until that triage is clear, or the signal will be ambiguous.
