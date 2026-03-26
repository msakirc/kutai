"""Search plan generation for the shopping intelligence system.

Takes a structured query analysis and produces an ordered list of search tasks,
respecting rate budgets and phasing dependent searches.
"""

from __future__ import annotations

import json
import re

from src.infra.logging_config import get_logger
from src.shopping.text_utils import generate_search_variants

logger = get_logger("shopping.intelligence.search_planner")

# ── Constants ────────────────────────────────────────────────────────────────

MAX_SEARCHES_PER_SESSION = 20
PHASE_1_BUDGET = 10  # immediate searches
PHASE_2_BUDGET = MAX_SEARCHES_PER_SESSION - PHASE_1_BUDGET

_SOURCES_BY_CATEGORY: dict[str, list[str]] = {
    "electronics": ["akakce", "trendyol", "hepsiburada", "amazon_tr", "n11"],
    "appliances": ["akakce", "trendyol", "hepsiburada", "teknosa"],
    "furniture": ["trendyol", "hepsiburada", "ikea_tr", "koçtaş"],
    "grocery": ["trendyol", "migros", "a101", "getir"],
    "clothing": ["trendyol", "hepsiburada", "n11"],
}
_DEFAULT_SOURCES = ["akakce", "trendyol", "hepsiburada"]


from ._llm import _llm_call

# ── Rule-based plan generation ───────────────────────────────────────────────

def _sources_for(category: str | None) -> list[str]:
    if category and category in _SOURCES_BY_CATEGORY:
        return _SOURCES_BY_CATEGORY[category]
    return _DEFAULT_SOURCES


def _build_rule_based_plan(analyzed: dict) -> list[dict]:
    """Generate a search plan from the query analysis using rules."""
    tasks: list[dict] = []
    category = analyzed.get("category")
    intent = analyzed.get("intent", "explore")
    products = analyzed.get("products_mentioned", [])
    constraints = analyzed.get("constraints", [])
    budget = analyzed.get("budget")
    sources = _sources_for(category)

    raw_query = analyzed.get("raw_query", "")

    # Phase 1: direct product searches
    if products:
        for product in products[:3]:  # cap at 3 products
            variants = generate_search_variants(product)
            for variant in variants[:2]:  # max 2 variants per product
                tasks.append({
                    "query": variant,
                    "sources": sources[:3],
                    "purpose": f"Find listings for '{variant}'",
                    "phase": 1,
                })
    elif raw_query:
        variants = generate_search_variants(raw_query)
        for variant in variants[:2]:
            tasks.append({
                "query": variant,
                "sources": sources[:3],
                "purpose": f"Search for '{variant}'",
                "phase": 1,
            })

    # Phase 1: price comparison across sources
    if intent in ("find_cheapest", "compare") and products:
        main_product = products[0]
        for source in sources:
            if len(tasks) >= PHASE_1_BUDGET:
                break
            tasks.append({
                "query": main_product,
                "sources": [source],
                "purpose": f"Price check on {source}",
                "phase": 1,
            })

    # Phase 1: budget-filtered search
    if budget:
        budget_query = f"{raw_query or (products[0] if products else '')} {int(budget)} TL altı"
        tasks.append({
            "query": budget_query,
            "sources": sources[:2],
            "purpose": "Budget-filtered search",
            "phase": 1,
        })

    # Phase 2: review / spec deep-dives (depend on Phase 1 results)
    if intent in ("find_best", "compare"):
        tasks.append({
            "query": "{top_product} inceleme yorum",
            "sources": ["google", "youtube"],
            "purpose": "Gather reviews for top result (template: fill after Phase 1)",
            "phase": 2,
        })
        tasks.append({
            "query": "{top_product} vs {runner_up}",
            "sources": sources[:2],
            "purpose": "Head-to-head comparison (template: fill after Phase 1)",
            "phase": 2,
        })

    # Phase 2: alternative search
    if intent != "specific_product":
        tasks.append({
            "query": "{category_or_query} alternatif",
            "sources": sources[:2],
            "purpose": "Discover alternatives (template: fill after Phase 1)",
            "phase": 2,
        })

    # Enforce budget limits
    phase1 = [t for t in tasks if t["phase"] == 1][:PHASE_1_BUDGET]
    phase2 = [t for t in tasks if t["phase"] == 2][:PHASE_2_BUDGET]
    return phase1 + phase2


# ── LLM-based plan ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = f"""\
You are a search planner for a Turkish shopping assistant.  Given a query
analysis JSON, produce a list of search tasks as a JSON array.

Each task object must have:
- query: the search string
- sources: list of source identifiers
- purpose: one-line explanation
- phase: 1 (immediate) or 2 (depends on phase-1 results)

Rules:
- Maximum {PHASE_1_BUDGET} phase-1 tasks, {PHASE_2_BUDGET} phase-2 tasks
- Phase-2 queries may use {{top_product}} / {{runner_up}} placeholders
- Return ONLY the JSON array, no markdown
"""


async def _llm_plan(analyzed: dict) -> list[dict] | None:
    """Ask the LLM to generate a search plan."""
    try:
        response = await _llm_call(
            prompt=json.dumps(analyzed, ensure_ascii=False),
            system=_SYSTEM_PROMPT,
            temperature=0.3,
        )
        if not response:
            return None

        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        tasks = json.loads(cleaned)
        if not isinstance(tasks, list):
            return None

        # Validate structure
        valid: list[dict] = []
        for t in tasks:
            if isinstance(t, dict) and "query" in t and "phase" in t:
                t.setdefault("sources", _DEFAULT_SOURCES)
                t.setdefault("purpose", "")
                t["phase"] = int(t["phase"])
                valid.append(t)

        return valid if valid else None

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("LLM search plan parsing failed: %s", exc)
        return None


# ── Public API ───────────────────────────────────────────────────────────────

async def generate_search_plan(analyzed_query: dict) -> list[dict]:
    """Generate a list of search tasks from a query analysis.

    Parameters
    ----------
    analyzed_query:
        Output of ``analyze_query`` -- a dict with intent, category,
        products_mentioned, constraints, etc.

    Returns
    -------
    List of task dicts, each with keys: query, sources, purpose, phase.
    Phase 1 tasks are immediate; Phase 2 tasks depend on Phase 1 results.
    Total tasks are capped at ``MAX_SEARCHES_PER_SESSION``.
    """
    if not analyzed_query:
        logger.warning("generate_search_plan called with empty analysis")
        return []

    try:
        llm_plan = await _llm_plan(analyzed_query)
        if llm_plan:
            logger.info(
                "Search plan from LLM: %d phase-1, %d phase-2",
                sum(1 for t in llm_plan if t["phase"] == 1),
                sum(1 for t in llm_plan if t["phase"] == 2),
            )
            return llm_plan
    except Exception as exc:
        logger.warning("LLM plan generation failed: %s", exc)

    plan = _build_rule_based_plan(analyzed_query)
    logger.info(
        "Rule-based search plan: %d phase-1, %d phase-2",
        sum(1 for t in plan if t["phase"] == 1),
        sum(1 for t in plan if t["phase"] == 2),
    )
    return plan
