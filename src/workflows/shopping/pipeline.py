"""ShoppingPipeline — mechanical executor for shopping workflow steps.

Mirrors CodingPipeline's interface but handles shopping-specific steps
without any LLM calls. Steps map to Python functions that use existing
shopping modules.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("workflows.shopping.pipeline")


# ── Artifact helper ─────────────────────────────────────────────────────────

async def _read_artifacts(mission_id: int | str, artifact_names: list[str]) -> dict:
    """Read artifacts from the mission blackboard."""
    from src.workflows.engine.artifacts import ArtifactStore
    store = ArtifactStore()
    result: dict[str, Any] = {}
    for name in artifact_names:
        value = await store.retrieve(mission_id, name)
        if value is not None:
            result[name] = value
    return result


# ── Step handlers ────────────────────────────────────────────────────────────

async def _step_search(task: dict, artifacts: dict) -> str:
    """Search products and community data. Returns JSON."""
    query = artifacts.get("user_query", "")
    if not query:
        query = task.get("description", "")

    from src.shopping.resilience.fallback_chain import (
        get_product_with_fallback,
        get_community_data,
    )

    product_task = asyncio.ensure_future(
        asyncio.wait_for(get_product_with_fallback(query), timeout=30)
    )
    community_task = asyncio.ensure_future(
        asyncio.wait_for(get_community_data(query), timeout=20)
    )

    products: list = []
    community: list = []
    try:
        products = await product_task
    except (asyncio.TimeoutError, Exception):
        pass
    try:
        community = await community_task
    except (asyncio.TimeoutError, Exception):
        pass

    if not isinstance(products, list):
        products = []
    if not isinstance(community, list):
        community = []

    products_dicts = [
        dataclasses.asdict(p) if dataclasses.is_dataclass(p) else p
        for p in products
    ]
    community_dicts = [
        dataclasses.asdict(c) if dataclasses.is_dataclass(c) else c
        for c in community
    ]

    result = {
        "formatted_text": "",
        "products": products_dicts,
        "community": community_dicts,
        "product_count": len(products_dicts),
        "community_count": len(community_dicts),
        "escalation_needed": len(products_dicts) == 0,
    }
    return json.dumps(result, default=str)


async def _step_search_and_reviews(task: dict, artifacts: dict) -> str:
    """Search products, community, and fetch reviews for top products."""
    query = artifacts.get("clarified_query") or artifacts.get("user_query", "")
    if not query:
        query = task.get("description", "")

    from src.shopping.resilience.fallback_chain import (
        get_product_with_fallback,
        get_community_data,
    )

    product_task = asyncio.ensure_future(
        asyncio.wait_for(get_product_with_fallback(query), timeout=30)
    )
    community_task = asyncio.ensure_future(
        asyncio.wait_for(get_community_data(query), timeout=20)
    )

    products: list = []
    community: list = []
    try:
        products = await product_task
    except (asyncio.TimeoutError, Exception):
        pass
    try:
        community = await community_task
    except (asyncio.TimeoutError, Exception):
        pass

    if not isinstance(products, list):
        products = []
    if not isinstance(community, list):
        community = []

    products_dicts = [
        dataclasses.asdict(p) if dataclasses.is_dataclass(p) else p
        for p in products
    ]
    community_dicts = [
        dataclasses.asdict(c) if dataclasses.is_dataclass(c) else c
        for c in community
    ]

    # Fetch reviews for top 3 products with URLs
    from src.shopping.scrapers import get_scraper

    reviews: list = []
    for p in products_dicts[:3]:
        url = p.get("url", "")
        if not url:
            continue
        source = None
        for domain, key in [
            ("trendyol", "trendyol"),
            ("hepsiburada", "hepsiburada"),
            ("amazon.com.tr", "amazon_tr"),
        ]:
            if domain in url:
                source = key
                break
        if source:
            try:
                scraper_cls = get_scraper(source)
                if scraper_cls:
                    scraper = scraper_cls()
                    r = await asyncio.wait_for(scraper.get_reviews(url), timeout=15)
                    if r:
                        reviews.extend(r[:10])
            except Exception:
                pass

    review_dicts = [
        dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r
        for r in reviews
    ]

    result = {
        "formatted_text": "",
        "products": products_dicts,
        "community": community_dicts,
        "reviews": review_dicts,
        "product_count": len(products_dicts),
        "community_count": len(community_dicts),
        "escalation_needed": len(products_dicts) == 0,
    }
    return json.dumps(result, default=str)


async def _step_format(task: dict, artifacts: dict) -> str:
    """Format search results into a Telegram-ready message."""
    from src.shopping.output.summary import format_recommendation_summary
    from src.shopping.output.formatters import format_price

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

    if not products:
        if community:
            text = (
                f"No product listings found, but found {len(community)} "
                f"community discussions:\n\n"
            )
            for c in community[:5]:
                text += f"• {c.get('name', 'Discussion')}\n  {c.get('url', '')}\n"
            return text
        return "No results found. Try a more specific search."

    # Sort by price to pick cheapest
    priced = [p for p in products if p.get("original_price")]
    priced.sort(key=lambda p: p.get("original_price", float("inf")))

    top = priced[0] if priced else products[0]
    top_pick = {
        "name": top.get("name", ""),
        "price": top.get("discounted_price") or top.get("original_price"),
        "source": top.get("source", ""),
        "url": top.get("url", ""),
        "reason": "Best price found",
    }

    where_to_buy = [
        {
            "source": p.get("source"),
            "price": p.get("discounted_price") or p.get("original_price"),
            "url": p.get("url"),
        }
        for p in priced[:5]
    ]

    warnings: list[str] = []
    sikayetvar = sum(1 for c in community if c.get("source") == "sikayetvar")
    if sikayetvar >= 2:
        warnings.append(f"⚠️ {sikayetvar} complaints on Şikayetvar")

    results = {
        "top_pick": top_pick,
        "budget_option": None,
        "alternatives": [],
        "warnings": warnings,
        "timing": {},
        "where_to_buy": where_to_buy,
        "confidence": min(
            len(set(p.get("source", "") for p in products)) * 0.25, 1.0
        ),
        "sources": len(set(p.get("source", "") for p in products)),
    }

    formatted = format_recommendation_summary(results, format="telegram")
    if not formatted:
        lines = [
            f"🏆 *{top_pick['name']}* — "
            + (format_price(top_pick["price"]) if top_pick.get("price") else "?")
        ]
        if top_pick.get("url"):
            lines.append(f"   {top_pick['source']}: {top_pick['url']}")
        if where_to_buy and len(where_to_buy) > 1:
            lines.append("\n📍 *Prices:*")
            for w in where_to_buy:
                p = format_price(w.get("price", 0)) if w.get("price") else "?"
                lines.append(f"  • {w.get('source', '?')} — {p}")
        for w in warnings:
            lines.append(f"\n{w}")
        formatted = "\n".join(lines)

    return formatted


async def _step_analyze_query(task: dict, artifacts: dict) -> str:
    """Analyze the shopping query — intent, category, constraints, vagueness."""
    from src.shopping.intelligence.query_analyzer import _fallback_analyze

    query = artifacts.get("user_query", "") or task.get("description", "")
    if isinstance(query, str) and query.startswith("{"):
        try:
            query = json.loads(query).get("user_query", query)
        except (json.JSONDecodeError, ValueError):
            pass

    analysis = _fallback_analyze(query)
    # Output must match artifact_schema: requires "query" and "needs_clarification"
    analysis["query"] = query
    return json.dumps(analysis, ensure_ascii=False, default=str)


# ── Step registry ────────────────────────────────────────────────────────────

_STEP_HANDLERS = {
    "execute_product_search": _step_search,
    "format_and_deliver": _step_format,
    "search_and_collect_reviews": _step_search_and_reviews,
    "understand_query_check_clarity": _step_analyze_query,
}


# ── Pipeline class ───────────────────────────────────────────────────────────

class ShoppingPipeline:
    """Mechanical executor for shopping workflow steps.

    Runs Python functions for data-fetch/format steps — no LLM involved.
    Intelligence steps (scoring, alternatives, budget analysis) are handled
    by separate LLM agent tasks in the workflow.
    """

    async def run(self, task: dict) -> dict:
        """Execute a mechanical shopping workflow step.

        Returns dict with:
            status: "completed" or "failed"
            result: str (the output text/data)
            model: "shopping_pipeline" (no LLM used)
            cost: 0.0
            iterations: 1
        """
        context = task.get("context", {})
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except (json.JSONDecodeError, ValueError):
                context = {}

        step_name = context.get("step_name", "") or context.get("workflow_step_id", "")
        mission_id = task.get("mission_id")
        input_artifacts = context.get("input_artifacts", [])

        artifacts = (
            await _read_artifacts(mission_id, input_artifacts)
            if mission_id
            else {}
        )

        handler = _STEP_HANDLERS.get(step_name)
        if not handler:
            return {
                "status": "failed",
                "result": f"Unknown step: {step_name!r}",
                "model": "shopping_pipeline",
                "cost": 0.0,
                "iterations": 1,
            }

        try:
            result_text = await handler(task, artifacts)
            return {
                "status": "completed",
                "result": result_text,
                "model": "shopping_pipeline",
                "cost": 0.0,
                "iterations": 1,
            }
        except Exception as exc:
            logger.exception("ShoppingPipeline step %r failed", step_name)
            return {
                "status": "failed",
                "result": f"Pipeline error: {exc}",
                "model": "shopping_pipeline",
                "cost": 0.0,
                "iterations": 1,
            }
