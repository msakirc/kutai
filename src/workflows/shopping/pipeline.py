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


def _extract_query(artifacts: dict, task: dict) -> str:
    """Extract the actual search query string from artifacts.

    Artifacts may be raw strings ("siemens s100") or JSON-encoded dicts
    (``{"clarified_query": "siemens s100", "skipped": true}``).
    """
    # Try clarified_query first (from step 1.1), then user_query (initial input)
    for key in ("clarified_query", "user_query"):
        raw = artifacts.get(key, "")
        if not raw:
            continue
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                parsed = json.loads(raw)
                # The artifact might wrap the query in a field
                q = parsed.get("clarified_query") or parsed.get("query") or parsed.get("user_query", "")
                if q:
                    return q
            except (json.JSONDecodeError, ValueError):
                pass
        # Raw string — use as-is
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    # Last resort: task description
    return task.get("description", "")


# ── Step handlers ────────────────────────────────────────────────────────────

async def _step_search(task: dict, artifacts: dict) -> str:
    """Search products and community data. Returns JSON."""
    query = _extract_query(artifacts, task)
    logger.info("search step starting", query=query[:100])

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
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning("product search failed for _step_search", error=str(exc))
    try:
        community = await community_task
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning("community search failed for _step_search", error=str(exc))

    logger.info("search step done", product_count=len(products or []), community_count=len(community or []))

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
    query = _extract_query(artifacts, task)

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
    """Format search results into a Telegram-ready message.

    If a 'recommendation' artifact exists (from the LLM synthesis step in
    the full shopping workflow), use it directly — it's already a rich
    text recommendation.  Otherwise fall back to formatting raw products.
    """
    from src.shopping.output.summary import format_recommendation_summary
    from src.shopping.output.formatters import format_price

    # Prefer the LLM-synthesized recommendation if available
    recommendation = artifacts.get("recommendation", "")
    if isinstance(recommendation, str) and len(recommendation.strip()) > 20:
        return recommendation.strip()

    raw = artifacts.get("search_results", "{}")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            data = {}
    else:
        data = raw if isinstance(raw, dict) else {}

    # Also read review_data artifact if available (from step 2.1 in full workflow)
    review_raw = artifacts.get("review_data", "{}")
    if isinstance(review_raw, str):
        try:
            review_data = json.loads(review_raw)
        except (json.JSONDecodeError, ValueError):
            review_data = {}
    else:
        review_data = review_raw if isinstance(review_raw, dict) else {}

    products = data.get("products", [])
    community = data.get("community", [])
    reviews = review_data.get("reviews", []) if isinstance(review_data, dict) else []

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

    # ── Build formatted output directly ──
    # The output formatters (format_recommendation_summary) expect a specific
    # dict structure that doesn't map well to raw scraper data. Build the
    # Telegram message directly — it's clearer and shows all available data.

    lines: list[str] = []

    # ── Prices across sources ──
    priced = sorted(
        [p for p in products if p.get("original_price") or p.get("discounted_price")],
        key=lambda p: p.get("discounted_price") or p.get("original_price") or float("inf"),
    )

    if priced:
        best = priced[0]
        best_price = best.get("discounted_price") or best.get("original_price")
        lines.append(
            f"🏆 *{best.get('name', '')}*\n"
            f"   💰 {format_price(best_price)} — {best.get('source', '')}"
        )
        if best.get("url"):
            lines.append(f"   🔗 {best['url']}")

        # Rating if available
        if best.get("rating"):
            stars = "⭐" * int(best["rating"])
            review_count = best.get("review_count", "")
            rc_str = f" ({review_count} değerlendirme)" if review_count else ""
            lines.append(f"   {stars} {best['rating']}/5{rc_str}")

        # Other sources with prices
        others = [p for p in priced[1:] if p.get("source") != best.get("source")]
        if others:
            lines.append("\n📍 *Diğer Fiyatlar:*")
            for p in others[:5]:
                price = p.get("discounted_price") or p.get("original_price")
                orig = p.get("original_price")
                disc = p.get("discounted_price")
                price_str = format_price(price)
                if disc and orig and disc < orig:
                    price_str += f" ~~{format_price(orig)}~~"
                src = p.get("source", "?")
                url = p.get("url", "")
                url_str = f"  {url}" if url else ""
                lines.append(f"  • {src} — {price_str}{url_str}")
    else:
        # Products found but no prices
        top = products[0]
        lines.append(f"🔍 *{top.get('name', '')}*")
        lines.append(f"   {top.get('source', '')} — fiyat bilgisi yok")
        if top.get("url"):
            lines.append(f"   🔗 {top['url']}")

    # ── Community data ──
    if community:
        by_source: dict[str, list] = {}
        for c in community:
            src = c.get("source", "other")
            by_source.setdefault(src, []).append(c)

        lines.append("\n💬 *Topluluk:*")
        for src, items in by_source.items():
            src_label = {
                "teknopat": "Technopat",
                "sikayetvar": "Şikayetvar",
                "donanimhaber": "DonanımHaber",
                "eksisozluk": "Ekşi Sözlük",
            }.get(src, src)

            if src == "sikayetvar":
                lines.append(f"  ⚠️ {len(items)} şikayet — {src_label}")
            else:
                lines.append(f"  💭 {len(items)} tartışma — {src_label}")

            # Show top 2 thread titles
            for item in items[:2]:
                name = item.get("name", "")[:60]
                url = item.get("url", "")
                if name:
                    lines.append(f"     • {name}")

    # ── Reviews ──
    if reviews:
        lines.append("\n📝 *Kullanıcı Yorumları:*")
        # Group by positive/negative sentiment if rating available
        positive = [r for r in reviews if (r.get("rating") or 0) >= 4]
        negative = [r for r in reviews if (r.get("rating") or 5) <= 2]
        neutral = [r for r in reviews if r not in positive and r not in negative]

        if positive:
            lines.append(f"  👍 {len(positive)} olumlu")
            for r in positive[:2]:
                text = r.get("text", "")[:80]
                if text:
                    lines.append(f"     \"{text}\"")
        if negative:
            lines.append(f"  👎 {len(negative)} olumsuz")
            for r in negative[:2]:
                text = r.get("text", "")[:80]
                if text:
                    lines.append(f"     \"{text}\"")
        if not positive and not negative and neutral:
            for r in neutral[:2]:
                text = r.get("text", "")[:80]
                if text:
                    lines.append(f"  💬 \"{text}\"")

    # ── Source count ──
    sources = set(p.get("source", "") for p in products)
    sources.discard("")
    if len(sources) >= 3:
        lines.append(f"\n✅ {len(sources)} kaynaktan karşılaştırma")
    elif len(sources) == 2:
        lines.append(f"\n🟡 {len(sources)} kaynak")
    elif len(sources) == 1:
        lines.append(f"\n🟠 Tek kaynak ({list(sources)[0]})")

    return "\n".join(lines)


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

async def _step_clarify(task: dict, artifacts: dict) -> str | dict:
    """Check if clarification is needed. If not, pass the query through.

    Returns str (artifact) for pass-through, or dict with
    status="needs_clarification" to trigger the Telegram pause mechanism.
    """
    parsed = artifacts.get("parsed_intent", "{}")
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (json.JSONDecodeError, ValueError):
            parsed = {}

    needs_clarification = parsed.get("needs_clarification", False)
    query = _extract_query(artifacts, task)

    if not needs_clarification:
        return json.dumps({
            "clarified_query": query,
            "skipped": True,
        }, ensure_ascii=False)

    # Vague query — trigger the Telegram clarification mechanism.
    # Return a special dict (not str) that run() will detect and
    # pass through as the result, with status="needs_clarification".
    category = parsed.get("category", "")
    cat_hint = f" ({category})" if category else ""
    return {
        "_needs_clarification": True,
        "clarification": (
            f"'{query}' biraz geniş bir arama{cat_hint}. "
            f"Aradığınız ürünü daraltabilir misiniz?\n\n"
            f"Örneğin:\n"
            f"• Marka veya model (ör: Nike Air Max)\n"
            f"• Bütçe (ör: 2000 TL altı)\n"
            f"• Kullanım amacı (ör: koşu ayakkabısı)"
        ),
        "query": query,
    }


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

        step_name = context.get("step_name", "")
        if not step_name:
            # Fallback: parse from title "[0.1] step_name_here"
            title = task.get("title", "")
            if "] " in title:
                step_name = title.split("] ", 1)[1]
        if not step_name:
            step_name = context.get("workflow_step_id", "")

        logger.info("step dispatch", step_name=step_name, task_id=task.get("id"))

        mission_id = task.get("mission_id")
        input_artifacts = context.get("input_artifacts", [])

        artifacts = (
            await _read_artifacts(mission_id, input_artifacts)
            if mission_id
            else {}
        )

        logger.info(
            "artifacts loaded",
            step_name=step_name,
            artifact_keys=list(artifacts.keys()),
            mission_id=mission_id,
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
            result = await handler(task, artifacts)

            # Handler can return a dict to signal special statuses
            # (e.g. needs_clarification from _step_clarify)
            if isinstance(result, dict) and result.get("_needs_clarification"):
                return {
                    "status": "needs_clarification",
                    "clarification": result.get("clarification", "More info needed"),
                    "result": json.dumps(result, ensure_ascii=False, default=str),
                    "model": "shopping_pipeline",
                    "cost": 0.0,
                    "iterations": 1,
                }

            return {
                "status": "completed",
                "result": result if isinstance(result, str) else json.dumps(result, default=str),
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
