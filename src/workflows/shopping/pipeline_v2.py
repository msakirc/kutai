"""Shopping pipeline v2 — trust site ordering, LLM-based grouping + review synthesis.

See docs/superpowers/specs/2026-04-21-shopping-trust-sites-synthesize-reviews-design.md
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
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


async def _fetch_products(query: str) -> list:
    """Thin wrapper around the shopping scraper fleet — mocked in tests."""
    from src.shopping.resilience.fallback_chain import get_product_with_fallback
    return await asyncio.wait_for(get_product_with_fallback(query), timeout=45)


def _attr(obj, name: str, default=None):
    """Read attribute from dataclass OR dict — scrapers mix both."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


_REVIEW_FETCH_TIMEOUT_S = 15
_MAX_SNIPPETS_PER_PRODUCT = 10


async def _fetch_reviews(products: list) -> dict[str, list[str]]:
    """For each product, fetch review snippets via its scraper. Keyed by URL.

    Concurrent; individual failures/timeouts yield empty list, never raise.
    Separate from ``_fetch_products`` so tests can mock independently.
    """
    if not products:
        return {}
    from src.shopping.scrapers import get_scraper

    async def _one(p) -> tuple[str, list[str]]:
        url = str(_attr(p, "url") or "")
        source = str(_attr(p, "source") or _attr(p, "site") or "")
        if not url or not source:
            return url, []
        scraper_cls = get_scraper(source)
        if scraper_cls is None:
            return url, []
        try:
            scraper = scraper_cls()
            reviews = await asyncio.wait_for(
                scraper.get_reviews(url, max_pages=1),
                timeout=_REVIEW_FETCH_TIMEOUT_S,
            )
        except Exception as exc:
            logger.debug("review fetch failed", source=source, url=url, err=str(exc))
            return url, []
        snippets: list[str] = []
        for r in reviews or []:
            if not isinstance(r, dict):
                continue
            text = r.get("text") or r.get("content") or r.get("comment") or ""
            text = str(text).strip()
            if text:
                snippets.append(text)
        return url, snippets[:_MAX_SNIPPETS_PER_PRODUCT]

    tasks = [asyncio.create_task(_one(p)) for p in products]
    out: dict[str, list[str]] = {}
    for coro in asyncio.as_completed(tasks):
        try:
            url, snips = await coro
        except Exception:
            continue
        if url and snips:
            out[url] = snips
    return out


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
        site = _attr(p, "source") or _attr(p, "site") or "unknown"
        by_site.setdefault(site, []).append(p)

    # Keep only top-N per site before fetching reviews (reviews are expensive)
    kept_products: list = []
    site_rank_map: list[tuple[str, int]] = []  # parallel to kept_products
    for site, products in by_site.items():
        for rank, p in enumerate(products[:per_site_n], start=1):
            kept_products.append(p)
            site_rank_map.append((site, rank))

    # Fetch review snippets only for products that don't already carry them
    # (tests inject them inline via the fixture products).
    needs_fetch = [p for p in kept_products if not _attr(p, "review_snippets")]
    try:
        reviews_map = await _fetch_reviews(needs_fetch) if needs_fetch else {}
    except Exception as exc:
        logger.warning("review fetch stage failed, continuing without snippets: %s", exc)
        reviews_map = {}

    cands: list[Candidate] = []
    for p, (site, rank) in zip(kept_products, site_rank_map):
        url = str(_attr(p, "url") or "")
        inline = list(_attr(p, "review_snippets") or [])
        snippets = inline or reviews_map.get(url, [])
        cands.append(
            Candidate(
                title=str(_attr(p, "name") or ""),
                site=site,
                site_rank=rank,
                price=(_attr(p, "discounted_price") or _attr(p, "original_price") or _attr(p, "price")),
                original_price=_attr(p, "original_price"),
                url=url,
                rating=_attr(p, "rating"),
                review_count=_attr(p, "review_count"),
                review_snippets=list(snippets),
            )
        )
    logger.info(
        "step_resolve done",
        candidate_count=len(cands),
        with_snippets=sum(1 for c in cands if c.review_snippets),
    )
    return cands


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


def _fmt_price_tr(value: float | None) -> str:
    if value is None:
        return "—"
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

    seen_sites: set[str] = set()
    price_rows: list[tuple[str, float | None, str]] = []
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


_STEP_HANDLERS_V2 = {
    "resolve_candidates": _handler_resolve_candidates,
    "group_and_synthesize": _handler_group_and_synthesize,
    "format_response": _handler_format_response,
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
