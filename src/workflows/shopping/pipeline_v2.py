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

    cands: list[Candidate] = []
    for site, products in by_site.items():
        for rank, p in enumerate(products[:per_site_n], start=1):
            cands.append(
                Candidate(
                    title=str(_attr(p, "name") or ""),
                    site=site,
                    site_rank=rank,
                    price=(_attr(p, "discounted_price") or _attr(p, "original_price") or _attr(p, "price")),
                    original_price=_attr(p, "original_price"),
                    url=str(_attr(p, "url") or ""),
                    rating=_attr(p, "rating"),
                    review_count=_attr(p, "review_count"),
                    review_snippets=list(_attr(p, "review_snippets") or []),
                )
            )
    logger.info("step_resolve done", candidate_count=len(cands))
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
