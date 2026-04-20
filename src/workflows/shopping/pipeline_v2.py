"""Shopping pipeline v2 — trust site ordering, LLM-based grouping + review synthesis.

See docs/superpowers/specs/2026-04-21-shopping-trust-sites-synthesize-reviews-design.md
"""
from __future__ import annotations

import asyncio
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
