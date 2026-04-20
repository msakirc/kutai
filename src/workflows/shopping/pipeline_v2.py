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
