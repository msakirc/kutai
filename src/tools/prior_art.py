"""Z1 Tier 6B (P5) — LLM-callable prior-art search tool.

Thin async wrapper around :func:`vecihi.find_prior_art`. Returns a JSON
string so it slots into the existing tool-result pipe (every tool returns
``str``; agents parse JSON).
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("tools.prior_art")


async def find_prior_art_tool(
    idea_summary: str,
    domain_keywords: str | list[str] = "",
    k: int = 10,
    ambition_tier: str = "private_beta",
) -> str:
    """Run a web-grounded prior-art search.

    Args:
        idea_summary: One-paragraph summary of the idea (drives query
            construction + relevance split).
        domain_keywords: Either a comma-separated string of keywords
            (the LLM may emit a string) or a list of keyword strings.
        k: Top-k attempted_solutions to surface (default 10).
        ambition_tier: ``"private_beta"`` / ``"public_launch"`` /
            ``"revenue_product"``. Higher tiers add Wayback validation
            and Product Hunt probes.

    Returns:
        JSON string with the prior_art_report schema. On error returns a
        JSON object with an ``error`` key.
    """
    try:
        from vecihi.prior_art import find_prior_art
    except Exception as exc:  # pragma: no cover — import path issue
        return json.dumps({"error": f"prior_art module unavailable: {exc}"})

    if isinstance(domain_keywords, str):
        keywords = [
            k.strip() for k in domain_keywords.split(",") if k.strip()
        ]
    else:
        keywords = [str(k).strip() for k in (domain_keywords or []) if str(k).strip()]

    try:
        report: dict[str, Any] = await find_prior_art(
            idea_summary=idea_summary or "",
            domain_keywords=keywords,
            k=int(k or 10),
            ambition_tier=str(ambition_tier or "private_beta"),
        )
        return json.dumps(report, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.warning("find_prior_art failed: %s", exc)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
