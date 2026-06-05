"""Purpose-built image-model scorer. Sibling to selector.py (text). Cloud-only
in Plan 1; eviction-cost is a stub and never fires because the catalog has no
local entries (Plan 2 adds it)."""
from __future__ import annotations

import os

from .image_providers import image_catalog
from .registry import ImageModelInfo
from .types import Pick, SelectionFailure


def _provider_available(m: ImageModelInfo, hf_available: bool | None) -> bool:
    if m.provider == "huggingface":
        return os.getenv("HF_TOKEN") is not None if hf_available is None else hf_available
    if m.provider == "pollinations":
        return True
    return False  # local providers ineligible in Plan 1


def _eviction_cost(m: ImageModelInfo) -> float:
    """Stub — replaced in Plan 2 with the real formula reading nerd_herd.
    Plan 1 is cloud-only; cloud providers always score 0 here."""
    return 0.0


def select_image(
    *,
    quality_tier: str = "fast",
    failures: list[str] | None = None,
    hf_available: bool | None = None,
    remaining_budget_usd: float | None = None,
) -> Pick | SelectionFailure:
    # IMPORTANT: failures is a list of STRING provider/model names.
    failed = set(failures or [])
    candidates: list[tuple[float, ImageModelInfo]] = []
    for m in image_catalog():
        if m.name in failed:
            continue
        if not _provider_available(m, hf_available):
            continue
        if remaining_budget_usd is not None and m.cost_per_image > remaining_budget_usd:
            continue
        score = m.quality_rank - _eviction_cost(m)
        candidates.append((score, m))

    if not candidates:
        return SelectionFailure(reason="availability",
                                detail="no eligible image provider")
    candidates.sort(key=lambda t: t[0], reverse=True)
    top_summary = "; ".join(f"{m.name}:{s:.1f}" for s, m in candidates[:5])
    best_score, best = candidates[0]
    return Pick(model=best, min_time_seconds=0.0, score=best_score, top_summary=top_summary)
