"""Cloud benchmark match + review-gate logic.

Family-aware overlay on top of existing AA enrichment. For each cloud
ModelInfo, look up its family in the supplied ``aa_lookup``. Store match
in ``ModelInfo.benchmark_scores`` regardless. Promote to active
``ModelInfo.capabilities`` only when the family appears in the
operator-approved list at ``.benchmark_cache/cloud_match_approved.json``.

Until a family is approved, capabilities continue to come from the
fallback chain (CLOUD_PROFILES name-substring match → flat default)
which was applied at registration time by ``detect_cloud_model()``.

This module does NOT compute benchmarks itself — it consumes an
``aa_lookup`` supplied by the caller (typically built from the local
bench enricher's output keyed by family).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import logging

from .registry import ModelInfo

logger = logging.getLogger("fatih_hoca.benchmark_cloud_match")


def _load_approved(path: Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()))
    except Exception as e:  # noqa: BLE001
        logger.warning("failed to load approved families from %s: %s", p, e)
        return set()


def is_family_approved(family: str, approved_path: Path) -> bool:
    return family in _load_approved(approved_path)


def apply_cloud_benchmarks(
    models: Iterable[ModelInfo],
    aa_lookup: Mapping[str, Mapping[str, float]],
    approved_path: Path,
) -> None:
    """For each cloud model with a family, set ``benchmark_scores`` from
    ``aa_lookup``. Promote to active ``capabilities`` only for approved families.

    Local models, cloud models without a family, and families with no AA
    hit are left untouched.
    """
    approved = _load_approved(Path(approved_path))
    for m in models:
        if m.location != "cloud":
            continue
        if not m.family:
            continue
        scores = aa_lookup.get(m.family)
        if not scores:
            continue
        m.benchmark_scores = dict(scores)
        if m.family in approved:
            for cap, score in scores.items():
                m.capabilities[cap] = float(score)


def write_review_artifact(
    models: Iterable[ModelInfo],
    aa_lookup: Mapping[str, Mapping[str, float]],
    output_path: Path,
) -> None:
    """Dump (litellm_name, family, matched_aa_entry, source, final_caps) per
    cloud model so a human can sanity-check before flipping approval."""
    rows = []
    for m in models:
        if m.location != "cloud":
            continue
        aa_match = aa_lookup.get(m.family) if m.family else None
        rows.append({
            "litellm_name": m.litellm_name,
            "family": m.family,
            "matched_aa_entry": m.family if aa_match else None,
            "source": "aa" if aa_match else "profile_or_default",
            "final_capabilities": dict(m.capabilities),
            "benchmark_scores": dict(m.benchmark_scores or {}),
        })
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    logger.info("cloud match review artifact written to %s (%d rows)", out, len(rows))
