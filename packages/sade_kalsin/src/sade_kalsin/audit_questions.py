"""B12 four-question audit prompt — verbatim from master synthesis §4.

The questions are deliberately blunt; the goal is to make scaffolding
defend its existence against "Mini-SWE-agent: 65% SWE-bench in 100 LOC +
bash." Per-layer rendering substitutes the layer name into Q2 only.
"""
from __future__ import annotations

from sade_kalsin.inventory import LayerReport


AUDIT_QUESTIONS: tuple[str, ...] = (
    "What does this layer do that bash + Claude can't?",
    "Last time we changed {layer} for a model-capability reason vs an integration reason — when?",
    "If we deleted it tomorrow, what test would catch it?",
    "Is the abstraction earning its keep, or did we lock in 2024-era constraints?",
)


def audit_questions_for(layer: LayerReport) -> list[str]:
    return [q.format(layer=layer.name) for q in AUDIT_QUESTIONS]
