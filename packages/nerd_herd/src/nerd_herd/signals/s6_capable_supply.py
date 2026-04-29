"""S6 — capable-capacity overlap.

For each capability the queue needs (vision, thinking, function_calling, hard
difficulty tier), compute the ratio of queue demand vs available capacity
across models that can serve it. If the model under evaluation IS eligible
for an under-supplied capability, it carries conserve-pressure proportional
to the shortage.

Today's S6 is a coarse first pass: takes the worst capability ratio across
demands the model can serve. Per-capability attribution weighting is left
for calibration Phase 2.
"""
from __future__ import annotations

from typing import Any


THRESHOLD = 0.70
SLOPE = 2.0


def _model_capabilities(model: Any) -> set[str]:
    caps = getattr(model, "capabilities", None) or set()
    if isinstance(caps, dict):
        return {k for k, v in caps.items() if v}
    return set(caps)


def _supply_for(capability: str, eligible_models: list, iter_avg: float) -> float:
    """Sum of remaining call-capacity across models capable of capability."""
    total = 0.0
    for m in eligible_models:
        if capability not in _model_capabilities(m):
            continue
        rem = float(getattr(m, "rpd_remaining", 0) or 0)
        total += rem * max(1.0, iter_avg)
    return total


def s6_capable_supply(
    model: Any,
    *,
    queue: dict | Any,
    eligible_models: list,
    iter_avg: float = 8.0,
) -> float:
    """Compute S6 for `model` given the current queue and capable-model pool.

    queue: dict-like with `by_capability: {cap: count}` and `by_difficulty: {d: count}`.
    eligible_models: full list of models eligible for the relevant capabilities.
    iter_avg: average iterations per task (multiplies queue counts to call demand).
    """
    by_capability = (
        queue.get("by_capability", {}) if isinstance(queue, dict)
        else getattr(queue, "by_capability", {}) or {}
    )
    if not by_capability:
        return 0.0
    model_caps = _model_capabilities(model)
    worst = 0.0
    for capability, count in by_capability.items():
        if capability not in model_caps:
            continue
        if count <= 0:
            continue
        demand = float(count) * max(1.0, iter_avg)
        supply = _supply_for(capability, eligible_models, iter_avg)
        if supply <= 0:
            continue
        ratio = demand / supply
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
    return worst
