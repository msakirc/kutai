"""S1 — per-axis remaining pressure.

For every populated cell in a RateLimitMatrix:
  effective = max(0, remaining - in_flight)
  frac = effective / limit
  Map to two-arm pressure curve based on pool profile.

Fold:
  - If any cell is negative: take min (worst-axis-wins for depletion).
  - Else: take max (best-axis-wins for abundance).
"""
from __future__ import annotations

import math

from nerd_herd.types import RateLimit, RateLimitMatrix


# Profile-driven thresholds
PROFILE_PARAMS: dict[str, dict[str, float]] = {
    # per_call (paid cloud, pay-per-token): conservation-only. Positive
    # abundance lives exclusively in S9 right-tool-perishability for hard
    # tasks (d>=7). Earlier abundance_max=1.0 made paid cloud win even
    # easy router/classifier calls on remaining budget alone, wasting
    # quota; ranking.py used to gate that with a d<7 suppression band-aid
    # — root fix is here so the signal contract stays clean (S1 = stress,
    # S9 = right-tool).
    "per_call":      {"depletion_threshold": 0.15, "depletion_max": -1.0,
                      "abundance_mode": "flat", "abundance_max": 0.0,
                      "time_scale_secs": 86400.0, "exhausted_neutral": False},
    # time_bucketed (free cloud, periodic reset): when remaining=0 we
    # MUST fire negative — pre-2026-04-30 production triage showed
    # exhausted_neutral=True silently zeroed S1 when daily quota hit 0,
    # selector kept picking the model, dispatcher kept 429'ing on
    # KDV.pre_call's daily_exhausted gate, task DLQ'd. Returning -1.0
    # via depletion_max (frac=0 < depletion_threshold=0.30 → intensity=1
    # → depletion_max applied) makes selector route to peers naturally.
    "time_bucketed": {"depletion_threshold": 0.30, "depletion_max": -1.0,
                      "abundance_mode": "time_decay", "abundance_max": 1.0,
                      "time_scale_secs": 86400.0, "exhausted_neutral": False},
}


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _cell_pressure(rl: RateLimit, *, reset_in_secs: float, profile: str) -> float:
    if rl.limit is None or rl.limit <= 0:
        return 0.0
    p = PROFILE_PARAMS[profile]
    effective = max(0, (rl.remaining or 0) - rl.in_flight)
    if p["exhausted_neutral"] and effective <= 0:
        return 0.0
    frac = min(1.0, effective / rl.limit)

    if frac < p["depletion_threshold"]:
        intensity = (p["depletion_threshold"] - frac) / p["depletion_threshold"]
        return _clamp(p["depletion_max"] * intensity, -1.0, 0.0)

    # Abundance arm
    if p["abundance_mode"] == "time_decay":
        if rl.reset_at and rl.reset_at > 0:
            time_weight = math.exp(-max(0.0, reset_in_secs) / p["time_scale_secs"])
            return _clamp(frac * time_weight * p["abundance_max"], 0.0, 1.0)
        return 0.0
    elif p["abundance_mode"] == "flat":
        return _clamp(p["abundance_max"], 0.0, 1.0)
    return 0.0


def s1_remaining(
    matrix: RateLimitMatrix,
    *,
    reset_in_secs: float = 0,
    in_flight: int = 0,
    profile: str = "per_call",
) -> float:
    """Compute S1 across all populated cells. Fold = worst-of-negatives or max-of-positives."""
    cell_pressures = [_cell_pressure(rl, reset_in_secs=reset_in_secs, profile=profile)
                      for _, rl in matrix.populated_cells()]
    if not cell_pressures:
        return 0.0
    negs = [p for p in cell_pressures if p < 0]
    if negs:
        return min(negs)
    return max(cell_pressures, default=0.0)
