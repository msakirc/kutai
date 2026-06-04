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

from nerd_herd.types import RateLimit, RateLimitMatrix


# Profile-driven thresholds
#
# Signal contract (2026-05-03 separation):
#   S1 = STOCK    — how much capacity remains (frac of limit)
#   S9 = TIMING   — how soon what's left will vanish (proximity to reset)
# Pre-2026-05-03 S1 also baked in time_decay for time_bucketed pools,
# duplicating S9's job. Now S1 is purely stock-based; abundance arm
# returns frac × abundance_max (proportional). S9 alone owns timing.
# combine.py composes S1+ and S9+ via noisy-OR so reinforcement when
# both fire is preserved.
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
                      "exhausted_neutral": False},
    # time_bucketed (free cloud, periodic reset): when remaining=0 we
    # MUST fire negative — pre-2026-04-30 production triage showed
    # exhausted_neutral=True silently zeroed S1 when daily quota hit 0,
    # selector kept picking the model, dispatcher kept 429'ing on
    # KDV.pre_call's daily_exhausted gate, task DLQ'd. Returning -1.0
    # via depletion_max (frac=0 < depletion_threshold=0.30 → intensity=1
    # → depletion_max applied) makes selector route to peers naturally.
    # Abundance arm is "proportional" (frac × abundance_max) so S1 carries
    # only the stock signal — timing is now S9's job exclusively.
    #
    # abundance_max 1.0→0.0 (2026-06-04): frac-of-own-limit abundance is
    # scale-invariant — a giant-tank provider (groq 14400/day, gemini
    # 1500/day) sits near frac=1.0 indefinitely and pinned the noisy-OR
    # positive arm at ~1.0, leaving NO headroom for the fleet-comparative
    # allocation signal (S12) to steer load toward under-used providers.
    # Frac is now conservation-ONLY (depletion arm unchanged); positive
    # pull lives in S9 (timing) + S12 (absolute fleet-relative under-use).
    # See docs/superpowers/specs/2026-06-04-cloud-utilization-continuity-design.md.
    "time_bucketed": {"depletion_threshold": 0.30, "depletion_max": -1.0,
                      "abundance_mode": "proportional", "abundance_max": 0.0,
                      "exhausted_neutral": False},
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
    if p["abundance_mode"] == "proportional":
        # Pure stock: returns frac of capacity, no timing component.
        # S9 owns timing.
        return _clamp(frac * p["abundance_max"], 0.0, 1.0)
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
