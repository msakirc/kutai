"""S5 — queue request pressure.

Same shape as S4 but on the request axis. Per-task call cost = est_iterations
(folded into queue.projected_calls at queue_profile_push time).
"""
from __future__ import annotations

from nerd_herd.signals.s4_queue_tokens import SLOPE, THRESHOLD
from nerd_herd.types import QueueProfile, RateLimitMatrix


def s5_queue_calls(
    matrix: RateLimitMatrix, *, queue: QueueProfile,
    fleet_remaining: dict[str, int] | None = None,
) -> float:
    projected = queue.projected_calls
    if projected <= 0:
        return 0.0
    worst = 0.0
    # Cycle axes only (excludes rpm). Denominator is the FLEET's cycle-remaining
    # on this request axis (see s4_queue_tokens) — fleet_remaining=None / axis
    # absent -> per-model fallback (fleet-of-one / unit tests == old behavior).
    for name, rl in matrix.cycle_request_cells():
        if fleet_remaining is not None and name in fleet_remaining:
            remaining = fleet_remaining[name]
        else:
            remaining = max(0, (rl.remaining or 0) - rl.in_flight)
        if remaining <= 0:
            continue
        ratio = projected / remaining
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
    return worst
