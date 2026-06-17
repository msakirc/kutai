"""S5 — queue request pressure.

Same shape as S4 but on the request axis. Per-task call cost = est_iterations
(folded into queue.projected_calls at queue_profile_push time).
"""
from __future__ import annotations

from nerd_herd.signals.s4_queue_tokens import SLOPE, THRESHOLD
from nerd_herd.types import QueueProfile, RateLimitMatrix


def s5_queue_calls(matrix: RateLimitMatrix, *, queue: QueueProfile) -> float:
    projected = queue.projected_calls
    if projected <= 0:
        return 0.0
    worst = 0.0
    # Cycle axes only (excludes rpm): a per-minute request window paces, it does
    # not conserve. Per-minute pacing is owned by lane caps + in-flight; the
    # genuine conserve case is daily exhaustion (e.g. gemini 20/day = rpd).
    for _, rl in matrix.cycle_request_cells():
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
