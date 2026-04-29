"""S7 — burn-rate extrapolation.

Compare recent (5min) consumption rate × seconds-to-reset against remaining
budget. Independent of S4/S5 — captures off-queue demand and validates queue
projections against historical truth.

Cold-start (no history) → 0.
"""
from __future__ import annotations

import time

from nerd_herd.burn_log import BurnLog
from nerd_herd.types import RateLimitMatrix


THRESHOLD = 0.50
SLOPE = 2.0


def s7_burn_rate(
    matrix: RateLimitMatrix,
    *,
    provider: str,
    model: str,
    burn_log: BurnLog,
    now: float | None = None,
) -> float:
    ts = now if now is not None else time.time()
    rate = burn_log.rate(provider=provider, model=model, now=ts)
    if rate.tokens_per_min <= 0 and rate.calls_per_min <= 0:
        return 0.0
    worst = 0.0
    for axis, rl in matrix.populated_cells():
        if rl.reset_at is None or rl.reset_at <= ts:
            continue
        secs_to_reset = max(0.0, rl.reset_at - ts)
        if axis.startswith("tp") or axis.startswith("itp") or axis.startswith("otp"):
            extrapolated = rate.tokens_per_min * (secs_to_reset / 60.0)
        elif axis.startswith("rp"):
            extrapolated = rate.calls_per_min * (secs_to_reset / 60.0)
        else:
            continue
        remaining = max(1, (rl.remaining or 0) - rl.in_flight)
        ratio = extrapolated / remaining
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
    return worst
