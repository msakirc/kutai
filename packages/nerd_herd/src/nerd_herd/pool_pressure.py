"""Pure pool-pressure computation. Consumed by SystemSnapshot.pressure_for()."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

DEPLETION_THRESHOLD = 0.15
TIME_SCALE_SECS = 86400.0


@dataclass
class PoolPressure:
    value: float
    depletion: float
    abundance: float
    time_weight: float
    in_flight_count: int


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_pool_pressure(
    remaining: int | None,
    limit: int | None,
    reset_at: int | None,
    in_flight_count: int,
) -> PoolPressure:
    """Signed pressure in [-1, +1] derived from pool state.

    Inputs:
      remaining         confirmed remaining quota
      limit             total quota in current window
      reset_at          absolute epoch seconds of next reset; None -> no time weight
      in_flight_count   calls dispatched but not yet confirmed
    """
    if not limit or limit <= 0:
        return PoolPressure(0.0, 0.0, 0.0, 0.0, in_flight_count)
    effective = max(0, (remaining or 0) - in_flight_count)
    remaining_frac = min(1.0, effective / limit)

    depletion = 0.0
    abundance = 0.0
    time_weight = 0.0

    if remaining_frac < DEPLETION_THRESHOLD:
        intensity = (DEPLETION_THRESHOLD - remaining_frac) / DEPLETION_THRESHOLD
        depletion = _clamp(-1.0 * intensity, -1.0, 0.0)
    elif reset_at is not None and reset_at > 0:
        reset_in = max(0.0, reset_at - time.time())
        time_weight = math.exp(-reset_in / TIME_SCALE_SECS)
        abundance = _clamp(remaining_frac * time_weight, 0.0, 1.0)

    value = _clamp(depletion + abundance)
    return PoolPressure(
        value=value,
        depletion=depletion,
        abundance=abundance,
        time_weight=time_weight,
        in_flight_count=in_flight_count,
    )
