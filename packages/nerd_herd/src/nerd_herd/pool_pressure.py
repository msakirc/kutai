"""Pure pool-pressure computation. Consumed by SystemSnapshot.pressure_for().

Supports two pool profiles via kwargs:
  per_call       threshold=0.15, depletion_max=-1.0, abundance_mode="flat"
  time_bucketed  threshold=0.30, depletion_max=-0.5, abundance_mode="time_decay"

Caller (SystemSnapshot.pressure_for) selects the profile based on model.is_free.
"""
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
    *,
    depletion_threshold: float = DEPLETION_THRESHOLD,
    depletion_max: float = -1.0,
    abundance_mode: str = "time_decay",
    abundance_max: float = 1.0,
    time_scale_secs: float = TIME_SCALE_SECS,
    exhausted_neutral: bool = False,
) -> PoolPressure:
    """Signed pressure in [-1, +1] derived from pool state.

    Parameters
    ----------
    remaining         confirmed remaining quota
    limit             total quota in current window
    reset_at          absolute epoch seconds of next reset; None -> no time_decay
    in_flight_count   calls dispatched but not yet confirmed
    depletion_threshold
        remaining_frac below this triggers depletion arm.
    depletion_max
        floor of depletion arm (e.g. -1.0 for per_call, -0.5 for time_bucketed).
    abundance_mode
        "flat"        abundance = abundance_max whenever remaining_frac >= threshold.
        "time_decay"  abundance = remaining_frac * exp(-reset_in / time_scale_secs)
                      * abundance_max. Requires reset_at.
    abundance_max
        ceiling of abundance arm.
    time_scale_secs
        decay characteristic time for "time_decay" mode.
    """
    if not limit or limit <= 0:
        return PoolPressure(0.0, 0.0, 0.0, 0.0, in_flight_count)
    effective = max(0, (remaining or 0) - in_flight_count)
    # time_bucketed pools are excluded from selection once exhausted
    # (daily_exhausted flag). No point signalling conservation for a pool
    # that's already off the board — treat as neutral.
    if exhausted_neutral and effective <= 0:
        return PoolPressure(0.0, 0.0, 0.0, 0.0, in_flight_count)
    remaining_frac = min(1.0, effective / limit)

    depletion = 0.0
    abundance = 0.0
    time_weight = 0.0

    if remaining_frac < depletion_threshold:
        intensity = (depletion_threshold - remaining_frac) / depletion_threshold
        depletion = _clamp(depletion_max * intensity, -1.0, 0.0)
    else:
        if abundance_mode == "time_decay":
            if reset_at is not None and reset_at > 0:
                reset_in = max(0.0, reset_at - time.time())
                time_weight = math.exp(-reset_in / time_scale_secs)
                abundance = _clamp(
                    remaining_frac * time_weight * abundance_max, 0.0, 1.0
                )
        elif abundance_mode == "flat":
            abundance = _clamp(abundance_max, 0.0, 1.0)

    value = _clamp(depletion + abundance)
    return PoolPressure(
        value=value,
        depletion=depletion,
        abundance=abundance,
        time_weight=time_weight,
        in_flight_count=in_flight_count,
    )
