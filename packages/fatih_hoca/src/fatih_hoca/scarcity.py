"""Pool scarcity signal for Phase 2d unified utilization equation.

Returns a float in [-1, +1] describing the opportunity cost of using
a given model right now:

    +1   "use it or lose it" — time_bucketed pool with reset imminent
     0   neutral — no preference
    -1   "conserve" — per_call pool with hard tasks queued

Consumed by ranking._apply_utilization_layer as:
    composite *= 1 + UTILIZATION_K * scarcity * (1 - max(0, fit_excess))
"""
from __future__ import annotations

from typing import Any

from fatih_hoca.pools import (
    LOCAL_IDLE_SATURATION_SECS,
    Pool,
    classify_pool,
)

# Soft cap on local-idle scarcity (matches spec §4 range 0.3-0.5)
LOCAL_IDLE_SCARCITY_MAX: float = 0.5
# Penalty when a loaded local is actively processing another request
LOCAL_BUSY_PENALTY: float = -0.10


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _local_scarcity(model: Any, snapshot: Any) -> float:
    local = getattr(snapshot, "local", None)
    if local is None:
        return 0.0

    loaded_name = getattr(local, "model_name", "") or ""
    is_this_model_loaded = (
        getattr(model, "is_loaded", False)
        and loaded_name == getattr(model, "name", None)
    )

    if is_this_model_loaded:
        requests_processing = int(getattr(local, "requests_processing", 0) or 0)
        if requests_processing > 0:
            return LOCAL_BUSY_PENALTY

        idle = float(getattr(local, "idle_seconds", 0.0) or 0.0)
        if idle <= 0:
            return 0.0
        frac = min(1.0, idle / LOCAL_IDLE_SATURATION_SECS)
        return _clamp(frac * LOCAL_IDLE_SCARCITY_MAX)

    # Not loaded — no idle signal, neutral
    return 0.0


def pool_scarcity(model: Any, snapshot: Any, queue_state: Any = None) -> float:
    """Compute signed scarcity in [-1, +1] for (model, snapshot, queue_state)."""
    pool = classify_pool(model)
    if pool is Pool.LOCAL:
        return _local_scarcity(model, snapshot)
    # Time-bucketed + per_call added in Tasks 4 + 5
    return 0.0
