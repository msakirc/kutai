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

import time
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
# Conservation: loaded local on a task above this difficulty gets a
# negative scarcity so the equation demotes it in favor of cloud
# candidates that clear the capability bar. Gated by cap curve
# comparison: we only demote when the local is actually below
# `cap_needed_for_difficulty`.
LOCAL_HARD_TASK_PENALTY: float = -1.0

# Time-bucketed pool tunables
RESET_IMMINENT_SECS: float = 3600.0       # "imminent" threshold (1h)
RESET_FAR_SECS: float = 14400.0            # "far" threshold (4h)
TIME_BUCKETED_BOOST_MAX: float = 1.0       # max positive when burning
TIME_BUCKETED_CONSERVE_MAX: float = -0.5   # max negative when saving

# Per-call pool tunables
PER_CALL_RESERVE_MAX: float = -1.0   # strongest conservation signal
PER_CALL_HARD_QUEUE_RATIO: float = 0.1  # 10% hard tasks in queue → strong pressure


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


def _time_bucketed_scarcity(model: Any, snapshot: Any) -> float:
    provider = getattr(model, "provider", "") or ""
    prov_state = getattr(snapshot, "cloud", {}).get(provider)
    if prov_state is None:
        return 0.0

    model_id = getattr(model, "name", None) or getattr(model, "litellm_name", "")
    model_state = prov_state.models.get(model_id) if hasattr(prov_state, "models") else None
    source = model_state if model_state is not None else prov_state

    limits = getattr(source, "limits", None)
    if limits is None:
        return 0.0
    rpd = getattr(limits, "rpd", None)
    if rpd is None:
        return 0.0

    remaining = getattr(rpd, "remaining", None)
    limit = getattr(rpd, "limit", None)
    reset_at = getattr(rpd, "reset_at", None)
    if remaining is None or limit is None or limit <= 0 or remaining <= 0:
        return 0.0

    remaining_frac = min(1.0, remaining / limit)

    if reset_at is not None and reset_at > 0:
        reset_in = max(0.0, reset_at - time.time())
    else:
        return 0.0

    if reset_in <= RESET_IMMINENT_SECS:
        # Reset imminent: "use it or lose it"
        # proximity: 0 at full hour remaining → 1 at reset moment
        proximity = 1.0 - (reset_in / RESET_IMMINENT_SECS)  # 0..1
        # Use max(proximity, remaining_frac) so that high remaining_frac
        # alone (even with modest proximity) produces a strong signal.
        # Example: reset_in=1800 (proximity=0.5), remaining_frac=0.85
        #   → max(0.5, 0.85) = 0.85, result = 1.0 × 0.85 = 0.85 ✓ (>= 0.6)
        return _clamp(TIME_BUCKETED_BOOST_MAX * max(proximity, remaining_frac))

    if reset_in >= RESET_FAR_SECS:
        # Reset far: conserve when remaining fraction is low (< 30%)
        if remaining_frac < 0.3:
            depletion = (0.3 - remaining_frac) / 0.3  # 0..1
            return _clamp(TIME_BUCKETED_CONSERVE_MAX * depletion)
        # Daily quota that'll otherwise reset unused — encourage burn
        # proportional to remaining fraction. A full bucket gets the full
        # soft boost; as we draw it down, scarcity decays toward neutral.
        return _clamp(TIME_BUCKETED_BOOST_MAX * remaining_frac * 0.75)

    # Between imminent and far: neutral
    return 0.0


def _per_call_scarcity(queue_state: Any, task_difficulty: int) -> float:
    if queue_state is None:
        return 0.0
    total = int(getattr(queue_state, "total_tasks", 0) or 0)
    hard = int(getattr(queue_state, "hard_tasks_count", 0) or 0)
    if total <= 0 or hard <= 0:
        return 0.0

    # If the CURRENT task is itself hard, no reason for it to be rationed
    if task_difficulty >= 7:
        return 0.0

    hard_ratio = hard / total
    # Saturate pressure at PER_CALL_HARD_QUEUE_RATIO
    pressure = min(1.0, hard_ratio / PER_CALL_HARD_QUEUE_RATIO)
    # Scale by how far below "hard" the current task is (d=1 → full, d=7 → 0)
    easiness = max(0.0, (7 - task_difficulty)) / 6.0  # d=1→1.0, d=7→0
    return _clamp(PER_CALL_RESERVE_MAX * pressure * easiness)


def pool_scarcity(
    model: Any,
    snapshot: Any,
    queue_state: Any = None,
    task_difficulty: int = 0,
) -> float:
    """Compute signed scarcity in [-1, +1].

    Parameters
    ----------
    model : ModelInfo-like
        Must expose `is_local`, `is_free`, `provider`, `name`.
    snapshot : SystemSnapshot-like
        Has `.local` and `.cloud` attrs.
    queue_state : QueueProfile or None
        Optional; used by per_call branch.
    task_difficulty : int
        Current task difficulty (1-10); used by per_call branch.
    """
    pool = classify_pool(model)
    if pool is Pool.LOCAL:
        return _local_scarcity(model, snapshot)
    if pool is Pool.TIME_BUCKETED:
        return _time_bucketed_scarcity(model, snapshot)
    if pool is Pool.PER_CALL:
        return _per_call_scarcity(queue_state, task_difficulty)
    return 0.0
