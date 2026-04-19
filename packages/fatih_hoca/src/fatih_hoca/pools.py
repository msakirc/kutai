"""Pool taxonomy and urgency math for Fatih Hoca Phase 2c.

Three pools: LOCAL (sunk-cost GPU), TIME_BUCKETED (free-tier or prepaid cloud
with reset timer), PER_CALL (paid cloud, no bucket).

Urgency in [0, 1]. Applied at Layer 3 in ranking.py as:
    composite *= (1 + UTILIZATION_K * urgency)
when capability gate allows.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any

# ── Tunables ───────────────────────────────────────────────────────────
UTILIZATION_K: float = 0.5
LOCAL_IDLE_SATURATION_SECS: float = 600.0
RESET_HORIZON_SECS: float = 3600.0


class Pool(str, Enum):
    LOCAL = "local"
    TIME_BUCKETED = "time_bucketed"
    PER_CALL = "per_call"


def classify_pool(model: Any) -> Pool:
    """Classify a ModelInfo-like object into its utilization pool."""
    if getattr(model, "is_local", False):
        return Pool.LOCAL
    if getattr(model, "is_free", False):
        return Pool.TIME_BUCKETED
    # Future: prepaid cloud lands here too via `getattr(model, "prepaid_remaining", 0) > 0`
    return Pool.PER_CALL


def _midnight_utc_reset_in_seconds() -> float:
    """Fallback: assume quotas reset at 00:00 UTC daily."""
    now = time.time()
    seconds_today = now % 86400
    return 86400 - seconds_today


def _bucketed_urgency(model: Any, snapshot: Any) -> float:
    provider = getattr(model, "provider", "") or ""
    prov_state = getattr(snapshot, "cloud", {}).get(provider)
    if prov_state is None:
        return 0.0

    # Prefer per-model limits, fall back to provider-wide
    model_id = getattr(model, "name", None) or getattr(model, "litellm_name", "")
    model_state = prov_state.models.get(model_id) if hasattr(prov_state, "models") else None
    source = model_state if model_state is not None else prov_state

    limits = getattr(source, "limits", None)
    if limits is None:
        return 0.0
    # Daily bucket (rpd). If unavailable, tpm/rpm can be added later.
    rpd = getattr(limits, "rpd", None)
    if rpd is None:
        return 0.0

    remaining = getattr(rpd, "remaining", None)
    limit = getattr(rpd, "limit", None)
    reset_at = getattr(rpd, "reset_at", None)

    if remaining is None or limit is None or limit <= 0:
        return 0.0
    if remaining <= 0:
        return 0.0

    if reset_at is not None and reset_at > 0:
        reset_in = max(0.0, reset_at - time.time())
    else:
        reset_in = _midnight_utc_reset_in_seconds()

    remaining_frac = min(1.0, max(0.0, remaining / limit))
    reset_proximity = max(0.0, 1.0 - min(1.0, reset_in / RESET_HORIZON_SECS))
    return remaining_frac * reset_proximity


def _local_urgency(snapshot: Any) -> float:
    local = getattr(snapshot, "local", None)
    if local is None:
        return 0.0
    idle = float(getattr(local, "idle_seconds", 0.0) or 0.0)
    if idle <= 0:
        return 0.0
    return min(1.0, idle / LOCAL_IDLE_SATURATION_SECS)


def compute_urgency(model: Any, snapshot: Any) -> float:
    """Return urgency in [0, 1] for this model under the current snapshot."""
    pool = classify_pool(model)
    if pool is Pool.LOCAL:
        return _local_urgency(snapshot)
    if pool is Pool.TIME_BUCKETED:
        return _bucketed_urgency(model, snapshot)
    return 0.0
