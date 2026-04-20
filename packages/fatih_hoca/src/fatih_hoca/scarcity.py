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

# Time-bucketed pool tunables
RESET_IMMINENT_SECS: float = 3600.0       # "imminent" threshold (1h)
RESET_FAR_SECS: float = 14400.0            # "far" threshold (4h)
TIME_BUCKETED_BOOST_MAX: float = 1.0       # max positive when burning
TIME_BUCKETED_CONSERVE_MAX: float = -0.5   # max negative when saving

# Per-call pool tunables
PER_CALL_RESERVE_MAX: float = -1.0      # strongest conservation signal
PER_CALL_ABUNDANCE_MAX: float = 1.0     # strongest "use-it" signal when budget flush + hard task
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

    # Depletion always applies: low remaining → conserve regardless of timing.
    if remaining_frac < 0.3:
        depletion = (0.3 - remaining_frac) / 0.3  # 0..1
        return _clamp(TIME_BUCKETED_CONSERVE_MAX * depletion)

    # Burn signal is continuous across the full reset horizon. A quota
    # sitting idle is waste — 24h to reset is a weaker signal than 1h,
    # but stronger than 72h. Exponential decay with 24h characteristic
    # time gives:
    #   reset_in  =   1h → weight ≈ 0.96  (strong "use it now")
    #   reset_in  =  12h → weight ≈ 0.61
    #   reset_in  =  24h → weight ≈ 0.37  (meaningful; daily quota)
    #   reset_in  =  48h → weight ≈ 0.14
    #   reset_in  =  72h → weight ≈ 0.05  (near-zero; long-horizon)
    # Scaled by remaining_frac so full-but-far pools still register,
    # while low-but-near pools don't get double-counted (depletion
    # arm already handled that above).
    import math
    time_weight = math.exp(-reset_in / 86400.0)  # 24h scale
    # Use max(time_weight, remaining_frac × some factor)? No — signal
    # stays meaningful as a product. High-remaining imminent-reset pools
    # score ~1.0; low-remaining far pools fade naturally.
    return _clamp(TIME_BUCKETED_BOOST_MAX * remaining_frac * time_weight)


def _per_call_scarcity(
    model: Any,
    snapshot: Any,
    queue_state: Any,
    task_difficulty: int,
) -> float:
    """Three arms: depletion (conservation), abundance (promotion), queue pressure.

    Arm 1 (depletion): as budget drops below 70% remaining, scarcity trends
    negative. Mirror of time_bucketed's conservation signal — but driven by
    per-call budget state, not reset timer.

    Arm 2 (abundance, NEW): when budget is flush (>70% remaining) AND the
    current task is hard (d≥7), scarcity goes positive. This is the mirror
    of time_bucketed's imminent-reset boost: both express "use-it-or-lose-
    it" — one for perishable quota (resets unused), the other for budget
    we won't get refunded on unspent. Together with the over-qualification
    dampener, abundance activates only when the task actually needs the
    expensive model — "Claude on d=8 with budget" yes, "Claude on d=3 with
    budget" no (dampened).

    Arm 3 (queue pressure): if upcoming queue has hard tasks and current
    task is easy, conserve. Overrides abundance on easy tasks because
    reserving for known hard demand is the stronger signal.

    Returns a float in [PER_CALL_RESERVE_MAX, PER_CALL_ABUNDANCE_MAX].
    """
    # ── Read remaining/limit from snapshot ───────────────────────────
    remaining_frac: float | None = None
    if model is not None and snapshot is not None:
        provider = getattr(model, "provider", "") or ""
        prov_state = getattr(snapshot, "cloud", {}) or {}
        prov_state = prov_state.get(provider) if isinstance(prov_state, dict) else None
        if prov_state is None:
            prov_state = getattr(snapshot, "cloud", None)
            if prov_state is not None and hasattr(prov_state, "get"):
                prov_state = prov_state.get(provider)
            else:
                prov_state = None
        model_id = getattr(model, "name", None) or getattr(model, "litellm_name", "")
        model_state = None
        if prov_state is not None and hasattr(prov_state, "models"):
            try:
                model_state = prov_state.models.get(model_id)
            except Exception:
                model_state = None
        source = model_state if model_state is not None else prov_state
        limits = getattr(source, "limits", None) if source is not None else None
        rpd = getattr(limits, "rpd", None) if limits is not None else None
        if rpd is not None:
            remaining = getattr(rpd, "remaining", None)
            limit = getattr(rpd, "limit", None)
            if remaining is not None and limit is not None and limit > 0:
                remaining_frac = min(1.0, max(0.0, remaining / limit))

    # ── Arm 1: depletion (conservation) ──────────────────────────────
    # Activates below 15% remaining (mutually exclusive with abundance arm
    # which activates above). Threshold kept low — budget exists to be
    # used. The arms form a signal across the budget range: full
    # abundance at 100% remaining → zero at 15% → full conservation at
    # 0%. Early depletion (30%+) was wasting hard-task coverage by
    # reserving budget that was never needed.
    DEPLETION_THRESHOLD = 0.15
    depletion_scarcity = 0.0
    if remaining_frac is not None and remaining_frac < DEPLETION_THRESHOLD:
        intensity = (DEPLETION_THRESHOLD - remaining_frac) / DEPLETION_THRESHOLD
        depletion_scarcity = PER_CALL_RESERVE_MAX * intensity

    # ── Arm 2: abundance (promotion) — NEW ────────────────────────────
    # Flush budget + hard task → positive signal. Capability wins when
    # we have headroom and the task actually needs it. Fit dampener in
    # ranking's utilization layer ensures over-qualified paid models
    # don't get boosted onto easy tasks.
    #
    # Scales smoothly across the budget range: at remaining=1.0 full
    # abundance (+1), fading to 0 at the depletion boundary (30%
    # remaining). This closes the old >70%-only "abundance gap" that
    # left the middle zone (30-70% remaining) with no signal —
    # important when a pool's limit is small (30 req/day) and many
    # hard tasks exist: all 30 should be used, not just the first 9.
    abundance_scarcity = 0.0
    if remaining_frac is not None and remaining_frac > DEPLETION_THRESHOLD and task_difficulty >= 7:
        # Full abundance as long as budget has headroom (>30% remaining).
        # User's framing: "if quota is available, capability wins." A
        # fading curve leaves a middle zone where capability loses to
        # base-ranking noise on razor-thin margins. Keep the push strong
        # until the depletion arm takes over at 30% remaining.
        abundance_scarcity = PER_CALL_ABUNDANCE_MAX

    # ── Arm 3: queue pressure (conservation on easy tasks) ───────────
    queue_scarcity = 0.0
    if queue_state is not None and task_difficulty < 7:
        total = int(getattr(queue_state, "total_tasks", 0) or 0)
        hard = int(getattr(queue_state, "hard_tasks_count", 0) or 0)
        if total > 0 and hard > 0:
            hard_ratio = hard / total
            pressure = min(1.0, hard_ratio / PER_CALL_HARD_QUEUE_RATIO)
            easiness = max(0.0, (7 - task_difficulty)) / 6.0
            queue_scarcity = PER_CALL_RESERVE_MAX * pressure * easiness

    # ── Combine ────────────────────────────────────────────────────────
    # Conservation signals (negative) stack — take the most conservative.
    # Abundance (positive) is mutually exclusive with depletion by
    # construction (one requires <70% remaining, the other >70%).
    # Queue pressure activates only on easy tasks where abundance is off.
    neg_signals = [s for s in (depletion_scarcity, queue_scarcity) if s < 0]
    if neg_signals:
        return _clamp(min(neg_signals))
    return _clamp(abundance_scarcity)


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
        return _per_call_scarcity(model, snapshot, queue_state, task_difficulty)
    return 0.0
