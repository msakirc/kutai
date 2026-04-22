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
    # Supply side relocated to nerd_herd. pressure_for() picks the
    # time_bucketed profile (threshold 0.3, depletion_max -0.5,
    # abundance_mode time_decay) based on model.is_free.
    if snapshot is None or not hasattr(snapshot, "pressure_for"):
        return 0.0
    return _clamp(snapshot.pressure_for(model))


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
    # ── Supply signal (depletion + abundance) from nerd_herd ─────────
    # pressure_for() picks the per_call profile (threshold 0.15,
    # depletion_max -1.0, abundance_mode flat) based on model.is_free.
    supply: float = 0.0
    if snapshot is not None and hasattr(snapshot, "pressure_for"):
        supply = float(snapshot.pressure_for(model) or 0.0)

    # Demand-side gate: abundance (+) only applies when the task is
    # actually hard enough to need the expensive model. Dampener in the
    # utilization layer handles fine-grained fit, but suppressing
    # abundance on easy tasks avoids boosting paid-cloud onto trivial
    # work regardless of fit noise.
    if supply > 0 and task_difficulty < 7:
        supply = 0.0

    depletion_scarcity = min(0.0, supply)
    abundance_scarcity = max(0.0, supply)

    # ── Arm 3: queue pressure (conservation on easy tasks) ───────────
    # Accept either the local `fatih_hoca.requirements.QueueProfile`
    # (`total_tasks`) or `nerd_herd.types.QueueProfile` pushed by
    # Beckman (`total_ready_count`). If `queue_state` is absent, fall
    # back to snapshot.queue_profile — wired automatically once Beckman
    # pushes land.
    qp = queue_state
    if qp is None:
        qp = getattr(snapshot, "queue_profile", None)
    queue_scarcity = 0.0
    if qp is not None and task_difficulty < 7:
        total = int(
            getattr(qp, "total_tasks", 0)
            or getattr(qp, "total_ready_count", 0)
            or 0
        )
        hard = int(getattr(qp, "hard_tasks_count", 0) or 0)
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
