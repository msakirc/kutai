"""Beckman's retry policy (replaces the inline `_quality_retry_flow`).

Pure decision function. Callers (apply.py) do the DB work and DLQ writes.

The quality bonus-attempt heuristic is preserved (flagged in the spec for
a sideways look during migration, but not removed — it solves real
DLQ-too-eagerly incidents).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

_BACKOFF_SECONDS = [0, 10, 30, 120, 600]
# "no_model" is the dispatcher's signal that fatih_hoca.select() returned
# None — every cloud provider rate-limited / dead AND local pool busy.
# This is environmental scarcity, not a worker bug. Persistent saturation
# (e.g. gemini free-tier 20-req/day cap hit at noon → unavailable until
# midnight reset, ~12h) is common, so the budget needs to span much
# longer than a single quota window. The earlier 10-attempt × 30/60/120/
# 300/600 ladder DLQ'd at ~58min — well before a daily reset. Production
# 2026-05-02 14:44-15:51: 20 no_model tasks DLQ'd at exactly 10/10
# attempts.
#
# Rebalanced: cap backoff at 1h (3600s), allow up to 30 attempts. Total
# patience window ~25h before DLQ, which spans any single-day reset.
# accelerate_retries (Beckman.on_model_swap / KDV capacity_restored
# events) wake deferred tasks early when capacity actually frees, so
# the 1h cap is just an upper bound, not a typical wait.
_NO_MODEL_BACKOFF_SECONDS = [
    30, 60, 120, 300, 600, 1200, 1800, 3600,
    3600, 3600, 3600, 3600, 3600, 3600, 3600,
    3600, 3600, 3600, 3600, 3600, 3600, 3600,
    3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600,
]
_NO_MODEL_MAX_ATTEMPTS = 30
_MAX_BONUS = 2


@dataclass(frozen=True)
class RetryDecision:
    action: str  # "immediate" | "delayed"
    delay_seconds: int = 0
    bonus_used: bool = False


@dataclass(frozen=True)
class DLQAction:
    action: str = "dlq"
    category: str = "unknown"
    reason: str = ""


Decision = Union[RetryDecision, DLQAction]


def decide_retry(
    failure: dict,
    progress: float | None = None,
    bonus_count: int = 0,
) -> Decision:
    """Decide whether to retry a failed task.

    ``failure`` carries category, worker_attempts, max_worker_attempts, model.
    ``progress`` is the executor's self-assessed progress (0.0–1.0). Only
    considered when the category is ``quality`` and the task is otherwise
    exhausted.
    ``bonus_count`` is the number of bonus attempts already granted for
    this task (lives in task_ctx). Capped at ``_MAX_BONUS``.

    Backoff is for transient/availability failures (rate limits, network
    timeouts, model-swap failures) where wall-clock waiting lets the
    underlying condition clear. Quality failures (schema validation,
    grader rejection, degenerate output) are deterministic — same input
    plus same model produces the same failure. Backing off just burns
    real time without changing the next attempt's outcome. Quality
    retries fire immediately so the new sampling pass / retry-hint
    checklist / agent_type-refresh has a chance to land a different
    output sooner.
    """
    attempts = int(failure.get("worker_attempts", 0))
    max_attempts = int(failure.get("max_worker_attempts", 3))
    category = failure.get("category", "unknown")

    # "no_model" (dispatcher's pick=None signal) is pure environmental
    # backpressure — the worker never got a model to call. User design
    # call 2026-05-02 18:15 UTC: "Sleeping tasks awake with wake signals,
    # but there is only one picking mechanism."
    #
    # Architecture: when the pool is empty, the task sleeps with a small
    # constant backoff. Wake signals (Beckman.on_model_swap, KDV
    # capacity_restored events) accelerate the next_retry_at. The single
    # picker (Beckman.next_task) re-evaluates each ready task on every
    # tick — if the pool is still empty, the picker skips silently
    # without burning anything. If the pool has capacity, the task
    # admits naturally.
    #
    # Backoff is just a "don't hammer selector with the same id every
    # 3s" damper, not a budget. No counter. No DLQ. The single-picker
    # semantics make a permanent-empty pool a no-op (task sits pending
    # forever) which is the right behaviour — a structural failure to
    # have ANY model is beyond a single task's responsibility to flag.
    if category == "no_model":
        return RetryDecision(action="delayed", delay_seconds=60)

    if attempts < max_attempts:
        # Quality / deterministic failures: immediate retry, no backoff.
        # The next attempt benefits from the per-artifact retry-hint
        # checklist (hooks.py), possible model swap (failed_models),
        # and potential agent_type refresh from live JSON (orchestrator).
        # None of those benefit from waiting.
        if category == "quality":
            return RetryDecision(action="immediate", delay_seconds=0)

        # Availability / transient: exponential backoff.
        # attempts=1 (first failure) → idx=0 → immediate;
        # attempts=2 → idx=1 → 10s; etc.
        idx = min(max(0, attempts - 1), len(_BACKOFF_SECONDS) - 1)
        delay = _BACKOFF_SECONDS[idx]
        return RetryDecision(
            action="immediate" if delay == 0 else "delayed",
            delay_seconds=delay,
        )

    # Exhausted. Consider quality bonus.
    if (
        category == "quality"
        and progress is not None
        and progress >= 0.5
        and bonus_count < _MAX_BONUS
    ):
        return RetryDecision(action="immediate", bonus_used=True)

    return DLQAction(
        category=category,
        reason=failure.get("error", "")[:300] or f"exhausted after {attempts} attempts",
    )
