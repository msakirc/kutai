"""Beckman's retry policy (replaces the inline `_quality_retry_flow`).

Pure decision function. Callers (apply.py) do the DB work and DLQ writes.

The quality bonus-attempt heuristic is preserved (flagged in the spec for
a sideways look during migration, but not removed — it solves real
DLQ-too-eagerly incidents).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

# Shared availability/transient backoff ladder. 15 entries; the tail
# (2h → 24h) lets daily_exhausted ride out a real quota-reset window
# rather than DLQ'ing inside a 1h cap. accelerate_retries (Beckman.on_
# model_swap / KDV capacity_restored events) wakes deferred tasks early
# when capacity actually frees, so longer ladder steps are upper bounds
# rather than typical waits — the long tail only affects the no-recovery
# worst case.
#
# Cumulative wall-clock from att=1 to DLQ at att=15: ~2 days. Production
# 2026-05-03: tasks DLQ'd in ~4min on a 10-step + max=6 combo because
# total backoff bottomed out at 220s. Lengthening the ladder closes
# that gap; the larger fix (max_worker_attempts default 6→15) keeps
# attempts in lockstep with the new ladder size.
_BACKOFF_SECONDS = [
    0, 10, 30, 60, 120,             # 0-4: minutes-scale
    300, 600, 1200, 1800, 3600,     # 5-9: 5min-1h
    7200, 14400, 28800, 43200,      # 10-13: 2h-12h
    86400,                          # 14: 24h — past daily-quota reset
]
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

    # Transient / availability failures must ride the WHOLE backoff ladder
    # (tail = 24h, past a daily-quota reset) so a task that can't get capacity
    # WAITS for it instead of DLQ'ing. The add_task default cap (6) is sized
    # for quality retries; with it the ladder bottoms out at idx 5 (300s, ~9min
    # cumulative) and an availability failure DLQ'd long before a daily-exhausted
    # provider could reset (mission_79 2026-05-30 reviewer #225600: max=6 →
    # DLQ in minutes against daily-exhausted gemini). Give transient categories
    # an effective cap of the ladder length; quality/deterministic keep the
    # task's own (smaller) cap — retrying a deterministic failure 15× just burns
    # tokens on the same output.
    _TRANSIENT = {
        "availability", "daily_exhausted", "rate_limited", "no_model",
        "timeout", "loading", "server_error",
    }
    if category in _TRANSIENT:
        max_attempts = max(max_attempts, len(_BACKOFF_SECONDS))

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
