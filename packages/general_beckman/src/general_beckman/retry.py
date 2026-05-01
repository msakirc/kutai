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
# This is not a worker bug; it's transient pool exhaustion. Default
# availability ladder (0/10/30) burns the 3 worker_attempts in <60s and
# DLQs the task before any provider can recover. Use a longer ladder
# AND a higher attempt cap so a slow gemini quota reset / one local
# release lets the task progress instead of dying. Production 2026-05-02:
# all gemini + openrouter ids dead, every executor task hit DLQ within
# 40s of the same "No model candidates available" error.
_NO_MODEL_BACKOFF_SECONDS = [30, 60, 120, 300, 600, 600, 600, 600, 600, 600]
_NO_MODEL_MAX_ATTEMPTS = 10
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

    # "no_model" (dispatcher's pick=None signal) gets its own ladder and
    # attempt cap. Failure is environmental — every provider exhausted
    # in the same instant. The default 3-attempt × 0/10/30 ladder DLQs
    # in under 60s before any quota or local slot can recover.
    if category == "no_model":
        no_model_max = max(max_attempts, _NO_MODEL_MAX_ATTEMPTS)
        if attempts < no_model_max:
            idx = min(max(0, attempts - 1), len(_NO_MODEL_BACKOFF_SECONDS) - 1)
            delay = _NO_MODEL_BACKOFF_SECONDS[idx]
            return RetryDecision(action="delayed", delay_seconds=delay)
        return DLQAction(
            category=category,
            reason=failure.get("error", "")[:300]
            or f"no_model exhausted after {attempts} attempts",
        )

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
