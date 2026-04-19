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
    """
    attempts = int(failure.get("worker_attempts", 0))
    max_attempts = int(failure.get("max_worker_attempts", 3))
    category = failure.get("category", "unknown")

    if attempts < max_attempts:
        # attempts=1 (first failure) → idx=0 → immediate; attempts=2 → idx=1 → 10s, etc.
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
