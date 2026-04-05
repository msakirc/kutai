"""
Unified retry logic for all task failure types.

Two failure types:
  quality     — output bad/missing. Immediate retry, then delay. Model escalation.
  availability — couldn't execute. Signal-aware backoff. No model change.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RetryDecision:
    action: str  # "immediate", "delayed", "terminal"
    delay_seconds: int = 0

    @staticmethod
    def immediate() -> RetryDecision:
        return RetryDecision(action="immediate", delay_seconds=0)

    @staticmethod
    def delayed(seconds: int) -> RetryDecision:
        return RetryDecision(action="delayed", delay_seconds=seconds)

    @staticmethod
    def terminal() -> RetryDecision:
        return RetryDecision(action="terminal", delay_seconds=0)


def compute_retry_timing(
    failure_type: str,
    attempts: int = 0,
    max_attempts: int = 6,
    last_avail_delay: int = 0,
) -> RetryDecision:
    if failure_type == "quality":
        if attempts >= max_attempts:
            return RetryDecision.terminal()
        if attempts < 3:
            return RetryDecision.immediate()
        return RetryDecision.delayed(600)
    elif failure_type == "availability":
        if last_avail_delay >= 7200:
            return RetryDecision.terminal()
        new_delay = max(60, min(last_avail_delay * 2, 7200))
        return RetryDecision.delayed(new_delay)
    raise ValueError(f"Unknown failure_type: {failure_type}")


def update_exclusions_on_failure(task_context: dict, failed_model: str, attempts: int) -> None:
    failed = task_context.setdefault("failed_models", [])
    if failed_model and failed_model not in failed:
        failed.append(failed_model)


def get_model_constraints(task_context: dict, attempts: int) -> tuple[list[str], int]:
    failed = task_context.get("failed_models", [])
    excluded = list(failed) if attempts >= 3 else []
    difficulty_bump = max(0, (attempts - 3) * 2) if attempts >= 4 else 0
    return excluded, difficulty_bump
