# error_policy.py
"""
Error taxonomy & retry policies.

Defines per-error-category retry strategies that replace the generic
retry_count < max_retries check in the orchestrator.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.infra.logging_config import get_logger
from ..core.state_machine import ErrorCategory

logger = get_logger("infra.error_policy")


@dataclass
class RetryPolicy:
    """Defines how to handle a specific error category."""
    max_retries: int = 3
    retry_with_fallback_model: bool = False
    increase_timeout: bool = False
    timeout_multiplier: float = 1.5
    inject_correction_prompt: bool = False
    pause_and_notify: bool = False
    fail_immediately: bool = False
    downgrade_tier: bool = False
    description: str = ""


# ─── Policy definitions per error category ────────────────────────────────────

ERROR_POLICIES: dict[str, RetryPolicy] = {
    ErrorCategory.MODEL_ERROR: RetryPolicy(
        max_retries=3,
        retry_with_fallback_model=True,
        description="Retry with next fallback model from the tier",
    ),
    ErrorCategory.TOOL_ERROR: RetryPolicy(
        max_retries=2,
        retry_with_fallback_model=False,
        description="Retry same model — tool failures may be transient",
    ),
    ErrorCategory.TIMEOUT: RetryPolicy(
        max_retries=2,
        increase_timeout=True,
        timeout_multiplier=1.5,
        description="Retry with increased timeout",
    ),
    ErrorCategory.INVALID_OUTPUT: RetryPolicy(
        max_retries=2,
        inject_correction_prompt=True,
        description="Retry with correction prompt containing parse error",
    ),
    ErrorCategory.BUDGET_EXCEEDED: RetryPolicy(
        max_retries=0,
        pause_and_notify=True,
        fail_immediately=False,
        description="Pause task and notify via Telegram",
    ),
    ErrorCategory.DEPENDENCY_FAILED: RetryPolicy(
        max_retries=0,
        fail_immediately=True,
        description="Fail immediately — dependency is broken",
    ),
    ErrorCategory.CANCELLED: RetryPolicy(
        max_retries=0,
        fail_immediately=True,
        description="Cancellation — do not retry",
    ),
    ErrorCategory.UNKNOWN: RetryPolicy(
        max_retries=2,
        retry_with_fallback_model=False,
        description="Unknown error — generic retry",
    ),
}


def get_retry_policy(error_category: str) -> RetryPolicy:
    """Get the retry policy for a given error category."""
    return ERROR_POLICIES.get(error_category, ERROR_POLICIES[ErrorCategory.UNKNOWN])


def should_retry(error_category: str, current_retry_count: int) -> bool:
    """
    Determine if a task should be retried based on error category
    and current retry count.
    """
    policy = get_retry_policy(error_category)

    if policy.fail_immediately:
        return False

    if policy.pause_and_notify:
        return False

    return current_retry_count < policy.max_retries


def get_adjusted_timeout(
    error_category: str,
    current_timeout: int,
    retry_count: int,
) -> int:
    """
    Calculate adjusted timeout for retry based on error category.
    """
    policy = get_retry_policy(error_category)
    if policy.increase_timeout and retry_count > 0:
        return int(current_timeout * policy.timeout_multiplier)
    return current_timeout


def get_retry_action(error_category: str) -> dict:
    """
    Get a description of the retry action to take.

    Returns dict with fields:
      - should_retry: bool
      - action: str (retry, pause, fail)
      - use_fallback_model: bool
      - inject_correction: bool
      - description: str
    """
    policy = get_retry_policy(error_category)

    if policy.fail_immediately:
        return {
            "should_retry": False,
            "action": "fail",
            "use_fallback_model": False,
            "inject_correction": False,
            "description": policy.description,
        }

    if policy.pause_and_notify:
        return {
            "should_retry": False,
            "action": "pause",
            "use_fallback_model": False,
            "inject_correction": False,
            "description": policy.description,
        }

    return {
        "should_retry": True,
        "action": "retry",
        "use_fallback_model": policy.retry_with_fallback_model,
        "inject_correction": policy.inject_correction_prompt,
        "description": policy.description,
    }
