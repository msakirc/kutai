# state_machine.py
"""
Task state machine — validates all status transitions.

States:
  pending → processing → completed / failed / needs_clarification /
                          needs_review / needs_subtasks / waiting_subtasks
  pending → cancelled
  processing → cancelled
  failed → pending  (retry)
  needs_clarification → pending  (user answered)
  needs_review → pending
  waiting_subtasks → completed / failed
  paused → pending
  pending → paused
  processing → paused

Error categories attached to failed tasks:
  model_error, tool_error, timeout, budget_exceeded,
  invalid_output, dependency_failed, cancelled
"""

from enum import Enum
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("core.state_machine")


# ─── States ──────────────────────────────────────────────────────────────────

class TaskState(str, Enum):
    """All valid task states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_CLARIFICATION = "needs_clarification"
    NEEDS_REVIEW = "needs_review"
    NEEDS_SUBTASKS = "needs_subtasks"
    WAITING_SUBTASKS = "waiting_subtasks"
    PAUSED = "paused"
    REJECTED = "rejected"


# ─── Error categories ────────────────────────────────────────────────────────

class ErrorCategory(str, Enum):
    """Error classification for failed tasks."""
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    INVALID_OUTPUT = "invalid_output"
    DEPENDENCY_FAILED = "dependency_failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


# ─── Valid transitions ────────────────────────────────────────────────────────

TRANSITIONS: dict[str, set[str]] = {
    TaskState.PENDING: {
        TaskState.PROCESSING,
        TaskState.CANCELLED,
        TaskState.PAUSED,
    },
    TaskState.PROCESSING: {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.NEEDS_CLARIFICATION,
        TaskState.NEEDS_REVIEW,
        TaskState.NEEDS_SUBTASKS,
        TaskState.WAITING_SUBTASKS,
        TaskState.PENDING,  # reset on watchdog stuck-detection
        TaskState.PAUSED,
    },
    TaskState.COMPLETED: set(),  # terminal
    TaskState.FAILED: {
        TaskState.PENDING,  # retry
    },
    TaskState.CANCELLED: set(),  # terminal
    TaskState.NEEDS_CLARIFICATION: {
        TaskState.PENDING,  # user responded
        TaskState.CANCELLED,
    },
    TaskState.NEEDS_REVIEW: {
        TaskState.PENDING,  # reviewer done
        TaskState.CANCELLED,
        TaskState.COMPLETED,  # review created a subtask, original done
    },
    TaskState.NEEDS_SUBTASKS: {
        TaskState.WAITING_SUBTASKS,
        TaskState.CANCELLED,
    },
    TaskState.WAITING_SUBTASKS: {
        TaskState.COMPLETED,  # all children done
        TaskState.FAILED,     # a child failed
        TaskState.CANCELLED,
    },
    TaskState.PAUSED: {
        TaskState.PENDING,  # resume
        TaskState.CANCELLED,
    },
    TaskState.REJECTED: set(),  # terminal
}


class InvalidTransition(Exception):
    """Raised when a state transition is not allowed."""

    def __init__(self, task_id: int, from_state: str, to_state: str):
        self.task_id = task_id
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid transition for task #{task_id}: "
            f"'{from_state}' → '{to_state}'"
        )


def validate_transition(from_state: str, to_state: str) -> bool:
    """Check if a state transition is valid."""
    allowed = TRANSITIONS.get(from_state, set())
    return to_state in allowed


async def transition_task(
    task_id: int,
    to_state: str,
    error: Optional[str] = None,
    error_category: Optional[str] = None,
    **extra_fields,
) -> None:
    """
    Transition a task to a new state with validation.

    Loads current state from DB, validates the transition is legal,
    then updates. Raises InvalidTransition if the move is not allowed.

    Additional fields (result, completed_at, retry_count, etc.) can be
    passed as **extra_fields and will be included in the DB update.
    """
    # Lazy import to avoid circular dependency
    from ..infra.db import get_task, update_task

    task = await get_task(task_id)
    if not task:
        raise ValueError(f"Task #{task_id} not found")

    current_state = task.get("status", "pending")

    if not validate_transition(current_state, to_state):
        raise InvalidTransition(task_id, current_state, to_state)

    update_kwargs = {"status": to_state, **extra_fields}

    if error is not None:
        update_kwargs["error"] = error

    if error_category is not None:
        update_kwargs["error_category"] = error_category

    logger.info(
        "state transition",
        task_id=task_id,
        from_state=current_state,
        to_state=to_state,
        error_category=error_category,
    )

    await update_task(task_id, **update_kwargs)


def classify_error(exception: Exception) -> str:
    """
    Classify an exception into an error category.

    Returns an ErrorCategory value string.
    """
    import asyncio

    exc_type = type(exception).__name__
    exc_msg = str(exception).lower()

    # Timeout
    if isinstance(exception, (asyncio.TimeoutError,)):
        return ErrorCategory.TIMEOUT
    if "timeout" in exc_msg:
        return ErrorCategory.TIMEOUT

    # Budget
    if "budget" in exc_msg or "cost" in exc_msg:
        return ErrorCategory.BUDGET_EXCEEDED

    # Cancellation
    if isinstance(exception, (asyncio.CancelledError,)):
        return ErrorCategory.CANCELLED

    # Model errors — API failures, rate limits, auth
    model_indicators = [
        "rate_limit", "ratelimit", "429", "api_error",
        "authentication", "401", "403", "api_key",
        "litellm", "openai", "anthropic", "groq",
        "model", "completion",
    ]
    if any(ind in exc_msg for ind in model_indicators):
        return ErrorCategory.MODEL_ERROR

    # Tool errors — command failures, file issues
    tool_indicators = [
        "tool error", "command failed", "file not found",
        "permission denied", "not found", "no such file",
        "subprocess", "docker",
    ]
    if any(ind in exc_msg for ind in tool_indicators):
        return ErrorCategory.TOOL_ERROR

    # Invalid output
    if "parse" in exc_msg or "json" in exc_msg or "invalid" in exc_msg:
        return ErrorCategory.INVALID_OUTPUT

    return ErrorCategory.UNKNOWN
