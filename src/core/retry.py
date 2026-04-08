"""
Unified retry logic for all task failure types.

Two failure types:
  quality     — output bad/missing. Immediate retry, then delay. Model escalation.
  availability — couldn't execute. Signal-aware backoff. No model change.
"""

from __future__ import annotations
import json as _json
from dataclasses import dataclass, field


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


@dataclass
class RetryContext:
    """Centralised retry / iteration / escalation state for a task."""

    # ── Task-level (persisted as DB columns) ──
    worker_attempts: int = 0
    infra_resets: int = 0
    max_worker_attempts: int = 6
    grade_attempts: int = 0
    max_grade_attempts: int = 3
    next_retry_at: str | None = None
    retry_reason: str | None = None
    failed_in_phase: str | None = None

    # ── Model tracking (persisted in task.context JSON) ──
    failed_models: list[str] = field(default_factory=list)
    grade_excluded_models: list[str] = field(default_factory=list)

    # ── Iteration-level (persisted in checkpoint) ──
    iteration: int = 0
    max_iterations: int = 8
    format_corrections: int = 0
    consecutive_tool_failures: int = 0
    model_escalated: bool = False
    guard_burns: int = 0
    useful_iterations: int = 0

    # ── Exhaustion tracking ──
    exhaustion_reason: str | None = None

    # ── Construction ────────────────────────────────────────────────

    @classmethod
    def from_task(cls, task: dict) -> RetryContext:
        """Reconstruct from a task DB record.

        Handles backwards compat: reads ``attempts`` if ``worker_attempts``
        is missing, ``max_attempts`` if ``max_worker_attempts`` is missing.
        ``context`` may be a dict or a JSON string.
        """
        # Parse context
        raw_ctx = task.get("context")
        if isinstance(raw_ctx, str):
            try:
                ctx = _json.loads(raw_ctx)
            except (ValueError, TypeError):
                ctx = {}
        elif isinstance(raw_ctx, dict):
            ctx = raw_ctx
        else:
            ctx = {}

        # worker_attempts: prefer new name, fall back to legacy
        if "worker_attempts" in task:
            worker_attempts = task["worker_attempts"]
        else:
            worker_attempts = task.get("attempts", 0)

        # max_worker_attempts: prefer new name, fall back to legacy
        if "max_worker_attempts" in task:
            max_worker_attempts = task["max_worker_attempts"]
        else:
            max_worker_attempts = task.get("max_attempts", 6)

        return cls(
            worker_attempts=worker_attempts or 0,
            infra_resets=task.get("infra_resets", 0) or 0,
            max_worker_attempts=max_worker_attempts or 6,
            grade_attempts=task.get("grade_attempts", 0) or 0,
            max_grade_attempts=task.get("max_grade_attempts", 3) or 3,
            next_retry_at=task.get("next_retry_at"),
            retry_reason=task.get("retry_reason"),
            failed_in_phase=task.get("failed_in_phase"),
            exhaustion_reason=task.get("exhaustion_reason"),
            failed_models=list(ctx.get("failed_models", [])),
            grade_excluded_models=list(ctx.get("grade_excluded_models", [])),
        )

    # ── Properties ──────────────────────────────────────────────────

    @property
    def total_attempts(self) -> int:
        return self.worker_attempts + self.infra_resets

    @property
    def effective_difficulty_bump(self) -> int:
        if self.worker_attempts >= 4:
            return max(0, (self.worker_attempts - 3) * 2)
        return 0

    @property
    def excluded_models(self) -> list[str]:
        if self.worker_attempts >= 3:
            return list(self.failed_models)
        return []

    # ── Mutation ────────────────────────────────────────────────────

    def _track_model(self, model: str | None) -> None:
        if model and model not in self.failed_models:
            self.failed_models.append(model)

    def record_failure(self, failure_type: str, model: str | None = None) -> RetryDecision:
        """Single entry point for recording any failure type.

        Returns a RetryDecision produced by the existing compute_retry_timing().
        """
        self._track_model(model)

        if failure_type == "infrastructure":
            self.infra_resets += 1
            self.retry_reason = "infrastructure"
            self.failed_in_phase = "infrastructure"
            if self.infra_resets >= 3:
                return RetryDecision.terminal()
            return RetryDecision.immediate()

        if failure_type == "exhaustion":
            # Classify exhaustion reason first
            if self.guard_burns >= self.max_iterations * 0.5:
                self.exhaustion_reason = "guards"
            elif self.consecutive_tool_failures >= 3:
                self.exhaustion_reason = "tool_failures"
            else:
                self.exhaustion_reason = "budget"
            self.failed_in_phase = "worker"
            # Then handle as quality
            failure_type = "quality"

        if failure_type in ("quality", "timeout"):
            self.worker_attempts += 1
            self.retry_reason = failure_type
            self.failed_in_phase = "worker"
            return compute_retry_timing(
                failure_type="quality",
                attempts=self.worker_attempts,
                max_attempts=self.max_worker_attempts,
            )

        if failure_type == "availability":
            self.worker_attempts += 1
            self.retry_reason = "availability"
            return compute_retry_timing(
                failure_type="availability",
                last_avail_delay=0,
            )

        raise ValueError(f"Unknown failure_type: {failure_type}")

    def record_guard_burn(self, guard_name: str) -> None:
        self.guard_burns += 1

    def record_useful_iteration(self) -> None:
        self.useful_iterations += 1

    # ── Serialization ───────────────────────────────────────────────

    def to_db_fields(self) -> dict:
        """Task-level fields only (DB columns)."""
        return {
            "worker_attempts": self.worker_attempts,
            "infra_resets": self.infra_resets,
            "max_worker_attempts": self.max_worker_attempts,
            "grade_attempts": self.grade_attempts,
            "max_grade_attempts": self.max_grade_attempts,
            "next_retry_at": self.next_retry_at,
            "retry_reason": self.retry_reason,
            "failed_in_phase": self.failed_in_phase,
            "exhaustion_reason": self.exhaustion_reason,
        }

    def to_context_patch(self) -> dict:
        """Model-tracking fields (merged into task.context JSON)."""
        return {
            "failed_models": self.failed_models,
            "grade_excluded_models": self.grade_excluded_models,
        }

    def to_checkpoint(self) -> dict:
        """Iteration-level fields (agent checkpoint)."""
        return {
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "format_corrections": self.format_corrections,
            "consecutive_tool_failures": self.consecutive_tool_failures,
            "model_escalated": self.model_escalated,
            "guard_burns": self.guard_burns,
            "useful_iterations": self.useful_iterations,
        }


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
