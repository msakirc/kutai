"""v2 workflow policies: review cycles, revision, and onboarding."""

from collections import defaultdict

from src.infra.logging_config import get_logger

logger = get_logger("workflows.engine.policies")


APPROVAL_REQUIRED_ACTIONS: set[str] = {
    "database_schema_changes",
    "dependency_major_upgrades",
    "architecture_pattern_changes",
    "deletion_of_existing_code",
}


class ReviewTracker:
    """Tracks review cycle failures per step and escalates after max_cycles."""

    def __init__(self, max_cycles: int = 3) -> None:
        self.max_cycles = max_cycles
        self._failures: defaultdict[str, int] = defaultdict(int)

    def record_failure(self, step_id: str) -> None:
        """Increment failure count for the given step."""
        self._failures[step_id] += 1

    def should_escalate(self, step_id: str) -> bool:
        """Return True if failures >= max_cycles for the step."""
        return self._failures[step_id] >= self.max_cycles

    def escalation_message(self, step_id: str, issues: list[str]) -> str:
        """Return a human-readable escalation message matching v2's template."""
        issues_list = ", ".join(issues)
        return (
            f"Review cycle for step '{step_id}' has failed {self.max_cycles} times. "
            f"Issues: {issues_list}. "
            f"Human decision needed: fix manually, accept as-is, or provide direction."
        )

    def get_cycle_count(self, step_id: str) -> int:
        """Return the current failure count for a step."""
        return self._failures[step_id]

    def reset(self, step_id: str) -> None:
        """Reset failure count for a step (after successful review)."""
        self._failures[step_id] = 0


def check_onboarding_policy(
    action_type: str, is_existing_project: bool = False
) -> bool:
    """Check if human approval is needed for the given action.

    Returns True if the action requires approval (action is in
    APPROVAL_REQUIRED_ACTIONS AND is_existing_project is True).
    Returns False for greenfield projects or non-restricted actions.
    """
    return action_type in APPROVAL_REQUIRED_ACTIONS and is_existing_project
