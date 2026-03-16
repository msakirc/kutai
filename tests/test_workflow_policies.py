"""Tests for v2 workflow policies: review cycles, revision, and onboarding."""

import pytest

from src.workflows.engine.policies import (
    APPROVAL_REQUIRED_ACTIONS,
    ReviewTracker,
    check_onboarding_policy,
)


class TestReviewTracker:
    """Tests for ReviewTracker class."""

    def test_review_cycle_tracking(self):
        """3 failures then escalate."""
        tracker = ReviewTracker(max_cycles=3)
        step = "code_review"

        tracker.record_failure(step)
        tracker.record_failure(step)
        assert not tracker.should_escalate(step)

        tracker.record_failure(step)
        assert tracker.should_escalate(step)

    def test_no_escalation_below_max(self):
        """2 failures, no escalate."""
        tracker = ReviewTracker(max_cycles=3)
        step = "code_review"

        tracker.record_failure(step)
        tracker.record_failure(step)

        assert not tracker.should_escalate(step)
        assert tracker.get_cycle_count(step) == 2

    def test_escalation_message(self):
        """Message format matches v2's escalation_message_template."""
        tracker = ReviewTracker(max_cycles=3)
        step = "code_review"

        for _ in range(3):
            tracker.record_failure(step)

        issues = ["lint errors", "missing tests"]
        msg = tracker.escalation_message(step, issues)

        assert "code_review" in msg
        assert "3" in msg
        assert "lint errors" in msg
        assert "missing tests" in msg
        assert "Human decision needed" in msg
        assert "fix manually" in msg
        assert "accept as-is" in msg
        assert "provide direction" in msg

    def test_get_cycle_count(self):
        """Tracks per step."""
        tracker = ReviewTracker(max_cycles=3)

        assert tracker.get_cycle_count("step_a") == 0

        tracker.record_failure("step_a")
        assert tracker.get_cycle_count("step_a") == 1

        tracker.record_failure("step_a")
        assert tracker.get_cycle_count("step_a") == 2

    def test_reset(self):
        """Resets count for a step."""
        tracker = ReviewTracker(max_cycles=3)
        step = "code_review"

        tracker.record_failure(step)
        tracker.record_failure(step)
        assert tracker.get_cycle_count(step) == 2

        tracker.reset(step)
        assert tracker.get_cycle_count(step) == 0
        assert not tracker.should_escalate(step)

    def test_different_steps_independent(self):
        """Separate tracking per step."""
        tracker = ReviewTracker(max_cycles=3)

        tracker.record_failure("step_a")
        tracker.record_failure("step_a")
        tracker.record_failure("step_a")

        tracker.record_failure("step_b")

        assert tracker.should_escalate("step_a")
        assert not tracker.should_escalate("step_b")
        assert tracker.get_cycle_count("step_a") == 3
        assert tracker.get_cycle_count("step_b") == 1


class TestOnboardingPolicy:
    """Tests for check_onboarding_policy function."""

    def test_onboarding_policy_existing_project(self):
        """Schema change on existing project requires approval."""
        assert check_onboarding_policy("database_schema_changes", is_existing_project=True) is True
        assert check_onboarding_policy("dependency_major_upgrades", is_existing_project=True) is True
        assert check_onboarding_policy("architecture_pattern_changes", is_existing_project=True) is True
        assert check_onboarding_policy("deletion_of_existing_code", is_existing_project=True) is True

    def test_onboarding_policy_greenfield(self):
        """No approval needed for greenfield projects."""
        assert check_onboarding_policy("database_schema_changes", is_existing_project=False) is False
        assert check_onboarding_policy("dependency_major_upgrades", is_existing_project=False) is False
        assert check_onboarding_policy("architecture_pattern_changes", is_existing_project=False) is False
        assert check_onboarding_policy("deletion_of_existing_code", is_existing_project=False) is False

    def test_onboarding_policy_non_restricted_action(self):
        """Arbitrary action doesn't need approval."""
        assert check_onboarding_policy("add_new_file", is_existing_project=True) is False
        assert check_onboarding_policy("run_tests", is_existing_project=True) is False
        assert check_onboarding_policy("format_code", is_existing_project=False) is False


class TestApprovalRequiredActions:
    """Test the APPROVAL_REQUIRED_ACTIONS constant."""

    def test_contains_expected_actions(self):
        assert "database_schema_changes" in APPROVAL_REQUIRED_ACTIONS
        assert "dependency_major_upgrades" in APPROVAL_REQUIRED_ACTIONS
        assert "architecture_pattern_changes" in APPROVAL_REQUIRED_ACTIONS
        assert "deletion_of_existing_code" in APPROVAL_REQUIRED_ACTIONS

    def test_is_a_set(self):
        assert isinstance(APPROVAL_REQUIRED_ACTIONS, (set, frozenset))
