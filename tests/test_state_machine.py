import pytest
from src.core.state_machine import TaskState, TRANSITIONS, validate_transition, InvalidTransition


class TestTaskStates:
    def test_all_states_exist(self):
        expected = {
            "pending", "processing", "ungraded", "completed", "failed",
            "waiting_subtasks", "waiting_human", "cancelled", "skipped",
        }
        assert {s.value for s in TaskState} == expected

    def test_old_states_removed(self):
        values = {s.value for s in TaskState}
        for old in ("paused", "sleeping", "needs_clarification",
                     "needs_review", "needs_subtasks", "rejected", "done"):
            assert old not in values


class TestTransitions:
    @pytest.mark.parametrize("state", ["completed", "cancelled", "skipped"])
    def test_terminal_states_have_no_transitions(self, state):
        assert TRANSITIONS[state] == set()

    def test_processing_to_ungraded(self):
        assert validate_transition("processing", "ungraded")

    def test_ungraded_to_completed(self):
        assert validate_transition("ungraded", "completed")

    def test_ungraded_to_pending(self):
        assert validate_transition("ungraded", "pending")

    def test_ungraded_to_failed(self):
        assert validate_transition("ungraded", "failed")

    def test_failed_to_ungraded(self):
        assert validate_transition("failed", "ungraded")

    def test_failed_to_pending(self):
        assert validate_transition("failed", "pending")

    def test_completed_to_anything_invalid(self):
        assert not validate_transition("completed", "pending")
        assert not validate_transition("completed", "processing")

    def test_pending_to_completed_invalid(self):
        assert not validate_transition("pending", "completed")

    def test_ungraded_to_processing_invalid(self):
        assert not validate_transition("ungraded", "processing")
