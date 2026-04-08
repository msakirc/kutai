"""Tests for RetryContext dataclass."""

import json
import pytest
from src.core.retry import RetryContext, RetryDecision


class TestDefaultConstruction:
    def test_defaults(self):
        ctx = RetryContext()
        assert ctx.worker_attempts == 0
        assert ctx.infra_resets == 0
        assert ctx.max_worker_attempts == 6
        assert ctx.grade_attempts == 0
        assert ctx.max_grade_attempts == 3
        assert ctx.next_retry_at is None
        assert ctx.retry_reason is None
        assert ctx.failed_in_phase is None
        assert ctx.failed_models == []
        assert ctx.grade_excluded_models == []
        assert ctx.iteration == 0
        assert ctx.max_iterations == 8
        assert ctx.format_corrections == 0
        assert ctx.consecutive_tool_failures == 0
        assert ctx.model_escalated is False
        assert ctx.guard_burns == 0
        assert ctx.useful_iterations == 0
        assert ctx.exhaustion_reason is None

    def test_list_fields_are_independent(self):
        """Each instance must have its own lists (no shared default)."""
        a = RetryContext()
        b = RetryContext()
        a.failed_models.append("m1")
        assert b.failed_models == []


class TestFromTask:
    def test_fresh_task(self):
        task = {"id": 1, "status": "pending"}
        ctx = RetryContext.from_task(task)
        assert ctx.worker_attempts == 0
        assert ctx.max_worker_attempts == 6

    def test_legacy_attempts_field(self):
        """Old tasks have 'attempts' instead of 'worker_attempts'."""
        task = {"id": 1, "attempts": 4}
        ctx = RetryContext.from_task(task)
        assert ctx.worker_attempts == 4

    def test_legacy_max_attempts_field(self):
        """Old tasks have 'max_attempts' instead of 'max_worker_attempts'."""
        task = {"id": 1, "max_attempts": 10}
        ctx = RetryContext.from_task(task)
        assert ctx.max_worker_attempts == 10

    def test_new_worker_attempts_field(self):
        task = {"id": 1, "worker_attempts": 3, "attempts": 99}
        ctx = RetryContext.from_task(task)
        # worker_attempts takes precedence over legacy attempts
        assert ctx.worker_attempts == 3

    def test_new_max_worker_attempts_field(self):
        task = {"id": 1, "max_worker_attempts": 8, "max_attempts": 99}
        ctx = RetryContext.from_task(task)
        assert ctx.max_worker_attempts == 8

    def test_dict_context(self):
        task = {
            "id": 1,
            "context": {
                "failed_models": ["model-a", "model-b"],
                "grade_excluded_models": ["model-c"],
            },
        }
        ctx = RetryContext.from_task(task)
        assert ctx.failed_models == ["model-a", "model-b"]
        assert ctx.grade_excluded_models == ["model-c"]

    def test_json_string_context(self):
        context = json.dumps({
            "failed_models": ["model-x"],
            "grade_excluded_models": [],
        })
        task = {"id": 1, "context": context}
        ctx = RetryContext.from_task(task)
        assert ctx.failed_models == ["model-x"]

    def test_all_db_fields(self):
        task = {
            "id": 1,
            "worker_attempts": 2,
            "infra_resets": 1,
            "max_worker_attempts": 5,
            "grade_attempts": 1,
            "max_grade_attempts": 2,
            "next_retry_at": "2026-04-07 12:00:00",
            "retry_reason": "quality",
            "failed_in_phase": "execution",
        }
        ctx = RetryContext.from_task(task)
        assert ctx.worker_attempts == 2
        assert ctx.infra_resets == 1
        assert ctx.max_worker_attempts == 5
        assert ctx.grade_attempts == 1
        assert ctx.max_grade_attempts == 2
        assert ctx.next_retry_at == "2026-04-07 12:00:00"
        assert ctx.retry_reason == "quality"
        assert ctx.failed_in_phase == "execution"

    def test_none_context(self):
        task = {"id": 1, "context": None}
        ctx = RetryContext.from_task(task)
        assert ctx.failed_models == []


class TestProperties:
    def test_total_attempts(self):
        ctx = RetryContext(worker_attempts=3, infra_resets=2)
        assert ctx.total_attempts == 5

    def test_effective_difficulty_bump_below_threshold(self):
        ctx = RetryContext(worker_attempts=3)
        assert ctx.effective_difficulty_bump == 0

    def test_effective_difficulty_bump_at_4(self):
        ctx = RetryContext(worker_attempts=4)
        assert ctx.effective_difficulty_bump == 2

    def test_effective_difficulty_bump_at_5(self):
        ctx = RetryContext(worker_attempts=5)
        assert ctx.effective_difficulty_bump == 4

    def test_excluded_models_below_threshold(self):
        ctx = RetryContext(worker_attempts=2, failed_models=["m1", "m2"])
        assert ctx.excluded_models == []

    def test_excluded_models_at_threshold(self):
        ctx = RetryContext(worker_attempts=3, failed_models=["m1", "m2"])
        assert ctx.excluded_models == ["m1", "m2"]

    def test_excluded_models_returns_copy(self):
        ctx = RetryContext(worker_attempts=3, failed_models=["m1"])
        excluded = ctx.excluded_models
        excluded.append("m2")
        assert ctx.failed_models == ["m1"]


class TestRecordFailure:
    def test_quality_first_attempt(self):
        ctx = RetryContext()
        decision = ctx.record_failure("quality", model="model-a")
        assert ctx.worker_attempts == 1
        assert ctx.retry_reason == "quality"
        assert "model-a" in ctx.failed_models
        assert decision.action == "immediate"

    def test_quality_terminal(self):
        ctx = RetryContext(worker_attempts=5, max_worker_attempts=6)
        decision = ctx.record_failure("quality", model="model-z")
        assert ctx.worker_attempts == 6
        assert decision.action == "terminal"

    def test_quality_delayed(self):
        ctx = RetryContext(worker_attempts=2, max_worker_attempts=6)
        decision = ctx.record_failure("quality", model=None)
        assert ctx.worker_attempts == 3
        # attempts=3 → delayed(600)
        assert decision.action == "delayed"
        assert decision.delay_seconds == 600

    def test_timeout_treated_as_quality(self):
        ctx = RetryContext()
        decision = ctx.record_failure("timeout", model="model-t")
        assert ctx.worker_attempts == 1
        assert ctx.retry_reason == "timeout"
        assert decision.action == "immediate"

    def test_infrastructure_increment(self):
        ctx = RetryContext()
        decision = ctx.record_failure("infrastructure", model=None)
        assert ctx.infra_resets == 1
        assert ctx.retry_reason == "infrastructure"
        assert decision.action != "terminal"

    def test_infrastructure_terminal_at_3(self):
        ctx = RetryContext(infra_resets=2)
        decision = ctx.record_failure("infrastructure", model=None)
        assert ctx.infra_resets == 3
        assert decision.action == "terminal"

    def test_infrastructure_model_tracked(self):
        ctx = RetryContext()
        ctx.record_failure("infrastructure", model="bad-model")
        assert "bad-model" in ctx.failed_models

    def test_exhaustion_budget(self):
        ctx = RetryContext(guard_burns=1, consecutive_tool_failures=0, max_iterations=8)
        decision = ctx.record_failure("exhaustion", model="m1")
        assert ctx.exhaustion_reason == "budget"
        # Treated as quality after classification
        assert ctx.worker_attempts == 1

    def test_exhaustion_guards(self):
        ctx = RetryContext(guard_burns=4, max_iterations=8)
        decision = ctx.record_failure("exhaustion", model=None)
        assert ctx.exhaustion_reason == "guards"

    def test_exhaustion_tool_failures(self):
        ctx = RetryContext(consecutive_tool_failures=3, max_iterations=8)
        decision = ctx.record_failure("exhaustion", model=None)
        assert ctx.exhaustion_reason == "tool_failures"

    def test_availability(self):
        ctx = RetryContext()
        decision = ctx.record_failure("availability", model=None)
        assert decision.action == "delayed"
        assert decision.delay_seconds == 60

    def test_duplicate_model_not_added(self):
        ctx = RetryContext(failed_models=["m1"])
        ctx.record_failure("quality", model="m1")
        assert ctx.failed_models.count("m1") == 1

    def test_none_model_not_added(self):
        ctx = RetryContext()
        ctx.record_failure("quality", model=None)
        assert ctx.failed_models == []


class TestSerialization:
    def test_to_db_fields(self):
        ctx = RetryContext(
            worker_attempts=2,
            infra_resets=1,
            max_worker_attempts=6,
            grade_attempts=1,
            max_grade_attempts=3,
            next_retry_at="2026-04-07 12:00:00",
            retry_reason="quality",
            failed_in_phase="execution",
        )
        fields = ctx.to_db_fields()
        assert fields == {
            "worker_attempts": 2,
            "infra_resets": 1,
            "max_worker_attempts": 6,
            "grade_attempts": 1,
            "max_grade_attempts": 3,
            "next_retry_at": "2026-04-07 12:00:00",
            "retry_reason": "quality",
            "failed_in_phase": "execution",
        }
        # Must NOT contain iteration-level or model-tracking fields
        assert "failed_models" not in fields
        assert "iteration" not in fields

    def test_to_context_patch(self):
        ctx = RetryContext(
            failed_models=["m1", "m2"],
            grade_excluded_models=["m3"],
        )
        patch = ctx.to_context_patch()
        assert patch == {
            "failed_models": ["m1", "m2"],
            "grade_excluded_models": ["m3"],
        }

    def test_to_checkpoint(self):
        ctx = RetryContext(
            iteration=3,
            max_iterations=8,
            format_corrections=1,
            consecutive_tool_failures=2,
            model_escalated=True,
            guard_burns=4,
            useful_iterations=2,
        )
        cp = ctx.to_checkpoint()
        assert cp == {
            "iteration": 3,
            "max_iterations": 8,
            "format_corrections": 1,
            "consecutive_tool_failures": 2,
            "model_escalated": True,
            "guard_burns": 4,
            "useful_iterations": 2,
        }

    def test_roundtrip(self):
        """Build from task, modify, serialize, reconstruct — state preserved."""
        original = RetryContext(
            worker_attempts=2,
            infra_resets=1,
            failed_models=["m1"],
            grade_excluded_models=["m2"],
            iteration=5,
            guard_burns=3,
        )
        # Simulate persisting to task record
        task = {**original.to_db_fields(), "context": original.to_context_patch()}
        reconstructed = RetryContext.from_task(task)
        assert reconstructed.worker_attempts == original.worker_attempts
        assert reconstructed.infra_resets == original.infra_resets
        assert reconstructed.failed_models == original.failed_models
        assert reconstructed.grade_excluded_models == original.grade_excluded_models
        # Iteration-level fields are NOT in task record (they're in checkpoint)
        assert reconstructed.iteration == 0  # not persisted in task


class TestGuardTracking:
    def test_record_guard_burn(self):
        ctx = RetryContext()
        ctx.record_guard_burn()
        assert ctx.guard_burns == 1
        ctx.record_guard_burn()
        assert ctx.guard_burns == 2

    def test_record_useful_iteration(self):
        ctx = RetryContext()
        ctx.record_useful_iteration()
        assert ctx.useful_iterations == 1
        ctx.record_useful_iteration()
        assert ctx.useful_iterations == 2
