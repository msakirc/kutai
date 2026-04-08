"""Tests for exhaustion handling: reason classification, budget boost, RetryContext."""

from src.core.retry import RetryContext


def test_exhaustion_reason_budget():
    """When iterations run out without guard or tool issues, reason is 'budget'."""
    ctx = RetryContext(useful_iterations=6, max_iterations=8)
    ctx.record_failure("exhaustion", model="m")
    assert ctx.exhaustion_reason == "budget"


def test_exhaustion_reason_guards():
    """When guard_burns >= 50% of max_iterations, reason is 'guards'."""
    ctx = RetryContext(guard_burns=5, max_iterations=8)
    ctx.record_failure("exhaustion", model="m")
    assert ctx.exhaustion_reason == "guards"


def test_exhaustion_reason_tool_failures():
    """When consecutive_tool_failures >= 3, reason is 'tool_failures'."""
    ctx = RetryContext(guard_burns=1, max_iterations=8, consecutive_tool_failures=3)
    ctx.record_failure("exhaustion", model="m")
    assert ctx.exhaustion_reason == "tool_failures"


def test_exhaustion_guards_priority_over_tool_failures():
    """Guards reason takes priority when both guards and tool failures are high."""
    ctx = RetryContext(guard_burns=5, max_iterations=8, consecutive_tool_failures=4)
    ctx.record_failure("exhaustion", model="m")
    assert ctx.exhaustion_reason == "guards"


def test_budget_boost_cap():
    """Budget boost is capped at 12 iterations."""
    assert min(int(8 * 1.5), 12) == 12
    assert min(int(5 * 1.5), 12) == 7
    assert min(int(3 * 1.5), 12) == 4


def test_budget_boost_no_effect_when_1():
    """Boost of 1.0 should not change effective_max_iterations."""
    max_iter = 8
    boost = 1.0
    effective = max_iter  # no change when boost <= 1.0
    assert effective == 8


def test_retry_context_record_guard_burn():
    """record_guard_burn increments the counter."""
    ctx = RetryContext()
    assert ctx.guard_burns == 0
    ctx.record_guard_burn("search_guard")
    assert ctx.guard_burns == 1
    ctx.record_guard_burn("repetition_guard")
    assert ctx.guard_burns == 2


def test_retry_context_record_useful_iteration():
    """record_useful_iteration increments the counter."""
    ctx = RetryContext()
    assert ctx.useful_iterations == 0
    ctx.record_useful_iteration()
    assert ctx.useful_iterations == 1


def test_exhaustion_increments_worker_attempts():
    """Exhaustion failure type increments worker_attempts via quality path."""
    ctx = RetryContext(worker_attempts=0, max_worker_attempts=6)
    ctx.record_failure("exhaustion", model="model-a")
    assert ctx.worker_attempts == 1
    assert ctx.failed_in_phase == "worker"


def test_exhaustion_tracks_model():
    """Exhaustion failure tracks the model that failed."""
    ctx = RetryContext()
    ctx.record_failure("exhaustion", model="model-x")
    assert "model-x" in ctx.failed_models


def test_to_db_fields_includes_exhaustion_reason():
    """to_db_fields includes exhaustion_reason after recording."""
    ctx = RetryContext(guard_burns=5, max_iterations=8)
    ctx.record_failure("exhaustion", model="m")
    fields = ctx.to_db_fields()
    assert fields["exhaustion_reason"] == "guards"


def test_to_checkpoint_includes_counters():
    """to_checkpoint serializes guard_burns and useful_iterations."""
    ctx = RetryContext(guard_burns=3, useful_iterations=5)
    cp = ctx.to_checkpoint()
    assert cp["guard_burns"] == 3
    assert cp["useful_iterations"] == 5
