"""Tests that infrastructure resets and quality attempts are independent."""

from src.core.retry import RetryContext


def test_infra_does_not_increment_worker():
    ctx = RetryContext(worker_attempts=2, infra_resets=0)
    ctx.record_failure("infrastructure")
    assert ctx.worker_attempts == 2
    assert ctx.infra_resets == 1


def test_infra_terminal_at_3():
    ctx = RetryContext(infra_resets=2)
    d = ctx.record_failure("infrastructure")
    assert d.action == "terminal"
    assert ctx.failed_in_phase == "infrastructure"


def test_quality_budget_independent():
    """5 worker attempts + 2 infra resets: quality terminal, infra still has budget."""
    ctx = RetryContext(worker_attempts=5, infra_resets=2)
    qd = ctx.record_failure("quality", model="m")
    assert qd.action == "terminal"

    ctx2 = RetryContext(worker_attempts=5, infra_resets=1)
    id2 = ctx2.record_failure("infrastructure")
    assert id2.action == "immediate"
    assert ctx2.worker_attempts == 5  # untouched


def test_infra_reset_is_immediate_below_threshold():
    """Infra resets 0 and 1 should return immediate retry."""
    for start in (0, 1):
        ctx = RetryContext(infra_resets=start)
        d = ctx.record_failure("infrastructure")
        if start < 2:
            assert d.action == "immediate", f"Expected immediate for infra_resets={start}"


def test_infra_does_not_touch_grade_attempts():
    ctx = RetryContext(grade_attempts=1, infra_resets=0)
    ctx.record_failure("infrastructure")
    assert ctx.grade_attempts == 1
    assert ctx.infra_resets == 1
