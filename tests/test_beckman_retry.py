"""Unit tests for general_beckman.retry — pure retry/DLQ decision logic."""
import pytest
from general_beckman.retry import (
    RetryDecision, decide_retry, DLQAction,
)


def _failure(category="quality", attempts=1, max_attempts=3):
    return {
        "category": category,
        "worker_attempts": attempts,
        "max_worker_attempts": max_attempts,
        "model": "test-model",
    }


def test_first_failure_retries_immediately():
    decision = decide_retry(_failure(attempts=1))
    assert decision.action == "immediate"


def test_mid_attempts_retries_with_delay():
    decision = decide_retry(_failure(attempts=2, max_attempts=3))
    assert decision.action == "delayed"
    assert decision.delay_seconds > 0


def test_exhausted_attempts_become_dlq():
    decision = decide_retry(_failure(attempts=3, max_attempts=3))
    assert isinstance(decision, DLQAction)


def test_quality_bonus_granted_with_progress():
    # Task exhausted on quality, but progress >= 0.5: grant one bonus attempt.
    decision = decide_retry(
        _failure(category="quality", attempts=3, max_attempts=3),
        progress=0.75,
        bonus_count=0,
    )
    assert decision.action == "immediate"
    assert decision.bonus_used is True


def test_quality_bonus_caps_at_two():
    decision = decide_retry(
        _failure(category="quality", attempts=3, max_attempts=3),
        progress=0.75,
        bonus_count=2,  # already used 2 bonuses
    )
    assert isinstance(decision, DLQAction)


# ─── no_model category: pure environmental backpressure ─────────────────────
#
# User design 2026-05-02 18:15 UTC: "Counter is for valid worker and grader
# attempts. DLQ is for truly not recoverable tasks. Not being dispatched is
# neither." no_model is just a deferral — the worker never got a model to
# call. Constant 60s backoff, no counter advancement, no DLQ.


def test_no_model_uses_constant_60s_defer():
    """Every no_model failure defers 60s, regardless of attempt count."""
    for a in (1, 5, 50, 500):
        decision = decide_retry(_failure(category="no_model", attempts=a))
        assert decision.action == "delayed"
        assert decision.delay_seconds == 60


def test_no_model_never_dlqs():
    """No matter how many attempts have racked up, no_model never DLQs.
    Permanently empty pool = task sits pending forever; that's a
    structural failure beyond a single task's responsibility."""
    for a in (10, 100, 9999):
        decision = decide_retry(
            _failure(category="no_model", attempts=a, max_attempts=3),
        )
        assert not isinstance(decision, DLQAction), (
            f"no_model must never DLQ (attempts={a})"
        )


def test_no_model_does_not_affect_other_categories():
    """Quality / availability paths unchanged."""
    q = decide_retry(_failure(category="quality", attempts=1))
    assert q.action == "immediate" and q.delay_seconds == 0
    a = decide_retry(_failure(category="availability", attempts=2))
    assert a.action == "delayed" and a.delay_seconds == 10
