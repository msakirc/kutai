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


# ─── no_model category: dispatcher pick=None, transient pool exhaustion ─────


def test_no_model_first_attempt_uses_30s_delay():
    """First no_model failure must NOT retry immediately — provider pool
    needs time to recover. Default availability ladder starts at 0s
    which burned worker_attempts in <60s on production 2026-05-02."""
    decision = decide_retry(_failure(category="no_model", attempts=1))
    assert decision.action == "delayed"
    assert decision.delay_seconds == 30


def test_no_model_backoff_grows():
    """Ladder: 30, 60, 120, 300, 600+."""
    delays = [
        decide_retry(_failure(category="no_model", attempts=a)).delay_seconds
        for a in range(1, 6)
    ]
    assert delays == [30, 60, 120, 300, 600]


def test_no_model_extends_attempt_cap_above_default_max():
    """Default max_worker_attempts=3 must NOT trigger DLQ for no_model;
    the cap floors at 10 internally."""
    for attempts in range(1, 10):
        decision = decide_retry(
            _failure(category="no_model", attempts=attempts, max_attempts=3),
        )
        assert not isinstance(decision, DLQAction), (
            f"attempts={attempts} should NOT DLQ"
        )


def test_no_model_dlqs_after_ten_attempts():
    decision = decide_retry(
        _failure(category="no_model", attempts=10, max_attempts=3),
    )
    assert isinstance(decision, DLQAction)
    assert decision.category == "no_model"


def test_no_model_respects_higher_user_max():
    """If task has explicit max_worker_attempts > 10, honor it."""
    decision = decide_retry(
        _failure(category="no_model", attempts=10, max_attempts=20),
    )
    assert decision.action == "delayed"


def test_no_model_does_not_affect_other_categories():
    """Quality / availability paths unchanged."""
    q = decide_retry(_failure(category="quality", attempts=1))
    assert q.action == "immediate" and q.delay_seconds == 0
    a = decide_retry(_failure(category="availability", attempts=2))
    assert a.action == "delayed" and a.delay_seconds == 10
