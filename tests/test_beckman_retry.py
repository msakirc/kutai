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
