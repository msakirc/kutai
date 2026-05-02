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


# ─── Shared availability ladder + 10-attempt cap ────────────────────────────
#
# User design 2026-05-02 18:50 UTC: extend shared backoff ladder to 10
# entries up to 1h, bump default attempt cap to 10. no_model special
# branch dropped — pool-empty mid-task surfaces as availability and
# routes through the same path as any other transient failure.


def test_availability_ladder_climbs_to_1h():
    """Ladder: 0, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600."""
    delays = [
        decide_retry(_failure(category="availability", attempts=a, max_attempts=20)).delay_seconds
        for a in range(1, 11)
    ]
    assert delays == [0, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]


def test_availability_dlqs_after_cap():
    """At max_attempts, availability DLQs."""
    decision = decide_retry(
        _failure(category="availability", attempts=10, max_attempts=10),
    )
    assert isinstance(decision, DLQAction)


def test_no_model_treated_as_availability():
    """no_model is not a special category — falls through to shared
    availability path. attempts++ each defer, DLQ at cap."""
    decision = decide_retry(
        _failure(category="no_model", attempts=1, max_attempts=10),
    )
    # attempt 1 → idx 0 → 0s (immediate)
    assert decision.action == "immediate" and decision.delay_seconds == 0
