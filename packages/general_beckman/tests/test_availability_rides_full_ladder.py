"""Availability failures must ride the FULL backoff ladder, not the quality cap.

mission_79 (2026-05-30): reviewer #225600 had max_worker_attempts=6 (the
add_task default), but the availability backoff ladder has 15 steps and only
reaches a 24h wait at the tail. With max=6 the ladder bottoms out at idx 5
(300s) — cumulative ~9 min — so an availability failure DLQ'd long before a
daily-exhausted gemini quota could reset (~24h). The founder's rule: a task
that can't get capacity must WAIT for it, not DLQ.

Fix: decide_retry gives availability/transient categories an effective cap of
max(task_max, len(ladder)) so they ride the whole ladder (to the 24h step)
regardless of the task's quality-oriented worker-attempt budget. Quality /
deterministic failures keep the task's own (smaller) cap — retrying them 15x
just burns tokens on the same failure.
"""
from __future__ import annotations

import pytest

from general_beckman.retry import decide_retry, RetryDecision, DLQAction, _BACKOFF_SECONDS


_LADDER = len(_BACKOFF_SECONDS)  # 15


def _f(category, attempts, max_attempts=6):
    return {
        "category": category,
        "worker_attempts": attempts,
        "max_worker_attempts": max_attempts,
        "error": "No model candidates available",
    }


def test_availability_does_not_dlq_at_quality_cap():
    """At attempts=6 (the old DLQ point) availability must still retry."""
    d = decide_retry(_f("availability", attempts=6, max_attempts=6))
    assert isinstance(d, RetryDecision), f"expected retry, got {d}"
    assert d.action == "delayed"


def test_availability_rides_deep_into_ladder():
    """Deep into the ladder (attempts=14) availability is still WAITING, not DLQ.

    backoff idx = min(attempts-1, len-1); attempts=14 → idx=13 → 43200 (12h).
    The final 86400 (24h, idx=14) entry is the boundary: attempts=15 is the
    DLQ decision point (15 < 15 is False), so 24h is reached as elapsed wall-
    clock across the ladder, not as a single delay. The point that matters:
    at attempts=14, well past the old max=6 cap, the task still retries."""
    d = decide_retry(_f("availability", attempts=14, max_attempts=6))
    assert isinstance(d, RetryDecision)
    assert d.delay_seconds == 43200  # 12h — still riding the ladder, not DLQ


def test_availability_dlqs_only_after_full_ladder():
    d = decide_retry(_f("availability", attempts=_LADDER, max_attempts=6))
    assert isinstance(d, DLQAction)


@pytest.mark.parametrize("cat", ["availability", "daily_exhausted", "rate_limited", "no_model", "timeout"])
def test_transient_categories_all_ride_full_ladder(cat):
    d = decide_retry(_f(cat, attempts=6, max_attempts=6))
    assert isinstance(d, RetryDecision), f"{cat} should retry past quality cap"


def test_quality_still_dlqs_at_task_cap():
    """Quality keeps the task's own cap — no free 15x retries on a dead-end."""
    d = decide_retry({
        "category": "quality", "worker_attempts": 6, "max_worker_attempts": 6,
        "error": "Grader rejected output",
    })
    assert isinstance(d, DLQAction)


def test_quality_below_cap_still_immediate():
    d = decide_retry({
        "category": "quality", "worker_attempts": 2, "max_worker_attempts": 6,
        "error": "Grader rejected output",
    })
    assert isinstance(d, RetryDecision)
    assert d.action == "immediate"
    assert d.delay_seconds == 0
