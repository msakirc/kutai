"""InFlightTracker — Task 8 of pool-pressure-shared."""
import time

import pytest

from kuleden_donen_var.in_flight import InFlightHandle, InFlightTracker


def test_begin_call_increments_count():
    t = InFlightTracker()
    t.begin_call("anthropic", "claude-sonnet-4-6", ttl_s=60)
    assert t.count("anthropic", "claude-sonnet-4-6") == 1


def test_end_call_decrements_count():
    t = InFlightTracker()
    h = t.begin_call("anthropic", "claude-sonnet-4-6")
    t.end_call(h)
    assert t.count("anthropic", "claude-sonnet-4-6") == 0


def test_ttl_prunes_expired():
    t = InFlightTracker()
    t.begin_call("anthropic", "claude-sonnet-4-6", ttl_s=0.01)
    time.sleep(0.05)
    t.begin_call("anthropic", "claude-sonnet-4-6")     # triggers prune
    assert t.count("anthropic", "claude-sonnet-4-6") == 1


def test_end_call_is_idempotent():
    t = InFlightTracker()
    h = t.begin_call("anthropic", "claude-sonnet-4-6")
    t.end_call(h)
    t.end_call(h)
    assert t.count("anthropic", "claude-sonnet-4-6") == 0


def test_separate_keys_independent():
    t = InFlightTracker()
    t.begin_call("anthropic", "claude-sonnet-4-6")
    t.begin_call("groq", "llama-3.1-70b")
    assert t.count("anthropic", "claude-sonnet-4-6") == 1
    assert t.count("groq", "llama-3.1-70b") == 1
