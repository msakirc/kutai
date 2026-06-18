"""Transient empty-response handling in the ReAct loop.

Empties are transient (server warmup / --parallel 1 slot contention): the same
task recovers within ~8s with identical messages. The loop must back off between
empty retries (not hammer instantly) and surface finish_reason/usage so the next
empty is diagnosable. See 2026-06-18 debugging session.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from coulson.dispatch_helpers import result_to_response_dict
from coulson.react import _empty_retry_plan
from hallederiz_kadir.types import CallResult


def _make_call_result(finish_reason="stop"):
    return CallResult(
        content="", tool_calls=None, thinking=None, usage={}, cost=0.0,
        latency=0.0, model="m", model_name="m", is_local=True,
        provider="local", task="t", finish_reason=finish_reason,
    )


def test_result_dict_carries_finish_reason():
    """react reads the response dict; finish_reason must survive the mapping
    so the empty-response log can record it."""
    d = result_to_response_dict(_make_call_result(finish_reason="length"), model="m")
    assert d["finish_reason"] == "length"


def test_empty_backoff_escalates():
    """Backoff grows with consecutive empties — give the server time to recover
    instead of re-hammering at 0ms."""
    b1, _ = _empty_retry_plan(1)
    b2, _ = _empty_retry_plan(2)
    b3, _ = _empty_retry_plan(3)
    assert b1 > 0
    assert b2 > b1
    assert b3 > b2


def test_empty_backoff_capped():
    """Backoff is capped so a wedged server doesn't stall a task for minutes."""
    backoff, _ = _empty_retry_plan(20)
    assert backoff <= 8.0


def test_empty_does_not_give_up_before_threshold():
    """A handful of transient empties must NOT fail the task — 459147 recovered
    after ~10 empties with identical messages."""
    _, give_up = _empty_retry_plan(3)
    assert give_up is False


def test_empty_gives_up_eventually():
    """A genuinely dead model still terminates — backoff total stays bounded."""
    _, give_up = _empty_retry_plan(99)
    assert give_up is True
