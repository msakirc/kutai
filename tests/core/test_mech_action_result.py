"""The orchestrator must propagate a mechanical Action's needs_review status.

Bug (2026-05-26, post-restart): the mechanical dispatch branch mapped
Action.status to a result dict but had NO needs_review case — it fell through
to {"status": "failed"}. So find_similar_missions' needs_review (prior
missions matched → founder Continue/Branch/Abort) became a worker failure,
was retried 6× on the backoff ladder, and DLQ'd (#166396). result_router
never saw needs_review, so the downstream RequestReview / _apply_review
completion path never ran.
"""
from __future__ import annotations

import json

from src.core.orchestrator import _mech_action_to_result


class _Action:
    def __init__(self, status, result=None, error=""):
        self.status = status
        self.result = result if result is not None else {}
        self.error = error


def test_needs_review_propagates():
    r = _mech_action_to_result(
        _Action("needs_review", {"matches": [1, 2, 3]},
                "find_similar_missions: matches=3")
    )
    assert r["status"] == "needs_review", \
        "needs_review must propagate (not collapse to failed → worker DLQ)"


def test_completed_maps_to_completed():
    r = _mech_action_to_result(_Action("completed", {"snap": 1}))
    assert r["status"] == "completed"
    assert json.loads(r["result"]) == {"snap": 1}


def test_needs_clarification_preserved():
    r = _mech_action_to_result(_Action("needs_clarification", {"q": "?"}))
    assert r["status"] == "needs_clarification"


def test_failed_maps_to_failed_with_error():
    r = _mech_action_to_result(_Action("failed", None, "boom"))
    assert r["status"] == "failed"
    assert r["error"] == "boom"
