"""GAP-1 + GAP-2 — ``_stamp_retry_feedback`` is the single chokepoint that

(1) appends the quality rejection to the ledger (so EVERY re-pend applier gets
    ledger coverage, not just the 4 that had a scattered ``_ledger_reject``), and
(2) returns ``escalate: bool`` — True when THIS attempt's output byte-matches
    the immediately-prior ledger entry (degenerate repeat → caller DLQs instead
    of re-pending a non-converging attempt).

Compare-then-append: the prior comparison happens BEFORE the new entry is
appended, so a single attempt can never self-match. Pure ctx in/out — no DB,
no LLM (the lazy ``append_rejection`` / ``_output_hash`` imports are pure).
"""
from __future__ import annotations

import json

from general_beckman.apply import _stamp_retry_feedback
from src.workflows.engine.hooks import _output_hash


def test_chokepoint_reason_lands_in_ledger():
    ctx: dict = {}
    _stamp_retry_feedback(ctx, 2, reason="grade: weak analysis", prev_output="draft-A")
    led = ctx["_rejection_ledger"]
    assert len(led) == 1
    assert led[0]["attempt"] == 2
    assert led[0]["category"] == "quality"
    assert led[0]["reason"] == "grade: weak analysis"
    assert led[0]["out_hash"] == _output_hash("draft-A")


def test_chokepoint_identical_drafts_escalate_true():
    ctx: dict = {}
    first = _stamp_retry_feedback(ctx, 1, reason="r1", prev_output="same-draft")
    assert first is False  # no prior entry to match
    second = _stamp_retry_feedback(ctx, 2, reason="r2", prev_output="same-draft")
    assert second is True
    # still appended both
    assert [e["attempt"] for e in ctx["_rejection_ledger"]] == [1, 2]


def test_chokepoint_different_drafts_escalate_false():
    ctx: dict = {}
    _stamp_retry_feedback(ctx, 1, reason="r1", prev_output="draft-A")
    assert _stamp_retry_feedback(ctx, 2, reason="r2", prev_output="draft-B") is False


def test_chokepoint_first_attempt_never_escalates():
    ctx: dict = {}
    assert _stamp_retry_feedback(ctx, 1, reason="r", prev_output="draft") is False


def test_chokepoint_none_prior_hash_never_escalates():
    # empty/None output hashes to None on both sides -> no comparison -> no escalate
    ctx: dict = {}
    _stamp_retry_feedback(ctx, 1, reason="r1", prev_output="")
    assert _stamp_retry_feedback(ctx, 2, reason="r2", prev_output="") is False


def test_chokepoint_never_self_matches_single_call():
    # one call: the new entry must NOT be compared against itself.
    ctx: dict = {}
    assert _stamp_retry_feedback(ctx, 5, reason="r", prev_output="x") is False


def test_chokepoint_ledger_survives_json_roundtrip():
    ctx: dict = {}
    _stamp_retry_feedback(ctx, 1, reason="r1", prev_output="a")
    _stamp_retry_feedback(ctx, 2, reason="r2", prev_output="b")
    revived = json.loads(json.dumps(ctx))
    assert revived["_rejection_ledger"] == ctx["_rejection_ledger"]


def test_chokepoint_reason_falls_back_to_schema_error():
    # When no explicit reason kwarg, the chokepoint uses ctx["_schema_error"].
    ctx: dict = {"_schema_error": "schema: missing section"}
    _stamp_retry_feedback(ctx, 1)
    assert ctx["_rejection_ledger"][0]["reason"] == "schema: missing section"


def test_chokepoint_still_records_failed_model():
    # GAP-1/2 must not regress the existing model-exclusion behavior.
    ctx: dict = {"generating_model": "qwen-9b"}
    _stamp_retry_feedback(ctx, 1, reason="r", prev_output="x")
    assert ctx["failed_models"] == ["qwen-9b"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
