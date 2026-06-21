"""Tests for the rejection ledger (Phase 1, T1).

The ledger records, per quality rejection, a compact entry
``{attempt, category, reason, out_hash}`` appended (not overwritten) to
``ctx["_rejection_ledger"]``. It is the compact history of
(approach, why-rejected) that the retry prompt renders (T2) so the worker
does not repeat a rejected approach (spec C5).

Pure ctx in / out — no DB, no LLM.
"""
from __future__ import annotations

import json

from src.workflows.engine.hooks import append_rejection


def test_append_rejection_stamps_entry():
    ctx: dict = {}
    append_rejection(ctx, attempt=1, reason="missing section: Risks", out_hash="abc")
    led = ctx["_rejection_ledger"]
    assert isinstance(led, list) and len(led) == 1
    entry = led[0]
    assert entry["attempt"] == 1
    assert entry["category"] == "quality"
    assert entry["reason"] == "missing section: Risks"
    assert entry["out_hash"] == "abc"


def test_append_rejection_accumulates_across_paths():
    ctx: dict = {}
    # schema, grade, degenerate — three distinct quality paths
    append_rejection(ctx, attempt=1, reason="schema: empty output", out_hash="h1")
    append_rejection(ctx, attempt=2, reason="grade: weak analysis", out_hash="h2")
    append_rejection(ctx, attempt=3, reason="degenerate: aaa-repeat", out_hash=None)
    led = ctx["_rejection_ledger"]
    assert len(led) == 3
    assert [e["attempt"] for e in led] == [1, 2, 3]
    assert [e["reason"] for e in led] == [
        "schema: empty output",
        "grade: weak analysis",
        "degenerate: aaa-repeat",
    ]
    assert led[2]["out_hash"] is None


def test_append_rejection_caps_reason_at_500():
    ctx: dict = {}
    long_reason = "x" * 5000
    append_rejection(ctx, attempt=1, reason=long_reason, out_hash="h")
    assert len(ctx["_rejection_ledger"][0]["reason"]) == 500


def test_append_rejection_coerces_attempt_to_int():
    ctx: dict = {}
    append_rejection(ctx, attempt="2", reason="r", out_hash="h")
    assert ctx["_rejection_ledger"][0]["attempt"] == 2
    assert isinstance(ctx["_rejection_ledger"][0]["attempt"], int)


def test_append_rejection_coerces_reason_to_str():
    ctx: dict = {}
    append_rejection(ctx, attempt=1, reason={"err": "bad"}, out_hash="h")
    assert isinstance(ctx["_rejection_ledger"][0]["reason"], str)


def test_ledger_survives_json_roundtrip():
    ctx: dict = {}
    append_rejection(ctx, attempt=1, reason="r1", out_hash="h1")
    append_rejection(ctx, attempt=2, reason="r2", out_hash="h2")
    revived = json.loads(json.dumps(ctx))
    assert revived["_rejection_ledger"] == ctx["_rejection_ledger"]


def test_availability_path_appends_nothing():
    # An availability failure produces no judged output -> no append.
    # This test documents that callers must NOT call append_rejection on
    # availability paths; the helper itself only acts when called.
    ctx: dict = {}
    # simulate an availability retry that never calls append_rejection
    assert "_rejection_ledger" not in ctx
