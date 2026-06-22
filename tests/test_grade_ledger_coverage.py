"""GAP-1 + GAP-2 — ledger coverage now lives in the SINGLE chokepoint.

The proven 48x competitive_positioning loop was a GRADE-rejection loop, but the
ledger (Phase 1) was originally only wired into 4 appliers via a scattered
``_ledger_reject`` — grade/grounding/verify/code_review. Every OTHER quality
re-pend (test_run, semgrep, type_sync, migration, mobile_smoke, the shared
blocker/review helpers, …) re-pended with NO ledger entry.

GAP-1 relocated the append INTO ``_stamp_retry_feedback`` — the sole chokepoint
that EVERY quality re-pend funnels through — so coverage is universal by
construction. GAP-2 made the same chokepoint return ``escalate`` and each
producer re-pend DLQs on a byte-identical degenerate repeat.

This is a source-audit guard (the behaviour is pinned by
``tests/test_chokepoint_stamp_escalate.py``). It fails if the append regresses
back out of the chokepoint, if a scattered ``_ledger_reject`` is reintroduced
(double-append risk), or if a producer re-pend stops honoring the escalate
return.
"""
from __future__ import annotations

from pathlib import Path

_APPLY = Path(__file__).resolve().parents[1] / (
    "packages/general_beckman/src/general_beckman/apply.py"
)


def test_chokepoint_appends_to_ledger():
    src = _APPLY.read_text(encoding="utf-8")
    chokepoint = src.index("def _stamp_retry_feedback(")
    nxt = src.index("\ndef ", chokepoint + 1)
    body = src[chokepoint:nxt]
    assert "append_rejection(" in body, (
        "_stamp_retry_feedback must append to the rejection ledger so EVERY "
        "quality re-pend (not just the 4 historical ones) gets coverage"
    )
    # compare-then-append: the degenerate-repeat comparison reads the prior
    # ledger entry BEFORE the new one is appended (never self-matches).
    assert body.index('ledger[-1].get("out_hash")') < body.index("append_rejection(")


def test_no_scattered_ledger_reject_remains():
    src = _APPLY.read_text(encoding="utf-8")
    # The 4 scattered appends were relocated into the chokepoint. Any
    # surviving call would double-append (the chokepoint already appends).
    assert "_ledger_reject(" not in src
    assert "def _ledger_reject(" not in src


def test_grade_repend_routes_through_chokepoint():
    src = _APPLY.read_text(encoding="utf-8")
    grade = src.index('if a.kind == "grade" and not a.passed:')
    nxt = src.index("\nasync def ", grade)
    body = src[grade:nxt]
    # both grade re-pend arms (bonus + normal) stamp with the grade reason.
    assert body.count('_stamp_retry_feedback(ctx, attempts, reason=f"grade:') == 2


def test_producer_repends_honor_escalate_dlq():
    src = _APPLY.read_text(encoding="utf-8")
    # Each producer-output re-pend that consumes the escalate return pairs a
    # `if _stamp_retry_feedback(...):` guard with a degenerate-repeat DLQ.
    guards = src.count("if _stamp_retry_feedback(ctx, attempts, reason=")
    dlqs = src.count("degenerate repeat: identical output across attempts")
    assert guards >= 18, f"expected the ~20 producer re-pends wired, got {guards}"
    # Every escalate guard DLQs on a degenerate repeat, PLUS the one
    # pre-existing T7 detector in _retry_or_dlq (the empty-result / Failed /
    # Exhausted funnel that the verdict appliers never route through).
    assert dlqs == guards + 1, (
        "each escalate guard must DLQ on degenerate repeat (+1 for the "
        "pre-existing _retry_or_dlq T7 detector)"
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
