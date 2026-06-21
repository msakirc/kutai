"""GAP 1 — the GRADE quality applier must append to the rejection ledger.

The proven 48× competitive_positioning loop was a GRADE-rejection loop, but
the ledger (Phase 1) was only wired into the grounding/verify/code_review
appliers — the grade applier re-pended with no ledger entry, so the retry
prompt's "Prior attempts" render stayed empty exactly where it mattered.

This pins that every quality re-pend applier records a ledger entry. It is a
source-audit guard (the behavioural append is covered by
``tests/test_rejection_ledger.py``); it fails if the `_ledger_reject` call is
removed from any of these appliers.
"""
from __future__ import annotations

import re
from pathlib import Path

_APPLY = Path(__file__).resolve().parents[1] / (
    "packages/general_beckman/src/general_beckman/apply.py"
)


def _applier_body(src: str, marker: str) -> str:
    """Return the slice of apply.py from a verdict-kind marker to the next."""
    start = src.index(marker)
    # next applier branch or end
    nxt = re.search(r'\n        if a\.kind == "', src[start + len(marker):])
    end = start + len(marker) + (nxt.start() if nxt else len(src))
    return src[start:end]


def test_grade_applier_appends_to_rejection_ledger():
    src = _APPLY.read_text(encoding="utf-8")
    grade = _applier_body(src, 'if a.kind == "grade" and not a.passed:')
    assert "_ledger_reject(" in grade, (
        "grade applier must call _ledger_reject so the GRADE loop "
        "(the 48x symptom) lands in the rejection ledger"
    )
    # quality-only: the ledger call must sit AFTER the availability-masquerade
    # guard (availability re-dispatches carry no judged output).
    assert grade.index("_grade_verdict_is_availability") < grade.index("_ledger_reject(")


def test_all_quality_appliers_keep_ledger_coverage():
    src = _APPLY.read_text(encoding="utf-8")
    for marker in (
        'if a.kind == "grade" and not a.passed:',
        '_ledger_reject(ctx, attempts, f"grounding:',
        '_ledger_reject(ctx, attempts, f"verify_artifacts:',
        '_ledger_reject(ctx, attempts, f"code_review:',
    ):
        assert marker.split("(")[0] in src or marker in src


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
