"""z10-wire-fixes F8 — confidence-outcome wires beyond code_review.

The audit found `_record_and_resolve_confidence` only fired on
`_apply_code_review_verdict`. Code-review verdicts are too sparse to hit
the 30-sample-per-cell threshold the prompt builder needs. Extend to
grade + grounding verdicts so the calibration table fills up.

These tests are STATIC — they inspect the apply.py source for the wire
sites. The runtime path is exercised by existing
``tests/infra/test_confidence_outcomes.py`` plus the verdict-apply tests.
A source-level guard is the cheapest regression catch.
"""
from __future__ import annotations

import inspect

from general_beckman import apply as apply_mod


def _src() -> str:
    return inspect.getsource(apply_mod)


def test_code_review_wire_still_present():
    """Z10 T4B baseline wire must remain."""
    src = _src()
    # Both pass + fail branches should record.
    assert src.count('source="reviewer_verdict"') >= 2, (
        "F8: reviewer_verdict wire regressed — need pass AND fail branches"
    )


def test_grade_wire_added_on_pass_and_fail():
    """F8: _apply_posthook_verdict grade branches record + resolve.

    Three wires expected:
      - grade-PASS (final pending cleared OR not yet)
      - grade-FAIL retry path
      - grade-FAIL DLQ path
    """
    src = _src()
    grade_hits = src.count('source="grade"')
    assert grade_hits >= 3, (
        f"F8: expected >=3 grade wires (pass + fail retry + fail DLQ), "
        f"got {grade_hits}"
    )


def test_grounding_wire_added_on_pass_and_fail():
    """F8: _apply_grounding_verdict records on both verdict branches."""
    src = _src()
    grounding_hits = src.count('source="grounding"')
    assert grounding_hits >= 2, (
        f"F8: expected >=2 grounding wires (pass + fail), "
        f"got {grounding_hits}"
    )


def test_wires_call_record_and_resolve_helper():
    """Each new wire must use _record_and_resolve_confidence — the helper
    no-ops cleanly when source task has no confidence signal, so it's
    safe to fire on every verdict."""
    src = _src()
    # The helper should appear at minimum 5 times: 2 (reviewer) + 3 (grade)
    # + 2 (grounding) = 7 expected. Allow some headroom for refactors.
    hits = src.count("_record_and_resolve_confidence(")
    assert hits >= 7, (
        f"F8: _record_and_resolve_confidence call count regression: {hits}"
    )
