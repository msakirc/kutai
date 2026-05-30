"""_grader_verdict_text must surface the auto-fail message under ``raw``.

mission_79 (2026-05-30): direct_competitor_identification (#225586) DLQ'd with
"grader verdict unavailable" — an opaque sentinel, not the real reason. The
grade child kept failing to get a model (cloud daily-exhausted), so
_grade_resume_err / _grade_resume built an AUTO-FAIL verdict of shape
``{"passed": False, "raw": "auto-fail: grader call failed (No model
candidates available)"}``. _grader_verdict_text's dict branch reads
insight/strategy/situation/message/error and the failed-axis booleans, but
NEVER the ``raw`` key — so the auto-fail message was dropped and the function
returned its "grader verdict unavailable" fallback.

(My earlier "BUG B retracted" was premature: the parser handles the *normal*
RELEVANT/VERDICT grade-FAIL shape but not this auto-fail shape.)

Fix: read ``raw`` as a last meaningful-text candidate before the sentinel.
"""
from __future__ import annotations

import pytest

from general_beckman.apply import _grader_verdict_text


def test_autofail_grader_call_failed_surfaces_message():
    raw = {"passed": False, "raw": "auto-fail: grader call failed (No model candidates available)"}
    out = _grader_verdict_text(raw)
    assert out != "grader verdict unavailable"
    assert "grader call failed" in out
    assert "No model candidates" in out


def test_autofail_grader_incapable_surfaces_message():
    raw = {"passed": False, "raw": "auto-fail: grader_incapable after 2 attempts: junk"}
    out = _grader_verdict_text(raw)
    assert "grader_incapable" in out


def test_normal_fail_still_prefers_situation_over_raw():
    """The structured fields still win — raw is only a last resort."""
    raw = {
        "passed": False, "relevant": True, "complete": False,
        "situation": "Identify direct competitors",
        "raw": "RELEVANT: YES\nCOMPLETE: NO\n...",
    }
    assert _grader_verdict_text(raw) == "Identify direct competitors"


def test_failed_axes_still_win_over_raw_when_no_text():
    raw = {"passed": False, "relevant": False, "complete": False,
           "raw": "RELEVANT: NO\nCOMPLETE: NO"}
    assert _grader_verdict_text(raw) == "grader rejected: relevant, complete"


def test_truly_empty_still_unavailable():
    assert _grader_verdict_text({"passed": False}) == "grader verdict unavailable"
    assert _grader_verdict_text({"passed": False, "raw": ""}) == "grader verdict unavailable"
    assert _grader_verdict_text({"passed": False, "raw": "   "}) == "grader verdict unavailable"
