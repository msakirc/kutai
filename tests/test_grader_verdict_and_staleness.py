"""Regressions for two correctness fixes in general_beckman.apply:

1. Grader verdict text leaks task title — when a thinking-model grader echoes
   the input ``Task: <title>`` line as its SITUATION/INSIGHT field, the
   verdict-text extractor used to surface it as the rejection reason.
2. Retry feedback staleness — ``_schema_error``/``_prev_output`` written for
   one attempt could leak into the next prompt as "your last output failed:
   <unrelated error>" when re-queue happened via a non-schema path.

Both are observable on task 2889 (mission 46) before the fix.
"""
from __future__ import annotations

import pytest

from general_beckman.apply import (
    _grader_verdict_text,
    _is_title_echo,
    _stamp_retry_feedback,
    _drop_stale_retry_feedback,
)


class TestGraderTitleEcho:
    def test_title_echo_in_insight_falls_through(self):
        title = "MVP scope definition task"
        raw = {"insight": "MVP scope definition task", "strategy": "", "situation": ""}
        assert _grader_verdict_text(raw, source_title=title) == "grader verdict unavailable"

    def test_title_echo_skipped_then_next_field_used(self):
        title = "MVP scope definition task"
        raw = {
            "insight": "MVP scope definition task",
            "strategy": "missing acceptance criteria for scope items",
        }
        assert (
            _grader_verdict_text(raw, source_title=title)
            == "missing acceptance criteria for scope items"
        )

    def test_no_title_provided_preserves_old_behavior(self):
        raw = {"insight": "something useful"}
        assert _grader_verdict_text(raw) == "something useful"

    def test_substring_echo_detected(self):
        assert _is_title_echo("MVP scope", "MVP scope definition task")
        assert _is_title_echo("MVP SCOPE DEFINITION TASK", "MVP scope definition task")

    def test_long_text_not_treated_as_echo(self):
        long_val = "this is a real analytic paragraph " + ("x" * 100)
        assert not _is_title_echo(long_val, "MVP")

    def test_failed_axes_fallback_still_works(self):
        title = "Some task"
        raw = {"insight": "Some task", "relevant": False, "complete": False}
        assert (
            _grader_verdict_text(raw, source_title=title)
            == "grader rejected: relevant, complete"
        )


class TestRetryFeedbackStaleness:
    def test_stamp_only_when_feedback_present(self):
        ctx = {}
        _stamp_retry_feedback(ctx, 3)
        assert "_schema_error_for_attempt" not in ctx

        ctx = {"_schema_error": "x"}
        _stamp_retry_feedback(ctx, 3)
        assert ctx["_schema_error_for_attempt"] == 3

    def test_fresh_feedback_kept(self):
        ctx = {
            "_schema_error": "still fresh",
            "_prev_output": "...",
            "_schema_error_for_attempt": 5,
        }
        _drop_stale_retry_feedback(ctx, 5)
        assert ctx["_schema_error"] == "still fresh"
        assert ctx["_prev_output"] == "..."

    def test_stale_feedback_dropped(self):
        ctx = {
            "_schema_error": "from 3 attempts ago",
            "_prev_output": "...",
            "_schema_error_for_attempt": 2,
        }
        _drop_stale_retry_feedback(ctx, 5)
        assert "_schema_error" not in ctx
        assert "_prev_output" not in ctx
        assert "_schema_error_for_attempt" not in ctx

    def test_untagged_legacy_feedback_dropped(self):
        # Pre-staleness-scheme rows have no stamp. Conservative drop.
        ctx = {"_schema_error": "legacy untagged"}
        _drop_stale_retry_feedback(ctx, 1)
        assert "_schema_error" not in ctx

    def test_no_op_when_nothing_to_drop(self):
        ctx = {"some_other_key": "preserved"}
        _drop_stale_retry_feedback(ctx, 5)
        assert ctx == {"some_other_key": "preserved"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
