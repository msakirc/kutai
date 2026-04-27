"""Test that reviewer agents skip the grade post-hook (handoff D).

Mission 57 task 4391 (1.13 research_quality_review) burned 5 retries
on "High-severity findings often stem from uncited statistics rather
than logical fallacies" — the reviewer's structural verdict was clean,
but the grader was second-guessing the rubric. Reviewers ARE the
quality judge; grading them is double-judgment that burns retry
budget on style.

Architectural fix: ``reviewer`` joins ``grader`` and
``artifact_summarizer`` in ``_NO_POSTHOOKS_AGENT_TYPES``. Reviewer
steps complete on their own verdict; downstream consumers act on
``verdict + issues``.
"""
from __future__ import annotations

import pytest

from general_beckman.posthooks import determine_posthooks


def test_reviewer_no_grade():
    task = {"agent_type": "reviewer"}
    ctx = {}
    assert determine_posthooks(task, ctx, {"status": "completed"}) == []


def test_grader_no_grade():
    """Existing behaviour preserved."""
    task = {"agent_type": "grader"}
    assert determine_posthooks(task, {}, {}) == []


def test_artifact_summarizer_no_grade():
    """Existing behaviour preserved."""
    task = {"agent_type": "artifact_summarizer"}
    assert determine_posthooks(task, {}, {}) == []


def test_mechanical_no_grade():
    task = {"agent_type": "mechanical"}
    assert determine_posthooks(task, {}, {}) == []


def test_shopping_pipeline_no_grade():
    task = {"agent_type": "shopping_pipeline_v2"}
    assert determine_posthooks(task, {}, {}) == []


def test_normal_agent_still_grades():
    """Non-reviewer-class agents still get a grade post-hook by default."""
    for agent in ("coder", "implementer", "analyst", "writer", "executor",
                  "researcher", "planner", "architect", "fixer"):
        task = {"agent_type": agent}
        result = determine_posthooks(task, {}, {})
        assert result == ["grade"], (
            f"agent={agent}: expected ['grade'], got {result}"
        )


def test_explicit_requires_grading_false_still_skips():
    """Per-step opt-out via ctx still works for non-reviewer agents."""
    task = {"agent_type": "analyst"}
    ctx = {"requires_grading": False}
    assert determine_posthooks(task, ctx, {}) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
