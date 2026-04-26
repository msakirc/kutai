"""Test negation-aware ``_is_disguised_failure`` (2026-04-27 fix).

Mission 57 task 4373 burned 5 retries because problem-statement output
contained ``"is not a critical blocker"`` and the substring
``"critical blocker"`` (from ``_FAILURE_PHRASES``) tripped detection
without seeing the negation.

Fix: per-occurrence sentence-bounded negation check. ``shell tool``
alone removed from hard list (was matching legit references in design
docs / post-mortems).
"""
from __future__ import annotations

import pytest

from src.workflows.engine.hooks import (
    _is_disguised_failure,
    _phrase_matches_unnegated,
)


# ── Phrase helper ──────────────────────────────────────────────────────


def test_phrase_unnegated_true_when_positive_use():
    assert _phrase_matches_unnegated(
        "the system has a critical blocker preventing rollout",
        "critical blocker",
    ) is True


def test_phrase_unnegated_false_when_only_negated():
    assert _phrase_matches_unnegated(
        "this is not a critical blocker for our use case",
        "critical blocker",
    ) is False


def test_phrase_unnegated_sentence_bounded():
    """Negation in earlier sentence does NOT mask positive use later."""
    assert _phrase_matches_unnegated(
        "no critical blocker here. but the next module has a critical blocker indeed.",
        "critical blocker",
    ) is True


def test_phrase_unnegated_multiple_negation_tokens():
    """Various negation tokens recognized."""
    cases = [
        ("isn't a critical blocker", "critical blocker"),
        ("wasn't a critical blocker", "critical blocker"),
        ("without a critical blocker", "critical blocker"),
        ("never a critical blocker", "critical blocker"),
    ]
    for text, phrase in cases:
        assert _phrase_matches_unnegated(text, phrase) is False, (
            f"failed for: {text}"
        )


# ── Real-world canaries ────────────────────────────────────────────────


def test_real_4373_problem_statement_passes():
    """Mission 57 task 4373 — actual artifact that was DLQ'd 5x."""
    output = (
        '{"severity": "Medium - Causes inconvenience and inefficiency '
        'but is not a critical blocker. Frustration increases with '
        'household complexity."}'
    )
    assert _is_disguised_failure(output) is False


def test_positive_shell_tool_reference_passes():
    """Bare ``shell tool`` was removed from hard list — safe to mention."""
    text = "The shell tool is working fine — no issues."
    assert _is_disguised_failure(text) is False


def test_design_doc_about_failure_modes_passes():
    """``failure mode`` is in false-positive list — design docs OK."""
    text = (
        "Failure mode 1: cache miss. Failure mode 2: rate limit. "
        "Both have proper error handling and graceful degradation."
    )
    assert _is_disguised_failure(text) is False


# ── Real failures must still trip ──────────────────────────────────────


def test_blocked_shell_tool_still_trips():
    text = "Cannot run tests because the shell tool is blocked in this environment."
    assert _is_disguised_failure(text) is True


def test_two_failure_phrases_still_trip():
    text = "Verification status: failed. Task failed."
    assert _is_disguised_failure(text) is True


def test_raw_tool_call_envelope_still_trips():
    """Agent emitted tool_call instead of final_answer."""
    text = '{"action": "tool_call", "tool": "x", "args": {}}'
    assert _is_disguised_failure(text) is True


def test_one_positive_use_in_mixed_sentences_trips():
    """When one sentence negates the phrase but another uses it positively,
    the positive use should win."""
    text = "Severity: not a critical blocker. Status: critical blocker for downstream tasks."
    assert _is_disguised_failure(text) is True


def test_short_output_passes():
    """Below 10 chars is too short to evaluate."""
    assert _is_disguised_failure("ok") is False
    assert _is_disguised_failure("") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
