"""Availability failures must never be mis-driven down the quality fast-DLQ path.

mission_79 (2026-05-30): reviewer task #225600 failed with
"All models failed for 'reviewer': No model candidates available" — an
AVAILABILITY failure (cloud daily-exhausted) — but its stored error_category
was 'quality' (the stale value from a prior grader-FAIL row; the
ModelCallFailed(error_category='availability') raised in react.py was lost
before reaching _apply_failed, which falls back to the stale row category).

decide_retry only fast-paths category=='quality' (immediate retry, NO backoff,
no capacity-restored wake-up). So an availability failure wearing a 'quality'
label burned all 6 worker_attempts in seconds against an exhausted pool and
DLQ'd — instead of backing off up the ladder (to 24h) and riding out the
quota-reset window.

Fix: _classify_availability_text() sniffs the error string for unambiguous
availability markers ("no model candidates", "no models available",
"all models failed", "rate limit", "daily", "quota", "no candidates") and
overrides a stale non-availability category. An availability failure can then
never take the quality fast-DLQ path regardless of where the category was
dropped upstream.
"""
from __future__ import annotations

import pytest

from general_beckman.apply import _classify_availability_text


# ── the literal mission_79 strings ──────────────────────────────────────────

@pytest.mark.parametrize("err", [
    "All models failed for 'reviewer': No model candidates available",
    "No model candidates available",
    "No model candidates after 3 failure(s)",
    "no models available",
    "Rate limit exceeded (per day)",
    "daily quota exhausted",
])
def test_availability_strings_detected(err):
    assert _classify_availability_text(err) == "availability"


# ── genuine quality / other failures are NOT hijacked ───────────────────────

@pytest.mark.parametrize("err", [
    "Grader rejected output: missing required section",
    "Schema validation: 'foo' has ~0 list items, need >= 5",
    "grader verdict unavailable",     # 'unavailable' alone must NOT trip it
    "degenerate output detected",
    "",
    None,
])
def test_non_availability_strings_pass_through(err):
    assert _classify_availability_text(err) is None
