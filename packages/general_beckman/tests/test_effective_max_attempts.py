"""Transient failures get an effective cap of the full ladder, everywhere.

mission_79 (2026-05-30): reviewer #225600 was correctly classified
error_category=availability, yet still DLQ'd with "Worker attempts exceeded:
6/6". decide_retry gives transient categories an effective cap of the full
backoff ladder (15) — but the admission cap-guard (__init__.py) and sweep
section 8 force-DLQ on the RAW max_worker_attempts (6), bypassing that
extension. The two enforcement sites disagreed with decide_retry, so an
availability task that decide_retry would keep retrying got killed at the
admission gate.

effective_max_attempts() centralizes the rule so all three sites agree:
transient/availability → max(cap, ladder length); quality/other → cap as-is.
"""
from __future__ import annotations

import pytest

from general_beckman.retry import (
    effective_max_attempts,
    TRANSIENT_CATEGORIES,
    _BACKOFF_SECONDS,
)

_LADDER = len(_BACKOFF_SECONDS)


@pytest.mark.parametrize("cat", sorted(TRANSIENT_CATEGORIES))
def test_transient_categories_extended_to_ladder(cat):
    assert effective_max_attempts(cat, 6) == _LADDER


def test_transient_keeps_larger_explicit_cap():
    assert effective_max_attempts("availability", _LADDER + 5) == _LADDER + 5


@pytest.mark.parametrize("cat", ["quality", "worker", "unknown", None, ""])
def test_non_transient_keeps_raw_cap(cat):
    assert effective_max_attempts(cat, 6) == 6


def test_availability_at_6_is_below_effective_cap():
    """The exact #225600 case: att=6, raw cap=6, but availability → cap 15."""
    assert 6 < effective_max_attempts("availability", 6)


def test_zero_cap_unchanged():
    # max_att<=0 means "no cap" sentinel — don't inflate it.
    assert effective_max_attempts("availability", 0) == _LADDER
    assert effective_max_attempts("quality", 0) == 0
