# packages/fatih_hoca/tests/test_capability_curve.py
"""Tests for capability_curve module (Phase 2d)."""
from fatih_hoca.capability_curve import (
    CAP_NEEDED_BY_DIFFICULTY,
    cap_needed_for_difficulty,
)


def test_dict_has_all_difficulties_1_through_10():
    for d in range(1, 11):
        assert d in CAP_NEEDED_BY_DIFFICULTY
        assert 0 <= CAP_NEEDED_BY_DIFFICULTY[d] <= 100


def test_monotonic_non_decreasing():
    prev = -1.0
    for d in range(1, 11):
        v = CAP_NEEDED_BY_DIFFICULTY[d]
        assert v >= prev, f"d={d} ({v}) < d={d-1} ({prev})"
        prev = v


def test_lookup_returns_dict_value():
    assert cap_needed_for_difficulty(1) == CAP_NEEDED_BY_DIFFICULTY[1]
    assert cap_needed_for_difficulty(5) == CAP_NEEDED_BY_DIFFICULTY[5]
    assert cap_needed_for_difficulty(10) == CAP_NEEDED_BY_DIFFICULTY[10]


def test_lookup_clamps_below_range():
    assert cap_needed_for_difficulty(0) == CAP_NEEDED_BY_DIFFICULTY[1]
    assert cap_needed_for_difficulty(-5) == CAP_NEEDED_BY_DIFFICULTY[1]


def test_lookup_clamps_above_range():
    assert cap_needed_for_difficulty(11) == CAP_NEEDED_BY_DIFFICULTY[10]
    assert cap_needed_for_difficulty(99) == CAP_NEEDED_BY_DIFFICULTY[10]
