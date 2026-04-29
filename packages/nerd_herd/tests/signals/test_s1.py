import pytest

from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.signals.s1_remaining import s1_remaining


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s1_empty_matrix_returns_zero():
    assert s1_remaining(_matrix(), reset_in_secs=0, in_flight=0) == 0.0


def test_s1_50pct_remaining_negative_below_depletion_threshold():
    m = _matrix(rpd=RateLimit(limit=100, remaining=50))
    p = s1_remaining(m, reset_in_secs=3600, in_flight=0, profile="per_call")
    assert -0.05 <= p <= 1.0


def test_s1_5pct_remaining_negative():
    m = _matrix(rpd=RateLimit(limit=100, remaining=5))
    p = s1_remaining(m, reset_in_secs=3600, in_flight=0, profile="per_call")
    assert p < -0.5


def test_s1_multi_cell_returns_worst_axis():
    m = _matrix(
        rpm=RateLimit(limit=30, remaining=25),     # 83% remaining
        tpm=RateLimit(limit=6000, remaining=600),  # 10% remaining
    )
    p = s1_remaining(m, reset_in_secs=60, in_flight=0, profile="per_call")
    assert p < -0.3


def test_s1_in_flight_overlay_drops_effective():
    m = _matrix(rpd=RateLimit(limit=100, remaining=10, in_flight=8))
    p = s1_remaining(m, reset_in_secs=3600, in_flight=0, profile="per_call")
    # effective = 10 - 8 = 2 → 2% remaining → strong negative
    assert p < -0.7


def test_s1_abundance_uses_max_when_no_negative_cell():
    m = _matrix(
        rpm=RateLimit(limit=30, remaining=29, reset_at=int(__import__('time').time()) + 600),
        rpd=RateLimit(limit=14_400, remaining=14_300, reset_at=int(__import__('time').time()) + 600),
    )
    p = s1_remaining(m, reset_in_secs=600, in_flight=0, profile="time_bucketed")
    # All cells flush + reset imminent → strong positive abundance via time_decay
    assert p > 0.3
