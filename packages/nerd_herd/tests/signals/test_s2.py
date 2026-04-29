import pytest
from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.signals.s2_call_burden import s2_call_burden


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s2_zero_when_no_call():
    m = _matrix(tpm=RateLimit(limit=6000, remaining=6000))
    assert s2_call_burden(m, est_per_call_tokens=0) == 0.0


def test_s2_zero_when_below_30pct_threshold():
    m = _matrix(tpm=RateLimit(limit=6000, remaining=6000))
    assert s2_call_burden(m, est_per_call_tokens=1500) == 0.0  # 25%


def test_s2_negative_at_60pct_bite():
    m = _matrix(tpm=RateLimit(limit=6000, remaining=6000))
    p = s2_call_burden(m, est_per_call_tokens=3600)  # 60%
    assert -0.5 < p < -0.3


def test_s2_max_negative_at_100pct_bite():
    m = _matrix(tpm=RateLimit(limit=6000, remaining=6000))
    p = s2_call_burden(m, est_per_call_tokens=6000)
    assert p == pytest.approx(-1.0, abs=0.05)


def test_s2_picks_largest_bite_across_windows():
    m = _matrix(
        tpm=RateLimit(limit=6000, remaining=6000),
        tpd=RateLimit(limit=1_000_000, remaining=1_000_000),
    )
    p = s2_call_burden(m, est_per_call_tokens=3000)
    assert -0.5 < p < -0.1
