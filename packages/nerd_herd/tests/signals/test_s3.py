import pytest
from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.signals.s3_task_burden import s3_task_burden


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s3_zero_when_no_task_tokens():
    assert s3_task_burden(_matrix(tpm=RateLimit(limit=6000, remaining=6000)), est_per_task_tokens=0) == 0.0


def test_s3_uses_largest_window_bite():
    m = _matrix(
        tpm=RateLimit(limit=6000, remaining=6000),
        tpd=RateLimit(limit=1_000_000, remaining=1_000_000),
    )
    p = s3_task_burden(m, est_per_task_tokens=50_000)
    assert p < -0.5
