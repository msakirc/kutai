import pytest
from nerd_herd.types import QueueProfile, RateLimit, RateLimitMatrix
from nerd_herd.signals.s5_queue_calls import s5_queue_calls


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s5_zero_when_no_queue():
    m = _matrix(rpd=RateLimit(limit=1000, remaining=500))
    assert s5_queue_calls(m, queue=QueueProfile(projected_calls=0)) == 0.0


def test_s5_negative_at_120pct_demand():
    m = _matrix(rpd=RateLimit(limit=1000, remaining=500))
    qp = QueueProfile(projected_calls=600)  # 120%
    p = s5_queue_calls(m, queue=qp)
    assert p == pytest.approx(-1.0, abs=0.05)
