import pytest
from nerd_herd.types import QueueProfile, RateLimit, RateLimitMatrix
from nerd_herd.signals.s4_queue_tokens import s4_queue_tokens


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s4_zero_when_no_queue():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=0)
    assert s4_queue_tokens(m, queue=qp) == 0.0


def test_s4_zero_below_70pct_demand_ratio():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=300_000)  # 60% of 500k
    assert s4_queue_tokens(m, queue=qp) == 0.0


def test_s4_negative_at_95pct_demand():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=475_000)  # 95% of remaining
    p = s4_queue_tokens(m, queue=qp)
    assert -0.6 < p < -0.4


def test_s4_clipped_at_oversubscription():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=750_000)  # 150% — over budget
    p = s4_queue_tokens(m, queue=qp)
    assert p == pytest.approx(-1.0, abs=0.05)
