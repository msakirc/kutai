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


# ── Per-minute (rpm) is PACING, not conservation (2026-06-18) ─────────────────

def test_s5_ignores_per_minute_window():
    # rpm refills every minute — a deep queue never exhausts it. S5 must stay 0
    # even when projected calls dwarf the per-minute limit (this was the live
    # waste vector: free models floored on rpm=5-15 while their daily was full).
    m = _matrix(rpm=RateLimit(limit=10, remaining=10))
    qp = QueueProfile(projected_calls=100)
    assert s5_queue_calls(m, queue=qp) == 0.0


def test_s5_fires_on_daily_overshoot():
    # The genuine conservation case S5 exists for: queue projects to exhaust
    # the DAILY request budget (5 planners @ 40 reqs vs gemini's 20/day).
    m = _matrix(rpd=RateLimit(limit=20, remaining=20))
    qp = QueueProfile(projected_calls=40)  # 2x daily -> conserve
    p = s5_queue_calls(m, queue=qp)
    assert p == pytest.approx(-1.0, abs=0.05)
