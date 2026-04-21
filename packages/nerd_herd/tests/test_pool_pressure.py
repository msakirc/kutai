import time
from nerd_herd.pool_pressure import compute_pool_pressure


def test_depletion_dominates_below_15pct():
    now = int(time.time())
    p = compute_pool_pressure(remaining=5, limit=100, reset_at=now + 3600, in_flight_count=0)
    assert p.value < -0.5
    assert p.depletion < 0


def test_abundance_peaks_near_reset():
    now = int(time.time())
    p = compute_pool_pressure(remaining=90, limit=100, reset_at=now + 600, in_flight_count=0)
    assert p.value > 0.8
    assert p.abundance > 0.8


def test_no_reset_at_no_time_weight():
    p = compute_pool_pressure(remaining=90, limit=100, reset_at=None, in_flight_count=0)
    assert p.time_weight == 0.0
    assert p.abundance == 0.0


def test_in_flight_reduces_effective_remaining():
    now = int(time.time())
    p1 = compute_pool_pressure(remaining=30, limit=100, reset_at=now + 3600, in_flight_count=0)
    p2 = compute_pool_pressure(remaining=30, limit=100, reset_at=now + 3600, in_flight_count=20)
    assert p2.value < p1.value


def test_zero_limit_returns_neutral():
    p = compute_pool_pressure(remaining=0, limit=0, reset_at=None, in_flight_count=0)
    assert p.value == 0.0
