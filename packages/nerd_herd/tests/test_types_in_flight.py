"""RateLimit.in_flight — Task 7 of pool-pressure-shared."""
from nerd_herd.types import RateLimit


def test_rate_limit_default_in_flight_zero():
    rl = RateLimit(limit=100, remaining=50, reset_at=0)
    assert rl.in_flight == 0


def test_rate_limit_accepts_in_flight():
    rl = RateLimit(limit=100, remaining=50, reset_at=0, in_flight=3)
    assert rl.in_flight == 3
