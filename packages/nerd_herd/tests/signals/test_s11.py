import pytest
from nerd_herd.signals.s11_cost import s11_cost


def test_s11_zero_when_no_cap():
    p = s11_cost(est_call_cost=0.05, daily_cost_remaining=0.0)
    assert p == 0.0


def test_s11_zero_when_below_threshold():
    p = s11_cost(est_call_cost=0.05, daily_cost_remaining=10.0)
    assert p == 0.0  # 0.5% of remaining


def test_s11_negative_when_call_eats_majority():
    p = s11_cost(est_call_cost=8.0, daily_cost_remaining=10.0)
    assert p < -0.3
