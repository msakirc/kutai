import pytest
from nerd_herd.signals.s6_capable_supply import s6_capable_supply


class FakeModel:
    def __init__(self, name, capabilities, rpd_remaining, supports_difficulty):
        self.name = name
        self.capabilities = capabilities
        self.rpd_remaining = rpd_remaining
        self.supports_difficulty = supports_difficulty


def test_s6_zero_when_no_demand():
    model = FakeModel("a", {"vision"}, 100, lambda d: True)
    queue = {"by_capability": {}, "by_difficulty": {}}
    eligible = [model]
    p = s6_capable_supply(model, queue=queue, eligible_models=eligible)
    assert p == 0.0


def test_s6_zero_when_supply_meets_demand():
    m = FakeModel("a", {"vision"}, 100, lambda d: True)
    queue = {"by_capability": {"vision": 5}, "by_difficulty": {}}
    p = s6_capable_supply(m, queue=queue, eligible_models=[m], iter_avg=10)
    # demand 5 calls × 10 iters = 50; supply 100 × 10 = 1000; ratio 0.05 → no pressure
    assert p == 0.0


def test_s6_negative_when_demand_exceeds_supply():
    m = FakeModel("a", {"vision"}, 50, lambda d: True)
    queue = {"by_capability": {"vision": 50}, "by_difficulty": {}}
    p = s6_capable_supply(m, queue=queue, eligible_models=[m], iter_avg=10)
    # demand 500; supply 500; ratio 1.0; excess 0.3 * SLOPE 2 = 0.6 → -0.6
    # Adjust: with 500 demand and 500 supply, ratio = 1.0; pressure = -clip(0.3 * 2, 0, 1) = -0.6
    assert -1.0 <= p < 0.0


def test_s6_zero_when_model_not_eligible():
    m = FakeModel("a", {"function_calling"}, 100, lambda d: True)  # no vision
    queue = {"by_capability": {"vision": 50}, "by_difficulty": {}}
    p = s6_capable_supply(m, queue=queue, eligible_models=[], iter_avg=10)
    assert p == 0.0
