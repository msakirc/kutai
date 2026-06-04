from types import SimpleNamespace
from nerd_herd.types import QueueProfile
from nerd_herd.signals.s6_capable_supply import s6_capable_supply, SAT


def _vm(rpd=20):
    return SimpleNamespace(name="v/m", provider="p", is_local=False, is_free=False,
                           cap_score=8.5, capabilities={"vision"}, rpd_remaining=rpd)


def test_empty_demand_zero():
    assert s6_capable_supply(_vm(), queue=QueueProfile(), eligible_models=[_vm()]) == 0.0


def test_no_eligible_models_zero():
    vm = _vm()
    q = QueueProfile(by_capability={"vision": 100}, total_ready_count=100)
    assert s6_capable_supply(vm, queue=q, eligible_models=[]) == 0.0


def test_ramp_continuous_from_low_shortage():
    vm = _vm(rpd=20)
    # Light demand below the OLD 0.70 ratio must now be nonzero (de-blinded).
    # demand = 12*8 = 96; supply = 20*8 = 160; ratio = 0.6 < 0.70 → old gave 0.
    q = QueueProfile(by_capability={"vision": 12}, total_ready_count=12)
    s6 = s6_capable_supply(vm, queue=q, eligible_models=[vm], iter_avg=8.0)
    assert s6 < 0.0


def test_monotonic_in_shortage():
    vm = _vm(rpd=20)
    mags = []
    for demand in (12, 25, 50, 100, 400):
        q = QueueProfile(by_capability={"vision": demand}, total_ready_count=demand)
        mags.append(-s6_capable_supply(vm, queue=q, eligible_models=[vm], iter_avg=8.0))
    assert all(b >= a - 1e-9 for a, b in zip(mags, mags[1:]))
    assert mags[-1] == 1.0


def test_sat_constant_exists():
    assert SAT > 0
