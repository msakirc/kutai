import time
from types import SimpleNamespace

from nerd_herd.types import (
    SystemSnapshot, CloudProviderState, CloudModelState,
    RateLimit, RateLimitMatrix, QueueProfile,
)
from nerd_herd.burn_log import BurnLog


def _free_snapshot(remaining=18, limit=20, reset_in=3600):
    now = time.time()
    rpd = RateLimit(limit=limit, remaining=remaining, reset_at=int(now + reset_in))
    cms = CloudModelState(model_id="gem/flash", utilization_pct=0.0,
                          limits=RateLimitMatrix(rpd=rpd))
    prov = CloudProviderState(provider="gem", models={"gem/flash": cms})
    return SystemSnapshot(cloud={"gem": prov}), now


def _free_model():
    return SimpleNamespace(name="gem/flash", provider="gem", is_local=False,
                           is_free=True, cap_score=7.0, capabilities=set(),
                           rpd_remaining=18)


def test_now_kwarg_defaults_to_walltime_and_is_threaded():
    snap, now = _free_snapshot()
    m = _free_model()
    bd_default = snap.pressure_for(m, task_difficulty=5)
    bd_now = snap.pressure_for(m, task_difficulty=5, now=now)
    assert abs(bd_default.signals["S9"] - bd_now.signals["S9"]) < 0.05


def test_burn_log_kwarg_drives_s7():
    snap, now = _free_snapshot(remaining=18, limit=20, reset_in=3600)
    m = _free_model()
    assert snap.pressure_for(m, task_difficulty=5, now=now).signals["S7"] == 0.0
    bl = BurnLog(window_secs=300.0)
    for i in range(20):
        bl.record(provider="gem", model="gem/flash", tokens=1000, calls=1, now=now - i)
    s7 = snap.pressure_for(m, task_difficulty=5, now=now, burn_log=bl).signals["S7"]
    assert s7 < 0.0


def test_eligible_models_kwarg_drives_s6():
    snap, now = _free_snapshot()
    m = SimpleNamespace(name="gem/flash", provider="gem", is_local=False,
                        is_free=True, cap_score=7.0, capabilities={"vision"},
                        rpd_remaining=2)
    snap.queue_profile = QueueProfile(by_capability={"vision": 50},
                                      total_ready_count=50)
    assert snap.pressure_for(m, task_difficulty=5, now=now).signals["S6"] == 0.0
    s6 = snap.pressure_for(m, task_difficulty=5, now=now,
                           eligible_models=[m]).signals["S6"]
    assert s6 < 0.0
