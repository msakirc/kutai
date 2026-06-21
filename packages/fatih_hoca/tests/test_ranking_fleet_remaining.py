"""Ranking precomputes fleet_remaining and passes it to pressure_for; the result
must equal the snapshot's own internal build (perf path == correctness path)."""
import time
from types import SimpleNamespace

from nerd_herd.types import (
    CloudModelState, CloudProviderState, QueueProfile,
    RateLimit, RateLimitMatrix, SystemSnapshot,
)


def _snap():
    now = time.time()
    free_m = CloudModelState(model_id="free/m", limits=RateLimitMatrix(
        rpd=RateLimit(limit=20, remaining=20, reset_at=int(now + 86400))))
    prem_m = CloudModelState(model_id="prem/m", limits=RateLimitMatrix(
        rpd=RateLimit(limit=1000, remaining=1000, reset_at=int(now + 86400))))
    snap = SystemSnapshot(cloud={
        "free_prov": CloudProviderState(provider="free_prov", models={"free/m": free_m}),
        "prem_prov": CloudProviderState(provider="prem_prov", models={"prem/m": prem_m}),
    })
    snap.queue_profile = QueueProfile(total_ready_count=40, projected_calls=40)
    return snap


def test_ranking_passes_fleet_remaining_that_unfloors_small_free():
    # A captured pressure_for must receive a non-None fleet_remaining whose value
    # matches the snapshot's internal build.
    from fatih_hoca import ranking

    snap = _snap()
    expected_fleet = snap._build_fleet_cycle_remaining()
    captured = {}
    orig = SystemSnapshot.pressure_for

    def spy(self, model, **kw):
        captured[getattr(model, "name", "")] = kw.get("fleet_remaining")
        return orig(self, model, **kw)

    free_model = SimpleNamespace(
        name="free/m", provider="free_prov", is_free=True, is_local=False,
        cap_score=7.0, agent_type="researcher", context={},
        estimated_cost=lambda *_: 0.0,
    )
    scored = [ranking.ScoredModel(model=free_model, score=1.0, composite_score=1.0)]

    SystemSnapshot.pressure_for = spy
    try:
        ranking._apply_utilization_layer(
            scored, snap, task_difficulty=3, reqs=free_model, now=time.time(),
            burn_log=None,
        )
    finally:
        SystemSnapshot.pressure_for = orig

    assert captured.get("free/m") == expected_fleet
