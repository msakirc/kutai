"""Tests for scarcity module (Phase 2d)."""
from types import SimpleNamespace

from fatih_hoca.scarcity import pool_scarcity


def _local_model(is_loaded=False, requests_processing=0):
    return SimpleNamespace(
        name="test-local",
        is_local=True,
        is_free=False,
        is_loaded=is_loaded,
        provider="local",
    )


def _snapshot_with_local(idle_seconds=0.0, loaded_name="other", requests_processing=0):
    local = SimpleNamespace(
        model_name=loaded_name,
        idle_seconds=idle_seconds,
        measured_tps=20.0,
        thinking_enabled=False,
        requests_processing=requests_processing,
    )
    return SimpleNamespace(local=local, cloud={})


# ── Local pool ──────────────────────────────────────────────────────────

def test_local_busy_returns_negative_small():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=0.0, loaded_name="test-local", requests_processing=1)
    s = pool_scarcity(model, snap)
    assert -0.2 <= s < 0  # busy → mild negative


def test_local_cold_idle_returns_zero():
    model = _local_model(is_loaded=False)
    snap = _snapshot_with_local(idle_seconds=0.0, loaded_name="something_else")
    s = pool_scarcity(model, snap)
    assert s == 0.0  # not loaded, no idle info → neutral


def test_local_loaded_and_saturated_idle_returns_strong_positive():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=600.0, loaded_name="test-local")
    s = pool_scarcity(model, snap)
    assert 0.4 <= s <= 0.5


def test_local_loaded_partial_idle_scales_linearly():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=300.0, loaded_name="test-local")
    s = pool_scarcity(model, snap)
    assert 0.20 <= s <= 0.30


def test_local_scarcity_clamped_to_plus_one():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=999999.0, loaded_name="test-local")
    s = pool_scarcity(model, snap)
    assert s <= 1.0


# ── Time-bucketed pool ──────────────────────────────────────────────────

import time as _time


def _free_cloud_model(provider="groq", model_id="groq/llama-70b"):
    return SimpleNamespace(
        name=model_id,
        litellm_name=model_id,
        is_local=False,
        is_free=True,
        is_loaded=False,
        provider=provider,
    )


def _snapshot_with_cloud(provider, model_id, remaining, limit, reset_in_secs):
    # Uses real nerd_herd types so SystemSnapshot.pressure_for() works.
    from nerd_herd.types import (
        CloudModelState,
        CloudProviderState,
        LocalModelState,
        RateLimit,
        RateLimitMatrix,
        SystemSnapshot,
    )
    reset_at = int(_time.time() + reset_in_secs)
    rpd = RateLimit(limit=limit, remaining=remaining, reset_at=reset_at)
    model_state = CloudModelState(
        model_id=model_id,
        utilization_pct=0.0,
        limits=RateLimitMatrix(rpd=rpd),
    )
    prov_state = CloudProviderState(
        provider=provider,
        utilization_pct=0.0,
        consecutive_failures=0,
        limits=RateLimitMatrix(rpd=rpd),
        models={model_id: model_state},
    )
    return SystemSnapshot(local=LocalModelState(), cloud={provider: prov_state})


def test_time_bucketed_reset_imminent_high_remaining_returns_strong_positive():
    model = _free_cloud_model()
    # 30 min to reset, 85% remaining → strong positive
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=850, limit=1000, reset_in_secs=1800)
    s = pool_scarcity(model, snap)
    assert 0.6 <= s <= 1.0


def test_time_bucketed_reset_far_low_remaining_returns_negative():
    model = _free_cloud_model()
    # 5h to reset, 10% remaining → conserve (negative)
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=100, limit=1000, reset_in_secs=18000)
    s = pool_scarcity(model, snap)
    assert -0.5 <= s <= -0.2


def test_time_bucketed_balanced_gives_moderate_burn_signal():
    # Time-bucketed scarcity is continuous across the reset horizon:
    # wasting free-pool quota is a loss regardless of proximity to reset,
    # though urgency decays exponentially. At 4h to reset with 50%
    # remaining: time_weight = exp(-4h/24h) ≈ 0.846, scarcity ≈ 0.5 × 0.846 ≈ 0.42.
    model = _free_cloud_model()
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=500, limit=1000, reset_in_secs=14400)
    s = pool_scarcity(model, snap)
    assert 0.3 < s < 0.55


def test_time_bucketed_full_remaining_far_reset_still_has_signal():
    # 24h to reset, 100% remaining → weight ≈ 0.37, scarcity ≈ 0.37.
    # Wasting daily quota is still a loss, just a weaker signal than imminent reset.
    model = _free_cloud_model()
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=1000, limit=1000, reset_in_secs=86400)
    s = pool_scarcity(model, snap)
    assert 0.3 < s < 0.45


def test_time_bucketed_exhausted_returns_zero():
    model = _free_cloud_model()
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=0, limit=1000, reset_in_secs=3600)
    s = pool_scarcity(model, snap)
    assert s == 0.0


def test_time_bucketed_missing_provider_returns_zero():
    model = _free_cloud_model(provider="missing-provider")
    snap = SimpleNamespace(local=None, cloud={})
    s = pool_scarcity(model, snap)
    assert s == 0.0


# ── Per-call pool ───────────────────────────────────────────────────────


def _paid_cloud_model(provider="anthropic", model_id="anthropic/claude-sonnet"):
    return SimpleNamespace(
        name=model_id,
        litellm_name=model_id,
        is_local=False,
        is_free=False,
        is_loaded=False,
        provider=provider,
    )


def _queue_profile(total=0, hard=0, max_d=0):
    return SimpleNamespace(
        total_tasks=total,
        hard_tasks_count=hard,
        max_difficulty=max_d,
        needs_vision_count=0,
        needs_tools_count=0,
        needs_thinking_count=0,
        cloud_only_count=0,
    )


def _snap_with_queue(qp):
    return SimpleNamespace(local=None, cloud={}, queue_profile=qp)


def test_per_call_easy_task_with_hard_queue_returns_strong_negative():
    model = _paid_cloud_model()
    snap = _snap_with_queue(_queue_profile(total=20, hard=5, max_d=8))
    s = pool_scarcity(model, snap, task_difficulty=3)
    assert -1.0 <= s <= -0.6


def test_per_call_hard_task_with_hard_queue_returns_near_zero():
    # Current task is itself hard → no reason to conserve from it
    model = _paid_cloud_model()
    snap = _snap_with_queue(_queue_profile(total=20, hard=5, max_d=8))
    s = pool_scarcity(model, snap, task_difficulty=8)
    assert -0.2 <= s <= 0.0


def test_per_call_no_queue_pressure_returns_zero():
    model = _paid_cloud_model()
    snap = _snap_with_queue(_queue_profile(total=10, hard=0, max_d=4))
    s = pool_scarcity(model, snap, task_difficulty=3)
    assert s == 0.0


def test_per_call_no_queue_state_returns_zero():
    model = _paid_cloud_model()
    snap = SimpleNamespace(local=None, cloud={})
    s = pool_scarcity(model, snap, task_difficulty=3)
    assert s == 0.0


def test_per_call_never_positive():
    model = _paid_cloud_model()
    for qp in [_queue_profile(), _queue_profile(total=50, hard=20, max_d=10)]:
        snap = _snap_with_queue(qp)
        for d in range(1, 11):
            s = pool_scarcity(model, snap, task_difficulty=d)
            assert s <= 0.0, f"per_call positive for d={d} qp.hard={qp.hard_tasks_count}"
