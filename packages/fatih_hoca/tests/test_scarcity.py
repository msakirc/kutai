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
    s = pool_scarcity(model, snap, queue_state=None)
    assert -0.2 <= s < 0  # busy → mild negative


def test_local_cold_idle_returns_zero():
    model = _local_model(is_loaded=False)
    snap = _snapshot_with_local(idle_seconds=0.0, loaded_name="something_else")
    s = pool_scarcity(model, snap, queue_state=None)
    assert s == 0.0  # not loaded, no idle info → neutral


def test_local_loaded_and_saturated_idle_returns_strong_positive():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=600.0, loaded_name="test-local")
    s = pool_scarcity(model, snap, queue_state=None)
    assert 0.4 <= s <= 0.5


def test_local_loaded_partial_idle_scales_linearly():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=300.0, loaded_name="test-local")
    s = pool_scarcity(model, snap, queue_state=None)
    assert 0.20 <= s <= 0.30


def test_local_scarcity_clamped_to_plus_one():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=999999.0, loaded_name="test-local")
    s = pool_scarcity(model, snap, queue_state=None)
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
    reset_at = _time.time() + reset_in_secs
    rpd = SimpleNamespace(remaining=remaining, limit=limit, reset_at=reset_at)
    limits = SimpleNamespace(rpd=rpd)
    model_state = SimpleNamespace(limits=limits, utilization_pct=0.0, daily_exhausted=False)
    prov_state = SimpleNamespace(
        models={model_id: model_state},
        limits=limits,
        utilization_pct=0.0,
        consecutive_failures=0,
    )
    return SimpleNamespace(local=None, cloud={provider: prov_state})


def test_time_bucketed_reset_imminent_high_remaining_returns_strong_positive():
    model = _free_cloud_model()
    # 30 min to reset, 85% remaining → strong positive
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=850, limit=1000, reset_in_secs=1800)
    s = pool_scarcity(model, snap, queue_state=None)
    assert 0.6 <= s <= 1.0


def test_time_bucketed_reset_far_low_remaining_returns_negative():
    model = _free_cloud_model()
    # 5h to reset, 10% remaining → conserve (negative)
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=100, limit=1000, reset_in_secs=18000)
    s = pool_scarcity(model, snap, queue_state=None)
    assert -0.5 <= s <= -0.2


def test_time_bucketed_balanced_returns_near_zero():
    model = _free_cloud_model()
    # 4h to reset, 50% remaining → neutral
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=500, limit=1000, reset_in_secs=14400)
    s = pool_scarcity(model, snap, queue_state=None)
    assert -0.2 <= s <= 0.2


def test_time_bucketed_exhausted_returns_zero():
    model = _free_cloud_model()
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=0, limit=1000, reset_in_secs=3600)
    s = pool_scarcity(model, snap, queue_state=None)
    assert s == 0.0


def test_time_bucketed_missing_provider_returns_zero():
    model = _free_cloud_model(provider="missing-provider")
    snap = SimpleNamespace(local=None, cloud={})
    s = pool_scarcity(model, snap, queue_state=None)
    assert s == 0.0
