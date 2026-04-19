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
