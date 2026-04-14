"""Tests for NerdHerd.snapshot() and push_*_state() methods."""
from nerd_herd import NerdHerd, SystemSnapshot, LocalModelState, CloudProviderState


def test_snapshot_returns_system_snapshot():
    nh = NerdHerd()
    snap = nh.snapshot()
    assert isinstance(snap, SystemSnapshot)


def test_snapshot_reflects_pushed_local_state():
    nh = NerdHerd()
    state = LocalModelState(
        model_name="qwen3-8b",
        thinking_enabled=True,
        measured_tps=55.0,
        context_length=32768,
    )
    nh.push_local_state(state)
    snap = nh.snapshot()
    assert snap.local.model_name == "qwen3-8b"
    assert snap.local.thinking_enabled is True
    assert snap.local.measured_tps == 55.0
    assert snap.local.context_length == 32768


def test_snapshot_reflects_pushed_cloud_state():
    nh = NerdHerd()
    provider = CloudProviderState(
        provider="anthropic",
        utilization_pct=25.0,
        consecutive_failures=1,
    )
    nh.push_cloud_state(provider)
    snap = nh.snapshot()
    assert "anthropic" in snap.cloud
    assert snap.cloud["anthropic"].provider == "anthropic"
    assert snap.cloud["anthropic"].utilization_pct == 25.0
    assert snap.cloud["anthropic"].consecutive_failures == 1


def test_push_local_state_clears_on_none():
    nh = NerdHerd()
    # Push a populated state first
    nh.push_local_state(LocalModelState(model_name="qwen3-8b", measured_tps=50.0))
    # Push a blank state (model unloaded / swapping out)
    nh.push_local_state(LocalModelState())
    snap = nh.snapshot()
    assert snap.local.model_name is None
    assert snap.local.measured_tps == 0.0


def test_push_cloud_state_updates_existing():
    nh = NerdHerd()
    nh.push_cloud_state(CloudProviderState(provider="openai", utilization_pct=10.0))
    nh.push_cloud_state(CloudProviderState(provider="openai", utilization_pct=90.0, consecutive_failures=3))
    snap = nh.snapshot()
    # Should have exactly one entry for openai with the latest values
    assert len([k for k in snap.cloud if k == "openai"]) == 1
    assert snap.cloud["openai"].utilization_pct == 90.0
    assert snap.cloud["openai"].consecutive_failures == 3
