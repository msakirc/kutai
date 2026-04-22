import time
from unittest.mock import MagicMock
from nerd_herd.types import (
    SystemSnapshot, CloudProviderState, CloudModelState, RateLimit, RateLimits,
    LocalModelState,
)


def _snap_with_cloud(provider, model_name, remaining, limit, reset_at, in_flight=0):
    m = CloudModelState(model_id=model_name)
    m.limits.rpd = RateLimit(limit=limit, remaining=remaining, reset_at=reset_at, in_flight=in_flight)
    prov = CloudProviderState(provider=provider, models={model_name: m})
    return SystemSnapshot(cloud={provider: prov})


def test_pressure_for_cloud_model_depletion_negative():
    snap = _snap_with_cloud("anthropic", "claude-sonnet-4-6",
                            remaining=5, limit=100, reset_at=int(time.time()) + 3600)
    # is_free=False → per_call profile (depletion_max=-1.0).
    fake = MagicMock(is_local=False, is_free=False, provider="anthropic")
    fake.name = "claude-sonnet-4-6"
    assert snap.pressure_for(fake) < -0.5


def test_pressure_for_missing_model_returns_zero():
    snap = SystemSnapshot()
    fake = MagicMock(is_local=False, provider="unknown"); fake.name = "x"
    assert snap.pressure_for(fake) == 0.0


def test_pressure_for_cached_after_first_read():
    snap = _snap_with_cloud("anthropic", "claude-sonnet-4-6",
                            remaining=50, limit=100, reset_at=int(time.time()) + 3600)
    fake = MagicMock(is_local=False, is_free=False, provider="anthropic")
    fake.name = "claude-sonnet-4-6"
    _ = snap.pressure_for(fake)
    first_obj = snap.cloud["anthropic"].models["claude-sonnet-4-6"].pool_pressure
    _ = snap.pressure_for(fake)
    second_obj = snap.cloud["anthropic"].models["claude-sonnet-4-6"].pool_pressure
    assert first_obj is second_obj


def test_pressure_for_local_busy_negative_or_zero():
    snap = SystemSnapshot(local=LocalModelState(model_name="qwen3-8b"))
    fake = MagicMock(is_local=True); fake.name = "qwen3-8b"
    val = snap.pressure_for(fake)
    assert -1.0 <= val <= 1.0
