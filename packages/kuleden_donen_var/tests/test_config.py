"""Tests for config dataclasses."""
from kuleden_donen_var.config import (
    KuledenConfig,
    ProviderStatus,
    ModelStatus,
    CapacityEvent,
    PreCallResult,
)


def test_kuleden_config_defaults():
    cfg = KuledenConfig()
    assert cfg.circuit_breaker_threshold == 3
    assert cfg.circuit_breaker_cooldown_seconds == 600.0
    assert cfg.circuit_breaker_window_seconds == 300.0
    assert cfg.on_capacity_change is None


def test_provider_status_defaults():
    ps = ProviderStatus(provider="groq")
    assert ps.circuit_breaker_open is False
    assert ps.utilization_pct == 0.0
    assert ps.rpm_remaining is None
    assert ps.tpm_remaining is None
    assert ps.rpd_remaining is None
    assert ps.reset_in_seconds is None
    assert ps.models == {}


def test_model_status_defaults():
    ms = ModelStatus(model_id="groq/llama-8b")
    assert ms.utilization_pct == 0.0
    assert ms.has_capacity is True
    assert ms.daily_exhausted is False


def test_capacity_event():
    ps = ProviderStatus(provider="groq")
    evt = CapacityEvent(
        provider="groq",
        model_id="groq/llama-8b",
        event_type="capacity_restored",
        snapshot=ps,
    )
    assert evt.event_type == "capacity_restored"
    assert evt.snapshot.provider == "groq"


def test_pre_call_result_allowed():
    r = PreCallResult(allowed=True, wait_seconds=0.0, daily_exhausted=False)
    assert r.allowed is True
    assert r.wait_seconds == 0.0


def test_pre_call_result_denied():
    r = PreCallResult(allowed=False, wait_seconds=12.5, daily_exhausted=False)
    assert r.allowed is False
    assert r.wait_seconds == 12.5


def test_pre_call_result_daily_exhausted():
    r = PreCallResult(allowed=False, wait_seconds=0.0, daily_exhausted=True)
    assert r.daily_exhausted is True
