from nerd_herd.types import (
    GPUState,
    SystemState,
    ExternalGPUUsage,
    HealthStatus,
    RateLimit,
    RateLimits,
    CloudModelState,
    CloudProviderState,
    LocalModelState,
    SystemSnapshot,
)


def test_gpu_state_defaults():
    g = GPUState(available=False)
    assert g.vram_total_mb == 0
    assert g.vram_usage_pct == 0.0
    assert not g.is_throttling
    assert not g.is_busy


def test_gpu_state_throttling():
    g = GPUState(available=True, temperature_c=90)
    assert g.is_throttling


def test_gpu_state_busy():
    g = GPUState(available=True, gpu_utilization_pct=85)
    assert g.is_busy


def test_gpu_state_vram_usage_pct():
    g = GPUState(available=True, vram_total_mb=8000, vram_used_mb=4000)
    assert g.vram_usage_pct == 50.0


def test_system_state_can_load_model():
    s = SystemState(ram_available_mb=8000)
    assert s.can_load_model
    s2 = SystemState(ram_available_mb=2000)
    assert not s2.can_load_model


def test_external_gpu_usage_fraction():
    e = ExternalGPUUsage(
        detected=True,
        external_vram_mb=2000,
        total_vram_mb=8000,
    )
    assert e.external_vram_fraction == 0.25


def test_external_gpu_usage_zero_total():
    e = ExternalGPUUsage()
    assert e.external_vram_fraction == 0.0


def test_health_status_degraded_list():
    h = HealthStatus(
        boot_time="2026-04-13T00:00:00Z",
        capabilities={"telegram": True, "llm": False, "sandbox": False},
    )
    assert h.degraded == ["llm", "sandbox"]


def test_health_status_all_healthy():
    h = HealthStatus(
        boot_time="2026-04-13T00:00:00Z",
        capabilities={"telegram": True, "llm": True},
    )
    assert h.degraded == []


# --- SystemSnapshot / cloud-state types ---

def test_rate_limit_defaults():
    r = RateLimit()
    assert r.limit is None
    assert r.remaining is None
    assert r.reset_at is None


def test_system_snapshot_defaults():
    s = SystemSnapshot()
    assert s.vram_available_mb == 0
    assert isinstance(s.local, LocalModelState)
    assert s.local.model_name is None
    assert s.local.thinking_enabled is False
    assert s.local.measured_tps == 0.0
    assert s.cloud == {}


def test_cloud_model_state():
    m = CloudModelState(model_id="gpt-4o", utilization_pct=42.5)
    assert m.model_id == "gpt-4o"
    assert m.utilization_pct == 42.5
    assert isinstance(m.limits, RateLimits)
    assert isinstance(m.limits.rpm, RateLimit)


def test_cloud_provider_state():
    p = CloudProviderState(
        provider="openai",
        utilization_pct=30.0,
        consecutive_failures=2,
        last_failure_at=1713000000,
    )
    assert p.provider == "openai"
    assert p.consecutive_failures == 2
    assert p.last_failure_at == 1713000000
    assert p.models == {}
    assert isinstance(p.limits, RateLimits)


def test_snapshot_with_cloud():
    provider = CloudProviderState(
        provider="anthropic",
        utilization_pct=10.0,
        models={
            "claude-3-5-sonnet": CloudModelState(
                model_id="claude-3-5-sonnet",
                utilization_pct=10.0,
            )
        },
    )
    local = LocalModelState(
        model_name="qwen3-8b",
        thinking_enabled=True,
        measured_tps=45.2,
        context_length=32768,
    )
    snap = SystemSnapshot(
        vram_available_mb=4096,
        local=local,
        cloud={"anthropic": provider},
    )
    assert snap.vram_available_mb == 4096
    assert snap.local.model_name == "qwen3-8b"
    assert snap.local.thinking_enabled is True
    assert "anthropic" in snap.cloud
    assert snap.cloud["anthropic"].provider == "anthropic"
    assert "claude-3-5-sonnet" in snap.cloud["anthropic"].models
