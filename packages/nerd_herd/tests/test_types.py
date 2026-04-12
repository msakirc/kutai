from nerd_herd.types import GPUState, SystemState, ExternalGPUUsage, HealthStatus


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
