# tests/test_gpu.py
from unittest.mock import MagicMock, patch
from nerd_herd.gpu import GPUCollector
from nerd_herd.types import GPUState, ExternalGPUUsage


def test_gpu_unavailable():
    """When pynvml fails to init, GPUState.available=False."""
    with patch("nerd_herd.gpu._pynvml", None):
        c = GPUCollector()
        state = c.gpu_state()
        assert not state.available
        assert state.vram_total_mb == 0


def test_collect_returns_dict():
    c = GPUCollector()
    result = c.collect()
    assert "gpu_available" in result
    assert "ram_available_mb" in result
    assert "cpu_percent" in result


def test_external_usage_no_gpu():
    c = GPUCollector()
    c._nvml_ok = False
    ext = c.detect_external_gpu_usage()
    assert not ext.detected
    assert ext.external_vram_fraction == 0.0


def test_prometheus_metrics_returns_list():
    c = GPUCollector()
    metrics = c.prometheus_metrics()
    assert isinstance(metrics, list)


def test_cache_returns_same_within_ttl():
    c = GPUCollector()
    s1 = c.gpu_state()
    s2 = c.gpu_state()
    assert s1 is s2  # same cached object


def test_invalidate_cache():
    c = GPUCollector()
    _ = c.gpu_state()
    c.invalidate_cache()
    assert c._last_poll == 0.0
