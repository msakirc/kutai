# tests/test_health.py
from nerd_herd.health import HealthRegistry


def test_initial_status_empty():
    h = HealthRegistry()
    status = h.get_status()
    assert status.capabilities == {}
    assert status.degraded == []
    assert status.boot_time  # non-empty


def test_mark_degraded():
    h = HealthRegistry()
    h.mark_degraded("telegram")
    assert not h.is_healthy("telegram")
    assert "telegram" in h.get_status().degraded


def test_mark_healthy():
    h = HealthRegistry()
    h.mark_degraded("llm")
    h.mark_healthy("llm")
    assert h.is_healthy("llm")
    assert "llm" not in h.get_status().degraded


def test_unknown_capability_is_healthy():
    h = HealthRegistry()
    assert h.is_healthy("never_registered")


def test_collect_dict():
    h = HealthRegistry()
    h.mark_healthy("telegram")
    h.mark_degraded("llm")
    result = h.collect()
    assert result["telegram"] == 1
    assert result["llm"] == 0


def test_prometheus_metrics():
    h = HealthRegistry()
    h.mark_healthy("sandbox")
    metrics = h.prometheus_metrics()
    assert isinstance(metrics, list)
    assert len(metrics) > 0
