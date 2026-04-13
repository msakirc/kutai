# tests/test_nerd_herd.py
import pytest
from unittest.mock import MagicMock
from nerd_herd import NerdHerd


def test_init_registers_builtin_collectors():
    nh = NerdHerd()
    names = nh.registry.names()
    assert "gpu" in names
    assert "load" in names
    assert "health" in names


def test_init_with_llama_url_registers_inference():
    nh = NerdHerd(llama_server_url="http://127.0.0.1:8080")
    assert "inference" in nh.registry.names()


def test_init_without_llama_url_no_inference():
    nh = NerdHerd(llama_server_url=None)
    assert "inference" not in nh.registry.names()


def test_gpu_state():
    nh = NerdHerd()
    state = nh.gpu_state()
    assert hasattr(state, "vram_free_mb")


def test_vram_budget_mb():
    nh = NerdHerd()
    budget = nh.get_vram_budget_mb()
    assert isinstance(budget, int)


def test_health_operations():
    nh = NerdHerd()
    nh.mark_degraded("test_cap")
    assert not nh.is_healthy("test_cap")
    nh.mark_healthy("test_cap")
    assert nh.is_healthy("test_cap")


def test_set_load_mode():
    nh = NerdHerd()
    result = nh.set_load_mode("shared", source="user")
    assert "shared" in result
    assert nh.get_load_mode() == "shared"


def test_register_custom_collector():
    nh = NerdHerd()
    mock = MagicMock()
    mock.name = "custom"
    nh.register_collector("custom", mock)
    assert "custom" in nh.registry.names()


def test_prometheus_lines():
    nh = NerdHerd()
    lines = nh.prometheus_lines()
    assert isinstance(lines, str)
