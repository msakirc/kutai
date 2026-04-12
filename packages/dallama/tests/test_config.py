"""Tests for DaLLaMa config dataclasses."""
import pytest
from dallama.config import (
    DaLLaMaConfig, ServerConfig, ServerStatus, InferenceSession, DaLLaMaLoadError,
)

def test_dallama_config_defaults():
    cfg = DaLLaMaConfig()
    assert cfg.llama_server_path == "llama-server"
    assert cfg.port == 8080
    assert cfg.host == "127.0.0.1"
    assert cfg.idle_timeout_seconds == 60.0
    assert cfg.circuit_breaker_threshold == 2
    assert cfg.circuit_breaker_cooldown_seconds == 300.0
    assert cfg.inference_drain_timeout_seconds == 30.0
    assert cfg.health_check_interval_seconds == 30.0
    assert cfg.health_fail_threshold == 3
    assert cfg.min_free_vram_mb == 4096
    assert cfg.on_ready is None
    assert cfg.get_vram_free_mb is None

def test_dallama_config_custom():
    cfg = DaLLaMaConfig(
        llama_server_path="/usr/bin/llama-server", port=9090,
        idle_timeout_seconds=120, on_ready=lambda m, r: None,
        get_vram_free_mb=lambda: 8000,
    )
    assert cfg.port == 9090
    assert cfg.idle_timeout_seconds == 120
    assert cfg.on_ready is not None
    assert cfg.get_vram_free_mb() == 8000

def test_server_config_minimal():
    sc = ServerConfig(model_path="/models/test.gguf", model_name="test-model", context_length=4096)
    assert sc.thinking is False
    assert sc.vision_projector == ""
    assert sc.extra_flags == []

def test_server_config_full():
    sc = ServerConfig(
        model_path="/models/qwen.gguf", model_name="qwen3-30b", context_length=16384,
        thinking=True, vision_projector="/models/mmproj.gguf",
        extra_flags=["--no-jinja", "--chat-template", "chatml"],
    )
    assert sc.thinking is True
    assert sc.vision_projector == "/models/mmproj.gguf"
    assert len(sc.extra_flags) == 3

def test_server_status_no_model():
    st = ServerStatus(model_name=None, healthy=False, busy=False, measured_tps=0.0, context_length=0)
    assert st.model_name is None
    assert st.healthy is False

def test_server_status_loaded():
    st = ServerStatus(model_name="qwen3-30b", healthy=True, busy=True, measured_tps=12.5, context_length=16384)
    assert st.busy is True
    assert st.measured_tps == 12.5

def test_inference_session():
    s = InferenceSession(url="http://127.0.0.1:8080", model_name="test")
    assert s.url == "http://127.0.0.1:8080"
    assert s.model_name == "test"

def test_dallama_load_error():
    err = DaLLaMaLoadError("qwen3-30b")
    assert "qwen3-30b" in str(err)
    assert isinstance(err, RuntimeError)
