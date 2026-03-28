"""Tests for model speed tracking (TPS feedback loop)."""
import pytest
from src.models.model_registry import ModelInfo, ModelRegistry


def _make_registry(*models):
    reg = ModelRegistry.__new__(ModelRegistry)
    reg.models = {m.name: m for m in models}
    return reg


def _make_model(name, tps=0.0):
    return ModelInfo(
        name=name, location="local", provider="llama_cpp",
        litellm_name=f"openai/{name}", capabilities={},
        context_length=8192, max_tokens=4096,
        tokens_per_second=tps,
    )


def test_update_measured_speed_first_measurement():
    """First measurement should be stored as-is (no EMA)."""
    info = _make_model("test-9b")
    reg = _make_registry(info)
    reg.update_measured_speed("test-9b", 42.5)
    assert info.tokens_per_second == 42.5


def test_update_measured_speed_ema():
    """Subsequent measurements use EMA with alpha=0.3."""
    info = _make_model("test-9b", tps=40.0)
    reg = _make_registry(info)
    reg.update_measured_speed("test-9b", 50.0)
    # EMA: 40 * 0.7 + 50 * 0.3 = 43.0
    assert abs(info.tokens_per_second - 43.0) < 0.1


def test_update_measured_speed_unknown_model():
    """Updating speed for unknown model is a no-op."""
    reg = _make_registry()
    reg.update_measured_speed("nonexistent", 42.5)  # should not raise
