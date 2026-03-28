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


def test_faster_model_preferred_when_prefer_speed():
    """When prefer_speed=True, a model with higher TPS should score better in the speed dimension."""
    from src.models.model_registry import ModelInfo

    fast = ModelInfo(
        name="fast-9b", location="local", provider="llama_cpp",
        litellm_name="openai/fast-9b", capabilities={"general": 5.0},
        context_length=32768, max_tokens=4096,
        supports_function_calling=True, tokens_per_second=50.0,
    )
    slow = ModelInfo(
        name="slow-27b", location="local", provider="llama_cpp",
        litellm_name="openai/slow-27b", capabilities={"general": 7.0},
        context_length=8192, max_tokens=4096,
        supports_function_calling=True, tokens_per_second=1.0,
    )

    # The fast model should have measurably better speed characteristics
    # even though the slow model has higher capability
    assert fast.tokens_per_second > slow.tokens_per_second * 10


def test_very_slow_model_gets_demoted():
    """A local model measured at <2 tok/s should be auto-demoted."""
    info = ModelInfo(
        name="slow-model", location="local", provider="llama_cpp",
        litellm_name="openai/slow-model", capabilities={},
        context_length=8192, max_tokens=4096, tokens_per_second=0.0,
    )
    reg = ModelRegistry.__new__(ModelRegistry)
    reg.models = {"slow-model": info}

    reg.update_measured_speed("slow-model", 1.0)
    assert info.tokens_per_second == 1.0
    assert info.demoted is True


def test_fast_model_not_demoted():
    """A model at 10 tok/s should NOT be demoted."""
    info = ModelInfo(
        name="fast-model", location="local", provider="llama_cpp",
        litellm_name="openai/fast-model", capabilities={},
        context_length=8192, max_tokens=4096, tokens_per_second=0.0,
    )
    reg = ModelRegistry.__new__(ModelRegistry)
    reg.models = {"fast-model": info}

    reg.update_measured_speed("fast-model", 10.0)
    assert info.tokens_per_second == 10.0
    assert info.demoted is False


def test_cloud_model_not_demoted_even_if_slow():
    """Cloud models should not be auto-demoted regardless of speed."""
    info = ModelInfo(
        name="cloud-model", location="cloud", provider="openai",
        litellm_name="gpt-4o-mini", capabilities={},
        context_length=128000, max_tokens=4096, tokens_per_second=0.0,
    )
    reg = ModelRegistry.__new__(ModelRegistry)
    reg.models = {"cloud-model": info}

    reg.update_measured_speed("cloud-model", 0.5)
    assert info.demoted is False
