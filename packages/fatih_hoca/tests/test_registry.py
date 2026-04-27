"""Tests for fatih_hoca.registry — ModelInfo, ModelRegistry, and helpers."""

import pytest
from fatih_hoca.registry import ModelInfo, ModelRegistry


# ─── ModelInfo properties ────────────────────────────────────────────────────

def test_model_info_is_local():
    m = ModelInfo(name="qwen3-30b", location="local", provider="local",
                  litellm_name="local/qwen3-30b")
    assert m.is_local is True


def test_model_info_ollama_is_local():
    m = ModelInfo(name="ollama-llama3", location="ollama", provider="ollama",
                  litellm_name="ollama/llama3")
    assert m.is_local is True


def test_model_info_is_cloud():
    m = ModelInfo(name="claude-sonnet", location="cloud", provider="anthropic",
                  litellm_name="anthropic/claude-sonnet-4-20250514")
    assert m.is_local is False


def test_model_info_is_free_local():
    m = ModelInfo(name="test", location="local", provider="llama_cpp",
                  litellm_name="openai/test")
    assert m.is_free is True


def test_model_info_is_free_cloud():
    m = ModelInfo(name="test", location="cloud", provider="groq",
                  litellm_name="groq/llama-8b",
                  cost_per_1k_input=0.0, cost_per_1k_output=0.0,
                  tier="free")
    assert m.is_free is True


def test_model_info_not_free():
    m = ModelInfo(name="test", location="cloud", provider="anthropic",
                  litellm_name="anthropic/claude-sonnet",
                  cost_per_1k_input=0.003, cost_per_1k_output=0.015,
                  tier="paid")
    assert m.is_free is False


def test_model_info_estimated_cost_local():
    m = ModelInfo(name="test", location="local", provider="local",
                  litellm_name="local/test",
                  cost_per_1k_input=0.003, cost_per_1k_output=0.015)
    assert m.estimated_cost(1000, 1000) == 0.0


def test_model_info_estimated_cost():
    m = ModelInfo(name="test", location="cloud", provider="anthropic",
                  litellm_name="anthropic/claude-sonnet",
                  cost_per_1k_input=0.003, cost_per_1k_output=0.015)
    cost = m.estimated_cost(1000, 1000)
    assert cost > 0
    assert abs(cost - 0.018) < 1e-9


def test_model_info_score_for():
    m = ModelInfo(name="test", location="local", provider="local",
                  litellm_name="local/test",
                  capabilities={"reasoning": 7.0, "code_generation": 8.0})
    assert m.score_for("reasoning") == 7.0
    assert m.score_for("nonexistent") == 0.0


def test_model_info_best_score():
    m = ModelInfo(name="test", location="local", provider="local",
                  litellm_name="local/test",
                  capabilities={"reasoning": 7.0, "code_generation": 8.5})
    assert m.best_score() == 8.5


def test_model_info_best_score_empty():
    m = ModelInfo(name="test", location="local", provider="local",
                  litellm_name="local/test")
    assert m.best_score() == 0.0


def test_model_info_operational_dict():
    m = ModelInfo(name="test", location="local", provider="llama_cpp",
                  litellm_name="openai/test",
                  context_length=16384, max_tokens=4096)
    d = m.operational_dict()
    assert d["location"] == "local"
    assert d["context_length"] == 16384
    assert d["max_tokens"] == 4096
    assert "capabilities" not in d  # operational dict excludes capabilities dict


# ─── ModelRegistry ────────────────────────────────────────────────────────────

def test_registry_init_empty():
    reg = ModelRegistry()
    assert reg.all_models() == []


def test_registry_register_and_get():
    reg = ModelRegistry()
    m = ModelInfo(name="test", location="local", provider="local",
                  litellm_name="local/test")
    reg.register(m)
    assert reg.get("test") is m
    assert reg.get("nonexistent") is None


def test_registry_all_models():
    reg = ModelRegistry()
    m1 = ModelInfo(name="a", location="local", provider="local", litellm_name="local/a")
    m2 = ModelInfo(name="b", location="cloud", provider="openai", litellm_name="openai/b")
    reg.register(m1)
    reg.register(m2)
    all_m = reg.all_models()
    assert len(all_m) == 2


def test_registry_by_litellm_name():
    reg = ModelRegistry()
    m = ModelInfo(name="test", location="cloud", provider="anthropic",
                  litellm_name="anthropic/claude-sonnet")
    reg.register(m)
    assert reg.by_litellm_name("anthropic/claude-sonnet") is m
    assert reg.by_litellm_name("groq/llama") is None


def test_registry_register_overwrites():
    reg = ModelRegistry()
    m1 = ModelInfo(name="test", location="local", provider="local",
                   litellm_name="local/test", context_length=4096)
    m2 = ModelInfo(name="test", location="local", provider="local",
                   litellm_name="local/test", context_length=8192)
    reg.register(m1)
    reg.register(m2)
    assert reg.get("test").context_length == 8192
    assert len(reg.all_models()) == 1


def test_registry_local_cloud_filter():
    reg = ModelRegistry()
    local = ModelInfo(name="local-model", location="local", provider="llama_cpp",
                      litellm_name="openai/local-model")
    cloud = ModelInfo(name="cloud-model", location="cloud", provider="anthropic",
                      litellm_name="anthropic/claude")
    reg.register(local)
    reg.register(cloud)
    assert len(reg.local_models()) == 1
    assert reg.local_models()[0].name == "local-model"
    assert len(reg.cloud_models()) == 1
    assert reg.cloud_models()[0].name == "cloud-model"


def test_registry_vision_models():
    reg = ModelRegistry()
    vision = ModelInfo(name="vmodel", location="local", provider="llama_cpp",
                       litellm_name="openai/vmodel", has_vision=True)
    plain = ModelInfo(name="plain", location="local", provider="llama_cpp",
                      litellm_name="openai/plain", has_vision=False)
    reg.register(vision)
    reg.register(plain)
    assert len(reg.vision_models()) == 1
    assert reg.vision_models()[0].name == "vmodel"


def test_registry_thinking_models():
    reg = ModelRegistry()
    thinker = ModelInfo(name="tmodel", location="local", provider="llama_cpp",
                        litellm_name="openai/tmodel", thinking_model=True)
    plain = ModelInfo(name="plain", location="local", provider="llama_cpp",
                      litellm_name="openai/plain", thinking_model=False)
    reg.register(thinker)
    reg.register(plain)
    assert len(reg.thinking_models()) == 1
    assert reg.thinking_models()[0].name == "tmodel"


def test_registry_models_with_capability():
    reg = ModelRegistry()
    strong = ModelInfo(name="strong", location="local", provider="local",
                       litellm_name="local/strong",
                       capabilities={"reasoning": 8.0, "code_generation": 7.0})
    weak = ModelInfo(name="weak", location="local", provider="local",
                     litellm_name="local/weak",
                     capabilities={"reasoning": 3.0, "code_generation": 2.0})
    reg.register(strong)
    reg.register(weak)
    good = reg.models_with_capability("reasoning", min_score=6.0)
    assert len(good) == 1
    assert good[0].name == "strong"


# ─── Standalone helpers ───────────────────────────────────────────────────────

def test_estimate_capabilities_returns_dict():
    from fatih_hoca.registry import estimate_capabilities
    caps = estimate_capabilities(
        family_key="llama31",
        total_params_b=8.0,
        active_params_b=None,
        quantization="Q4_K_M",
    )
    assert isinstance(caps, dict)
    assert len(caps) > 0
    for v in caps.values():
        assert 0.0 <= v <= 10.0


def test_estimate_capabilities_unknown_family():
    from fatih_hoca.registry import estimate_capabilities
    caps = estimate_capabilities(
        family_key=None,
        total_params_b=7.0,
        active_params_b=None,
        quantization="Q4_K_M",
    )
    assert isinstance(caps, dict)


def test_calculate_gpu_layers_zero_on_no_vram():
    from fatih_hoca.registry import calculate_gpu_layers
    layers = calculate_gpu_layers(
        file_size_mb=4000,
        n_layers=32,
        available_vram_mb=0,
        context_length=8192,
    )
    assert layers == 0


def test_calculate_gpu_layers_positive():
    from fatih_hoca.registry import calculate_gpu_layers
    layers = calculate_gpu_layers(
        file_size_mb=4000,
        n_layers=32,
        available_vram_mb=8192,
        context_length=8192,
    )
    assert layers > 0
    assert layers <= 32


def test_calculate_dynamic_context_minimum():
    from fatih_hoca.registry import calculate_dynamic_context
    ctx = calculate_dynamic_context(
        file_size_mb=0,
        n_layers=0,
        gpu_layers=0,
        available_ram_mb=32768,
        available_vram_mb=8192,
    )
    assert ctx == 8192  # safe minimum


def test_calculate_dynamic_context_positive():
    from fatih_hoca.registry import calculate_dynamic_context
    ctx = calculate_dynamic_context(
        file_size_mb=4000,
        n_layers=32,
        gpu_layers=20,
        available_ram_mb=32768,
        available_vram_mb=8192,
    )
    assert ctx >= 4096


def test_detect_function_calling_known_family():
    from fatih_hoca.registry import detect_function_calling
    assert detect_function_calling("qwen3", {}) is True
    assert detect_function_calling("unknown_family_xyz", {}) is False


def test_detect_thinking_model():
    from fatih_hoca.registry import detect_thinking_model
    assert detect_thinking_model("qwen3") is True
    assert detect_thinking_model("llama31") is False
    assert detect_thinking_model(None, "openai/o3-mini") is True


def test_resolve_provider_explicit():
    from fatih_hoca.registry import _resolve_provider
    assert _resolve_provider("anthropic/claude-sonnet") == "anthropic"
    assert _resolve_provider("groq/llama3") == "groq"
    assert _resolve_provider("unknown_prefix/model") is None


def test_resolve_provider_inferred():
    from fatih_hoca.registry import _resolve_provider
    assert _resolve_provider("claude-3-opus") == "anthropic"
    assert _resolve_provider("gpt-4o") == "openai"


# ─── register_cloud_from_discovered ──────────────────────────────────────────

def test_register_cloud_from_discovered_merges_scraped_fields():
    from fatih_hoca.cloud.types import DiscoveredModel
    from fatih_hoca.registry import ModelRegistry, register_cloud_from_discovered

    registry = ModelRegistry()
    discovered = DiscoveredModel(
        litellm_name="groq/llama-3.3-70b-versatile",
        raw_id="llama-3.3-70b-versatile",
        context_length=131072,
        max_output_tokens=32768,
        cost_per_1k_input=0.59,
        cost_per_1k_output=0.79,
        sampling_defaults={"temperature": 1.0},
        extra={"owned_by": "Meta"},
    )
    register_cloud_from_discovered(registry, "groq", discovered)
    m = registry.get("groq/llama-3.3-70b-versatile")
    assert m is not None
    assert m.location == "cloud"
    assert m.provider == "groq"
    assert m.context_length == 131072
    assert m.max_tokens == 32768
    assert m.cost_per_1k_input == 0.59
    assert m.cost_per_1k_output == 0.79
    assert m.family == "llama-3.3-70b"


def test_register_cloud_skips_inactive():
    from fatih_hoca.cloud.types import DiscoveredModel
    from fatih_hoca.registry import ModelRegistry, register_cloud_from_discovered

    registry = ModelRegistry()
    discovered = DiscoveredModel(
        litellm_name="groq/dead", raw_id="dead", active=False,
    )
    result = register_cloud_from_discovered(registry, "groq", discovered)
    assert result is None
    assert registry.get("groq/dead") is None


def test_register_cloud_uses_detect_defaults_when_scraped_missing():
    """Scraped fields are optional — when None, detect_cloud_model() defaults apply."""
    from fatih_hoca.cloud.types import DiscoveredModel
    from fatih_hoca.registry import ModelRegistry, register_cloud_from_discovered

    registry = ModelRegistry()
    discovered = DiscoveredModel(
        litellm_name="claude-sonnet-4-20250514",
        raw_id="claude-sonnet-4-20250514",
        # All scraped fields None.
    )
    m = register_cloud_from_discovered(registry, "anthropic", discovered)
    assert m is not None
    # detect_cloud_model() must have produced sane defaults.
    assert m.context_length > 0
    assert m.max_tokens > 0
    assert m.location == "cloud"
    assert m.provider == "anthropic"
    assert m.family == "claude-sonnet-4"
