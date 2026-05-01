# test_init.py
"""Tests for the fatih_hoca public API: init(), select(), all_models()."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

import fatih_hoca
from fatih_hoca import Pick, Failure, ModelInfo, ModelRequirements, ScoredModel
from nerd_herd.types import SystemSnapshot


# ─── Reset module state between tests ────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Reset module-level _selector and _registry between tests."""
    fatih_hoca._selector = None
    fatih_hoca._registry = None
    yield
    fatih_hoca._selector = None
    fatih_hoca._registry = None


# ─── Export tests ─────────────────────────────────────────────────────────────

def test_exports_exist():
    assert hasattr(fatih_hoca, "init")
    assert hasattr(fatih_hoca, "select")
    assert hasattr(fatih_hoca, "all_models")
    assert hasattr(fatih_hoca, "Pick")
    assert hasattr(fatih_hoca, "Failure")
    assert hasattr(fatih_hoca, "ModelInfo")
    assert hasattr(fatih_hoca, "ModelRequirements")
    assert hasattr(fatih_hoca, "ScoredModel")


def test_imports_from_package():
    """Named imports from the top-level package work."""
    from fatih_hoca import init, select, all_models
    assert callable(init)
    assert callable(select)
    assert callable(all_models)


def test_pick_is_dataclass():
    from fatih_hoca.registry import ModelInfo
    m = ModelInfo(
        name="test", location="local", provider="llama_cpp",
        litellm_name="openai/test",
    )
    p = Pick(model=m, min_time_seconds=5.0)
    assert p.model.name == "test"
    assert p.min_time_seconds == 5.0


def test_failure_is_dataclass():
    f = Failure(model="openai/test", reason="timeout")
    assert f.model == "openai/test"
    assert f.reason == "timeout"


# ─── all_models before init ───────────────────────────────────────────────────

def test_all_models_before_init():
    models = fatih_hoca.all_models()
    assert isinstance(models, list)
    assert len(models) == 0


# ─── select before init ───────────────────────────────────────────────────────

def test_select_before_init_returns_none():
    result = fatih_hoca.select(task="coder", difficulty=5)
    assert result is None


# ─── init with catalog_path ───────────────────────────────────────────────────

def test_init_with_yaml(tmp_path):
    # The registry uses 'cloud:' dict format (not 'cloud_models:' list)
    yaml_content = '''
cloud:
  test-model:
    litellm_name: anthropic/claude-3-5-haiku-20241022
'''
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)

    nerd_herd = MagicMock()
    nerd_herd.snapshot.return_value = SystemSnapshot()

    names = fatih_hoca.init(catalog_path=str(yaml_file), nerd_herd=nerd_herd)
    assert isinstance(names, list)

    models = fatih_hoca.all_models()
    assert any(m.name == "test-model" for m in models)


def test_init_returns_model_names(tmp_path):
    yaml_content = '''
cloud:
  model-a:
    litellm_name: anthropic/claude-3-5-haiku-20241022
  model-b:
    litellm_name: anthropic/claude-3-5-sonnet-20241022
'''
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)

    nerd_herd = MagicMock()
    nerd_herd.snapshot.return_value = SystemSnapshot()

    names = fatih_hoca.init(catalog_path=str(yaml_file), nerd_herd=nerd_herd)
    assert "model-a" in names
    assert "model-b" in names


def test_init_with_no_args_returns_empty_list():
    nerd_herd = MagicMock()
    nerd_herd.snapshot.return_value = SystemSnapshot()
    names = fatih_hoca.init(nerd_herd=nerd_herd)
    assert names == []
    assert fatih_hoca.all_models() == []


def test_init_without_nerd_herd_uses_noop():
    """init() without nerd_herd uses a no-op stub (does not raise)."""
    names = fatih_hoca.init()
    assert isinstance(names, list)
    # select() should work (returns None due to no models)
    result = fatih_hoca.select(task="coder", difficulty=5)
    assert result is None


# ─── select after init ────────────────────────────────────────────────────────

def test_select_after_init_with_registered_model():
    """Register a model directly, then select it."""
    from fatih_hoca.registry import ModelInfo, ModelRegistry
    from fatih_hoca.selector import Selector

    nerd_herd = MagicMock()
    nerd_herd.snapshot.return_value = SystemSnapshot(vram_available_mb=8192)
    nerd_herd.recent_swap_count.return_value = 0

    # Use init() to set up, then register manually
    fatih_hoca.init(nerd_herd=nerd_herd)

    model = ModelInfo(
        name="local-test",
        location="local",
        provider="llama_cpp",
        litellm_name="openai/local-test",
        capabilities={
            "reasoning": 7.0,
            "code_generation": 7.0,
            "tool_use": 6.0,
            "instruction_adherence": 6.0,
        },
        supports_function_calling=True,
        tokens_per_second=20.0,
        context_length=32768,
    )
    fatih_hoca._registry.register(model)

    result = fatih_hoca.select(task="coder", difficulty=5)
    assert result is not None
    assert isinstance(result, Pick)
    assert result.model.name == "local-test"


def test_select_after_init_with_yaml(tmp_path):
    """Full integration: init from YAML, then select."""
    yaml_content = '''
cloud:
  groq-llama:
    litellm_name: groq/llama-3.3-70b-versatile
'''
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)

    nerd_herd = MagicMock()
    nerd_herd.snapshot.return_value = SystemSnapshot(vram_available_mb=0)

    fatih_hoca.init(catalog_path=str(yaml_file), nerd_herd=nerd_herd)
    models = fatih_hoca.all_models()
    assert len(models) >= 1

    result = fatih_hoca.select(task="coder", difficulty=5)
    # Cloud model should be selectable (no VRAM check for cloud)
    if result is not None:
        assert isinstance(result, Pick)


def test_init_reinit_replaces_registry(tmp_path):
    """Calling init() twice resets the registry."""
    yaml_a = tmp_path / "a.yaml"
    yaml_a.write_text("cloud:\n  model-a:\n    litellm_name: groq/llama-3.3-70b-versatile\n")

    yaml_b = tmp_path / "b.yaml"
    yaml_b.write_text("cloud:\n  model-b:\n    litellm_name: anthropic/claude-3-5-haiku-20241022\n")

    nerd_herd = MagicMock()
    nerd_herd.snapshot.return_value = SystemSnapshot()

    fatih_hoca.init(catalog_path=str(yaml_a), nerd_herd=nerd_herd)
    first_models = [m.name for m in fatih_hoca.all_models()]
    assert "model-a" in first_models

    fatih_hoca.init(catalog_path=str(yaml_b), nerd_herd=nerd_herd)
    second_models = [m.name for m in fatih_hoca.all_models()]
    assert "model-b" in second_models
    # model-a should NOT be in the new registry (fresh init)
    assert "model-a" not in second_models


# ─── select passes kwargs through ─────────────────────────────────────────────

def test_select_passes_kwargs_to_selector():
    """select() forwards keyword args to Selector.select()."""
    nerd_herd = MagicMock()
    snap = SystemSnapshot(vram_available_mb=8192)
    nerd_herd.snapshot.return_value = snap
    nerd_herd.recent_swap_count.return_value = 0

    fatih_hoca.init(nerd_herd=nerd_herd)

    from fatih_hoca.registry import ModelInfo
    m = ModelInfo(
        name="func-model",
        location="local",
        provider="llama_cpp",
        litellm_name="openai/func-model",
        capabilities={"reasoning": 7.0, "code_generation": 7.0,
                      "tool_use": 6.0, "instruction_adherence": 6.0},
        supports_function_calling=True,
        tokens_per_second=20.0,
        context_length=32768,
    )
    fatih_hoca._registry.register(m)

    # This model supports function calling
    result = fatih_hoca.select(task="coder", needs_function_calling=True)
    assert result is not None
    assert result.model.supports_function_calling is True


# ─── Cloud discovery wiring ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_init_disables_provider_when_discovery_auth_fails(monkeypatch, tmp_path):
    """If groq discovery returns auth_fail, the selector's available_providers must NOT include 'groq'."""
    from fatih_hoca.cloud.types import DiscoveredModel, ProviderResult
    import fatih_hoca

    fake_results = {
        "groq": ProviderResult(provider="groq", status="auth_fail", auth_ok=False, error="401"),
        "openai": ProviderResult(
            provider="openai", status="ok", auth_ok=True,
            models=[DiscoveredModel(litellm_name="gpt-4o", raw_id="gpt-4o", context_length=128000)],
        ),
    }

    async def _fake_refresh(api_keys):
        return fake_results

    monkeypatch.setattr(fatih_hoca, "_run_cloud_discovery", _fake_refresh)

    alerts: list = []
    fatih_hoca.init(
        api_keys={"groq": "g", "openai": "o"},
        cloud_cache_dir=str(tmp_path / "cache"),
        cloud_alert_state_path=str(tmp_path / "throttle.json"),
        alert_fn=lambda provider, status, error: alerts.append((provider, status, error)),
    )
    sel = fatih_hoca._selector
    assert sel is not None
    # available_providers on the selector reflects discovery-confirmed providers only.
    assert "groq" not in (sel._available_providers or set())
    assert "openai" in (sel._available_providers or set())
    # Discovered openai model is registered.
    assert fatih_hoca._registry.get("gpt-4o") is not None
    # auth_fail surfaced through the injected alert callback.
    assert any(a[0] == "groq" and a[1] == "auth_fail" for a in alerts)
    # discovery_results exposed for the boot caller (KDV wiring etc.).
    assert "groq" in fatih_hoca.discovery_results
    assert fatih_hoca.discovery_results["openai"].auth_ok is True


@pytest.mark.asyncio
async def test_init_does_not_mark_dead_yaml_models_missing_from_discovery(
    monkeypatch, tmp_path,
):
    """Regression: a yaml-registered model that doesn't appear in the
    provider's /models response must NOT be auto-marked dead.

    Previously cross-validation marked any registered model missing from
    discovery as dead, then persisted across restarts. Production
    2026-05-02: 16 gemini + 33 openrouter ids all marked dead from one
    degraded discovery, killing every cloud-fallback path.

    Runtime 404 hook is the authoritative signal — discovery-diff is too
    coarse (tier-filtering, pagination, schema drift)."""
    from fatih_hoca.cloud.types import DiscoveredModel, ProviderResult
    from fatih_hoca.registry import ModelInfo
    import fatih_hoca

    # Discovery returns ONE model. Yaml has TWO models registered.
    fake_results = {
        "openai": ProviderResult(
            provider="openai", status="ok", auth_ok=True,
            models=[DiscoveredModel(
                litellm_name="openai/gpt-4o", raw_id="gpt-4o",
                context_length=128000,
            )],
        ),
    }

    async def _fake_refresh(api_keys):
        return fake_results

    monkeypatch.setattr(fatih_hoca, "_run_cloud_discovery", _fake_refresh)

    fatih_hoca.init(
        api_keys={"openai": "k"},
        cloud_cache_dir=str(tmp_path / "cache"),
        cloud_alert_state_path=str(tmp_path / "throttle.json"),
    )

    # Pre-register a yaml-style entry that is NOT in the discovery response.
    yaml_only = ModelInfo(
        name="openai/gpt-4-retired",
        location="cloud",
        provider="openai",
        litellm_name="openai/gpt-4-retired",
        capabilities={"reasoning": 7.0},
    )
    fatih_hoca._registry.register(yaml_only)

    # Re-run discovery cross-check by re-init-ing — yaml entry must NOT
    # be marked dead just because it's missing from /models.
    fatih_hoca.init(
        api_keys={"openai": "k"},
        cloud_cache_dir=str(tmp_path / "cache"),
        cloud_alert_state_path=str(tmp_path / "throttle.json"),
    )
    # Re-register after init wipes registry.
    fatih_hoca._registry.register(yaml_only)

    assert not fatih_hoca._registry.is_dead("openai/gpt-4-retired"), (
        "yaml-only model must not be auto-marked dead by discovery diff"
    )


@pytest.mark.asyncio
async def test_init_revives_previously_dead_id_when_back_in_discovery(
    monkeypatch, tmp_path,
):
    """Symmetric path: if a previously-dead id reappears in /models, revive."""
    from fatih_hoca.cloud.types import DiscoveredModel, ProviderResult
    import fatih_hoca

    fake_results = {
        "openai": ProviderResult(
            provider="openai", status="ok", auth_ok=True,
            models=[DiscoveredModel(
                litellm_name="openai/gpt-4o", raw_id="gpt-4o",
                context_length=128000,
            )],
        ),
    }

    async def _fake_refresh(api_keys):
        return fake_results

    monkeypatch.setattr(fatih_hoca, "_run_cloud_discovery", _fake_refresh)
    fatih_hoca.init(
        api_keys={"openai": "k"},
        cloud_cache_dir=str(tmp_path / "cache"),
        cloud_alert_state_path=str(tmp_path / "throttle.json"),
    )

    # Mark gpt-4o dead, then re-run init — should revive.
    fatih_hoca._registry.mark_dead("openai/gpt-4o")
    assert fatih_hoca._registry.is_dead("openai/gpt-4o")

    fatih_hoca.init(
        api_keys={"openai": "k"},
        cloud_cache_dir=str(tmp_path / "cache"),
        cloud_alert_state_path=str(tmp_path / "throttle.json"),
    )
    assert not fatih_hoca._registry.is_dead("openai/gpt-4o"), (
        "id present in fresh discovery must be revived"
    )
