# test_selector.py
"""Tests for fatih_hoca.selector — eligibility filtering and select()."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from fatih_hoca.selector import Selector
from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.requirements import ModelRequirements
from fatih_hoca.types import Failure, Pick
from nerd_herd.types import SystemSnapshot, LocalModelState, CloudProviderState


# ─── Fixtures / Helpers ───────────────────────────────────────────────────────

def _make_registry(*models: ModelInfo) -> ModelRegistry:
    reg = ModelRegistry()
    for m in models:
        reg.register(m)
    return reg


def _make_nerd_herd(
    vram_available_mb: int = 8192,
    local_model_name: str | None = None,
    local_loaded: bool = False,
) -> MagicMock:
    snap = SystemSnapshot(vram_available_mb=vram_available_mb)
    if local_model_name:
        snap.local = LocalModelState(
            model_name=local_model_name,
            measured_tps=20.0,
        )
    nh = MagicMock()
    nh.snapshot.return_value = snap
    return nh


def _make_model(
    name: str = "test-model",
    location: str = "local",
    provider: str = "llama_cpp",
    litellm_name: str | None = None,
    capabilities: dict | None = None,
    tps: float = 20.0,
    context: int = 32768,
    function_calling: bool = True,
    has_vision: bool = False,
    demoted: bool = False,
    specialty: str = "",
    is_loaded: bool = False,
    tier: str = "paid",
    cost_in: float = 0.0,
    cost_out: float = 0.0,
    thinking_model: bool = False,
) -> ModelInfo:
    caps = capabilities or {
        "reasoning": 7.0,
        "code_generation": 7.0,
        "tool_use": 6.0,
        "instruction_adherence": 6.0,
    }
    return ModelInfo(
        name=name,
        location=location,
        provider=provider,
        litellm_name=litellm_name or (f"openai/{name}" if location == "local" else f"{provider}/{name}"),
        capabilities=caps,
        tokens_per_second=tps,
        context_length=context,
        supports_function_calling=function_calling,
        has_vision=has_vision,
        demoted=demoted,
        specialty=specialty,
        is_loaded=is_loaded,
        tier=tier,
        cost_per_1k_input=cost_in,
        cost_per_1k_output=cost_out,
        thinking_model=thinking_model,
    )


def _make_selector(
    models: list[ModelInfo] | None = None,
    vram_available_mb: int = 8192,
) -> Selector:
    reg = _make_registry(*(models or []))
    nh = _make_nerd_herd(vram_available_mb=vram_available_mb)
    return Selector(registry=reg, nerd_herd=nh)


# ─── Basic contract tests ─────────────────────────────────────────────────────

def test_select_returns_pick_with_valid_model():
    model = _make_model("alpha")
    sel = _make_selector([model])
    result = sel.select(task="coder", difficulty=5)
    assert result is not None
    assert isinstance(result, Pick)
    assert result.model.name == "alpha"
    assert result.min_time_seconds > 0


def test_select_returns_none_when_no_models():
    sel = _make_selector([])
    result = sel.select(task="coder", difficulty=5)
    assert result is None


# ─── Eligibility: demoted models ─────────────────────────────────────────────

def test_select_excludes_demoted_models():
    demoted = _make_model("demoted", demoted=True)
    sel = _make_selector([demoted])
    result = sel.select(task="coder", difficulty=5)
    assert result is None


def test_select_picks_non_demoted_when_demoted_also_present():
    demoted = _make_model("demoted", demoted=True)
    good = _make_model("good", demoted=False)
    sel = _make_selector([demoted, good])
    result = sel.select(task="coder", difficulty=5)
    assert result is not None
    assert result.model.name == "good"


# ─── Eligibility: context length ─────────────────────────────────────────────

def test_select_filters_by_context():
    small_ctx = _make_model("small", context=4096)
    sel = _make_selector([small_ctx])
    # Request more context than the model supports
    result = sel.select(
        task="coder",
        difficulty=5,
        estimated_input_tokens=4000,
        estimated_output_tokens=2000,
        min_context_length=8192,
    )
    assert result is None


def test_select_passes_model_with_sufficient_context():
    large_ctx = _make_model("large", context=32768)
    sel = _make_selector([large_ctx])
    result = sel.select(
        task="coder",
        difficulty=5,
        min_context_length=8192,
    )
    assert result is not None
    assert result.model.name == "large"


# ─── Eligibility: failures ────────────────────────────────────────────────────

def test_select_with_failures_avoids_failed_model():
    model = _make_model("failed-model", litellm_name="openai/failed-model")
    sel = _make_selector([model])
    failures = [Failure(model="openai/failed-model", reason="timeout")]
    result = sel.select(task="coder", difficulty=5, failures=failures)
    assert result is None


def test_select_picks_unfailed_model():
    failed = _make_model("failed", litellm_name="openai/failed")
    good = _make_model("good", litellm_name="openai/good")
    reg = _make_registry(failed, good)
    nh = _make_nerd_herd()
    sel = Selector(registry=reg, nerd_herd=nh)
    failures = [Failure(model="openai/failed", reason="server_error")]
    result = sel.select(task="coder", difficulty=5, failures=failures)
    assert result is not None
    assert result.model.name == "good"


# ─── Eligibility: function calling ───────────────────────────────────────────

def test_select_respects_needs_function_calling():
    no_fc = _make_model("no-fc", function_calling=False)
    sel = _make_selector([no_fc])
    result = sel.select(task="coder", difficulty=5, needs_function_calling=True)
    assert result is None


def test_select_picks_model_with_function_calling():
    no_fc = _make_model("no-fc", function_calling=False)
    with_fc = _make_model("with-fc", function_calling=True)
    sel = _make_selector([no_fc, with_fc])
    result = sel.select(task="coder", difficulty=5, needs_function_calling=True)
    assert result is not None
    assert result.model.name == "with-fc"


# ─── Eligibility: vision ──────────────────────────────────────────────────────

def test_select_respects_needs_vision():
    no_vision = _make_model("no-vision", has_vision=False)
    sel = _make_selector([no_vision])
    result = sel.select(task="visual_reviewer", difficulty=5, needs_vision=True)
    assert result is None


def test_select_picks_vision_model():
    no_vision = _make_model("no-vision", has_vision=False)
    with_vision = _make_model(
        "vision",
        has_vision=True,
        capabilities={
            "reasoning": 7.0,
            "vision": 8.0,
            "code_generation": 6.0,
            "tool_use": 6.0,
            "instruction_adherence": 7.0,
        },
    )
    sel = _make_selector([no_vision, with_vision])
    result = sel.select(task="visual_reviewer", difficulty=5, needs_vision=True)
    assert result is not None
    assert result.model.name == "vision"


# ─── Eligibility: local_only ─────────────────────────────────────────────────

def test_select_local_only_rejects_cloud():
    cloud_model = _make_model(
        "cloud", location="cloud", provider="anthropic",
        litellm_name="anthropic/claude",
    )
    sel = _make_selector([cloud_model])
    result = sel.select(task="coder", difficulty=5, local_only=True)
    assert result is None


def test_select_local_only_picks_local():
    local = _make_model("local")
    cloud = _make_model("cloud", location="cloud", provider="anthropic",
                        litellm_name="anthropic/claude")
    sel = _make_selector([local, cloud])
    result = sel.select(task="coder", difficulty=5, local_only=True)
    assert result is not None
    assert result.model.name == "local"


# ─── Eligibility: exclude_models ─────────────────────────────────────────────

def test_select_respects_exclude_models():
    model = _make_model("excluded", litellm_name="openai/excluded")
    sel = _make_selector([model])
    result = sel.select(task="coder", exclude_models=["openai/excluded"])
    assert result is None


# ─── Swap budget ─────────────────────────────────────────────────────────────

def test_select_with_exhausted_swap_budget_prefers_loaded():
    """When swap budget is exhausted, prefer already-loaded model over swap."""
    loaded = _make_model("loaded", is_loaded=True)
    unloaded = _make_model(
        "unloaded",
        is_loaded=False,
        capabilities={
            "reasoning": 9.0,
            "code_generation": 9.0,
            "tool_use": 8.0,
            "instruction_adherence": 8.0,
        },
    )
    reg = _make_registry(loaded, unloaded)
    nh = _make_nerd_herd()
    sel = Selector(registry=reg, nerd_herd=nh)

    # Exhaust swap budget
    for _ in range(3):
        sel._swap_budget.record_swap()

    assert sel._swap_budget.exhausted

    result = sel.select(task="coder", difficulty=5, priority=5)
    # Should pick the loaded model (no swap needed), not unloaded
    assert result is not None
    assert result.model.name == "loaded"


def test_select_swap_budget_none_when_exhausted_and_no_alternative():
    """When swap budget is exhausted and no loaded/cloud model is available, return None."""
    unloaded = _make_model("unloaded", is_loaded=False)
    reg = _make_registry(unloaded)
    nh = _make_nerd_herd()
    sel = Selector(registry=reg, nerd_herd=nh)

    # Exhaust swap budget
    for _ in range(3):
        sel._swap_budget.record_swap()

    result = sel.select(task="coder", difficulty=5, priority=5)
    assert result is None


def test_select_swap_budget_exempt_for_high_priority():
    """Priority >= 9 is exempt from swap budget."""
    unloaded = _make_model("unloaded", is_loaded=False)
    sel = _make_selector([unloaded])

    # Exhaust swap budget
    for _ in range(3):
        sel._swap_budget.record_swap()

    # Priority 9 is exempt — should still work
    result = sel.select(task="coder", difficulty=5, priority=9)
    assert result is not None
    assert result.model.name == "unloaded"


def test_select_swap_budget_exempt_for_local_only():
    """local_only requests are exempt from swap budget."""
    unloaded = _make_model("unloaded", is_loaded=False)
    sel = _make_selector([unloaded])

    for _ in range(3):
        sel._swap_budget.record_swap()

    result = sel.select(task="coder", difficulty=5, local_only=True)
    assert result is not None


# ─── model_override ───────────────────────────────────────────────────────────

def test_select_model_override_returns_that_model():
    alpha = _make_model("alpha")
    beta = _make_model("beta")
    sel = _make_selector([alpha, beta])
    result = sel.select(task="coder", model_override="alpha")
    assert result is not None
    assert result.model.name == "alpha"


def test_select_model_override_by_litellm_name():
    model = _make_model("mymodel", litellm_name="openai/mymodel")
    sel = _make_selector([model])
    result = sel.select(task="coder", model_override="openai/mymodel")
    assert result is not None
    assert result.model.name == "mymodel"


def test_select_returns_none_when_model_override_not_found():
    model = _make_model("alpha")
    sel = _make_selector([model])
    result = sel.select(task="coder", model_override="nonexistent-model")
    assert result is None


# ─── min_time_seconds calculation ────────────────────────────────────────────

def test_select_min_time_calculation():
    model = _make_model("fast", tps=50.0)
    sel = _make_selector([model])
    result = sel.select(
        task="coder",
        estimated_output_tokens=1000,
        needs_thinking=False,
    )
    assert result is not None
    # 1000 tokens / 50 tps = 20s
    assert abs(result.min_time_seconds - 20.0) < 0.1


def test_select_min_time_thinking_multiplier():
    model = _make_model("thinker", tps=20.0, thinking_model=True)
    sel = _make_selector([model])
    result = sel.select(
        task="coder",
        estimated_output_tokens=200,
        needs_thinking=True,
    )
    assert result is not None
    # 200 / 20 = 10s, × 3 for thinking = 30s
    assert abs(result.min_time_seconds - 30.0) < 0.1


# ─── Circuit breaker ─────────────────────────────────────────────────────────

def test_select_rejects_cloud_with_circuit_breaker_open():
    cloud = _make_model(
        "claude",
        location="cloud",
        provider="anthropic",
        litellm_name="anthropic/claude",
    )
    snap = SystemSnapshot(vram_available_mb=8192)
    snap.cloud["anthropic"] = CloudProviderState(
        provider="anthropic",
        consecutive_failures=5,
    )
    nh = MagicMock()
    nh.snapshot.return_value = snap

    reg = _make_registry(cloud)
    sel = Selector(registry=reg, nerd_herd=nh)
    result = sel.select(task="coder", difficulty=5)
    assert result is None


def test_select_accepts_cloud_with_circuit_breaker_below_threshold():
    cloud = _make_model(
        "claude",
        location="cloud",
        provider="anthropic",
        litellm_name="anthropic/claude",
        capabilities={
            "reasoning": 9.0,
            "code_generation": 9.0,
            "tool_use": 8.0,
            "instruction_adherence": 8.0,
        },
    )
    snap = SystemSnapshot(vram_available_mb=8192)
    snap.cloud["anthropic"] = CloudProviderState(
        provider="anthropic",
        consecutive_failures=4,  # 4 < 5, circuit still closed
    )
    nh = MagicMock()
    nh.snapshot.return_value = snap

    reg = _make_registry(cloud)
    sel = Selector(registry=reg, nerd_herd=nh)
    result = sel.select(task="coder", difficulty=5)
    assert result is not None
    assert result.model.name == "claude"


# ─── VRAM availability ────────────────────────────────────────────────────────

def test_select_rejects_local_when_no_vram():
    model = _make_model("local")
    reg = _make_registry(model)
    nh = _make_nerd_herd(vram_available_mb=0)
    sel = Selector(registry=reg, nerd_herd=nh)
    result = sel.select(task="coder", difficulty=5)
    assert result is None


def test_select_local_with_vram():
    model = _make_model("local")
    sel = _make_selector([model], vram_available_mb=8192)
    result = sel.select(task="coder", difficulty=5)
    assert result is not None


# ─── Coding specialty mismatch ────────────────────────────────────────────────

def test_select_coding_specialty_rejected_for_non_code_task():
    coding_model = _make_model("code-specialist", specialty="coding")
    sel = _make_selector([coding_model])
    result = sel.select(task="planner", difficulty=5)
    assert result is None


def test_select_coding_specialty_accepted_for_code_task():
    coding_model = _make_model("code-specialist", specialty="coding")
    sel = _make_selector([coding_model])
    result = sel.select(task="coder", difficulty=5)
    assert result is not None
