# test_servable.py
"""Tests for the per-model servability primitive (RC-A, mission 74).

`is_servable(model, reqs)` answers "can the task KEEP running on the model
it already holds, right now?" — it reuses the selector's hard-eligibility
chain for ONE model against the current snapshot, WITHOUT the pool-pressure
gate (pressure governs starting *new* load, not continuing held load).

A held local model that's already loaded must survive vram_available_mb==0:
its own residency consumed the VRAM, that is continuation not contention.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import fatih_hoca
from fatih_hoca.selector import Selector
from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.requirements import ModelRequirements
from nerd_herd.types import (
    SystemSnapshot, CloudProviderState, CloudModelState,
)


# ─── Helpers (mirror test_selector.py) ───────────────────────────────────────

def _make_registry(*models: ModelInfo) -> ModelRegistry:
    reg = ModelRegistry()
    for m in models:
        reg.register(m)
    return reg


def _make_model(
    name: str = "test-model",
    location: str = "local",
    provider: str = "llama_cpp",
    litellm_name: str | None = None,
    function_calling: bool = True,
    demoted: bool = False,
    is_loaded: bool = False,
    context: int = 32768,
) -> ModelInfo:
    return ModelInfo(
        name=name,
        location=location,
        provider=provider,
        litellm_name=litellm_name or (
            f"openai/{name}" if location == "local" else f"{provider}/{name}"
        ),
        capabilities={"reasoning": 7.0, "code_generation": 7.0,
                      "tool_use": 6.0, "instruction_adherence": 6.0},
        tokens_per_second=20.0,
        context_length=context,
        supports_function_calling=function_calling,
        demoted=demoted,
        is_loaded=is_loaded,
    )


def _selector(models, *, vram_available_mb=8192, snap=None,
              available_providers=None) -> Selector:
    reg = _make_registry(*models)
    if snap is None:
        snap = SystemSnapshot(vram_available_mb=vram_available_mb)
    nh = MagicMock()
    nh.snapshot.return_value = snap
    return Selector(registry=reg, nerd_herd=nh,
                    available_providers=available_providers)


def _reqs(**kw) -> ModelRequirements:
    base = dict(task="coder", agent_type="coder", difficulty=5,
                estimated_input_tokens=1000, estimated_output_tokens=1000)
    base.update(kw)
    return ModelRequirements(**base)


# ─── Selector.is_servable ─────────────────────────────────────────────────────

def test_is_servable_true_for_eligible_local_model():
    m = _make_model("alpha")
    sel = _selector([m], vram_available_mb=8192)
    assert sel.is_servable(model=m, reqs=_reqs()) is True


def test_is_servable_false_for_demoted_model():
    m = _make_model("dead", demoted=True)
    sel = _selector([m])
    assert sel.is_servable(model=m, reqs=_reqs()) is False


def test_is_servable_false_for_daily_exhausted_cloud():
    m = _make_model("flash", location="cloud", provider="gemini",
                    litellm_name="gemini/flash")
    snap = SystemSnapshot(vram_available_mb=8192)
    snap.cloud["gemini"] = CloudProviderState(
        provider="gemini",
        models={"gemini/flash": CloudModelState(
            model_id="gemini/flash", daily_exhausted=True)},
    )
    sel = _selector([m], snap=snap, available_providers={"gemini"})
    assert sel.is_servable(model=m, reqs=_reqs()) is False


def test_is_servable_loaded_local_survives_zero_vram():
    """Held local model already resident — vram_available_mb==0 reflects
    its OWN occupancy, not a swap-out. It must remain servable to continue."""
    loaded = _make_model("resident", is_loaded=True)
    sel = _selector([loaded], vram_available_mb=0)
    assert sel.is_servable(model=loaded, reqs=_reqs()) is True


def test_is_servable_unloaded_local_blocked_at_zero_vram():
    """An UNloaded local at vram==0 is genuinely unservable (would need a
    swap into a full GPU) — the carve-out must not leak to it."""
    unloaded = _make_model("cold", is_loaded=False)
    sel = _selector([unloaded], vram_available_mb=0)
    assert sel.is_servable(model=unloaded, reqs=_reqs()) is False


# ─── module-level fatih_hoca.is_servable ──────────────────────────────────────

def test_module_is_servable_false_when_uninitialized(monkeypatch):
    """No selector built yet → fail-closed False (caller re-selects)."""
    monkeypatch.setattr(fatih_hoca, "_selector", None)
    m = _make_model("alpha")
    assert fatih_hoca.is_servable(model=m, reqs=_reqs()) is False


def test_module_is_servable_delegates_to_selector(monkeypatch):
    m = _make_model("alpha")
    sel = _selector([m])
    monkeypatch.setattr(fatih_hoca, "_selector", sel)
    assert fatih_hoca.is_servable(model=m, reqs=_reqs()) is True
