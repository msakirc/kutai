# test_select_diag.py
"""Tests for the ``diag_out`` out-param on Selector.select().

WS-1 (handoff 2026-05-25-researcher-no-candidates): when select() returns
an empty pool it must surface WHICH filter emptied it so the downstream
forensic logger writes a non-blank snapshot_summary. The selector already
computes the per-reason histogram + per-model reject reasons + pressure
scalars internally; ``diag_out`` (an optional caller-supplied dict, mutated
in place) carries them out without changing the Pick|None return contract.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fatih_hoca.selector import Selector
from fatih_hoca.registry import ModelInfo, ModelRegistry
from nerd_herd.types import (
    SystemSnapshot, CloudProviderState, CloudModelState,
)


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
        is_loaded=is_loaded,
    )


def _selector(models, *, vram_available_mb=8192, snap=None,
              available_providers=None) -> Selector:
    reg = _make_registry(*models)
    if snap is None:
        snap = SystemSnapshot(vram_available_mb=vram_available_mb)
    nh = MagicMock()
    nh.snapshot.return_value = snap
    nh.recent_swap_count.return_value = 0
    return Selector(registry=reg, nerd_herd=nh,
                    available_providers=available_providers)


# ─── eligibility-empty stage ──────────────────────────────────────────────────

def test_diag_out_eligibility_empty_names_fc_filter():
    """All models non-FC, task needs FC → empty pool. diag must name the
    eligibility stage and the no_function_calling histogram entry."""
    models = [
        _make_model("loc-a", function_calling=False, is_loaded=True),
        _make_model("loc-b", function_calling=False, is_loaded=True),
    ]
    sel = _selector(models)
    diag: dict = {}

    pick = sel.select(task="test", agent_type="", needs_function_calling=True,
                      diag_out=diag)

    assert pick is None
    assert diag["empty_stage"] == "eligibility"
    assert diag["eligible_count"] == 0
    assert diag["filter_reasons"].get("no_function_calling") == 2


def test_diag_out_fc_capable_rejected_names_daily_exhausted_cloud():
    """An FC-capable cloud model rejected for daily_exhausted must appear in
    fc_capable_rejected with its reason; a non-FC local must NOT (it was only
    rejected for lacking FC, not for a rate/pressure cause)."""
    cloud = _make_model("flash", location="cloud", provider="gemini",
                        litellm_name="gemini/flash", function_calling=True)
    local = _make_model("loc", function_calling=False, is_loaded=True)
    snap = SystemSnapshot(vram_available_mb=8192)
    snap.cloud["gemini"] = CloudProviderState(
        provider="gemini",
        models={"gemini/flash": CloudModelState(
            model_id="gemini/flash", daily_exhausted=True)},
    )
    sel = _selector([cloud, local], snap=snap, available_providers={"gemini"})
    diag: dict = {}

    pick = sel.select(task="test", agent_type="", needs_function_calling=True,
                      diag_out=diag)

    assert pick is None
    assert diag["empty_stage"] == "eligibility"
    fc = diag["fc_capable_rejected"]
    assert fc.get("gemini/flash", "").startswith("daily_exhausted")
    # non-FC local must not be listed as an FC-capable rejection
    assert "openai/loc" not in fc


def test_diag_out_pressure_stage_records_threshold_and_scalars(monkeypatch):
    """Candidates pass eligibility but all fall below the pressure threshold
    → empty pool at the pressure stage. diag records threshold + scalars."""
    m = _make_model("loaded-fc", function_calling=True, is_loaded=True)
    sel = _selector([m])

    def fake_rank(candidates, **_kw):
        s = MagicMock()
        s.model = candidates[0]
        s.score = 5.0
        s.urgency = -1.0  # below any threshold → filtered out
        return [s]

    monkeypatch.setattr("fatih_hoca.selector.rank_candidates", fake_rank)
    diag: dict = {}

    pick = sel.select(task="test", agent_type="", needs_function_calling=True,
                      diag_out=diag)

    assert pick is None
    assert diag["empty_stage"] == "pressure"
    assert "pressure_threshold" in diag
    assert diag["pressure_scalars"].get("loaded-fc") == -1.0


def test_diag_out_success_sets_empty_stage_none(monkeypatch):
    """A served pick must mark empty_stage None — never a false empty signal."""
    m = _make_model("loaded-fc", function_calling=True, is_loaded=True)
    sel = _selector([m])

    def fake_rank(candidates, **_kw):
        s = MagicMock()
        s.model = candidates[0]
        s.score = 8.0
        s.urgency = 0.0  # clears default threshold (-0.75)
        return [s]

    monkeypatch.setattr("fatih_hoca.selector.rank_candidates", fake_rank)
    diag: dict = {}

    pick = sel.select(task="test", agent_type="", needs_function_calling=True,
                      diag_out=diag)

    assert pick is not None
    assert diag["empty_stage"] is None


def test_diag_out_none_is_safe():
    """diag_out defaulting to None must not raise on the empty path."""
    models = [_make_model("loc", function_calling=False, is_loaded=True)]
    sel = _selector(models)
    assert sel.select(task="test", needs_function_calling=True) is None
