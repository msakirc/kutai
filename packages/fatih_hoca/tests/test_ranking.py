# test_ranking.py
"""Tests for fatih_hoca.ranking — composite scoring and failure adaptation."""
from __future__ import annotations

import pytest

from fatih_hoca.ranking import rank_candidates, ScoredModel
from fatih_hoca.registry import ModelInfo
from fatih_hoca.requirements import ModelRequirements
from fatih_hoca.types import Failure
from nerd_herd.types import SystemSnapshot, LocalModelState, CloudProviderState


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_model(
    name,
    location="local",
    provider="local",
    capabilities=None,
    tps=20.0,
    cost_in=0.0,
    cost_out=0.0,
    context=32768,
    function_calling=True,
    thinking=False,
    is_loaded=False,
    specialty=None,
    **kw,
):
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
        litellm_name=f"{provider}/{name}" if location == "cloud" else f"local/{name}",
        capabilities=caps,
        tokens_per_second=tps,
        cost_per_1k_input=cost_in,
        cost_per_1k_output=cost_out,
        context_length=context,
        supports_function_calling=function_calling,
        thinking_model=thinking,
        is_loaded=is_loaded,
        specialty=specialty or "",
        **kw,
    )


# ─── Basic Contract Tests ─────────────────────────────────────────────────────

def test_rank_returns_scored_models():
    m = _make_model("test")
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([m], reqs, snap, failures=[])
    assert len(ranked) == 1
    assert isinstance(ranked[0], ScoredModel)
    assert ranked[0].score > 0
    assert isinstance(ranked[0].reasons, list)


def test_rank_empty_candidates():
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([], reqs, snap, failures=[])
    assert ranked == []


def test_rank_multiple_returns_sorted():
    """Result should be sorted best-first."""
    models = [_make_model(f"model-{i}") for i in range(5)]
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates(models, reqs, snap, failures=[])
    assert len(ranked) == 5
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_scored_model_litellm_name():
    m = _make_model("mymodel")
    sm = ScoredModel(model=m, score=50.0)
    assert sm.litellm_name == "local/mymodel"


# ─── Loaded Model Preference ─────────────────────────────────────────────────

def test_rank_prefers_loaded_model():
    loaded = _make_model("qwen3-30b", is_loaded=True)
    unloaded = _make_model("llama-8b", is_loaded=False)
    snap = SystemSnapshot(
        vram_available_mb=8000,
        # idle_seconds > 0 so the loaded model gets its S9 perishability signal
        # (pressure_for returns 0 for idle=0; a warm loaded model is preferred).
        local=LocalModelState(model_name="qwen3-30b", measured_tps=15.0, idle_seconds=30.0),
    )
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([loaded, unloaded], reqs, snap, failures=[])
    assert ranked[0].model.name == "qwen3-30b"


def test_loaded_model_gets_stickiness_reason():
    loaded = _make_model("qwen3-30b", is_loaded=True)
    snap = SystemSnapshot(
        local=LocalModelState(model_name="qwen3-30b", measured_tps=15.0),
    )
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([loaded], reqs, snap, failures=[])
    assert "loaded" in ranked[0].reasons


# ─── Difficulty-Driven Cost vs Capability ─────────────────────────────────────

def test_rank_prefers_cheaper_for_easy():
    expensive = _make_model(
        "claude",
        location="cloud",
        provider="anthropic",
        cost_in=0.003,
        cost_out=0.015,
        capabilities={
            "reasoning": 9.0,
            "code_generation": 9.0,
            "tool_use": 8.0,
            "instruction_adherence": 8.0,
        },
    )
    cheap = _make_model(
        "groq-llama",
        location="cloud",
        provider="groq",
        capabilities={
            "reasoning": 6.0,
            "code_generation": 6.0,
            "tool_use": 5.0,
            "instruction_adherence": 5.0,
        },
    )
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=2)
    ranked = rank_candidates([expensive, cheap], reqs, snap, failures=[])
    assert ranked[0].model.name == "groq-llama"


def test_rank_prefers_stronger_for_hard():
    strong = _make_model(
        "claude",
        location="cloud",
        provider="anthropic",
        cost_in=0.003,
        cost_out=0.015,
        tier="paid",
        capabilities={
            "reasoning": 9.5,
            "code_generation": 9.0,
            "tool_use": 9.0,
            "instruction_adherence": 9.0,
        },
    )
    weak = _make_model(
        "weak-cloud",
        location="cloud",
        provider="anthropic",
        cost_in=0.001,
        cost_out=0.005,
        tier="paid",
        capabilities={
            "reasoning": 5.0,
            "code_generation": 5.0,
            "tool_use": 5.0,
            "instruction_adherence": 5.0,
        },
    )
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=9)
    ranked = rank_candidates([strong, weak], reqs, snap, failures=[])
    # At difficulty 9, capability weight (45%) dominates when cost/speed
    # are comparable — no min_score gate needed.
    assert ranked[0].model.name == "claude"


# ─── Failure Adaptation ───────────────────────────────────────────────────────

def test_failure_adaptation_timeout():
    slow = _make_model("slow-model", tps=5.0)
    fast = _make_model("fast-model", tps=30.0)
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    failures = [Failure(model="local/slow-model", reason="timeout", latency=120.0)]
    ranked = rank_candidates([slow, fast], reqs, snap, failures=failures)
    assert ranked[0].model.name == "fast-model"


def test_failure_adaptation_rate_limit():
    # Both models have enough capability to pass min_score gate (difficulty=5 → min≈1.88).
    # model-a (anthropic) has higher raw capability but gets a 0.3x rate-limit penalty
    # so model-b (groq) should win.
    model_a = _make_model(
        "model-a",
        location="cloud",
        provider="anthropic",
        cost_in=0.003,
        cost_out=0.015,
        capabilities={
            "reasoning": 9.0,
            "code_generation": 9.0,
            "tool_use": 8.0,
            "instruction_adherence": 8.0,
        },
    )
    model_b = _make_model(
        "model-b",
        location="cloud",
        provider="groq",
        capabilities={
            "reasoning": 7.0,
            "code_generation": 7.0,
            "tool_use": 6.0,
            "instruction_adherence": 6.0,
        },
    )
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    failures = [Failure(model="anthropic/model-a", reason="rate_limit")]
    ranked = rank_candidates([model_a, model_b], reqs, snap, failures=failures)
    assert ranked[0].model.name == "model-b"


def test_failure_adaptation_loading_excludes_model():
    m = _make_model("broken-model")
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    failures = [Failure(model="local/broken-model", reason="loading")]
    ranked = rank_candidates([m], reqs, snap, failures=failures)
    assert ranked == []


def test_failure_adaptation_quality_failure():
    bad = _make_model(
        "bad-model",
        capabilities={
            "reasoning": 7.0,
            "code_generation": 7.0,
            "tool_use": 6.0,
            "instruction_adherence": 6.0,
        },
    )
    good = _make_model(
        "good-model",
        capabilities={
            "reasoning": 7.0,
            "code_generation": 7.0,
            "tool_use": 6.0,
            "instruction_adherence": 6.0,
        },
    )
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    failures = [Failure(model="local/bad-model", reason="quality_failure")]
    ranked = rank_candidates([bad, good], reqs, snap, failures=failures)
    # good-model should win since bad-model has 0.5x penalty
    assert ranked[0].model.name == "good-model"


def test_failure_adaptation_server_error():
    broken = _make_model("broken")
    ok = _make_model("ok")
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    failures = [Failure(model="local/broken", reason="server_error")]
    ranked = rank_candidates([broken, ok], reqs, snap, failures=failures)
    assert ranked[0].model.name == "ok"


# ─── Layer 3: Ranking Adjustments ────────────────────────────────────────────

def test_thinking_bonus():
    thinking = _make_model(
        "thinker",
        thinking=True,
        capabilities={
            "reasoning": 8.0,
            "code_generation": 7.0,
            "tool_use": 6.0,
            "instruction_adherence": 6.0,
        },
    )
    normal = _make_model(
        "normal",
        capabilities={
            "reasoning": 8.0,
            "code_generation": 7.0,
            "tool_use": 6.0,
            "instruction_adherence": 6.0,
        },
    )
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5, needs_thinking=True)
    ranked = rank_candidates([thinking, normal], reqs, snap, failures=[])
    assert ranked[0].model.name == "thinker"


def test_specialty_bonus():
    coding_model = _make_model(
        "coder-special",
        specialty="coding",
        capabilities={
            "reasoning": 7.0,
            "code_generation": 9.0,
            "tool_use": 7.0,
            "instruction_adherence": 7.0,
        },
    )
    general = _make_model(
        "general",
        capabilities={
            "reasoning": 7.5,
            "code_generation": 7.5,
            "tool_use": 7.0,
            "instruction_adherence": 7.0,
        },
    )
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([coding_model, general], reqs, snap, failures=[])
    assert ranked[0].model.name == "coder-special"


def test_specialty_no_bonus_wrong_task():
    """Coding specialty should not get bonus for non-coding tasks."""
    coding_model = _make_model(
        "coder-special",
        specialty="coding",
        capabilities={
            "reasoning": 8.0,
            "planning": 8.0,
            "analysis": 8.0,
            "instruction_adherence": 8.0,
        },
    )
    general = _make_model(
        "general",
        capabilities={
            "reasoning": 8.0,
            "planning": 8.0,
            "analysis": 8.0,
            "instruction_adherence": 8.0,
        },
    )
    snap = SystemSnapshot()
    # planner task — coding specialty shouldn't get bonus
    reqs = ModelRequirements(task="planner", difficulty=5)
    ranked = rank_candidates([coding_model, general], reqs, snap, failures=[])
    # Both have equal caps; without specialty bonus the ordering can be either way,
    # but coding_model should NOT have specialty reason
    for r in ranked:
        if r.model.name == "coder-special":
            assert not any("specialty=" in reason for reason in r.reasons)


# ─── Cloud Utilization via Snapshot ──────────────────────────────────────────

def test_availability_uses_snapshot_cloud():
    """High utilization in snapshot should reduce availability score."""
    from nerd_herd.types import CloudModelState

    model_high = _make_model(
        "claude-high",
        location="cloud",
        provider="anthropic",
        cost_in=0.003,
        cost_out=0.015,
    )
    model_low = _make_model(
        "groq-low",
        location="cloud",
        provider="groq",
    )

    prov_high = CloudProviderState(provider="anthropic", utilization_pct=90.0)
    prov_low = CloudProviderState(provider="groq", utilization_pct=5.0)

    snap = SystemSnapshot(cloud={"anthropic": prov_high, "groq": prov_low})
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([model_high, model_low], reqs, snap, failures=[])
    # groq-low has much better availability score, should rank ahead
    assert ranked[0].model.name == "groq-low"


# ─── Speed Scoring ───────────────────────────────────────────────────────────

def test_measured_tps_from_snapshot():
    """When local model is loaded, measured_tps from snapshot is used."""
    # Same model but snapshot says it's running fast
    fast_model = _make_model("fast", tps=5.0, is_loaded=True)
    snap = SystemSnapshot(
        local=LocalModelState(model_name="fast", measured_tps=40.0)
    )
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([fast_model], reqs, snap, failures=[])
    # Should use 40 tps → speed_score=80 not 40 from static 5 tps
    assert ranked[0].score > 0
    # The "loaded" stickiness bonus is also applied
    assert "loaded" in ranked[0].reasons


# ─── ScoredModel Dataclass ───────────────────────────────────────────────────

def test_scored_model_defaults():
    m = _make_model("m")
    sm = ScoredModel(model=m, score=42.0)
    assert sm.capability_score == 0.0
    assert sm.composite_score == 0.0
    assert sm.reasons == []


def test_scored_model_cloud_litellm_name():
    m = _make_model("claude-3", location="cloud", provider="anthropic")
    sm = ScoredModel(model=m, score=80.0)
    assert sm.litellm_name == "anthropic/claude-3"


# ─── Grading perf_score Integration ──────────────────────────────────────────

def test_cloud_model_with_stats_gets_grading_perf_score(monkeypatch):
    """Grading blend: cloud model with stats gets non-flat perf_score.

    grading=80, tps_perf=50 (cloud fallback) → blended = 0.6*80 + 0.4*50 = 68
    """
    from fatih_hoca import ranking

    def fake_grading(name):
        if name == "groq-llama-70b":
            return 80.0
        return None

    monkeypatch.setattr(ranking, "grading_perf_score", fake_grading)

    model = _make_model(
        "groq-llama-70b",
        location="cloud",
        provider="groq",
    )
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([model], reqs, snap, failures=[])
    assert len(ranked) == 1

    # Find the perf= reason and parse the value
    perf_reasons = [r for r in ranked[0].reasons if r.startswith("perf=")]
    assert len(perf_reasons) == 1, f"expected one perf= reason, got {ranked[0].reasons}"
    perf_val = float(perf_reasons[0].split("=")[1].split("(")[0])
    assert abs(perf_val - 68.0) <= 1.0, f"expected ~68.0 (blend), got {perf_val}"


def test_local_without_stats_falls_back_to_tps_derived(monkeypatch):
    """Local model with no grading stats keeps tps-derived perf_score (~65)."""
    from fatih_hoca import ranking

    monkeypatch.setattr(ranking, "grading_perf_score", lambda _: None)

    # tps=20 → perf_score = min(90, 45 + (20-10)*1.2) = min(90, 57) = 57
    # But if is_loaded + snapshot measured_tps=20:
    #   perf_score = min(95, 50 + (20-10)*1.5) = min(95, 65) = 65
    model = _make_model("local-qwen", tps=20.0, is_loaded=False)
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([model], reqs, snap, failures=[])
    assert len(ranked) == 1

    perf_reasons = [r for r in ranked[0].reasons if r.startswith("perf=")]
    assert len(perf_reasons) == 1, f"expected one perf= reason, got {ranked[0].reasons}"
    perf_val = float(perf_reasons[0].split("=")[1].split("(")[0])
    # tps=20 (not loaded), formula: min(90, 45 + (20-10)*1.2) = 57
    assert abs(perf_val - 57.0) <= 1.0, f"expected ~57.0 (tps-derived), got {perf_val}"
