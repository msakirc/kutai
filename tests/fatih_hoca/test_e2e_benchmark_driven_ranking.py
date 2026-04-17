# tests/fatih_hoca/test_e2e_benchmark_driven_ranking.py
"""End-to-end: AA benchmark signal flows through init → blend → ranking → correct winner.

This is the integration test for Tasks 1–9 combined. If any link breaks
(alias matching, enrich, blend, ranking math), this test exposes it.
"""
from __future__ import annotations

import pytest

import fatih_hoca
from fatih_hoca.registry import ModelInfo


def _isolate_aa_only(monkeypatch):
    """Disable all non-AA benchmark fetchers so enrichment only uses canned AA cache."""
    from src.models.benchmark import benchmark_fetcher as bf

    for cls_name in [
        "LMArenaFetcher", "BFCLFetcher", "OpenRouterRankingsFetcher",
        "SenecaTRBenchFetcher", "TurkishMMLUFetcher", "UGILeaderboardFetcher",
    ]:
        cls = getattr(bf, cls_name, None)
        if cls is not None:
            monkeypatch.setattr(cls, "fetch_bulk", lambda self, cache: {})


def _seed(monkeypatch):
    from fatih_hoca import registry as reg_mod
    seeded = {
        "qwen3-30b-a3b": ModelInfo(
            name="qwen3-30b-a3b", location="local",
            provider="llama_cpp", litellm_name="openai/qwen3-30b-a3b",
            path="/fake/Qwen3-30B-A3B-Instruct-Q4_K_M.gguf",
            total_params_b=30.0, active_params_b=3.0, family="qwen3",
            tokens_per_second=10.0,
            capabilities={
                "reasoning": 6.0, "code_generation": 5.5, "code_reasoning": 5.5,
                "analysis": 6.0, "instruction_adherence": 6.5,
                "tool_use": 5.5, "domain_knowledge": 6.0,
            },
            is_variant=True, variant_flags={"thinking"},
        ),
        "qwen3-32b": ModelInfo(
            name="qwen3-32b", location="local",
            provider="llama_cpp", litellm_name="openai/qwen3-32b",
            path="/fake/Qwen3-32B-Instruct-Q4_K_M.gguf",
            total_params_b=32.0, active_params_b=32.0, family="qwen3",
            tokens_per_second=10.0,
            capabilities={
                "reasoning": 6.5, "code_generation": 5.5, "code_reasoning": 5.5,
                "analysis": 6.0, "instruction_adherence": 6.5,
                "tool_use": 5.5, "domain_knowledge": 6.0,
            },
        ),
    }

    def fake_load_gguf_dir(self, models_dir):
        self._models.update(seeded)
        return list(seeded.values())

    monkeypatch.setattr(reg_mod.ModelRegistry, "load_gguf_dir", fake_load_gguf_dir)


def test_aa_signal_promotes_dense_32b_for_coder_over_base_a3b(
    monkeypatch, canned_aa_cache, tmp_path, fake_nerd_herd
):
    """After AA enrichment and blending, dense 32B (AA code_gen=6.5) must beat
    base A3B (AA code_gen=5.0) for coder task."""
    _isolate_aa_only(monkeypatch)
    _seed(monkeypatch)
    monkeypatch.chdir(canned_aa_cache.parent)

    # Reset fatih_hoca singletons so init runs fresh
    fatih_hoca._registry = None
    fatih_hoca._selector = None

    fake_nerd_herd.vram_available_mb = 24000
    fatih_hoca.init(models_dir=str(tmp_path / "fake"), nerd_herd=fake_nerd_herd)

    models = {m.name: m for m in fatih_hoca.all_models()}
    a3b_code = models["qwen3-30b-a3b"].capabilities["code_generation"]
    d32_code = models["qwen3-32b"].capabilities["code_generation"]
    assert d32_code > a3b_code, (
        f"AA signal must promote dense 32B coder above base A3B: "
        f"32b={d32_code:.2f} a3b={a3b_code:.2f}"
    )

    pick = fatih_hoca.select(
        task="coder", agent_type="coder", difficulty=5,
        estimated_input_tokens=500, estimated_output_tokens=1000,
        call_category="main_work",
        prefer_quality=True,
    )
    assert pick is not None
    assert pick.model.name == "qwen3-32b", (
        f"expected qwen3-32b (AA-stronger coder), got {pick.model.name}"
    )
