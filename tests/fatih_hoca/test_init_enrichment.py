# tests/fatih_hoca/test_init_enrichment.py
"""init() must populate benchmark_scores on registered models from cached AA data."""
from __future__ import annotations

import logging
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


def _seed_registry_with_aa_matching_models(monkeypatch, canned_aa_cache):
    """Monkeypatch ModelRegistry.load_gguf_dir to return three Qwen locals whose names match the canned AA cache."""
    from fatih_hoca import registry as reg_mod
    _isolate_aa_only(monkeypatch)

    def fake_load_gguf_dir(self, models_dir):
        self._models["qwen3-30b-a3b"] = ModelInfo(
            name="qwen3-30b-a3b", location="local",
            provider="llama_cpp", litellm_name="openai/qwen3-30b-a3b",
            path="/fake/Qwen3-30B-A3B-Instruct-Q4_K_M.gguf",
            total_params_b=30.0, active_params_b=3.0,
            family="qwen3",
            capabilities={"reasoning": 5.0, "code_generation": 5.0},
            is_variant=True, variant_flags={"thinking"},
        )
        self._models["qwen3-32b"] = ModelInfo(
            name="qwen3-32b", location="local",
            provider="llama_cpp", litellm_name="openai/qwen3-32b",
            path="/fake/Qwen3-32B-Instruct-Q4_K_M.gguf",
            total_params_b=32.0, active_params_b=32.0,
            family="qwen3",
            capabilities={"reasoning": 5.0, "code_generation": 5.0},
        )
        return list(self._models.values())

    monkeypatch.setattr(reg_mod.ModelRegistry, "load_gguf_dir", fake_load_gguf_dir)


def test_init_populates_benchmark_scores(monkeypatch, canned_aa_cache, tmp_path):
    _seed_registry_with_aa_matching_models(monkeypatch, canned_aa_cache)

    # Point enrichment at our canned cache dir
    monkeypatch.chdir(canned_aa_cache.parent)

    fatih_hoca.init(models_dir=str(tmp_path / "fake_gguf_dir"))

    models = {m.name: m for m in fatih_hoca.all_models()}
    assert "qwen3-30b-a3b" in models
    assert "qwen3-32b" in models

    # Benchmark scores should be populated from AA cache
    a3b = models["qwen3-30b-a3b"]
    d32 = models["qwen3-32b"]
    assert a3b.benchmark_scores, "qwen3-30b-a3b should have benchmark_scores from AA"
    assert d32.benchmark_scores, "qwen3-32b should have benchmark_scores from AA"

    # After blending, capabilities should reflect AA signal (AA has d32 code_gen=6.5 vs a3b=5.0)
    assert d32.capabilities["code_generation"] > a3b.capabilities["code_generation"], \
        "post-blend capabilities must reflect AA's stronger coder signal for dense 32B"


def test_init_logs_coverage(monkeypatch, canned_aa_cache, tmp_path, caplog):
    _seed_registry_with_aa_matching_models(monkeypatch, canned_aa_cache)
    monkeypatch.chdir(canned_aa_cache.parent)

    with caplog.at_level(logging.INFO, logger="fatih_hoca"):
        fatih_hoca.init(models_dir=str(tmp_path / "fake_gguf_dir"))

    coverage_logs = [r for r in caplog.records if "benchmark coverage" in r.message.lower()]
    assert coverage_logs, "init() must emit a 'benchmark coverage: N/M matched' log line"
    msg = coverage_logs[0].message
    assert "2/2" in msg or "matched=2" in msg or "2 matched" in msg, \
        f"expected 2/2 coverage, got: {msg}"


def test_init_warns_on_unmatched_models(monkeypatch, canned_aa_cache, tmp_path, caplog):
    """If a model doesn't match any AA entry, init must log the unmatched name at WARNING."""
    from fatih_hoca import registry as reg_mod

    def fake_load_gguf_dir(self, models_dir):
        self._models["some-obscure-local"] = ModelInfo(
            name="some-obscure-local", location="local",
            provider="llama_cpp", litellm_name="openai/some-obscure-local",
            path="/fake/some-obscure-local.gguf",
            total_params_b=7.0, family="unknown",
            capabilities={"reasoning": 5.0},
        )
        return list(self._models.values())

    monkeypatch.setattr(reg_mod.ModelRegistry, "load_gguf_dir", fake_load_gguf_dir)
    _isolate_aa_only(monkeypatch)
    monkeypatch.chdir(canned_aa_cache.parent)

    with caplog.at_level(logging.WARNING, logger="fatih_hoca"):
        fatih_hoca.init(models_dir=str(tmp_path / "fake_gguf_dir"))

    unmatched_logs = [r for r in caplog.records if "unmatched" in r.message.lower()]
    assert unmatched_logs, "init() must warn on models without benchmark coverage"
    assert any("some-obscure-local" in r.message for r in unmatched_logs)
