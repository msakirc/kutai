# tests/fatih_hoca/test_scoring_hygiene.py
"""Scoring hygiene: no clamp-flattening, real perf_score, narrowed failure penalty."""
from __future__ import annotations

import pytest

from fatih_hoca.ranking import rank_candidates
from fatih_hoca.registry import ModelInfo
from fatih_hoca.requirements import ModelRequirements
from nerd_herd.types import SystemSnapshot


def _make_model(name: str, caps: dict, **kw) -> ModelInfo:
    """Shared helper for tests — creates a ModelInfo with specified capabilities."""
    defaults = dict(
        location=kw.get("location", "local"),
        provider=kw.get("provider", "llama_cpp" if kw.get("location", "local") == "local" else kw.get("provider", "openai")),
        litellm_name=kw.get("litellm_name", f"openai/{name}"),
        path=kw.get("path", f"/fake/{name}.gguf"),
        total_params_b=kw.get("total_params_b", 8.0),
        active_params_b=kw.get("active_params_b", 8.0),
        tokens_per_second=kw.get("tps", 20.0),
        capabilities=caps,
    )
    # Merge any extra kwargs (skip the already-placed ones)
    extras = {k: v for k, v in kw.items() if k not in {"location", "provider", "litellm_name", "path", "total_params_b", "active_params_b", "tps"}}
    defaults.update(extras)
    return ModelInfo(name=name, **defaults)


class TestCapabilityClampRemoval:
    """After removing the ×10-then-clamp, a very-strong model must beat a merely-strong model."""

    def test_very_strong_model_separates_from_strong_model(self):
        all_caps = [
            "reasoning","planning","analysis","code_generation","code_reasoning",
            "system_design","prose_quality","instruction_adherence","domain_knowledge",
            "context_utilization","structured_output","tool_use","conversation","turkish","vision",
        ]
        ten = _make_model("ten", {c: 10.0 for c in all_caps})
        nine = _make_model("nine", {c: 9.0 for c in all_caps})
        reqs = ModelRequirements(
            primary_capability="reasoning",
            difficulty=7,
            estimated_input_tokens=500,
            estimated_output_tokens=500,
        )
        snap = SystemSnapshot()
        ranked = rank_candidates([ten, nine], reqs, snap, failures=[])
        assert ranked[0].model.name == "ten", \
            "after clamp removal, the stronger model must beat the weaker one"
        # Meaningful separation: clamp removal must not be a no-op here
        assert ranked[0].composite_score > ranked[1].composite_score, \
            "ten must have strictly higher composite than nine"


class TestPerfScoreFromTps:
    """perf_score must reflect measured tps, not a hardcoded 50."""

    def test_loaded_model_with_high_tps_scores_above_50(self):
        m = _make_model("fast", {"reasoning": 7.0}, tps=40.0)
        m.is_loaded = True
        reqs = ModelRequirements(primary_capability="reasoning", difficulty=5)

        from nerd_herd.types import LocalModelState
        snap = SystemSnapshot()
        snap.local = LocalModelState(model_name="fast", measured_tps=40.0)

        ranked = rank_candidates([m], reqs, snap, failures=[])
        reasons = ranked[0].reasons
        perf_reason = next((r for r in reasons if r.startswith("perf=")), None)
        assert perf_reason is not None, "ranking must expose perf= in reasons"
        perf_val = float(perf_reason.split("=")[1])
        assert perf_val > 50, f"40 tps should beat baseline 50, got perf={perf_val}"

    def test_unmeasured_cloud_model_falls_back_to_50(self):
        m = _make_model("cloud", {"reasoning": 7.0}, location="cloud",
                        provider="anthropic", litellm_name="claude/sonnet",
                        path="")
        reqs = ModelRequirements(primary_capability="reasoning", difficulty=5)
        snap = SystemSnapshot()

        ranked = rank_candidates([m], reqs, snap, failures=[])
        reasons = ranked[0].reasons
        perf_reason = next((r for r in reasons if r.startswith("perf=")), None)
        assert perf_reason is not None
        perf_val = float(perf_reason.split("=")[1])
        assert perf_val == 50.0, f"cloud model with no history should fall back to 50, got {perf_val}"


class TestFailurePenaltyScope:
    """A single model's 429 must not poison its provider siblings."""

    def test_single_model_rate_limit_does_not_penalize_siblings(self):
        from fatih_hoca.types import Failure
        from nerd_herd.types import CloudProviderState

        rate_limited = _make_model("groq-llama-70b",
            {"reasoning": 7.0}, location="cloud",
            provider="groq", litellm_name="groq/llama-3.3-70b",
            path=None)
        sibling = _make_model("groq-mixtral",
            {"reasoning": 7.0}, location="cloud",
            provider="groq", litellm_name="groq/mixtral-8x7b",
            path=None)

        reqs = ModelRequirements(primary_capability="reasoning", difficulty=5)
        snap = SystemSnapshot()
        snap.cloud["groq"] = CloudProviderState(provider="groq", consecutive_failures=0)

        failures = [Failure(
            model="groq/llama-3.3-70b",
            reason="rate_limit",
        )]

        ranked = rank_candidates([rate_limited, sibling], reqs, snap, failures=failures)
        by_name = {r.model.name: r for r in ranked}

        sibling_reasons = by_name["groq-mixtral"].reasons
        assert not any("rate_limit" in r.lower() for r in sibling_reasons), \
            f"sibling should not be penalized for another model's 429, got reasons={sibling_reasons}"

        rl_reasons = by_name["groq-llama-70b"].reasons
        assert any("rate_limit" in r.lower() for r in rl_reasons), \
            "rate-limited model must show rate_limit in reasons"

    def test_provider_wide_penalty_applies_when_circuit_breaker_trips(self):
        """When Nerd Herd reports consecutive_failures >= 3, treat as provider outage."""
        from fatih_hoca.types import Failure
        from nerd_herd.types import CloudProviderState

        m1 = _make_model("groq-a", {"reasoning": 7.0}, location="cloud",
                         provider="groq", litellm_name="groq/a", path=None)
        m2 = _make_model("groq-b", {"reasoning": 7.0}, location="cloud",
                         provider="groq", litellm_name="groq/b", path=None)

        reqs = ModelRequirements(primary_capability="reasoning", difficulty=5)
        snap = SystemSnapshot()
        snap.cloud["groq"] = CloudProviderState(provider="groq", consecutive_failures=3)

        failures = [Failure(model="groq/a", reason="rate_limit")]

        ranked = rank_candidates([m1, m2], reqs, snap, failures=failures)
        by_name = {r.model.name: r for r in ranked}
        assert any("provider" in r.lower() for r in by_name["groq-b"].reasons), \
            f"sibling should be penalized when circuit breaker shows provider outage, got {by_name['groq-b'].reasons}"
