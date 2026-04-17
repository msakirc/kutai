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
