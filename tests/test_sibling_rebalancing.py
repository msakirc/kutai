"""
Tests for S7: Multi-model provider sibling load balancing.

Covers:
  - Congested model (>70% util) with low-util sibling (<30%) → sibling gets nudge
  - No rebalancing when primary is not congested (<70%)
  - No rebalancing when sibling is also heavily loaded (>30%)
  - Nudge only applies to cloud models (local models skipped)
  - reason tag added to nudged candidate
  - Multiple siblings: only low-util ones get nudged
  - Re-sort happens after nudge (sibling can overtake congested model)
  - Single model per provider: no rebalancing attempted
"""
from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cloud_model(name, provider="gemini", ctx=128000, demoted=False,
                      is_free=True, tps=0):
    m = MagicMock()
    m.name = name
    m.litellm_name = f"{provider}/{name}"
    m.is_local = False
    m.is_loaded = False
    m.is_free = is_free
    m.demoted = demoted
    m.provider = provider
    m.context_length = ctx
    m.max_tokens = 4096
    m.tokens_per_second = tps
    m.total_params_b = 70.0
    m.active_params_b = 70.0
    m.specialty = None
    m.thinking_model = False
    m.has_vision = False
    m.supports_function_calling = True
    m.supports_json_mode = True
    m.model_type = "dense"
    m.capabilities = {cap: 3.5 for cap in [
        "reasoning", "code_generation", "analysis", "domain_knowledge",
        "instruction_adherence", "context_utilization", "structured_output",
        "tool_use", "planning", "prose_quality", "code_reasoning",
        "system_design", "conversation", "vision",
    ]}
    m.operational_dict.return_value = {
        "context_length": ctx,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "has_vision": False,
        "thinking_model": False,
        "is_local": False,
        "provider": provider,
    }
    m.estimated_cost.return_value = 0.0
    return m


def _make_local_model(name, loaded=False, ctx=8192):
    m = MagicMock()
    m.name = name
    m.litellm_name = f"openai/{name}"
    m.is_local = True
    m.is_loaded = loaded
    m.is_free = True
    m.demoted = False
    m.provider = "local"
    m.context_length = ctx
    m.max_tokens = 2048
    m.tokens_per_second = 20.0
    m.load_time_seconds = 20
    m.gpu_layers = 33
    m.file_size_mb = 5000
    m.vram_required_mb = 6000
    m.total_params_b = 8.0
    m.active_params_b = 8.0
    m.specialty = None
    m.thinking_model = False
    m.has_vision = False
    m.supports_function_calling = True
    m.supports_json_mode = True
    m.model_type = "dense"
    m.capabilities = {cap: 3.5 for cap in [
        "reasoning", "code_generation", "analysis", "domain_knowledge",
        "instruction_adherence", "context_utilization", "structured_output",
        "tool_use", "planning", "prose_quality", "code_reasoning",
        "system_design", "conversation", "vision",
    ]}
    m.operational_dict.return_value = {
        "context_length": ctx, "supports_function_calling": True,
        "supports_json_mode": True, "has_vision": False,
        "thinking_model": False, "is_local": True, "provider": "local",
    }
    m.estimated_cost.return_value = 0.0
    return m


def _run_select_s7(models, util_map: dict):
    """Run select_model with given cloud models and utilization mock."""
    from src.core.router import select_model, ModelRequirements

    reg = MagicMock()
    reg.models = {m.name: m for m in models}

    def _fake_score(model_capabilities, model_operational, requirements):
        return 3.5

    def _fake_util(litellm_name):
        return util_map.get(litellm_name, 0)

    mock_rl = MagicMock()
    mock_rl.has_capacity.return_value = True
    mock_rl.get_utilization.side_effect = _fake_util
    mock_rl.get_provider_utilization.return_value = 50

    reqs = ModelRequirements(task="assistant", difficulty=5)

    with patch("src.core.router.get_registry", return_value=reg), \
         patch("src.core.router.score_model_for_task", side_effect=_fake_score), \
         patch("src.core.router.get_rate_limit_manager", return_value=mock_rl), \
         patch("src.core.router.get_quota_planner") as mock_qp, \
         patch("src.infra.load_manager.is_local_inference_allowed", return_value=True), \
         patch("src.infra.load_manager.get_vram_budget_fraction", return_value=1.0), \
         patch("src.models.local_model_manager.get_runtime_state", return_value=None):
        mock_qp.return_value.expensive_threshold = 7
        return select_model(reqs)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSiblingRebalancing(unittest.TestCase):

    def _names(self, results):
        return [c.model.name for c in results]

    def _reasons_for(self, results, name):
        for c in results:
            if c.model.name == name:
                return c.reasons
        return []

    def test_congested_primary_sibling_gets_nudge(self):
        """
        When preferred model is >70% util and sibling is <30%, sibling gets
        score nudge and reason tag.
        """
        flash = _make_cloud_model("gemini-flash", provider="gemini")
        thinking = _make_cloud_model("gemini-flash-thinking", provider="gemini")

        # flash is congested, thinking is free
        util = {
            "gemini/gemini-flash": 80,
            "gemini/gemini-flash-thinking": 10,
        }
        results = _run_select_s7([flash, thinking], util)

        reasons_thinking = self._reasons_for(results, "gemini-flash-thinking")
        self.assertTrue(
            any("sibling_rebal" in r for r in reasons_thinking),
            f"Expected sibling_rebal in reasons, got: {reasons_thinking}"
        )

    def test_no_rebalancing_when_primary_not_congested(self):
        """If primary util < 70%, no nudge is applied."""
        flash = _make_cloud_model("gemini-flash", provider="gemini")
        thinking = _make_cloud_model("gemini-flash-thinking", provider="gemini")

        util = {
            "gemini/gemini-flash": 50,       # not congested
            "gemini/gemini-flash-thinking": 5,
        }
        results = _run_select_s7([flash, thinking], util)

        reasons_thinking = self._reasons_for(results, "gemini-flash-thinking")
        self.assertFalse(
            any("sibling_rebal" in r for r in reasons_thinking),
            "No nudge expected when primary not congested"
        )

    def test_no_rebalancing_when_sibling_also_loaded(self):
        """If sibling util >= 30%, no nudge."""
        flash = _make_cloud_model("gemini-flash", provider="gemini")
        thinking = _make_cloud_model("gemini-flash-thinking", provider="gemini")

        util = {
            "gemini/gemini-flash": 90,
            "gemini/gemini-flash-thinking": 40,   # sibling also loaded
        }
        results = _run_select_s7([flash, thinking], util)

        reasons_thinking = self._reasons_for(results, "gemini-flash-thinking")
        self.assertFalse(
            any("sibling_rebal" in r for r in reasons_thinking),
            "No nudge expected when sibling also heavily loaded"
        )

    def test_local_models_not_rebalanced(self):
        """Local models should never receive a sibling nudge."""
        llama_a = _make_local_model("llama-a", loaded=True)
        llama_b = _make_local_model("llama-b", loaded=False)

        # Both local — rebalancing shouldn't touch them
        util = {
            "openai/llama-a": 85,
            "openai/llama-b": 5,
        }
        results = _run_select_s7([llama_a, llama_b], util)

        for r in results:
            self.assertFalse(
                any("sibling_rebal" in reason for reason in r.reasons),
                f"Local model {r.model.name} should not get sibling_rebal"
            )

    def test_sibling_can_overtake_congested_model(self):
        """
        When scores are close and primary is heavily congested, the nudged
        sibling should appear before the congested model in final ranking.
        """
        flash = _make_cloud_model("gemini-flash", provider="gemini")
        thinking = _make_cloud_model("gemini-flash-thinking", provider="gemini")

        # Very high congestion on flash, thinking is free — nudge should reorder
        util = {
            "gemini/gemini-flash": 95,
            "gemini/gemini-flash-thinking": 5,
        }
        results = _run_select_s7([flash, thinking], util)
        names = self._names(results)

        # After rebalancing, thinking should rank above flash since it got +8%
        # and scores were equal before the nudge.
        thinking_pos = names.index("gemini-flash-thinking") if "gemini-flash-thinking" in names else 99
        flash_pos = names.index("gemini-flash") if "gemini-flash" in names else 99
        self.assertLess(thinking_pos, flash_pos,
                        "Nudged sibling should rank above congested model")

    def test_single_provider_model_no_rebalancing(self):
        """Provider with only one model should not error or apply nudge."""
        flash = _make_cloud_model("gemini-flash", provider="gemini")
        gpt4o = _make_cloud_model("gpt-4o", provider="openai")

        util = {
            "gemini/gemini-flash": 90,
            "openai/gpt-4o": 5,
        }
        # No exception, flash has no siblings so no nudge
        results = _run_select_s7([flash, gpt4o], util)
        reasons_flash = self._reasons_for(results, "gemini-flash")
        self.assertFalse(any("sibling_rebal" in r for r in reasons_flash))

    def test_both_siblings_congested_no_nudge(self):
        """When both siblings are congested, neither gets a nudge."""
        flash = _make_cloud_model("gemini-flash", provider="gemini")
        thinking = _make_cloud_model("gemini-flash-thinking", provider="gemini")

        util = {
            "gemini/gemini-flash": 80,
            "gemini/gemini-flash-thinking": 75,
        }
        results = _run_select_s7([flash, thinking], util)

        for r in results:
            self.assertFalse(
                any("sibling_rebal" in reason for reason in r.reasons),
                f"Both congested: {r.model.name} should not get sibling_rebal"
            )

    def test_cross_provider_no_interference(self):
        """Congestion in one provider should not affect models in another."""
        flash = _make_cloud_model("gemini-flash", provider="gemini")
        gpt4o = _make_cloud_model("gpt-4o", provider="openai")
        gpt4o_mini = _make_cloud_model("gpt-4o-mini", provider="openai")

        # gemini is fine, openai primary is congested
        util = {
            "gemini/gemini-flash": 5,
            "openai/gpt-4o": 85,
            "openai/gpt-4o-mini": 5,
        }
        results = _run_select_s7([flash, gpt4o, gpt4o_mini], util)

        # gemini-flash should not be nudged (it has no siblings in its provider)
        reasons_flash = self._reasons_for(results, "gemini-flash")
        self.assertFalse(any("sibling_rebal" in r for r in reasons_flash))

        # gpt-4o-mini should be nudged (sibling of congested gpt-4o)
        reasons_mini = self._reasons_for(results, "gpt-4o-mini")
        self.assertTrue(
            any("sibling_rebal" in r for r in reasons_mini),
            f"gpt-4o-mini should get sibling_rebal, got: {reasons_mini}"
        )


if __name__ == "__main__":
    unittest.main()
