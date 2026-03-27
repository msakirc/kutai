"""
Tests for S8: Scoring reorganization.

Covers Layer 1 (eligibility hard filter) changes:
  - Coding-specialty model hard-filtered for non-code tasks
  - Coding-specialty model NOT filtered for coding tasks
  - Non-coding specialty model unaffected (no false filter)
  - Layer 3 specialty match bonus still applies for matched tasks

Layer 3 multiplier simplification:
  - prefer_local no longer adds a 1.15x multiplier (weight modifiers handle it)
  - coding_model_mismatch 0.50x multiplier is gone (moved to hard filter)
  - Max 3 conceptual multiplier groups present
"""
from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cloud_model(name, provider="gemini", specialty=None, thinking_model=False):
    m = MagicMock()
    m.name = name
    m.litellm_name = f"{provider}/{name}"
    m.is_local = False
    m.is_loaded = False
    m.is_free = True
    m.demoted = False
    m.provider = provider
    m.context_length = 128000
    m.max_tokens = 4096
    m.tokens_per_second = 0
    m.total_params_b = 70.0
    m.active_params_b = 70.0
    m.specialty = specialty
    m.thinking_model = thinking_model
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
        "context_length": 128000, "supports_function_calling": True,
        "supports_json_mode": True, "has_vision": False,
        "thinking_model": thinking_model, "is_local": False,
        "provider": provider,
    }
    m.estimated_cost.return_value = 0.0
    return m


def _make_local_model(name, loaded=False, ctx=8192, specialty=None):
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
    m.specialty = specialty
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


def _run_select(reqs, models):
    from src.core.router import select_model

    reg = MagicMock()
    reg.models = {m.name: m for m in models}

    def _fake_score(*args, **kwargs):
        return 3.5

    mock_rl = MagicMock()
    mock_rl.has_capacity.return_value = True
    mock_rl.get_utilization.return_value = 20
    mock_rl.get_provider_utilization.return_value = 20

    with patch("src.core.router.get_registry", return_value=reg), \
         patch("src.core.router.score_model_for_task", side_effect=_fake_score), \
         patch("src.core.router.get_rate_limit_manager", return_value=mock_rl), \
         patch("src.core.router.get_quota_planner") as mock_qp, \
         patch("src.infra.load_manager.is_local_inference_allowed", return_value=True), \
         patch("src.infra.load_manager.get_vram_budget_fraction", return_value=1.0), \
         patch("src.models.local_model_manager.get_runtime_state", return_value=None):
        mock_qp.return_value.expensive_threshold = 7
        return select_model(reqs)


# ── Layer 1: coding model hard filter ─────────────────────────────────────────

class TestCodingModelHardFilter(unittest.TestCase):

    def test_coding_model_filtered_for_non_code_task(self):
        """Coding-specialty model must not appear in candidates for planner task."""
        from src.core.router import ModelRequirements
        coding_model = _make_cloud_model("code-specialist", specialty="coding")
        general_model = _make_cloud_model("general-model")

        reqs = ModelRequirements(task="planner", difficulty=5)
        results = _run_select(reqs, [coding_model, general_model])

        names = [c.model.name for c in results]
        self.assertNotIn("code-specialist", names,
                         "Coding model must be hard-filtered for planner task")
        self.assertIn("general-model", names)

    def test_coding_model_filtered_for_assistant_task(self):
        """Coding-specialty model filtered for conversation/assistant tasks."""
        from src.core.router import ModelRequirements
        coding_model = _make_cloud_model("code-specialist", specialty="coding")

        reqs = ModelRequirements(task="assistant", difficulty=5)
        results = _run_select(reqs, [coding_model])

        names = [c.model.name for c in results]
        self.assertNotIn("code-specialist", names)

    def test_coding_model_filtered_for_researcher_task(self):
        """Coding-specialty model filtered for researcher task."""
        from src.core.router import ModelRequirements
        coding_model = _make_cloud_model("code-specialist", specialty="coding")

        reqs = ModelRequirements(task="researcher", difficulty=5)
        results = _run_select(reqs, [coding_model])

        names = [c.model.name for c in results]
        self.assertNotIn("code-specialist", names)

    def test_coding_model_passes_for_coder_task(self):
        """Coding-specialty model must be allowed for coder task."""
        from src.core.router import ModelRequirements
        coding_model = _make_cloud_model("code-specialist", specialty="coding")

        reqs = ModelRequirements(task="coder", difficulty=5)
        results = _run_select(reqs, [coding_model])

        names = [c.model.name for c in results]
        self.assertIn("code-specialist", names,
                      "Coding model must pass filter for coder task")

    def test_coding_model_passes_for_fixer_task(self):
        """Coding-specialty model must be allowed for fixer/implementer tasks."""
        from src.core.router import ModelRequirements
        for task in ("fixer", "implementer", "test_generator"):
            with self.subTest(task=task):
                coding_model = _make_cloud_model("code-specialist", specialty="coding")
                reqs = ModelRequirements(task=task, difficulty=5)
                results = _run_select(reqs, [coding_model])
                names = [c.model.name for c in results]
                self.assertIn("code-specialist", names,
                              f"Coding model should pass for {task}")

    def test_non_coding_specialty_not_filtered(self):
        """Reasoning/vision specialty models are NOT hard-filtered for any task."""
        from src.core.router import ModelRequirements
        reasoning_model = _make_cloud_model("deepthink", specialty="reasoning")

        reqs = ModelRequirements(task="assistant", difficulty=5)
        results = _run_select(reqs, [reasoning_model])

        names = [c.model.name for c in results]
        self.assertIn("deepthink", names,
                      "Non-coding specialty models should not be hard-filtered")

    def test_no_specialty_model_never_filtered(self):
        """Models with no specialty are not filtered regardless of task."""
        from src.core.router import ModelRequirements
        generic = _make_cloud_model("llama3", specialty=None)

        for task in ("planner", "coder", "assistant", "researcher"):
            with self.subTest(task=task):
                reqs = ModelRequirements(task=task, difficulty=5)
                results = _run_select(reqs, [generic])
                names = [c.model.name for c in results]
                self.assertIn("llama3", names)


# ── Layer 3: multiplier structure ─────────────────────────────────────────────

class TestLayerThreeMultipliers(unittest.TestCase):

    def _get_candidate(self, results, name):
        for c in results:
            if c.model.name == name:
                return c
        return None

    def test_specialty_match_bonus_still_applied(self):
        """Specialty-matched models still get the 1.15x bonus in Layer 3."""
        from src.core.router import ModelRequirements
        coding_model = _make_cloud_model("code-specialist", specialty="coding")
        generic = _make_cloud_model("generic")

        reqs = ModelRequirements(task="coder", difficulty=5)
        results = _run_select(reqs, [coding_model, generic])

        code_cand = self._get_candidate(results, "code-specialist")
        self.assertIsNotNone(code_cand)
        reasons = code_cand.reasons
        self.assertTrue(any("specialty" in r for r in reasons),
                        f"Expected specialty reason, got: {reasons}")

    def test_coding_mismatch_reason_absent(self):
        """coding_model_mismatch reason must never appear — it's now a hard filter."""
        from src.core.router import ModelRequirements
        # We can only verify on a passing candidate (coding model for coding task)
        coding_model = _make_cloud_model("code-specialist", specialty="coding")
        reqs = ModelRequirements(task="coder", difficulty=5)
        results = _run_select(reqs, [coding_model])

        for c in results:
            self.assertFalse(
                any("coding_model_mismatch" in r for r in c.reasons),
                "coding_model_mismatch reason should not appear (it's a hard filter)"
            )

    def test_prefer_local_no_multiplier_reason(self):
        """prefer_local must NOT add a 'prefer_local' multiplier reason anymore."""
        from src.core.router import ModelRequirements
        local_model = _make_local_model("llama3")

        reqs = ModelRequirements(task="assistant", difficulty=5, prefer_local=True)
        results = _run_select(reqs, [local_model])

        cand = self._get_candidate(results, "llama3")
        self.assertIsNotNone(cand)
        # The old behavior added a 1.15x multiplier with reason "prefer_local"
        # This should be absent now — preference is expressed through weight mods
        self.assertNotIn("prefer_local", cand.reasons,
                         "prefer_local multiplier should have been removed (S8)")

    def test_thinking_bonus_in_layer_3(self):
        """Thinking bonus still applied via Group A of Layer 3."""
        from src.core.router import ModelRequirements
        thinking_model = _make_cloud_model("qwq", thinking_model=True)
        non_thinking = _make_cloud_model("llama3")

        reqs = ModelRequirements(task="planner", difficulty=7, needs_thinking=True)
        results = _run_select(reqs, [thinking_model, non_thinking])

        thinking_cand = self._get_candidate(results, "qwq")
        self.assertIsNotNone(thinking_cand)
        self.assertIn("thinking_bonus", thinking_cand.reasons)

    def test_at_most_three_multiplier_groups_applied(self):
        """
        A single candidate should have at most 3 multiplier-related reasons.
        The 3 groups are: thinking_bonus, specialty=*, loaded/thinking_mismatch/needs_swap.
        """
        from src.core.router import ModelRequirements
        # A thinking, coding-specialty, loaded model scoring a coding task w/ thinking
        loaded_thinking = _make_local_model("magic-coder", loaded=True, specialty="coding")
        loaded_thinking.thinking_model = True

        reqs = ModelRequirements(task="coder", difficulty=8, needs_thinking=True)
        results = _run_select(reqs, [loaded_thinking])

        cand = self._get_candidate(results, "magic-coder")
        self.assertIsNotNone(cand)

        # Multiplier reasons are: thinking_bonus, specialty=coding, loaded
        # "loaded" may appear from both the availability and stickiness sections
        # so we use a set intersection to count distinct multiplier groups.
        _multiplier_reasons = {"thinking_bonus", "specialty=coding", "loaded",
                               "thinking_mismatch", "needs_swap"}
        applied = set(cand.reasons) & _multiplier_reasons
        self.assertLessEqual(len(applied), 3,
                             f"Expected ≤3 distinct multiplier reasons, got: {applied}")


if __name__ == "__main__":
    unittest.main()
