"""
Tests for S6: Runtime State Tracking.

Covers:
  - ModelRuntimeState dataclass: construction, field defaults
  - LocalModelManager.runtime_state: None initially, set after swap, cleared on stop
  - get_runtime_state() module helper
  - router.select_model() runtime context hard-filter: rejects loaded model when
    actual context < needed_ctx
  - router.select_model() thinking-mismatch stickiness: 1.10x when mismatch,
    1.40x when match
  - router.select_model() measured_tps: uses runtime measured_tps over registry tps
  - get_metrics() updates runtime_state.measured_tps
"""
from __future__ import annotations

import asyncio
import time
import sys
import os
import unittest
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── helpers ──────────────────────────────────────────────────────────────────

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_local_model(name="llama3", loaded=False, ctx=8192, tps=0.0,
                      thinking_model=False, demoted=False):
    """Return a minimal ModelInfo mock for a local model."""
    m = MagicMock()
    m.name = name
    m.litellm_name = f"openai/{name}"
    m.is_local = True
    m.is_loaded = loaded
    m.is_free = True
    m.demoted = demoted
    m.provider = "local"
    m.context_length = ctx
    m.max_tokens = 2048
    m.tokens_per_second = tps
    m.total_params_b = 8.0
    m.active_params_b = 8.0
    m.load_time_seconds = 15
    m.gpu_layers = 33
    m.file_size_mb = 5000
    m.vram_required_mb = 6000
    m.specialty = None
    m.thinking_model = thinking_model
    m.has_vision = False
    m.supports_function_calling = True
    m.supports_json_mode = True
    m.model_type = "dense"
    m.capabilities = {cap: 3.0 for cap in [
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
        "thinking_model": thinking_model,
        "is_local": True,
        "provider": "local",
    }
    m.estimated_cost.return_value = 0.0
    return m


# ── ModelRuntimeState unit tests ─────────────────────────────────────────────

class TestModelRuntimeState(unittest.TestCase):

    def test_fields_present(self):
        from src.models.local_model_manager import ModelRuntimeState
        field_names = {f.name for f in fields(ModelRuntimeState)}
        for expected in ("model_name", "thinking_enabled", "context_length",
                         "gpu_layers", "measured_tps", "loaded_at"):
            self.assertIn(expected, field_names)

    def test_defaults(self):
        from src.models.local_model_manager import ModelRuntimeState
        rs = ModelRuntimeState(
            model_name="llama3",
            thinking_enabled=False,
            context_length=8192,
            gpu_layers=33,
        )
        self.assertEqual(rs.measured_tps, 0.0)
        self.assertAlmostEqual(rs.loaded_at, time.time(), delta=2.0)

    def test_thinking_enabled_flag(self):
        from src.models.local_model_manager import ModelRuntimeState
        rs = ModelRuntimeState(
            model_name="qwq",
            thinking_enabled=True,
            context_length=32768,
            gpu_layers=33,
        )
        self.assertTrue(rs.thinking_enabled)

    def test_measured_tps_mutable(self):
        from src.models.local_model_manager import ModelRuntimeState
        rs = ModelRuntimeState("m", False, 4096, 20)
        rs.measured_tps = 42.5
        self.assertEqual(rs.measured_tps, 42.5)


# ── get_runtime_state helper ──────────────────────────────────────────────────

class TestGetRuntimeState(unittest.TestCase):

    def test_returns_none_when_no_manager(self):
        """get_runtime_state() returns None if the singleton has never been init'd."""
        import src.models.local_model_manager as lmm_mod
        original = lmm_mod._manager
        try:
            lmm_mod._manager = None
            from src.models.local_model_manager import get_runtime_state
            self.assertIsNone(get_runtime_state())
        finally:
            lmm_mod._manager = original

    def test_returns_runtime_state_from_manager(self):
        from src.models.local_model_manager import ModelRuntimeState, get_runtime_state
        import src.models.local_model_manager as lmm_mod

        fake_rs = ModelRuntimeState("llama3", False, 8192, 33)
        fake_mgr = MagicMock()
        fake_mgr.runtime_state = fake_rs

        original = lmm_mod._manager
        try:
            lmm_mod._manager = fake_mgr
            result = get_runtime_state()
            self.assertIs(result, fake_rs)
        finally:
            lmm_mod._manager = original


# ── get_metrics() updates measured_tps ────────────────────────────────────────

class TestGetMetricsUpdatesTps(unittest.TestCase):

    def test_measured_tps_updated_from_metrics(self):
        """get_metrics() should write measured_tps to runtime_state."""
        from src.models.local_model_manager import LocalModelManager, ModelRuntimeState

        mgr = LocalModelManager.__new__(LocalModelManager)
        mgr.current_model = "llama3"
        mgr.process = MagicMock()
        mgr.process.poll.return_value = None
        mgr.port = 8080
        mgr.api_base = "http://127.0.0.1:8080"
        mgr._last_request_time = time.time()
        mgr._started_at = time.time()
        mgr._total_swaps = 0
        mgr._active_inference_count = 0
        mgr._scheduler = MagicMock()
        mgr._scheduler.is_busy = False
        mgr.runtime_state = ModelRuntimeState("llama3", False, 8192, 33)

        # Prometheus-style response body
        metrics_body = (
            "# HELP llamacpp_tokens_predicted_seconds tok/s\n"
            "# TYPE llamacpp_tokens_predicted_seconds gauge\n"
            "llamacpp_tokens_predicted_seconds 37.4\n"
        )

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.text = metrics_body

        fake_client = AsyncMock()
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=False)
        fake_client.get = AsyncMock(return_value=fake_resp)

        with patch("src.models.local_model_manager.get_registry") as mock_reg, \
             patch("httpx.AsyncClient", return_value=fake_client):
            mock_info = MagicMock()
            mock_info.model_type = None
            mock_reg.return_value.get.return_value = mock_info

            run_async(mgr.get_metrics())

        self.assertAlmostEqual(mgr.runtime_state.measured_tps, 37.4, places=1)

    def test_measured_tps_not_updated_when_zero(self):
        """get_metrics() must not overwrite a positive measured_tps with zero."""
        from src.models.local_model_manager import LocalModelManager, ModelRuntimeState

        mgr = LocalModelManager.__new__(LocalModelManager)
        mgr.current_model = "llama3"
        mgr.process = MagicMock()
        mgr.process.poll.return_value = None
        mgr.port = 8080
        mgr.api_base = "http://127.0.0.1:8080"
        mgr._last_request_time = time.time()
        mgr._started_at = time.time()
        mgr._total_swaps = 0
        mgr._active_inference_count = 0
        mgr._scheduler = MagicMock()
        mgr._scheduler.is_busy = False
        mgr.runtime_state = ModelRuntimeState("llama3", False, 8192, 33)
        mgr.runtime_state.measured_tps = 25.0  # already populated

        # No tps in response
        metrics_body = "llamacpp_prompt_tokens_total 100\n"

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.text = metrics_body

        fake_client = AsyncMock()
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=False)
        fake_client.get = AsyncMock(return_value=fake_resp)

        with patch("src.models.local_model_manager.get_registry") as mock_reg, \
             patch("httpx.AsyncClient", return_value=fake_client):
            mock_info = MagicMock()
            mock_info.model_type = None
            mock_reg.return_value.get.return_value = mock_info

            run_async(mgr.get_metrics())

        # Should be unchanged since reported tps was 0
        self.assertEqual(mgr.runtime_state.measured_tps, 25.0)


# ── router.select_model() runtime-aware scoring ───────────────────────────────

def _make_registry_with_models(models):
    """Return a mock registry containing the given model mocks."""
    reg = MagicMock()
    reg.models = {m.name: m for m in models}
    return reg


def _run_select(reqs, models, runtime=None):
    """Run select_model() with mocked registry and runtime state.

    Note: is_local_inference_allowed / get_vram_budget_fraction are lazily
    imported inside select_model(), so we must patch them at their source
    module (src.infra.load_manager), not at src.core.router.
    Similarly get_runtime_state is lazily imported, so patch at its source.
    """
    from src.core.router import select_model

    reg = _make_registry_with_models(models)

    def _fake_score(model_capabilities, model_operational, requirements):
        return 3.5  # good enough score for all models

    with patch("src.core.router.get_registry", return_value=reg), \
         patch("src.core.router.score_model_for_task", side_effect=_fake_score), \
         patch("src.core.router.get_rate_limit_manager") as mock_rl, \
         patch("src.core.router.get_quota_planner") as mock_qp, \
         patch("src.infra.load_manager.is_local_inference_allowed", return_value=True), \
         patch("src.infra.load_manager.get_vram_budget_fraction", return_value=1.0), \
         patch("src.models.local_model_manager.get_runtime_state", return_value=runtime):
        mock_rl.return_value.has_capacity.return_value = True
        mock_rl.return_value.get_utilization.return_value = 20
        mock_rl.return_value.get_provider_utilization.return_value = 20
        mock_qp.return_value.expensive_threshold = 7
        return select_model(reqs)


class TestRuntimeContextFilter(unittest.TestCase):

    def test_loaded_model_filtered_when_runtime_ctx_too_small(self):
        """
        When the runtime context is smaller than needed_ctx, the loaded model
        should be excluded from candidates even if registry ctx is sufficient.
        """
        from src.core.router import ModelRequirements
        from src.models.local_model_manager import ModelRuntimeState

        # Registry says 16384 ctx — would pass static filter
        model = _make_local_model("llama3", loaded=True, ctx=16384)

        # But actually loaded with only 4096 (dynamic calc chose less)
        runtime = ModelRuntimeState("llama3", False, 4096, 33)

        reqs = ModelRequirements(
            task="coder",
            difficulty=5,
            estimated_input_tokens=3000,  # ~3900 effective ctx needed
            estimated_output_tokens=0,
            min_context_length=5000,      # explicitly needs 5000 > 4096
        )

        results = _run_select(reqs, [model], runtime=runtime)
        names = [c.model.name for c in results]
        self.assertNotIn("llama3", names,
                         "Loaded model with insufficient runtime ctx must be filtered out")

    def test_loaded_model_passes_when_runtime_ctx_sufficient(self):
        from src.core.router import ModelRequirements
        from src.models.local_model_manager import ModelRuntimeState

        model = _make_local_model("llama3", loaded=True, ctx=16384)
        runtime = ModelRuntimeState("llama3", False, 8192, 33)

        reqs = ModelRequirements(task="coder", difficulty=5,
                                 min_context_length=4096)

        results = _run_select(reqs, [model], runtime=runtime)
        names = [c.model.name for c in results]
        self.assertIn("llama3", names)

    def test_runtime_filter_only_applies_to_loaded_model(self):
        """
        An unloaded model should never be filtered by runtime ctx
        (it hasn't been loaded yet, so we don't know its actual ctx).
        """
        from src.core.router import ModelRequirements
        from src.models.local_model_manager import ModelRuntimeState

        # Model is unloaded — runtime is for a DIFFERENT loaded model
        unloaded = _make_local_model("llama3-unloaded", loaded=False, ctx=16384)
        runtime = ModelRuntimeState("other-model", False, 512, 10)

        reqs = ModelRequirements(task="coder", difficulty=5,
                                 min_context_length=1000)

        results = _run_select(reqs, [unloaded], runtime=runtime)
        names = [c.model.name for c in results]
        self.assertIn("llama3-unloaded", names)


class TestThinkingMismatchStickiness(unittest.TestCase):

    def _extract_score(self, results, name):
        for c in results:
            if c.model.name == name:
                return c.score
        return None

    def test_thinking_mismatch_reduces_stickiness(self):
        """
        When needs_thinking=True but loaded model has thinking_enabled=False,
        stickiness should be 1.10x not 1.40x — producing a lower final score.
        """
        from src.core.router import ModelRequirements
        from src.models.local_model_manager import ModelRuntimeState

        model = _make_local_model("qwq", loaded=True, ctx=32768,
                                   thinking_model=True)
        runtime_mismatch = ModelRuntimeState("qwq", thinking_enabled=False,
                                              context_length=32768, gpu_layers=33)
        runtime_match = ModelRuntimeState("qwq", thinking_enabled=True,
                                           context_length=32768, gpu_layers=33)

        reqs = ModelRequirements(task="planner", difficulty=7,
                                  needs_thinking=True)

        results_mismatch = _run_select(reqs, [model], runtime=runtime_mismatch)
        results_match = _run_select(reqs, [model], runtime=runtime_match)

        score_mismatch = self._extract_score(results_mismatch, "qwq")
        score_match = self._extract_score(results_match, "qwq")

        self.assertIsNotNone(score_mismatch)
        self.assertIsNotNone(score_match)
        self.assertLess(score_mismatch, score_match,
                        "Thinking mismatch should produce a lower stickiness score")

    def test_no_mismatch_uses_full_stickiness(self):
        """When thinking matches, reasons should contain 'loaded' not 'thinking_mismatch'."""
        from src.core.router import ModelRequirements
        from src.models.local_model_manager import ModelRuntimeState

        model = _make_local_model("qwq", loaded=True, ctx=32768,
                                   thinking_model=True)
        runtime = ModelRuntimeState("qwq", thinking_enabled=True,
                                     context_length=32768, gpu_layers=33)

        reqs = ModelRequirements(task="planner", difficulty=7, needs_thinking=True)
        results = _run_select(reqs, [model], runtime=runtime)

        self.assertTrue(len(results) > 0)
        reasons = results[0].reasons
        self.assertIn("loaded", reasons)
        self.assertNotIn("thinking_mismatch", reasons)

    def test_mismatch_reason_added(self):
        """thinking_mismatch reason should appear in candidate reasons."""
        from src.core.router import ModelRequirements
        from src.models.local_model_manager import ModelRuntimeState

        model = _make_local_model("qwq", loaded=True, ctx=32768,
                                   thinking_model=True)
        runtime = ModelRuntimeState("qwq", thinking_enabled=False,
                                     context_length=32768, gpu_layers=33)

        reqs = ModelRequirements(task="planner", difficulty=7, needs_thinking=True)
        results = _run_select(reqs, [model], runtime=runtime)

        self.assertTrue(len(results) > 0)
        reasons = results[0].reasons
        self.assertIn("thinking_mismatch", reasons)
        # "loaded" may appear from the availability section, but the stickiness
        # branch should NOT add a second "loaded" when thinking mismatches.
        # Count occurrences: availability adds one "loaded", stickiness should not.
        loaded_count = reasons.count("loaded")
        # With mismatch: stickiness uses 1.10x and appends "thinking_mismatch",
        # so only the availability "loaded" remains (count == 1).
        self.assertEqual(loaded_count, 1,
                         "Stickiness section must not append 'loaded' on thinking mismatch")

    def test_no_thinking_needed_no_mismatch(self):
        """If needs_thinking=False, thinking state doesn't cause mismatch."""
        from src.core.router import ModelRequirements
        from src.models.local_model_manager import ModelRuntimeState

        model = _make_local_model("qwq", loaded=True, ctx=32768,
                                   thinking_model=True)
        runtime = ModelRuntimeState("qwq", thinking_enabled=False,
                                     context_length=32768, gpu_layers=33)

        reqs = ModelRequirements(task="planner", difficulty=5, needs_thinking=False)
        results = _run_select(reqs, [model], runtime=runtime)

        self.assertTrue(len(results) > 0)
        reasons = results[0].reasons
        self.assertIn("loaded", reasons)
        self.assertNotIn("thinking_mismatch", reasons)


class TestMeasuredTpsInScoring(unittest.TestCase):

    def _get_speed_score_indirectly(self, tps_registry, tps_measured, loaded=True):
        """Run select_model and return the composite score (affected by speed)."""
        from src.core.router import ModelRequirements
        from src.models.local_model_manager import ModelRuntimeState

        model = _make_local_model("llama3", loaded=loaded, ctx=8192,
                                   tps=tps_registry)
        runtime = (
            ModelRuntimeState("llama3", False, 8192, 33,
                               measured_tps=tps_measured)
            if loaded else None
        )

        reqs = ModelRequirements(task="assistant", difficulty=3,
                                  prefer_speed=True,
                                  estimated_output_tokens=1000)
        results = _run_select(reqs, [model], runtime=runtime)
        return results[0].score if results else 0.0

    def test_measured_tps_used_for_loaded_model(self):
        """High measured_tps should yield higher score than low registry tps."""
        # Registry says 2 tok/s but we measured 45 tok/s — use measured
        score_high_measured = self._get_speed_score_indirectly(
            tps_registry=2.0, tps_measured=45.0
        )
        # Registry says 45 tok/s but measured only 2 tok/s
        score_low_measured = self._get_speed_score_indirectly(
            tps_registry=45.0, tps_measured=2.0
        )
        self.assertGreater(score_high_measured, score_low_measured,
                           "Measured tps should dominate registry tps for speed scoring")

    def test_registry_tps_used_when_measured_zero(self):
        """When measured_tps=0 (no generations yet), fall back to registry tps."""
        # High registry tps, no measurement yet
        score_high_reg = self._get_speed_score_indirectly(
            tps_registry=45.0, tps_measured=0.0
        )
        # Low registry tps, no measurement yet
        score_low_reg = self._get_speed_score_indirectly(
            tps_registry=2.0, tps_measured=0.0
        )
        self.assertGreater(score_high_reg, score_low_reg,
                           "Registry tps should be used when measured_tps is 0")


if __name__ == "__main__":
    unittest.main()
