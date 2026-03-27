"""
Tests for S9: Adaptive timeouts.

Covers LLMDispatcher._compute_timeout():
  - OVERHEAD always returns 20s hard cap
  - MAIN_WORK with measured_tps: uses (output_tokens / tps) * 2.0, clamped [20, 300]
  - MAIN_WORK without measured_tps: uses difficulty heuristics
  - Clamping: very slow model capped at 300s, very fast capped at 20s floor
  - call_model() receives timeout_override from dispatcher routing methods
"""
from __future__ import annotations

import sys
import os
import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_reqs(difficulty=5, output_tokens=1000):
    from src.core.router import ModelRequirements
    return ModelRequirements(
        task="assistant",
        difficulty=difficulty,
        estimated_output_tokens=output_tokens,
    )


class TestComputeTimeout(unittest.TestCase):

    def _dispatcher(self):
        from src.core.llm_dispatcher import LLMDispatcher
        return LLMDispatcher()

    # ── OVERHEAD hard cap ─────────────────────────────────────────────────────

    def test_overhead_always_20s(self):
        from src.core.llm_dispatcher import CallCategory
        d = self._dispatcher()
        for difficulty in (1, 3, 5, 7, 10):
            with self.subTest(difficulty=difficulty):
                reqs = _make_reqs(difficulty=difficulty)
                t = d._compute_timeout(CallCategory.OVERHEAD, reqs)
                self.assertEqual(t, 20.0,
                                 f"OVERHEAD must always be 20s, got {t}s "
                                 f"(difficulty={difficulty})")

    def test_overhead_ignores_output_tokens(self):
        from src.core.llm_dispatcher import CallCategory
        d = self._dispatcher()
        for tokens in (100, 2048, 8192):
            with self.subTest(tokens=tokens):
                reqs = _make_reqs(output_tokens=tokens)
                t = d._compute_timeout(CallCategory.OVERHEAD, reqs)
                self.assertEqual(t, 20.0)

    # ── MAIN_WORK with measured TPS ───────────────────────────────────────────

    def test_main_work_uses_measured_tps(self):
        from src.core.llm_dispatcher import CallCategory
        from src.models.local_model_manager import ModelRuntimeState

        d = self._dispatcher()
        reqs = _make_reqs(difficulty=5, output_tokens=1000)

        # 1000 tokens / 10 tps = 100s generation → * 2.0 = 200s
        runtime = ModelRuntimeState("llama3", False, 8192, 33, measured_tps=10.0)

        with patch("src.models.local_model_manager.get_runtime_state",
                   return_value=runtime):
            t = d._compute_timeout(CallCategory.MAIN_WORK, reqs)

        self.assertAlmostEqual(t, 200.0, places=1)

    def test_main_work_tps_clamped_to_max(self):
        from src.core.llm_dispatcher import CallCategory
        from src.models.local_model_manager import ModelRuntimeState

        d = self._dispatcher()
        # 2000 tokens at 1 tps = 2000s → clamped to 300s
        reqs = _make_reqs(output_tokens=2000)
        runtime = ModelRuntimeState("llama3", False, 8192, 33, measured_tps=1.0)

        with patch("src.models.local_model_manager.get_runtime_state",
                   return_value=runtime):
            t = d._compute_timeout(CallCategory.MAIN_WORK, reqs)

        self.assertEqual(t, 300.0, "Very slow model should be capped at 300s")

    def test_main_work_tps_floored_at_min(self):
        from src.core.llm_dispatcher import CallCategory
        from src.models.local_model_manager import ModelRuntimeState

        d = self._dispatcher()
        # 10 tokens at 1000 tps = 0.02s → clamped to 20s floor
        reqs = _make_reqs(output_tokens=10)
        runtime = ModelRuntimeState("llama3", False, 8192, 33, measured_tps=1000.0)

        with patch("src.models.local_model_manager.get_runtime_state",
                   return_value=runtime):
            t = d._compute_timeout(CallCategory.MAIN_WORK, reqs)

        self.assertEqual(t, 20.0, "Very fast model should still have 20s floor")

    def test_main_work_falls_back_when_tps_zero(self):
        from src.core.llm_dispatcher import CallCategory
        from src.models.local_model_manager import ModelRuntimeState

        d = self._dispatcher()
        # measured_tps=0 → should fall back to difficulty heuristic, not 0-division
        reqs = _make_reqs(difficulty=6, output_tokens=1000)
        runtime = ModelRuntimeState("llama3", False, 8192, 33, measured_tps=0.0)

        with patch("src.models.local_model_manager.get_runtime_state",
                   return_value=runtime):
            t = d._compute_timeout(CallCategory.MAIN_WORK, reqs)

        # difficulty=6 → heuristic fallback 120s
        self.assertEqual(t, 120.0)

    def test_main_work_falls_back_when_no_runtime(self):
        from src.core.llm_dispatcher import CallCategory

        d = self._dispatcher()
        reqs = _make_reqs(difficulty=8, output_tokens=1000)

        with patch("src.models.local_model_manager.get_runtime_state",
                   return_value=None):
            t = d._compute_timeout(CallCategory.MAIN_WORK, reqs)

        # difficulty=8 → heuristic 200s
        self.assertEqual(t, 200.0)

    # ── Difficulty heuristics ─────────────────────────────────────────────────

    def test_difficulty_heuristics_order(self):
        """Higher difficulty should yield longer timeout (without runtime TPS)."""
        from src.core.llm_dispatcher import CallCategory

        d = self._dispatcher()
        timeouts = []
        for diff in (1, 3, 5, 7, 9):
            reqs = _make_reqs(difficulty=diff)
            with patch("src.models.local_model_manager.get_runtime_state",
                       return_value=None):
                timeouts.append(d._compute_timeout(CallCategory.MAIN_WORK, reqs))

        for i in range(len(timeouts) - 1):
            self.assertLessEqual(
                timeouts[i], timeouts[i + 1],
                f"Expected timeout[{i}]≤timeout[{i+1}] "
                f"but got {timeouts[i]} > {timeouts[i+1]}"
            )

    def test_max_difficulty_hits_ceiling(self):
        from src.core.llm_dispatcher import CallCategory

        d = self._dispatcher()
        reqs = _make_reqs(difficulty=10)
        with patch("src.models.local_model_manager.get_runtime_state",
                   return_value=None):
            t = d._compute_timeout(CallCategory.MAIN_WORK, reqs)
        self.assertEqual(t, 300.0)


# ── Integration: dispatcher passes timeout to call_model ─────────────────────

class TestDispatcherPassesTimeout(unittest.TestCase):

    def test_main_work_passes_timeout_override(self):
        """_route_main_work must pass computed timeout to call_model."""
        from src.core.llm_dispatcher import LLMDispatcher, CallCategory
        from src.models.local_model_manager import ModelRuntimeState

        d = LLMDispatcher()

        # Arrange: runtime says 20 tps, 500 output tokens → 50s → *2 = 100s
        runtime = ModelRuntimeState("llama3", False, 8192, 33, measured_tps=20.0)

        captured_timeout = {}

        async def _fake_call_model(reqs, messages, tools=None, timeout_override=None):
            captured_timeout["val"] = timeout_override
            return {"content": "ok", "model": "llama3"}

        async def _run():
            with patch("src.models.local_model_manager.get_runtime_state",
                       return_value=runtime), \
                 patch("src.core.router.call_model",
                       side_effect=_fake_call_model):
                from src.core.router import ModelRequirements
                reqs = ModelRequirements(
                    task="coder", difficulty=6, estimated_output_tokens=500
                )
                await d._route_main_work(reqs, [{"role": "user", "content": "hi"}], None)

        run_async(_run())

        # 500/20 * 2 = 50s
        self.assertIsNotNone(captured_timeout.get("val"))
        self.assertAlmostEqual(captured_timeout["val"], 50.0, places=1)

    def test_overhead_passes_20s_cap(self):
        """_route_overhead must pass 20.0 as timeout to call_model."""
        from src.core.llm_dispatcher import LLMDispatcher

        d = LLMDispatcher()

        captured_timeout = {}

        async def _fake_call_model(reqs, messages, tools=None, timeout_override=None):
            captured_timeout["val"] = timeout_override
            return {"content": "ok", "model": "llama3"}

        async def _run():
            with patch("src.models.local_model_manager.get_runtime_state",
                       return_value=None), \
                 patch("src.core.router.call_model",
                       side_effect=_fake_call_model), \
                 patch.object(d, "_get_loaded_litellm_name", return_value=None), \
                 patch.object(d, "_force_cloud_only",
                              side_effect=lambda r: r):
                from src.core.router import ModelRequirements
                reqs = ModelRequirements(task="classifier", difficulty=3)
                await d._route_overhead(reqs, [{"role": "user", "content": "hi"}], None)

        run_async(_run())

        self.assertEqual(captured_timeout.get("val"), 20.0,
                         "OVERHEAD route must pass 20.0s timeout")


if __name__ == "__main__":
    unittest.main()
