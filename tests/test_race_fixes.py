"""
Tests for two race condition fixes:

1. Inference counter generation tracking (force-swap safety)
2. Atomic swap_started_at timestamp (replaces boolean swap_in_progress)
"""
from __future__ import annotations

import asyncio
import sys
import os
import time
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 1: Inference counter generation tracking
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferenceGeneration(unittest.TestCase):
    """Tests for the inference counter generation mechanism that prevents
    orphaned mark_inference_end() calls from corrupting the counter after
    a force-swap."""

    def _make_manager(self):
        """Create a minimal manager-like object for testing inference tracking."""
        # We can't easily construct LocalModelManager (heavy deps), so test
        # the logic directly by simulating the methods.
        class InferenceTracker:
            def __init__(self):
                self._active_inference_count = 0
                self._inference_idle = asyncio.Event()
                self._inference_idle.set()
                self._inference_generation = 0

            def mark_inference_start(self) -> int:
                self._active_inference_count += 1
                self._inference_idle.clear()
                return self._inference_generation

            def mark_inference_end(self, generation: int) -> None:
                if generation != self._inference_generation:
                    return  # orphaned — skip decrement
                self._active_inference_count = max(0, self._active_inference_count - 1)
                if self._active_inference_count == 0:
                    self._inference_idle.set()

            def force_swap_reset(self):
                """Simulate what _do_swap does on timeout."""
                self._inference_generation += 1
                self._active_inference_count = 0
                self._inference_idle.set()

        return InferenceTracker()

    def test_normal_start_end_cycle(self):
        """Normal inference: start → end works correctly."""
        t = self._make_manager()
        gen = t.mark_inference_start()
        self.assertEqual(t._active_inference_count, 1)
        self.assertFalse(t._inference_idle.is_set())

        t.mark_inference_end(gen)
        self.assertEqual(t._active_inference_count, 0)
        self.assertTrue(t._inference_idle.is_set())

    def test_multiple_concurrent_inferences(self):
        """Multiple inferences: counter tracks all of them."""
        t = self._make_manager()
        g1 = t.mark_inference_start()
        g2 = t.mark_inference_start()
        g3 = t.mark_inference_start()
        self.assertEqual(t._active_inference_count, 3)

        t.mark_inference_end(g1)
        self.assertEqual(t._active_inference_count, 2)
        t.mark_inference_end(g2)
        self.assertEqual(t._active_inference_count, 1)
        t.mark_inference_end(g3)
        self.assertEqual(t._active_inference_count, 0)
        self.assertTrue(t._inference_idle.is_set())

    def test_force_swap_then_orphan_end(self):
        """Critical test: force-swap bumps generation, orphaned end is ignored."""
        t = self._make_manager()
        gen_old = t.mark_inference_start()
        self.assertEqual(t._active_inference_count, 1)

        # Force-swap happens (30s drain timeout)
        t.force_swap_reset()
        self.assertEqual(t._active_inference_count, 0)
        self.assertEqual(t._inference_generation, 1)
        self.assertTrue(t._inference_idle.is_set())

        # Orphaned inference finally completes — this MUST be ignored
        t.mark_inference_end(gen_old)
        self.assertEqual(t._active_inference_count, 0,
                         "Orphaned end should NOT decrement counter below 0")
        self.assertTrue(t._inference_idle.is_set(),
                        "Idle flag should remain set after orphan end")

    def test_new_inference_after_force_swap_not_corrupted(self):
        """After force-swap, new inferences track correctly even if old ones finish."""
        t = self._make_manager()
        gen_old = t.mark_inference_start()  # gen 0
        t.force_swap_reset()  # gen → 1

        # New inference on new model
        gen_new = t.mark_inference_start()  # gen 1
        self.assertEqual(t._active_inference_count, 1)

        # Old inference finishes — should be ignored
        t.mark_inference_end(gen_old)
        self.assertEqual(t._active_inference_count, 1,
                         "Old gen end must not affect new gen count")

        # New inference finishes — should decrement normally
        t.mark_inference_end(gen_new)
        self.assertEqual(t._active_inference_count, 0)
        self.assertTrue(t._inference_idle.is_set())

    def test_double_force_swap(self):
        """Two consecutive force-swaps don't break the counter."""
        t = self._make_manager()
        g1 = t.mark_inference_start()
        t.force_swap_reset()  # gen → 1
        g2 = t.mark_inference_start()
        t.force_swap_reset()  # gen → 2

        # Both old generations' ends should be ignored
        t.mark_inference_end(g1)
        t.mark_inference_end(g2)
        self.assertEqual(t._active_inference_count, 0)

        # New gen works
        g3 = t.mark_inference_start()
        t.mark_inference_end(g3)
        self.assertEqual(t._active_inference_count, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 2: Atomic swap_started_at timestamp
# ═══════════════════════════════════════════════════════════════════════════════

class TestSwapStartedAtTimestamp(unittest.TestCase):
    """Tests for swap_started_at replacing the boolean swap_in_progress field."""

    def test_dispatcher_detects_swap_in_progress(self):
        """_is_swap_in_progress returns True when swap_started_at > 0."""
        from src.core.llm_dispatcher import LLMDispatcher
        d = LLMDispatcher()

        mock_mgr = MagicMock()
        mock_mgr.swap_started_at = 12345.678

        with patch("src.models.local_model_manager.get_local_manager",
                   return_value=mock_mgr):
            self.assertTrue(d._is_swap_in_progress())

    def test_dispatcher_detects_no_swap(self):
        """_is_swap_in_progress returns False when swap_started_at = 0."""
        from src.core.llm_dispatcher import LLMDispatcher
        d = LLMDispatcher()

        mock_mgr = MagicMock()
        mock_mgr.swap_started_at = 0.0

        with patch("src.models.local_model_manager.get_local_manager",
                   return_value=mock_mgr):
            self.assertFalse(d._is_swap_in_progress())

    def test_swap_version_returns_timestamp(self):
        """_swap_version returns the actual swap_started_at value."""
        from src.core.llm_dispatcher import LLMDispatcher
        d = LLMDispatcher()

        mock_mgr = MagicMock()
        mock_mgr.swap_started_at = 99.99

        with patch("src.models.local_model_manager.get_local_manager",
                   return_value=mock_mgr):
            self.assertEqual(d._swap_version(), 99.99)

    def test_swap_version_zero_when_no_swap(self):
        """_swap_version returns 0.0 when no swap in progress."""
        from src.core.llm_dispatcher import LLMDispatcher
        d = LLMDispatcher()

        mock_mgr = MagicMock()
        mock_mgr.swap_started_at = 0.0

        with patch("src.models.local_model_manager.get_local_manager",
                   return_value=mock_mgr):
            self.assertEqual(d._swap_version(), 0.0)

    def test_overhead_excludes_all_local_during_swap(self):
        """When swap in progress + no loaded model → exclude ALL local."""
        from src.core.llm_dispatcher import LLMDispatcher, CallCategory
        d = LLMDispatcher()

        mock_result = {"content": "ok", "model": "gemini/flash"}
        mock_call = AsyncMock(return_value=mock_result)

        exclude_all_called = []

        def _track_exclude_all(reqs):
            exclude_all_called.append(True)
            return reqs

        with patch.object(d, "_swap_version", return_value=100.0), \
             patch.object(d, "_get_loaded_model_name", return_value=None), \
             patch.object(d, "_should_wait_for_cold_start", return_value=False), \
             patch.object(d, "_exclude_all_local", side_effect=_track_exclude_all):
            from fatih_hoca.requirements import ModelRequirements
            reqs = ModelRequirements(task="classifier", difficulty=2)
            run_async(d._route_overhead(reqs, [{"role": "user", "content": "x"}], None))

        self.assertTrue(exclude_all_called,
                        "Should have called _exclude_all_local during active swap")

    def test_overhead_uses_exclude_unloaded_when_model_loaded(self):
        """When model loaded (even if swap_started_at > 0) → exclude unloaded only."""
        from src.core.llm_dispatcher import LLMDispatcher
        d = LLMDispatcher()

        mock_result = {"content": "ok", "model": "local/x"}
        mock_call = AsyncMock(return_value=mock_result)

        exclude_unloaded_called = []

        def _track_exclude_unloaded(reqs):
            exclude_unloaded_called.append(True)
            return reqs

        # swap in progress BUT model is loaded (swap between models, not cold start)
        with patch.object(d, "_swap_version", return_value=100.0), \
             patch.object(d, "_get_loaded_model_name", return_value="model-a"), \
             patch.object(d, "_should_wait_for_cold_start", return_value=False), \
             patch.object(d, "_exclude_unloaded_local",
                          side_effect=_track_exclude_unloaded):
            from fatih_hoca.requirements import ModelRequirements
            reqs = ModelRequirements(task="classifier", difficulty=2)
            run_async(d._route_overhead(reqs, [{"role": "user", "content": "x"}], None))

        self.assertTrue(exclude_unloaded_called,
                        "Should use _exclude_unloaded_local when model is loaded")


if __name__ == "__main__":
    unittest.main()
