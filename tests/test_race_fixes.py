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


# Old TestSwapStartedAtTimestamp removed: tested LLMDispatcher helper
# methods (_swap_version, _is_swap_in_progress, _exclude_all_local,
# _exclude_unloaded_local, _get_loaded_model_name, _should_wait_for_cold_start,
# _route_overhead) that were deleted during the dispatcher simplification
# refactor (1044→413 lines). swap_in_progress / swap_started_at on the
# DaLLaMa swap manager is exercised in packages/dallama/tests/test_swap.py.


if __name__ == "__main__":
    unittest.main()
