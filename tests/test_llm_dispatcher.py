# tests/test_llm_dispatcher.py
"""
Comprehensive tests for src/core/llm_dispatcher.py

Covers:
  - SwapBudget: allow/block/reset/exempt logic, remaining/exhausted properties
  - GradeQueue: enqueue, drain (local/cloud), self-grade skip, needs_drain, pending models
  - LLMDispatcher: OVERHEAD and MAIN_WORK routing, grade deferral, swap notification, stats
  - CallCategory enum values
  - Integration-style: ensure_model never called for OVERHEAD, on_graded callback fires
"""
from __future__ import annotations

import asyncio
import sys
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_reqs(**kwargs):
    """Create a minimal ModelRequirements-like object for testing."""
    from dataclasses import dataclass, field

    @dataclass
    class FakeReqs:
        task: str = ""
        primary_capability: str = "general"
        difficulty: int = 5
        min_score: float = 0.0
        estimated_input_tokens: int = 2000
        estimated_output_tokens: int = 1000
        min_context_length: int = 0
        needs_function_calling: bool = False
        needs_json_mode: bool = False
        needs_thinking: bool = False
        needs_vision: bool = False
        local_only: bool = False
        prefer_speed: bool = False
        prefer_quality: bool = False
        prefer_local: bool = False
        max_cost: float = 0.0
        priority: int = 5
        exclude_models: list = field(default_factory=list)
        model_override: str | None = None
        agent_type: str = ""
        effective_task: str = ""

    r = FakeReqs()
    for k, v in kwargs.items():
        setattr(r, k, v)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# SwapBudget Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSwapBudget(unittest.TestCase):

    def _make(self, max_swaps=3, window_seconds=300.0):
        from src.core.llm_dispatcher import SwapBudget
        return SwapBudget(max_swaps=max_swaps, window_seconds=window_seconds)

    # 1. 2 swaps when max is 3 → allowed
    def test_swap_budget_allows_under_limit(self):
        budget = self._make(max_swaps=3)
        budget.record_swap()
        budget.record_swap()
        self.assertTrue(budget.can_swap())

    # 2. 3 swaps → 4th blocked
    def test_swap_budget_blocks_over_limit(self):
        budget = self._make(max_swaps=3)
        budget.record_swap()
        budget.record_swap()
        budget.record_swap()
        self.assertFalse(budget.can_swap())

    # 3. swaps expire after window_seconds
    def test_swap_budget_resets_after_window(self):
        budget = self._make(max_swaps=2, window_seconds=0.05)
        budget.record_swap()
        budget.record_swap()
        self.assertFalse(budget.can_swap())
        # Wait for the window to expire
        time.sleep(0.1)
        self.assertTrue(budget.can_swap())

    # 4. local_only=True always allowed even over budget
    def test_swap_budget_exempts_local_only(self):
        budget = self._make(max_swaps=1)
        budget.record_swap()
        # Exhausted but local_only bypasses
        self.assertTrue(budget.can_swap(local_only=True))

    # 5. priority>=9 always allowed even over budget
    def test_swap_budget_exempts_urgent(self):
        budget = self._make(max_swaps=1)
        budget.record_swap()
        # Exhausted but priority 9 bypasses
        self.assertTrue(budget.can_swap(priority=9))
        self.assertTrue(budget.can_swap(priority=10))

    # 6. remaining count
    def test_swap_budget_remaining_property(self):
        budget = self._make(max_swaps=3)
        self.assertEqual(budget.remaining, 3)
        budget.record_swap()
        self.assertEqual(budget.remaining, 2)
        budget.record_swap()
        self.assertEqual(budget.remaining, 1)
        budget.record_swap()
        self.assertEqual(budget.remaining, 0)

    # 7. exhausted flag
    def test_swap_budget_exhausted_property(self):
        budget = self._make(max_swaps=2)
        self.assertFalse(budget.exhausted)
        budget.record_swap()
        self.assertFalse(budget.exhausted)
        budget.record_swap()
        self.assertTrue(budget.exhausted)


# ═══════════════════════════════════════════════════════════════════════════════
# GradeQueue Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradeQueue(unittest.TestCase):

    def _make(self, max_pending=20):
        from src.core.llm_dispatcher import GradeQueue
        return GradeQueue(max_pending=max_pending)

    def _make_grade(self, task_id="t1", generating_model="model-a", **kwargs):
        from src.core.llm_dispatcher import PendingGrade
        return PendingGrade(
            task_id=task_id,
            task_title="Test Task",
            task_description="Do something",
            response_text="The answer is 42",
            generating_model=generating_model,
            task_name="test",
            priority=kwargs.get("priority", 5),
            on_graded=kwargs.get("on_graded", None),
        )

    # 8. enqueue increments depth
    def test_grade_queue_enqueue(self):
        q = self._make()
        self.assertEqual(q.depth, 0)
        run_async(q.enqueue(self._make_grade("t1")))
        self.assertEqual(q.depth, 1)
        run_async(q.enqueue(self._make_grade("t2")))
        self.assertEqual(q.depth, 2)

    # 9. drain with matching model (available != generating) → processes grade
    def test_grade_queue_drain_with_matching_model(self):
        q = self._make()
        grade = self._make_grade(task_id="t1", generating_model="model-a")
        run_async(q.enqueue(grade))

        with patch("src.core.router.grade_response", new=AsyncMock(return_value=0.8)):
            count = run_async(q.drain(available_model="model-b"))

        self.assertEqual(count, 1)
        self.assertEqual(q.depth, 0)

    # 10. drain skips self-grade (available == generating stays queued)
    def test_grade_queue_drain_skips_self_grade(self):
        q = self._make()
        grade = self._make_grade(task_id="t1", generating_model="model-a")
        run_async(q.enqueue(grade))

        with patch("src.core.router.grade_response", new=AsyncMock(return_value=0.8)):
            count = run_async(q.drain(available_model="model-a"))

        self.assertEqual(count, 0)
        self.assertEqual(q.depth, 1)  # still queued

    # 11. use_cloud=True drains everything regardless of model
    def test_grade_queue_drain_with_cloud(self):
        q = self._make()
        run_async(q.enqueue(self._make_grade("t1", generating_model="model-a")))
        run_async(q.enqueue(self._make_grade("t2", generating_model="model-b")))

        with patch("src.core.router.grade_response", new=AsyncMock(return_value=0.7)):
            count = run_async(q.drain(available_model="model-a", use_cloud=True))

        self.assertEqual(count, 2)
        self.assertEqual(q.depth, 0)

    # 12. needs_drain triggers at max_pending
    def test_grade_queue_needs_drain_threshold(self):
        q = self._make(max_pending=3)
        self.assertFalse(q.needs_drain)
        run_async(q.enqueue(self._make_grade("t1")))
        run_async(q.enqueue(self._make_grade("t2")))
        self.assertFalse(q.needs_drain)
        run_async(q.enqueue(self._make_grade("t3")))
        self.assertTrue(q.needs_drain)

    # 13. get_pending_models returns set of generating models
    def test_grade_queue_get_pending_models(self):
        q = self._make()
        run_async(q.enqueue(self._make_grade("t1", generating_model="model-x")))
        run_async(q.enqueue(self._make_grade("t2", generating_model="model-y")))
        run_async(q.enqueue(self._make_grade("t3", generating_model="model-x")))

        models = run_async(q.get_pending_models())
        self.assertEqual(models, {"model-x", "model-y"})


# ═══════════════════════════════════════════════════════════════════════════════
# LLMDispatcher Routing Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMDispatcherRouting(unittest.TestCase):

    def _make_dispatcher(self):
        # Reset the singleton so each test gets a fresh instance
        import src.core.llm_dispatcher as mod
        mod._dispatcher = None
        from src.core.llm_dispatcher import LLMDispatcher
        return LLMDispatcher()

    def _fake_model_info(self, name="model-a", litellm_name="openai/model-a"):
        info = MagicMock()
        info.litellm_name = litellm_name
        info.is_local = True
        return info

    def _fake_manager(self, current_model="model-a"):
        mgr = MagicMock()
        mgr.current_model = current_model
        return mgr

    def _fake_registry(self, litellm_name="openai/model-a"):
        reg = MagicMock()
        info = self._fake_model_info(litellm_name=litellm_name)
        reg.get.return_value = info
        reg.all_models.return_value = [info]
        return reg

    # ── OVERHEAD routing ──

    # 14. OVERHEAD pinned to loaded model
    def test_dispatcher_overhead_uses_loaded_model(self):
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "classify this"}]
        mock_result = {"content": "ok", "model": "openai/model-a"}

        mock_call = AsyncMock(return_value=mock_result)

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_litellm_name",
                   return_value="openai/model-a"), \
             patch("src.core.router.call_model", mock_call), \
             patch("src.core.llm_dispatcher.LLMDispatcher._force_cloud_only",
                   side_effect=lambda r: r):

            from src.core.llm_dispatcher import CallCategory
            result = run_async(dispatcher.request(
                category=CallCategory.OVERHEAD,
                reqs=reqs,
                messages=messages,
            ))

        # Should have been called with the pinned (loaded) model
        self.assertTrue(mock_call.called)
        called_reqs = mock_call.call_args[0][0]
        self.assertEqual(called_reqs.model_override, "openai/model-a")

    # 15. no model loaded → cloud
    def test_dispatcher_overhead_falls_to_cloud_when_no_loaded(self):
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "classify this"}]
        mock_result = {"content": "ok", "model": "claude-3-haiku"}

        mock_call = AsyncMock(return_value=mock_result)

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_litellm_name",
                   return_value=None), \
             patch("src.core.router.call_model", mock_call), \
             patch("src.core.llm_dispatcher.LLMDispatcher._force_cloud_only",
                   side_effect=lambda r: r):

            from src.core.llm_dispatcher import CallCategory
            result = run_async(dispatcher.request(
                category=CallCategory.OVERHEAD,
                reqs=reqs,
                messages=messages,
            ))

        # Should fall through to cloud (call_model called once, no loaded model)
        self.assertTrue(mock_call.called)

    # ── MAIN_WORK routing ──

    # 16. MAIN_WORK calls call_model normally
    def test_dispatcher_main_work_passes_through(self):
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "do work"}]
        mock_result = {"content": "done", "model": "openai/model-a"}

        mock_call = AsyncMock(return_value=mock_result)

        with patch("src.core.router.call_model", mock_call):
            from src.core.llm_dispatcher import CallCategory
            result = run_async(dispatcher.request(
                category=CallCategory.MAIN_WORK,
                reqs=reqs,
                messages=messages,
            ))

        mock_call.assert_called_once()
        self.assertEqual(result, mock_result)

    # 17. exhausted budget → tries loaded model first
    def test_dispatcher_main_work_with_exhausted_budget(self):
        dispatcher = self._make_dispatcher()
        # Exhaust the budget
        dispatcher.swap_budget.record_swap()
        dispatcher.swap_budget.record_swap()
        dispatcher.swap_budget.record_swap()

        reqs = _make_reqs(local_only=False)
        messages = [{"role": "user", "content": "do work"}]
        mock_result = {"content": "done", "model": "openai/model-a"}

        mock_call = AsyncMock(return_value=mock_result)

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_litellm_name",
                   return_value="openai/model-a"), \
             patch("src.core.router.call_model", mock_call):

            from src.core.llm_dispatcher import CallCategory
            result = run_async(dispatcher.request(
                category=CallCategory.MAIN_WORK,
                reqs=reqs,
                messages=messages,
            ))

        # With exhausted budget, the loaded model is tried first via model_override
        self.assertTrue(mock_call.called)
        first_call_reqs = mock_call.call_args_list[0][0][0]
        self.assertEqual(first_call_reqs.model_override, "openai/model-a")

    # ── Grading ──

    # 18. priority>=8 grades immediately
    def test_dispatcher_request_grade_immediate_for_urgent(self):
        dispatcher = self._make_dispatcher()
        mock_grade = AsyncMock(return_value=0.9)

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_litellm_name",
                   return_value="openai/model-a"), \
             patch("src.core.router.grade_response", mock_grade):

            score = run_async(dispatcher.request_grade(
                task_id="t1",
                task_title="Test",
                task_description="Do X",
                response_text="Done",
                generating_model="openai/model-b",
                priority=8,
            ))

        mock_grade.assert_called_once()
        self.assertEqual(score, 0.9)
        self.assertEqual(dispatcher.grade_queue.depth, 0)  # not deferred

    # 19. loaded != generator → immediate grade
    def test_dispatcher_request_grade_immediate_when_different_model(self):
        dispatcher = self._make_dispatcher()
        mock_grade = AsyncMock(return_value=0.75)

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_litellm_name",
                   return_value="openai/model-a"), \
             patch("src.core.router.grade_response", mock_grade):

            score = run_async(dispatcher.request_grade(
                task_id="t1",
                task_title="Test",
                task_description="Do X",
                response_text="Done",
                generating_model="openai/model-b",  # different from loaded
                priority=5,
            ))

        mock_grade.assert_called_once()
        self.assertEqual(score, 0.75)
        self.assertEqual(dispatcher.grade_queue.depth, 0)  # not deferred

    # 20. loaded == generator → deferred
    def test_dispatcher_request_grade_defers_when_same_model(self):
        dispatcher = self._make_dispatcher()
        mock_grade = AsyncMock(return_value=0.8)

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_litellm_name",
                   return_value="openai/model-a"), \
             patch("src.core.router.grade_response", mock_grade):

            score = run_async(dispatcher.request_grade(
                task_id="t1",
                task_title="Test",
                task_description="Do X",
                response_text="Done",
                generating_model="openai/model-a",  # same as loaded
                priority=5,
            ))

        # grade_response should NOT have been called (deferred)
        mock_grade.assert_not_called()
        self.assertIsNone(score)
        self.assertEqual(dispatcher.grade_queue.depth, 1)

    # 21. swap notification triggers drain
    def test_dispatcher_on_model_swap_drains_grades(self):
        dispatcher = self._make_dispatcher()

        # Enqueue a grade that was generated by "old-model"
        from src.core.llm_dispatcher import PendingGrade
        grade = PendingGrade(
            task_id="t1",
            task_title="Test",
            task_description="Do X",
            response_text="Done",
            generating_model="old-model",
            task_name="test",
            priority=5,
        )
        run_async(dispatcher.grade_queue.enqueue(grade))
        self.assertEqual(dispatcher.grade_queue.depth, 1)

        mock_grade = AsyncMock(return_value=0.8)

        with patch("src.core.router.grade_response", mock_grade):
            run_async(dispatcher.on_model_swap(
                old_model="old-model",
                new_model="new-model",
            ))

        # The swap_budget should have recorded the swap
        self.assertEqual(len(dispatcher.swap_budget._timestamps), 1)
        # Grade should have been drained (during "before swap" drain with old_model available)
        # When old_model="old-model" and generating_model="old-model", they match so NOT drained.
        # After swap new_model="new-model" != "old-model" → drained.
        self.assertEqual(dispatcher.grade_queue.depth, 0)

    # 22. get_stats returns correct counters
    def test_dispatcher_stats(self):
        dispatcher = self._make_dispatcher()

        mock_call = AsyncMock(return_value={"content": "ok", "model": "m"})

        with patch("src.core.router.call_model", mock_call), \
             patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_litellm_name",
                   return_value=None), \
             patch("src.core.llm_dispatcher.LLMDispatcher._force_cloud_only",
                   side_effect=lambda r: r):

            from src.core.llm_dispatcher import CallCategory

            # 2 MAIN_WORK
            run_async(dispatcher.request(CallCategory.MAIN_WORK, _make_reqs(), []))
            run_async(dispatcher.request(CallCategory.MAIN_WORK, _make_reqs(), []))
            # 1 OVERHEAD
            run_async(dispatcher.request(CallCategory.OVERHEAD, _make_reqs(), []))

        stats = dispatcher.get_stats()
        self.assertEqual(stats["total_calls"], 3)
        self.assertEqual(stats["overhead_calls"], 1)
        self.assertIn("overhead_pct", stats)
        self.assertIn("swap_budget_remaining", stats)
        self.assertIn("grade_queue_depth", stats)
        # overhead_pct should be ~33.3%
        self.assertIn("33.3%", stats["overhead_pct"])


# ═══════════════════════════════════════════════════════════════════════════════
# CallCategory Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCallCategory(unittest.TestCase):

    # 23. enum values exist
    def test_call_category_enum_values(self):
        from src.core.llm_dispatcher import CallCategory
        self.assertEqual(CallCategory.MAIN_WORK.value, "main_work")
        self.assertEqual(CallCategory.OVERHEAD.value, "overhead")
        # Ensure both members exist
        members = {m.name for m in CallCategory}
        self.assertIn("MAIN_WORK", members)
        self.assertIn("OVERHEAD", members)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration-style Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMDispatcherIntegration(unittest.TestCase):

    def _make_dispatcher(self):
        import src.core.llm_dispatcher as mod
        mod._dispatcher = None
        from src.core.llm_dispatcher import LLMDispatcher
        return LLMDispatcher()

    # 24. ensure_model never called for OVERHEAD
    def test_overhead_never_triggers_ensure_model(self):
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "classify"}]

        mock_call = AsyncMock(return_value={"content": "ok", "model": "openai/m"})
        mock_ensure = AsyncMock()

        # Patch ensure_model at the local_model_manager level
        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_litellm_name",
                   return_value="openai/model-a"), \
             patch("src.core.router.call_model", mock_call), \
             patch("src.core.llm_dispatcher.LLMDispatcher._force_cloud_only",
                   side_effect=lambda r: r):

            # Patch ensure_model via the module that would be imported
            with patch.dict("sys.modules", {
                "src.models.local_model_manager": MagicMock(
                    get_local_manager=MagicMock(return_value=MagicMock(
                        current_model="model-a",
                        ensure_model=mock_ensure,
                    ))
                )
            }):
                from src.core.llm_dispatcher import CallCategory
                run_async(dispatcher.request(
                    category=CallCategory.OVERHEAD,
                    reqs=reqs,
                    messages=messages,
                ))

        # ensure_model (which causes swaps) must NOT have been called for OVERHEAD
        mock_ensure.assert_not_called()

    # 25. on_graded callback fires when grade completes
    def test_grade_callback_invoked_on_drain(self):
        dispatcher = self._make_dispatcher()

        callback_results = []

        async def on_graded(score: float):
            callback_results.append(score)

        from src.core.llm_dispatcher import PendingGrade
        grade = PendingGrade(
            task_id="t1",
            task_title="Test",
            task_description="Do X",
            response_text="Done",
            generating_model="model-a",
            task_name="test",
            priority=5,
            on_graded=on_graded,
        )
        run_async(dispatcher.grade_queue.enqueue(grade))

        mock_grade_fn = AsyncMock(return_value=0.88)

        with patch("src.core.router.grade_response", mock_grade_fn):
            # Drain with a different model so it can grade model-a's output
            count = run_async(dispatcher.grade_queue.drain(available_model="model-b"))

        self.assertEqual(count, 1)
        self.assertEqual(callback_results, [0.88])


# ─── Singleton Tests ──────────────────────────────────────────────────────────

class TestGetDispatcherSingleton(unittest.TestCase):

    def test_get_dispatcher_returns_same_instance(self):
        import src.core.llm_dispatcher as mod
        mod._dispatcher = None  # reset

        from src.core.llm_dispatcher import get_dispatcher
        d1 = get_dispatcher()
        d2 = get_dispatcher()
        self.assertIs(d1, d2)

    def test_get_dispatcher_returns_llm_dispatcher(self):
        import src.core.llm_dispatcher as mod
        mod._dispatcher = None

        from src.core.llm_dispatcher import get_dispatcher, LLMDispatcher
        d = get_dispatcher()
        self.assertIsInstance(d, LLMDispatcher)


if __name__ == "__main__":
    unittest.main()
