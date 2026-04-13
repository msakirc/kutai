# tests/test_llm_dispatcher.py
"""
Comprehensive tests for src/core/llm_dispatcher.py

Covers:
  - SwapBudget: allow/block/reset/exempt logic, remaining/exhausted properties
  - LLMDispatcher: OVERHEAD and MAIN_WORK routing, swap notification, stats
  - CallCategory enum values
  - Integration-style: ensure_model never called for OVERHEAD
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
# LLMDispatcher Routing Tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_fake_scored(name="model-a", litellm_name="openai/model-a", is_local=False):
    """Create a fake ScoredModel for tests using MagicMock."""
    fake_model = MagicMock()
    fake_model.name = name
    fake_model.litellm_name = litellm_name
    fake_model.is_local = is_local
    fake_model.location = "cloud"
    fake_model.thinking_model = False
    fake_model.has_vision = False

    fake_scored = MagicMock()
    fake_scored.model = fake_model
    fake_scored.score = 99.0
    fake_scored.capability_score = 0.8
    fake_scored.reasons = []
    return fake_scored


def _make_fake_call_result(content="ok", model="openai/model-a", model_name="model-a"):
    """Create a fake talking_layer.CallResult."""
    import talking_layer
    return talking_layer.CallResult(
        content=content,
        tool_calls=None,
        thinking=None,
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        cost=0.0,
        latency=0.1,
        model=model,
        model_name=model_name,
        is_local=False,
        provider="openai",
        task="test",
    )


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

    # 14. OVERHEAD routes through _exclude_unloaded_local (no pinning)
    def test_dispatcher_overhead_uses_loaded_model(self):
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "classify this"}]

        fake_scored = _make_fake_scored()
        fake_result = _make_fake_call_result("ok")

        mock_tl_call = AsyncMock(return_value=fake_result)

        with patch("src.core.llm_dispatcher.LLMDispatcher._exclude_unloaded_local",
                   side_effect=lambda r: r), \
             patch("src.core.llm_dispatcher.LLMDispatcher._select_candidates",
                   return_value=[fake_scored]), \
             patch("src.core.llm_dispatcher.LLMDispatcher._prepare_messages",
                   side_effect=lambda msgs, m: msgs), \
             patch("talking_layer.call", mock_tl_call):

            from src.core.llm_dispatcher import CallCategory
            result = run_async(dispatcher.request(
                category=CallCategory.OVERHEAD,
                reqs=reqs,
                messages=messages,
            ))

        # talking_layer.call was invoked once
        self.assertTrue(mock_tl_call.called)
        self.assertEqual(mock_tl_call.call_count, 1)
        # Result has content from the fake result
        self.assertEqual(result["content"], "ok")

    # 15. no model loaded → exclude_unloaded excludes all local → cloud only
    def test_dispatcher_overhead_falls_to_cloud_when_no_loaded(self):
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "classify this"}]

        fake_scored = _make_fake_scored(name="haiku", litellm_name="claude-3-haiku")
        fake_result = _make_fake_call_result("ok", model="claude-3-haiku", model_name="haiku")

        mock_tl_call = AsyncMock(return_value=fake_result)

        with patch("src.core.llm_dispatcher.LLMDispatcher._exclude_unloaded_local",
                   side_effect=lambda r: r), \
             patch("src.core.llm_dispatcher.LLMDispatcher._select_candidates",
                   return_value=[fake_scored]), \
             patch("src.core.llm_dispatcher.LLMDispatcher._prepare_messages",
                   side_effect=lambda msgs, m: msgs), \
             patch("talking_layer.call", mock_tl_call):

            from src.core.llm_dispatcher import CallCategory
            result = run_async(dispatcher.request(
                category=CallCategory.OVERHEAD,
                reqs=reqs,
                messages=messages,
            ))

        # talking_layer.call invoked — cloud model was used
        self.assertTrue(mock_tl_call.called)
        self.assertEqual(mock_tl_call.call_count, 1)

    # ── MAIN_WORK routing ──

    # 16. MAIN_WORK calls talking_layer.call normally
    def test_dispatcher_main_work_passes_through(self):
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "do work"}]

        fake_scored = _make_fake_scored()
        fake_result = _make_fake_call_result("done")

        mock_tl_call = AsyncMock(return_value=fake_result)

        with patch("src.core.llm_dispatcher.LLMDispatcher._select_candidates",
                   return_value=[fake_scored]), \
             patch("src.core.llm_dispatcher.LLMDispatcher._prepare_messages",
                   side_effect=lambda msgs, m: msgs), \
             patch("talking_layer.call", mock_tl_call):

            from src.core.llm_dispatcher import CallCategory
            result = run_async(dispatcher.request(
                category=CallCategory.MAIN_WORK,
                reqs=reqs,
                messages=messages,
            ))

        mock_tl_call.assert_called_once()
        self.assertEqual(result["content"], "done")

    # 17. exhausted budget → tries loaded model first via model_override pinning
    def test_dispatcher_main_work_with_exhausted_budget(self):
        dispatcher = self._make_dispatcher()
        # Exhaust the budget
        dispatcher.swap_budget.record_swap()
        dispatcher.swap_budget.record_swap()
        dispatcher.swap_budget.record_swap()

        reqs = _make_reqs(local_only=False)
        messages = [{"role": "user", "content": "do work"}]

        fake_scored = _make_fake_scored()
        fake_result = _make_fake_call_result("done")

        # Track what model_override was used in the first _select_candidates call
        select_calls = []

        def fake_select_candidates(r, tools=None):
            select_calls.append(getattr(r, "model_override", None))
            return [fake_scored]

        mock_tl_call = AsyncMock(return_value=fake_result)

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_litellm_name",
                   return_value="openai/model-a"), \
             patch("src.core.llm_dispatcher.LLMDispatcher._select_candidates",
                   side_effect=fake_select_candidates), \
             patch("src.core.llm_dispatcher.LLMDispatcher._prepare_messages",
                   side_effect=lambda msgs, m: msgs), \
             patch("talking_layer.call", mock_tl_call):

            from src.core.llm_dispatcher import CallCategory
            result = run_async(dispatcher.request(
                category=CallCategory.MAIN_WORK,
                reqs=reqs,
                messages=messages,
            ))

        # First _select_candidates call should have model_override set to loaded model
        self.assertTrue(len(select_calls) >= 1)
        self.assertEqual(select_calls[0], "openai/model-a")

    # 22. get_stats returns correct counters
    def test_dispatcher_stats(self):
        dispatcher = self._make_dispatcher()

        fake_scored = _make_fake_scored()
        fake_result = _make_fake_call_result("ok")
        mock_tl_call = AsyncMock(return_value=fake_result)

        with patch("src.core.llm_dispatcher.LLMDispatcher._select_candidates",
                   return_value=[fake_scored]), \
             patch("src.core.llm_dispatcher.LLMDispatcher._prepare_messages",
                   side_effect=lambda msgs, m: msgs), \
             patch("src.core.llm_dispatcher.LLMDispatcher._exclude_unloaded_local",
                   side_effect=lambda r: r), \
             patch("talking_layer.call", mock_tl_call):

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

        mock_ensure = AsyncMock()
        fake_scored = _make_fake_scored()
        fake_result = _make_fake_call_result("ok")
        mock_tl_call = AsyncMock(return_value=fake_result)

        # _exclude_unloaded_local is the mechanism that prevents swaps
        with patch("src.core.llm_dispatcher.LLMDispatcher._exclude_unloaded_local",
                   side_effect=lambda r: r), \
             patch("src.core.llm_dispatcher.LLMDispatcher._select_candidates",
                   return_value=[fake_scored]), \
             patch("src.core.llm_dispatcher.LLMDispatcher._prepare_messages",
                   side_effect=lambda msgs, m: msgs), \
             patch("talking_layer.call", mock_tl_call):

            # Patch ensure_model via the module that would be imported
            with patch.dict("sys.modules", {
                "src.models.local_model_manager": MagicMock(
                    get_local_manager=MagicMock(return_value=MagicMock(
                        current_model="model-a",
                        swap_started_at=0.0,
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

    # 26. _exclude_unloaded_local keeps loaded model + cloud
    def test_exclude_unloaded_local_keeps_loaded_model(self):
        dispatcher = self._make_dispatcher()

        from src.core.router import ModelRequirements
        reqs = ModelRequirements(task="classifier", difficulty=2)

        # Set up: model-a is loaded, model-b and model-c are unloaded
        mock_model_a = MagicMock()
        mock_model_a.is_local = True
        mock_model_a.name = "model-a"
        mock_model_a.litellm_name = "openai/model-a"

        mock_model_b = MagicMock()
        mock_model_b.is_local = True
        mock_model_b.name = "model-b"
        mock_model_b.litellm_name = "openai/model-b"

        mock_cloud = MagicMock()
        mock_cloud.is_local = False
        mock_cloud.name = "gemini-flash"
        mock_cloud.litellm_name = "gemini/flash"

        mock_reg = MagicMock()
        mock_reg.all_models.return_value = [mock_model_a, mock_model_b, mock_cloud]

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_model_name",
                   return_value="model-a"), \
             patch("src.models.model_registry.get_registry",
                   return_value=mock_reg):
            result = dispatcher._exclude_unloaded_local(reqs)

        # Unloaded local model-b should be excluded
        self.assertIn("openai/model-b", result.exclude_models)
        # Loaded model-a should NOT be excluded
        self.assertNotIn("openai/model-a", result.exclude_models)
        # Cloud should NOT be excluded
        self.assertNotIn("gemini/flash", result.exclude_models)
        # local_only should be False
        self.assertFalse(result.local_only)

    # 27. _exclude_unloaded_local with nothing loaded excludes all local
    def test_exclude_unloaded_local_no_model_loaded(self):
        dispatcher = self._make_dispatcher()

        from src.core.router import ModelRequirements
        reqs = ModelRequirements(task="classifier", difficulty=2)

        mock_model_a = MagicMock()
        mock_model_a.is_local = True
        mock_model_a.name = "model-a"
        mock_model_a.litellm_name = "openai/model-a"

        mock_reg = MagicMock()
        mock_reg.all_models.return_value = [mock_model_a]

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_model_name",
                   return_value=None), \
             patch("src.models.model_registry.get_registry",
                   return_value=mock_reg):
            result = dispatcher._exclude_unloaded_local(reqs)

        # All local models excluded (none loaded)
        self.assertIn("openai/model-a", result.exclude_models)


# ═══════════════════════════════════════════════════════════════════════════════
# Cold-Start Wait Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestColdStartWait(unittest.TestCase):
    """Test that OVERHEAD calls wait for proactive model load on cold start."""

    def _make_dispatcher(self):
        import src.core.llm_dispatcher as mod
        mod._dispatcher = None
        from src.core.llm_dispatcher import LLMDispatcher
        return LLMDispatcher()

    # 30. OVERHEAD waits for model load when no model loaded and no cloud
    def test_overhead_waits_for_cold_start_load(self):
        """When no model is loaded and no cloud available, OVERHEAD should
        wait for the proactive load to complete rather than failing immediately."""
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "classify this"}]

        # Simulate: no model loaded initially, then loaded after wait
        load_count = {"calls": 0}

        def fake_get_loaded_name():
            load_count["calls"] += 1
            if load_count["calls"] >= 3:  # loaded after a few polls
                return "model-a"
            return None

        mock_model_a = MagicMock()
        mock_model_a.is_local = True
        mock_model_a.name = "model-a"
        mock_model_a.litellm_name = "openai/model-a"

        mock_reg = MagicMock()
        mock_reg.all_models.return_value = [mock_model_a]

        mock_mgr = MagicMock()
        mock_mgr.swap_started_at = 1.0  # proactive load in progress

        fake_scored = _make_fake_scored()
        fake_result = _make_fake_call_result("ok")
        mock_tl_call = AsyncMock(return_value=fake_result)

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_model_name",
                   side_effect=fake_get_loaded_name), \
             patch("src.models.model_registry.get_registry",
                   return_value=mock_reg), \
             patch("src.models.local_model_manager.get_local_manager",
                   return_value=mock_mgr), \
             patch("src.core.llm_dispatcher.LLMDispatcher._select_candidates",
                   return_value=[fake_scored]), \
             patch("src.core.llm_dispatcher.LLMDispatcher._prepare_messages",
                   side_effect=lambda msgs, m: msgs), \
             patch("talking_layer.call", mock_tl_call):

            from src.core.llm_dispatcher import CallCategory
            result = run_async(dispatcher.request(
                category=CallCategory.OVERHEAD,
                reqs=reqs,
                messages=messages,
            ))

        # Should have succeeded after waiting for model load
        self.assertEqual(result["content"], "ok")
        mock_tl_call.assert_called_once()

    # 31. OVERHEAD times out waiting if model never loads
    def test_overhead_cold_start_timeout(self):
        """If model never loads within timeout, OVERHEAD should fail gracefully."""
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "classify this"}]

        mock_model_a = MagicMock()
        mock_model_a.is_local = True
        mock_model_a.name = "model-a"
        mock_model_a.litellm_name = "openai/model-a"

        mock_reg = MagicMock()
        mock_reg.all_models.return_value = [mock_model_a]

        mock_mgr = MagicMock()
        mock_mgr.swap_started_at = 1.0
        mock_mgr.current_model = None

        import talking_layer as _tl
        mock_tl_call = AsyncMock(return_value=_tl.CallError(
            category="no_model", message="no models", retryable=False))

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_model_name",
                   return_value=None), \
             patch("src.models.model_registry.get_registry",
                   return_value=mock_reg), \
             patch("src.models.local_model_manager.get_local_manager",
                   return_value=mock_mgr), \
             patch("src.core.llm_dispatcher.LLMDispatcher._select_candidates",
                   return_value=[_make_fake_scored()]), \
             patch("src.core.llm_dispatcher.LLMDispatcher._prepare_messages",
                   side_effect=lambda msgs, m: msgs), \
             patch("talking_layer.call", mock_tl_call), \
             patch("src.core.llm_dispatcher._COLD_START_WAIT_TIMEOUT", 0.3), \
             patch("src.core.llm_dispatcher._COLD_START_POLL_INTERVAL", 0.1):

            from src.core.llm_dispatcher import CallCategory
            with self.assertRaises(RuntimeError) as ctx:
                run_async(dispatcher.request(
                    category=CallCategory.OVERHEAD,
                    reqs=reqs,
                    messages=messages,
                ))
            self.assertIn("OVERHEAD call failed", str(ctx.exception))

    # 32. OVERHEAD skips wait when cloud is available
    def test_overhead_no_wait_when_cloud_available(self):
        """When cloud models are available, OVERHEAD should not wait — just use cloud."""
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "classify this"}]

        mock_model_a = MagicMock()
        mock_model_a.is_local = True
        mock_model_a.name = "model-a"
        mock_model_a.litellm_name = "openai/model-a"

        mock_cloud = MagicMock()
        mock_cloud.is_local = False
        mock_cloud.name = "haiku"
        mock_cloud.litellm_name = "claude-3-haiku"

        mock_reg = MagicMock()
        mock_reg.all_models.return_value = [mock_model_a, mock_cloud]

        fake_scored = _make_fake_scored(name="haiku", litellm_name="claude-3-haiku")
        fake_result = _make_fake_call_result("ok", model="claude-3-haiku", model_name="haiku")
        mock_tl_call = AsyncMock(return_value=fake_result)

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_model_name",
                   return_value=None), \
             patch("src.models.model_registry.get_registry",
                   return_value=mock_reg), \
             patch("src.core.llm_dispatcher.LLMDispatcher._select_candidates",
                   return_value=[fake_scored]), \
             patch("src.core.llm_dispatcher.LLMDispatcher._prepare_messages",
                   side_effect=lambda msgs, m: msgs), \
             patch("talking_layer.call", mock_tl_call):

            from src.core.llm_dispatcher import CallCategory
            result = run_async(dispatcher.request(
                category=CallCategory.OVERHEAD,
                reqs=reqs,
                messages=messages,
            ))

        # Should succeed immediately without waiting
        self.assertEqual(result["content"], "ok")
        mock_tl_call.assert_called_once()

    # 33. OVERHEAD skips wait when no swap in progress (nothing loading)
    def test_overhead_no_wait_when_nothing_loading(self):
        """When no model is loaded and nothing is loading, don't wait forever."""
        dispatcher = self._make_dispatcher()
        reqs = _make_reqs()
        messages = [{"role": "user", "content": "classify this"}]

        mock_model_a = MagicMock()
        mock_model_a.is_local = True
        mock_model_a.name = "model-a"
        mock_model_a.litellm_name = "openai/model-a"

        mock_reg = MagicMock()
        mock_reg.all_models.return_value = [mock_model_a]

        mock_mgr = MagicMock()
        mock_mgr.swap_started_at = 0.0  # nothing loading
        mock_mgr.current_model = None

        import talking_layer as _tl
        mock_tl_call = AsyncMock(return_value=_tl.CallError(
            category="no_model", message="no models", retryable=False))

        with patch("src.core.llm_dispatcher.LLMDispatcher._get_loaded_model_name",
                   return_value=None), \
             patch("src.models.model_registry.get_registry",
                   return_value=mock_reg), \
             patch("src.models.local_model_manager.get_local_manager",
                   return_value=mock_mgr), \
             patch("src.core.llm_dispatcher.LLMDispatcher._select_candidates",
                   return_value=[_make_fake_scored()]), \
             patch("src.core.llm_dispatcher.LLMDispatcher._prepare_messages",
                   side_effect=lambda msgs, m: msgs), \
             patch("talking_layer.call", mock_tl_call):

            from src.core.llm_dispatcher import CallCategory
            with self.assertRaises(RuntimeError):
                run_async(dispatcher.request(
                    category=CallCategory.OVERHEAD,
                    reqs=reqs,
                    messages=messages,
                ))


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
