# tests/test_llm_dispatcher.py
"""
Tests for src/core/llm_dispatcher.py (new simplified API).

Tests the thin ask-load-call-retry loop:
  - CallCategory enum
  - request() happy path / no-model / retry-on-error / max-retries
  - needs_thinking=False for OVERHEAD
  - needs_function_calling propagation
  - on_model_swap()
  - get_stats() counters
  - get_dispatcher() singleton
"""
from __future__ import annotations

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Fake lightweight types ───────────────────────────────────────────────────

class FakeCallResult:
    """Lightweight stand-in that satisfies isinstance(result, hallederiz_kadir.CallResult)."""
    def __init__(self, **kwargs):
        self.content = kwargs.get("content", "test response")
        self.model = kwargs.get("model", "test/model")
        self.model_name = kwargs.get("model_name", "test-model")
        self.cost = kwargs.get("cost", 0.001)
        self.usage = kwargs.get("usage", {"prompt_tokens": 100, "completion_tokens": 50})
        self.tool_calls = kwargs.get("tool_calls", [])
        self.latency = kwargs.get("latency", 1.5)
        self.thinking = kwargs.get("thinking", None)
        self.is_local = kwargs.get("is_local", False)
        self.provider = kwargs.get("provider", "test")
        self.task = kwargs.get("task", "test")


class FakeCallError:
    retryable = True
    message = "timeout"
    category = "timeout"

    def __init__(self, retryable=True, message="timeout", category="timeout"):
        self.retryable = retryable
        self.message = message
        self.category = category


# ─── Factories ────────────────────────────────────────────────────────────────

def _make_mock_model(
    name="test-model",
    is_local=False,
    litellm_name="test/model",
    thinking_model=False,
    has_vision=False,
    is_loaded=True,
):
    m = MagicMock()
    m.name = name
    m.is_local = is_local
    m.litellm_name = litellm_name
    m.thinking_model = thinking_model
    m.has_vision = has_vision
    m.is_loaded = is_loaded
    m.location = ""
    return m


def _make_mock_pick(model=None, min_time=30.0):
    p = MagicMock()
    p.model = model or _make_mock_model()
    p.min_time_seconds = min_time
    return p


def _fresh_dispatcher():
    """Return a new LLMDispatcher with reset singleton."""
    import src.core.llm_dispatcher as mod
    mod._dispatcher = None
    from src.core.llm_dispatcher import LLMDispatcher
    return LLMDispatcher()


# ─── Patch helpers ────────────────────────────────────────────────────────────

def _patch_select(return_value):
    return patch("fatih_hoca.select", return_value=return_value)


def _patch_call(return_value=None, side_effect=None):
    if side_effect is not None:
        return patch("hallederiz_kadir.call", new=AsyncMock(side_effect=side_effect))
    return patch("hallederiz_kadir.call", new=AsyncMock(return_value=return_value))


def _patch_call_result_class(cls):
    """Patch hallederiz_kadir.CallResult so isinstance() checks in request() work."""
    return patch("hallederiz_kadir.CallResult", cls)


def _patch_failure_class(cls=None):
    """Patch fatih_hoca.types.Failure used inside request()."""
    if cls is None:
        from fatih_hoca.types import Failure
        cls = Failure
    return patch("fatih_hoca.types.Failure", cls)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CallCategory enum
# ═══════════════════════════════════════════════════════════════════════════════

class TestCallCategory:

    def test_main_work_value(self):
        from src.core.llm_dispatcher import CallCategory
        assert CallCategory.MAIN_WORK.value == "main_work"

    def test_overhead_value(self):
        from src.core.llm_dispatcher import CallCategory
        assert CallCategory.OVERHEAD.value == "overhead"

    def test_both_members_exist(self):
        from src.core.llm_dispatcher import CallCategory
        names = {m.name for m in CallCategory}
        assert "MAIN_WORK" in names
        assert "OVERHEAD" in names


# ═══════════════════════════════════════════════════════════════════════════════
# 2. request() — happy path
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequestHappyPath:

    @pytest.mark.asyncio
    async def test_returns_dict_with_content(self):
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult(content="hello")

        with _patch_select(pick), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            result = await dispatcher.request(
                category=CallCategory.MAIN_WORK,
                task="coder",
                messages=[{"role": "user", "content": "write code"}],
            )

        assert result["content"] == "hello"

    @pytest.mark.asyncio
    async def test_result_dict_has_expected_keys(self):
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        with _patch_select(pick), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            result = await dispatcher.request(
                category=CallCategory.MAIN_WORK,
                messages=[{"role": "user", "content": "do it"}],
            )

        for key in ("content", "model", "model_name", "cost", "usage",
                    "tool_calls", "latency", "thinking", "is_local",
                    "ran_on", "provider", "task"):
            assert key in result, f"missing key: {key}"

    @pytest.mark.asyncio
    async def test_overhead_happy_path(self):
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult(content="classified")

        with _patch_select(pick), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            result = await dispatcher.request(
                category=CallCategory.OVERHEAD,
                task="router",
                messages=[{"role": "user", "content": "classify"}],
            )

        assert result["content"] == "classified"

    @pytest.mark.asyncio
    async def test_ran_on_is_local_for_local_model(self):
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult(is_local=True, provider="llama.cpp")

        with _patch_select(pick), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            result = await dispatcher.request(
                category=CallCategory.MAIN_WORK,
                messages=[],
            )

        assert result["ran_on"] == "local"

    @pytest.mark.asyncio
    async def test_ran_on_is_provider_for_cloud_model(self):
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult(is_local=False, provider="anthropic")

        with _patch_select(pick), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            result = await dispatcher.request(
                category=CallCategory.MAIN_WORK,
                messages=[],
            )

        assert result["ran_on"] == "anthropic"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. request() — no model available
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequestNoModel:

    @pytest.mark.asyncio
    async def test_main_work_raises_model_call_failed(self):
        from src.core.llm_dispatcher import CallCategory
        from src.core.router import ModelCallFailed

        dispatcher = _fresh_dispatcher()

        with _patch_select(None):
            with pytest.raises(ModelCallFailed):
                await dispatcher.request(
                    category=CallCategory.MAIN_WORK,
                    task="coder",
                    messages=[],
                )

    @pytest.mark.asyncio
    async def test_overhead_raises_runtime_error(self):
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()

        with _patch_select(None):
            with pytest.raises(RuntimeError, match="OVERHEAD call failed"):
                await dispatcher.request(
                    category=CallCategory.OVERHEAD,
                    task="router",
                    messages=[],
                )

    @pytest.mark.asyncio
    async def test_main_work_error_message_contains_task(self):
        from src.core.llm_dispatcher import CallCategory
        from src.core.router import ModelCallFailed

        dispatcher = _fresh_dispatcher()

        with _patch_select(None):
            with pytest.raises(ModelCallFailed) as exc_info:
                await dispatcher.request(
                    category=CallCategory.MAIN_WORK,
                    task="my-task",
                    messages=[],
                )
        assert exc_info.value.call_id == "my-task"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. request() — retry on retryable CallError
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequestRetry:

    @pytest.mark.asyncio
    async def test_retries_on_call_error_then_succeeds(self):
        """First call returns retryable error, second call succeeds."""
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()

        error = FakeCallError(retryable=True, message="timeout", category="timeout")
        success = FakeCallResult(content="retry worked")

        call_count = {"n": 0}

        async def fake_call(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return error
            return success

        with _patch_select(pick), \
             patch("hallederiz_kadir.call", new=AsyncMock(side_effect=fake_call)), \
             _patch_call_result_class(FakeCallResult), \
             patch("hallederiz_kadir.CallError", FakeCallError):
            result = await dispatcher.request(
                category=CallCategory.MAIN_WORK,
                messages=[],
            )

        assert result["content"] == "retry worked"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self):
        """Non-retryable CallError should raise without retrying."""
        from src.core.llm_dispatcher import CallCategory
        from src.core.router import ModelCallFailed

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        error = FakeCallError(retryable=False, message="bad request", category="bad_request")

        call_count = {"n": 0}

        async def fake_call(**kwargs):
            call_count["n"] += 1
            return error

        with _patch_select(pick), \
             patch("hallederiz_kadir.call", new=AsyncMock(side_effect=fake_call)), \
             _patch_call_result_class(FakeCallResult), \
             patch("hallederiz_kadir.CallError", FakeCallError):
            with pytest.raises(ModelCallFailed):
                await dispatcher.request(
                    category=CallCategory.MAIN_WORK,
                    messages=[],
                )

        assert call_count["n"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. request() — max retries exhausted
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequestMaxRetries:

    @pytest.mark.asyncio
    async def test_raises_after_5_retryable_failures(self):
        """After 5 accumulated failures (max_recursion), should raise."""
        from src.core.llm_dispatcher import CallCategory
        from src.core.router import ModelCallFailed

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        error = FakeCallError(retryable=True, message="timeout", category="timeout")

        async def always_error(**kwargs):
            return error

        with _patch_select(pick), \
             patch("hallederiz_kadir.call", new=AsyncMock(side_effect=always_error)), \
             _patch_call_result_class(FakeCallResult), \
             patch("hallederiz_kadir.CallError", FakeCallError):
            with pytest.raises(ModelCallFailed):
                await dispatcher.request(
                    category=CallCategory.MAIN_WORK,
                    messages=[],
                )

    @pytest.mark.asyncio
    async def test_overhead_raises_runtime_error_after_max_retries(self):
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        error = FakeCallError(retryable=True, message="timeout", category="timeout")

        async def always_error(**kwargs):
            return error

        with _patch_select(pick), \
             patch("hallederiz_kadir.call", new=AsyncMock(side_effect=always_error)), \
             _patch_call_result_class(FakeCallResult), \
             patch("hallederiz_kadir.CallError", FakeCallError):
            with pytest.raises(RuntimeError, match="OVERHEAD call failed"):
                await dispatcher.request(
                    category=CallCategory.OVERHEAD,
                    messages=[],
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. needs_thinking=False for OVERHEAD
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeedsThinkingOverhead:

    @pytest.mark.asyncio
    async def test_overhead_always_passes_needs_thinking_false(self):
        """OVERHEAD category must always pass needs_thinking=False to select()."""
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        select_kwargs = {}

        def fake_select(**kwargs):
            select_kwargs.update(kwargs)
            return pick

        with patch("fatih_hoca.select", side_effect=fake_select), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            await dispatcher.request(
                category=CallCategory.OVERHEAD,
                messages=[],
                needs_thinking=True,  # even if caller passes True, should be overridden
            )

        assert select_kwargs.get("needs_thinking") is False

    @pytest.mark.asyncio
    async def test_main_work_defaults_needs_thinking_true(self):
        """MAIN_WORK with no explicit needs_thinking should default to True."""
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        select_kwargs = {}

        def fake_select(**kwargs):
            select_kwargs.update(kwargs)
            return pick

        with patch("fatih_hoca.select", side_effect=fake_select), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            await dispatcher.request(
                category=CallCategory.MAIN_WORK,
                messages=[],
            )

        assert select_kwargs.get("needs_thinking") is True


# ═══════════════════════════════════════════════════════════════════════════════
# 7. tools → needs_function_calling
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeedsFunctionCalling:

    @pytest.mark.asyncio
    async def test_tools_sets_needs_function_calling_true(self):
        """Passing tools must cause needs_function_calling=True in select()."""
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        select_kwargs = {}

        def fake_select(**kwargs):
            select_kwargs.update(kwargs)
            return pick

        tools = [{"name": "search", "description": "search the web"}]

        with patch("fatih_hoca.select", side_effect=fake_select), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            await dispatcher.request(
                category=CallCategory.MAIN_WORK,
                messages=[],
                tools=tools,
            )

        assert select_kwargs.get("needs_function_calling") is True

    @pytest.mark.asyncio
    async def test_no_tools_does_not_force_function_calling(self):
        """Without tools, needs_function_calling should default to False."""
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        select_kwargs = {}

        def fake_select(**kwargs):
            select_kwargs.update(kwargs)
            return pick

        with patch("fatih_hoca.select", side_effect=fake_select), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            await dispatcher.request(
                category=CallCategory.MAIN_WORK,
                messages=[],
            )

        assert select_kwargs.get("needs_function_calling") is False



# ═══════════════════════════════════════════════════════════════════════════════
# 9. get_stats()
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetStats:

    def test_initial_stats_are_zero(self):
        dispatcher = _fresh_dispatcher()
        stats = dispatcher.get_stats()
        assert stats["total_calls"] == 0
        assert stats["overhead_calls"] == 0
        assert stats["overhead_pct"] == "0%"

    @pytest.mark.asyncio
    async def test_total_calls_increment(self):
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        with _patch_select(pick), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            await dispatcher.request(CallCategory.MAIN_WORK, messages=[])
            await dispatcher.request(CallCategory.MAIN_WORK, messages=[])

        stats = dispatcher.get_stats()
        assert stats["total_calls"] == 2
        assert stats["overhead_calls"] == 0

    @pytest.mark.asyncio
    async def test_overhead_calls_tracked_separately(self):
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        with _patch_select(pick), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            await dispatcher.request(CallCategory.MAIN_WORK, messages=[])
            await dispatcher.request(CallCategory.MAIN_WORK, messages=[])
            await dispatcher.request(CallCategory.OVERHEAD, messages=[])

        stats = dispatcher.get_stats()
        assert stats["total_calls"] == 3
        assert stats["overhead_calls"] == 1

    @pytest.mark.asyncio
    async def test_overhead_pct_correct(self):
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        with _patch_select(pick), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            # 1 overhead out of 3 total = 33.3%
            await dispatcher.request(CallCategory.MAIN_WORK, messages=[])
            await dispatcher.request(CallCategory.MAIN_WORK, messages=[])
            await dispatcher.request(CallCategory.OVERHEAD, messages=[])

        stats = dispatcher.get_stats()
        assert "33.3%" in stats["overhead_pct"]

    @pytest.mark.asyncio
    async def test_overhead_pct_increments_even_on_failed_call(self):
        """Even when select() returns None, counters should still increment."""
        from src.core.llm_dispatcher import CallCategory
        from src.core.router import ModelCallFailed

        dispatcher = _fresh_dispatcher()

        with _patch_select(None):
            try:
                await dispatcher.request(CallCategory.MAIN_WORK, messages=[])
            except ModelCallFailed:
                pass
            try:
                await dispatcher.request(CallCategory.OVERHEAD, messages=[])
            except RuntimeError:
                pass

        stats = dispatcher.get_stats()
        assert stats["total_calls"] == 2
        assert stats["overhead_calls"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Singleton — get_dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetDispatcherSingleton:

    def test_returns_same_instance(self):
        import src.core.llm_dispatcher as mod
        mod._dispatcher = None

        from src.core.llm_dispatcher import get_dispatcher
        d1 = get_dispatcher()
        d2 = get_dispatcher()
        assert d1 is d2

    def test_returns_llm_dispatcher_instance(self):
        import src.core.llm_dispatcher as mod
        mod._dispatcher = None

        from src.core.llm_dispatcher import get_dispatcher, LLMDispatcher
        d = get_dispatcher()
        assert isinstance(d, LLMDispatcher)

    def test_reset_creates_new_instance(self):
        import src.core.llm_dispatcher as mod
        from src.core.llm_dispatcher import get_dispatcher

        mod._dispatcher = None
        d1 = get_dispatcher()
        mod._dispatcher = None
        d2 = get_dispatcher()
        assert d1 is not d2


# ═══════════════════════════════════════════════════════════════════════════════
# 11. _timeout_floor()
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeoutFloor:

    def test_overhead_floor(self):
        from src.core.llm_dispatcher import CallCategory
        dispatcher = _fresh_dispatcher()
        assert dispatcher._timeout_floor(CallCategory.OVERHEAD) == 45.0

    def test_main_work_floor(self):
        from src.core.llm_dispatcher import CallCategory
        dispatcher = _fresh_dispatcher()
        assert dispatcher._timeout_floor(CallCategory.MAIN_WORK) == 60.0


# ═══════════════════════════════════════════════════════════════════════════════
# 12. local model load path
# ═══════════════════════════════════════════════════════════════════════════════

class TestLocalModelLoad:

    @pytest.mark.asyncio
    async def test_local_model_load_failure_raises_model_call_failed(self):
        """If _ensure_local_model returns False, should raise ModelCallFailed."""
        from src.core.llm_dispatcher import CallCategory
        from src.core.router import ModelCallFailed

        dispatcher = _fresh_dispatcher()
        # Local model, not ollama
        model = _make_mock_model(is_local=True, is_loaded=False)
        model.location = ""
        model.thinking_model = False
        pick = _make_mock_pick(model=model)

        with _patch_select(pick), \
             patch.object(dispatcher, "_ensure_local_model", AsyncMock(return_value=False)):
            with pytest.raises(ModelCallFailed):
                await dispatcher.request(
                    category=CallCategory.MAIN_WORK,
                    messages=[],
                )

    @pytest.mark.asyncio
    async def test_local_model_load_failure_overhead_raises_runtime_error(self):
        """If _ensure_local_model returns False for OVERHEAD, RuntimeError."""
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        model = _make_mock_model(is_local=True, is_loaded=False)
        model.location = ""
        model.thinking_model = False
        pick = _make_mock_pick(model=model)

        with _patch_select(pick), \
             patch.object(dispatcher, "_ensure_local_model", AsyncMock(return_value=False)):
            with pytest.raises(RuntimeError, match="OVERHEAD call failed"):
                await dispatcher.request(
                    category=CallCategory.OVERHEAD,
                    messages=[],
                )

    @pytest.mark.asyncio
    async def test_ollama_model_skips_local_load(self):
        """Models with location='ollama' should skip _ensure_local_model."""
        from src.core.llm_dispatcher import CallCategory

        dispatcher = _fresh_dispatcher()
        model = _make_mock_model(is_local=True)
        model.location = "ollama"
        pick = _make_mock_pick(model=model)
        result_obj = FakeCallResult()

        mock_ensure = AsyncMock(return_value=True)

        with _patch_select(pick), \
             _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult), \
             patch.object(dispatcher, "_ensure_local_model", mock_ensure):
            await dispatcher.request(
                category=CallCategory.MAIN_WORK,
                messages=[],
            )

        mock_ensure.assert_not_called()


def test_dispatcher_has_no_on_model_swap():
    """Dispatcher is a pure pipe; swap-event handling lives in Beckman."""
    from src.core.llm_dispatcher import LLMDispatcher
    assert not hasattr(LLMDispatcher, "on_model_swap")
