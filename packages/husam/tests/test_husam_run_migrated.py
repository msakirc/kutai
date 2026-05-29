"""Migrated from tests/test_llm_dispatcher.py (SP3b Task 2).

These cases used to drive ``LLMDispatcher._do_dispatch`` directly. That method
moved into ``husam.run`` (select → dispatcher.execute → map). The assertions
are preserved verbatim; only the call surface changed:

    dispatcher._do_dispatch(category=..., task=..., messages=...)
        -> husam.run({"context": {"llm_call": {...}}, "kind": ...})

``husam.run`` bumps the real dispatcher's call counters (get_stats stays on the
dispatcher), so stats assertions exercise ``get_dispatcher().get_stats()``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── Fake lightweight types (ported from tests/test_llm_dispatcher.py) ────────

class FakeCallResult:
    """Stand-in that satisfies isinstance(result, hallederiz_kadir.CallResult)."""
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
    p.estimated_load_seconds = 0.0
    p.score = 0.5
    p.top_summary = ""
    return p


def _fresh_dispatcher():
    """Reset the dispatcher singleton and return a fresh instance.

    husam.run uses get_dispatcher() (the singleton), so resetting + reading
    via get_dispatcher() keeps counter assertions consistent.
    """
    import src.core.llm_dispatcher as mod
    mod._dispatcher = None
    return mod.get_dispatcher()


def _task(category="main_work", task="", agent_type="", difficulty=5,
          messages=None, tools=None, failures=None, preselected_pick=None,
          mission_id=None, **llm_call_extra):
    """Wrap former _do_dispatch kwargs into a husam.run task spec."""
    llm_call = {
        "raw_dispatch": True,
        "call_category": category,
        "task": task,
        "agent_type": agent_type,
        "difficulty": difficulty,
        "messages": messages if messages is not None else [],
        "tools": tools,
        "failures": failures if failures is not None else [],
    }
    llm_call.update(llm_call_extra)
    spec: dict = {
        "context": {"llm_call": llm_call},
        "kind": category,
        "preselected_pick": preselected_pick,
    }
    if mission_id is not None:
        spec["mission_id"] = mission_id
    return spec


# ─── Patch helpers ────────────────────────────────────────────────────────────

def _patch_select(return_value):
    return patch("fatih_hoca.select", return_value=return_value)


def _patch_call(return_value=None, side_effect=None):
    if side_effect is not None:
        return patch("hallederiz_kadir.call", new=AsyncMock(side_effect=side_effect))
    return patch("hallederiz_kadir.call", new=AsyncMock(return_value=return_value))


def _patch_call_result_class(cls):
    return patch("hallederiz_kadir.CallResult", cls)


# ═══════════════════════════════════════════════════════════════════════════════
# request() — happy path
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunHappyPath:

    @pytest.mark.asyncio
    async def test_returns_dict_with_content(self):
        import husam
        _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult(content="hello")

        with _patch_select(pick), _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            result = await husam.run(_task(
                category="main_work", task="coder",
                messages=[{"role": "user", "content": "write code"}],
            ))

        assert result["content"] == "hello"

    @pytest.mark.asyncio
    async def test_result_dict_has_expected_keys(self):
        import husam
        _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        with _patch_select(pick), _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            result = await husam.run(_task(
                category="main_work",
                messages=[{"role": "user", "content": "do it"}],
            ))

        for key in ("content", "model", "model_name", "cost", "usage",
                    "tool_calls", "latency", "thinking", "is_local",
                    "ran_on", "provider", "task"):
            assert key in result, f"missing key: {key}"

    @pytest.mark.asyncio
    async def test_overhead_happy_path(self):
        import husam
        _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult(content="classified")

        with _patch_select(pick), _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            result = await husam.run(_task(
                category="overhead", task="router",
                messages=[{"role": "user", "content": "classify"}],
            ))

        assert result["content"] == "classified"

    @pytest.mark.asyncio
    async def test_ran_on_is_local_for_local_model(self):
        import husam
        _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult(is_local=True, provider="llama.cpp")

        with _patch_select(pick), _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            result = await husam.run(_task(category="main_work", messages=[]))

        assert result["ran_on"] == "local"

    @pytest.mark.asyncio
    async def test_ran_on_is_provider_for_cloud_model(self):
        import husam
        _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult(is_local=False, provider="anthropic")

        with _patch_select(pick), _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            result = await husam.run(_task(category="main_work", messages=[]))

        assert result["ran_on"] == "anthropic"


# ═══════════════════════════════════════════════════════════════════════════════
# request() — no model available
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunNoModel:

    @pytest.mark.asyncio
    async def test_main_work_raises_model_call_failed(self):
        import husam
        from src.core.router import ModelCallFailed
        _fresh_dispatcher()

        with _patch_select(None):
            with pytest.raises(ModelCallFailed):
                await husam.run(_task(category="main_work", task="coder", messages=[]))

    @pytest.mark.asyncio
    async def test_overhead_raises_runtime_error(self):
        import husam
        _fresh_dispatcher()

        with _patch_select(None):
            with pytest.raises(RuntimeError, match="OVERHEAD call failed"):
                await husam.run(_task(category="overhead", task="router", messages=[]))

    @pytest.mark.asyncio
    async def test_main_work_error_message_contains_task(self):
        import husam
        from src.core.router import ModelCallFailed
        _fresh_dispatcher()

        with _patch_select(None):
            with pytest.raises(ModelCallFailed) as exc_info:
                await husam.run(_task(category="main_work", task="my-task", messages=[]))
        assert exc_info.value.call_id == "my-task"


# ═══════════════════════════════════════════════════════════════════════════════
# CallError handling (single attempt — no internal retry post-SP3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunCallError:

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self):
        """A CallError surfaces as ModelCallFailed after exactly one attempt.

        Pre-SP3 the dispatcher retried internally; husam does a single
        attempt and surfaces the failure (retries live in coulson / Beckman).
        """
        import husam
        from src.core.router import ModelCallFailed
        _fresh_dispatcher()
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
                await husam.run(_task(category="main_work", messages=[]))

        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_retryable_error_also_raises_single_attempt(self):
        """Even a retryable CallError raises after one attempt (no internal retry)."""
        import husam
        from src.core.router import ModelCallFailed
        _fresh_dispatcher()
        pick = _make_mock_pick()
        error = FakeCallError(retryable=True, message="timeout", category="timeout")

        call_count = {"n": 0}

        async def always_error(**kwargs):
            call_count["n"] += 1
            return error

        with _patch_select(pick), \
             patch("hallederiz_kadir.call", new=AsyncMock(side_effect=always_error)), \
             _patch_call_result_class(FakeCallResult), \
             patch("hallederiz_kadir.CallError", FakeCallError):
            with pytest.raises(ModelCallFailed):
                await husam.run(_task(category="main_work", messages=[]))
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_overhead_call_error_raises_runtime_error(self):
        import husam
        from src.core.router import ModelCallFailed
        _fresh_dispatcher()
        pick = _make_mock_pick()
        error = FakeCallError(retryable=True, message="timeout", category="timeout")

        async def always_error(**kwargs):
            return error

        with _patch_select(pick), \
             patch("hallederiz_kadir.call", new=AsyncMock(side_effect=always_error)), \
             _patch_call_result_class(FakeCallResult), \
             patch("hallederiz_kadir.CallError", FakeCallError):
            # OVERHEAD CallError surfaces as ModelCallFailed with an
            # "OVERHEAD call failed: ..." message (SP3 unified the exception
            # shape so the orchestrator's category-aware retry catches both).
            with pytest.raises(ModelCallFailed, match="OVERHEAD call failed"):
                await husam.run(_task(category="overhead", messages=[]))


# ═══════════════════════════════════════════════════════════════════════════════
# needs_thinking=False for OVERHEAD
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeedsThinkingOverhead:

    @pytest.mark.asyncio
    async def test_overhead_always_passes_needs_thinking_false(self):
        import husam
        _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        select_kwargs = {}

        def fake_select(**kwargs):
            select_kwargs.update(kwargs)
            return pick

        with patch("fatih_hoca.select", side_effect=fake_select), \
             _patch_call(result_obj), _patch_call_result_class(FakeCallResult):
            await husam.run(_task(
                category="overhead", messages=[], needs_thinking=True,
            ))

        assert select_kwargs.get("needs_thinking") is False

    @pytest.mark.asyncio
    async def test_main_work_defaults_needs_thinking_true(self):
        import husam
        _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        select_kwargs = {}

        def fake_select(**kwargs):
            select_kwargs.update(kwargs)
            return pick

        with patch("fatih_hoca.select", side_effect=fake_select), \
             _patch_call(result_obj), _patch_call_result_class(FakeCallResult):
            await husam.run(_task(category="main_work", messages=[]))

        assert select_kwargs.get("needs_thinking") is True


# ═══════════════════════════════════════════════════════════════════════════════
# tools → needs_function_calling
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeedsFunctionCalling:

    @pytest.mark.asyncio
    async def test_tools_sets_needs_function_calling_true(self):
        import husam
        _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        select_kwargs = {}

        def fake_select(**kwargs):
            select_kwargs.update(kwargs)
            return pick

        tools = [{"name": "search", "description": "search the web"}]

        with patch("fatih_hoca.select", side_effect=fake_select), \
             _patch_call(result_obj), _patch_call_result_class(FakeCallResult):
            await husam.run(_task(category="main_work", messages=[], tools=tools))

        assert select_kwargs.get("needs_function_calling") is True

    @pytest.mark.asyncio
    async def test_no_tools_does_not_force_function_calling(self):
        import husam
        _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        select_kwargs = {}

        def fake_select(**kwargs):
            select_kwargs.update(kwargs)
            return pick

        with patch("fatih_hoca.select", side_effect=fake_select), \
             _patch_call(result_obj), _patch_call_result_class(FakeCallResult):
            await husam.run(_task(category="main_work", messages=[]))

        assert select_kwargs.get("needs_function_calling") is False


# ═══════════════════════════════════════════════════════════════════════════════
# get_stats() — counters bump on the (real) dispatcher singleton
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetStats:

    @pytest.mark.asyncio
    async def test_total_calls_increment(self):
        import husam
        d = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        with _patch_select(pick), _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            await husam.run(_task(category="main_work", messages=[]))
            await husam.run(_task(category="main_work", messages=[]))

        stats = d.get_stats()
        assert stats["total_calls"] == 2
        assert stats["overhead_calls"] == 0

    @pytest.mark.asyncio
    async def test_overhead_calls_tracked_separately(self):
        import husam
        d = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        with _patch_select(pick), _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            await husam.run(_task(category="main_work", messages=[]))
            await husam.run(_task(category="main_work", messages=[]))
            await husam.run(_task(category="overhead", messages=[]))

        stats = d.get_stats()
        assert stats["total_calls"] == 3
        assert stats["overhead_calls"] == 1

    @pytest.mark.asyncio
    async def test_overhead_pct_correct(self):
        import husam
        d = _fresh_dispatcher()
        pick = _make_mock_pick()
        result_obj = FakeCallResult()

        with _patch_select(pick), _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult):
            await husam.run(_task(category="main_work", messages=[]))
            await husam.run(_task(category="main_work", messages=[]))
            await husam.run(_task(category="overhead", messages=[]))

        stats = d.get_stats()
        assert "33.3%" in stats["overhead_pct"]

    @pytest.mark.asyncio
    async def test_counters_increment_even_on_failed_call(self):
        """Even when select() returns None, counters still increment."""
        import husam
        from src.core.router import ModelCallFailed
        d = _fresh_dispatcher()

        with _patch_select(None):
            try:
                await husam.run(_task(category="main_work", messages=[]))
            except ModelCallFailed:
                pass
            try:
                await husam.run(_task(category="overhead", messages=[]))
            except RuntimeError:
                pass

        stats = d.get_stats()
        assert stats["total_calls"] == 2
        assert stats["overhead_calls"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# local model load path (execute() consults dispatcher._ensure_local_model)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLocalModelLoad:

    @pytest.mark.asyncio
    async def test_local_model_load_failure_raises_model_call_failed(self):
        import husam
        from src.core.router import ModelCallFailed
        d = _fresh_dispatcher()
        model = _make_mock_model(is_local=True, is_loaded=False)
        model.location = ""
        model.thinking_model = False
        pick = _make_mock_pick(model=model)

        with _patch_select(pick), \
             patch.object(d, "_ensure_local_model", AsyncMock(return_value=(False, False))):
            with pytest.raises(ModelCallFailed):
                await husam.run(_task(category="main_work", messages=[]))

    @pytest.mark.asyncio
    async def test_local_model_load_failure_overhead_raises_runtime_error(self):
        import husam
        from src.core.router import ModelCallFailed
        d = _fresh_dispatcher()
        model = _make_mock_model(is_local=True, is_loaded=False)
        model.location = ""
        model.thinking_model = False
        pick = _make_mock_pick(model=model)

        with _patch_select(pick), \
             patch.object(d, "_ensure_local_model", AsyncMock(return_value=(False, False))):
            # Loading failure is a non-retryable CallError(loading); for
            # OVERHEAD husam surfaces it as ModelCallFailed with an
            # "OVERHEAD call failed" message.
            with pytest.raises(ModelCallFailed, match="OVERHEAD call failed"):
                await husam.run(_task(category="overhead", messages=[]))

    @pytest.mark.asyncio
    async def test_ollama_model_skips_local_load(self):
        import husam
        d = _fresh_dispatcher()
        model = _make_mock_model(is_local=True)
        model.location = "ollama"
        pick = _make_mock_pick(model=model)
        result_obj = FakeCallResult()

        mock_ensure = AsyncMock(return_value=(True, False))

        with _patch_select(pick), _patch_call(result_obj), \
             _patch_call_result_class(FakeCallResult), \
             patch.object(d, "_ensure_local_model", mock_ensure):
            await husam.run(_task(category="main_work", messages=[]))

        mock_ensure.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Hoca returns None → husam raises (seam test)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunNoneFromHocaContract:

    @pytest.mark.asyncio
    async def test_main_work_none_pick_raises_model_call_failed(self):
        import husam
        from src.core.router import ModelCallFailed
        _fresh_dispatcher()
        with _patch_select(None):
            with pytest.raises(ModelCallFailed):
                await husam.run(_task(category="main_work", task="coder", messages=[]))

    @pytest.mark.asyncio
    async def test_overhead_none_pick_raises_runtime_error(self):
        import husam
        _fresh_dispatcher()
        with _patch_select(None):
            with pytest.raises(RuntimeError, match="OVERHEAD call failed"):
                await husam.run(_task(category="overhead", task="router", messages=[]))
