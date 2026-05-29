# tests/test_llm_dispatcher.py
"""
Tests for src/core/llm_dispatcher.py — the dumb-pipe dispatcher (post-SP3b).

The select → execute → map orchestration that used to live in
``LLMDispatcher._do_dispatch`` / ``dispatch`` moved to ``husam.run``; those
cases now live in ``packages/husam/tests/test_husam_run_migrated.py``.

What remains here are the surviving dispatcher surfaces:
  - CallCategory enum
  - get_dispatcher() singleton
  - dispatcher has no on_model_swap (pure pipe)
  - _ensure_local_model tuple contract (a surviving primitive helper)

execute() begin/end_call internals are covered by
``tests/core/test_dispatcher_in_flight.py`` and
``tests/core/test_dispatcher_records_swap.py``.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def test_dispatcher_has_no_on_model_swap():
    """Dispatcher is a pure pipe; swap-event handling lives in Beckman."""
    from src.core.llm_dispatcher import LLMDispatcher
    assert not hasattr(LLMDispatcher, "on_model_swap")


def test_dispatcher_is_a_dumb_pipe():
    """Post-SP3b: select/map orchestration moved to husam.

    dispatch / _do_dispatch / _result_to_dict are gone; execute (the dumb
    one-attempt primitive) plus its pipe helpers remain.
    """
    from src.core.llm_dispatcher import LLMDispatcher
    assert not hasattr(LLMDispatcher, "dispatch")
    assert not hasattr(LLMDispatcher, "_do_dispatch")
    assert not hasattr(LLMDispatcher, "_result_to_dict")
    assert hasattr(LLMDispatcher, "execute")
    assert hasattr(LLMDispatcher, "_record_pick")
    assert hasattr(LLMDispatcher, "_ensure_local_model")
    assert hasattr(LLMDispatcher, "_prepare_messages")


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRACT TESTS — module-boundary invariants
#
# Purpose: catch regressions at seams where past bugs have slipped through
# (ensure_model return-type drift). These pin the contracts at
# dispatcher ↔ local_model_manager.
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnsureLocalModelContract:
    """Contract: _ensure_local_model returns (ok: bool, swap_happened: bool).

    Past failure: tests mocked this with return_value=True/False (bool),
    which broke `ok, swap_happened = await self._ensure_local_model(...)`
    tuple-unpacking at the call site. This test pins the tuple contract
    so drift is caught at the seam.
    """

    @pytest.mark.asyncio
    async def test_returns_tuple_when_already_loaded(self):
        """Already-loaded model path returns (True, False) — no swap."""
        from src.core.llm_dispatcher import LLMDispatcher

        dispatcher = LLMDispatcher()
        model = _make_mock_model(is_local=True, is_loaded=True)
        model.thinking_model = False
        model.has_vision = False

        fake_manager = MagicMock()
        fake_manager._thinking_enabled = False
        fake_manager._vision_enabled = False
        # DaLLaMa is the source of truth: dispatcher consults
        # manager.is_loaded + manager.current_model, not the registry's
        # ModelInfo.is_loaded flag.
        fake_manager.is_loaded = True
        fake_manager.current_model = model.name

        with patch("src.models.local_model_manager.get_local_manager",
                   return_value=fake_manager):
            result = await dispatcher._ensure_local_model(model)
        assert isinstance(result, tuple) and len(result) == 2
        assert result == (True, False)

    @pytest.mark.asyncio
    async def test_returns_tuple_when_swap_happens(self):
        """Swap path returns (True, True) when before != after."""
        from src.core.llm_dispatcher import LLMDispatcher

        dispatcher = LLMDispatcher()
        model = _make_mock_model(name="new-model", is_local=True, is_loaded=False)
        model.thinking_model = False
        model.has_vision = False

        fake_manager = MagicMock()
        fake_manager._thinking_enabled = False
        fake_manager._vision_enabled = False
        fake_manager.current_model = "old-model"

        async def _fake_ensure_model(name, **kw):
            fake_manager.current_model = name
            return True
        fake_manager.ensure_model = _fake_ensure_model

        with patch("src.models.local_model_manager.get_local_manager",
                   return_value=fake_manager):
            result = await dispatcher._ensure_local_model(model)
        assert isinstance(result, tuple) and len(result) == 2
        assert result == (True, True)

    @pytest.mark.asyncio
    async def test_returns_tuple_when_load_fails(self):
        """Load failure returns (False, False)."""
        from src.core.llm_dispatcher import LLMDispatcher

        dispatcher = LLMDispatcher()
        model = _make_mock_model(name="doomed", is_local=True, is_loaded=False)
        model.thinking_model = False
        model.has_vision = False

        fake_manager = MagicMock()
        fake_manager._thinking_enabled = False
        fake_manager._vision_enabled = False
        fake_manager.is_loaded = False
        fake_manager.current_model = "doomed"

        async def _fake_ensure_model(name, **kw):
            return False
        fake_manager.ensure_model = _fake_ensure_model

        with patch("src.models.local_model_manager.get_local_manager",
                   return_value=fake_manager):
            result = await dispatcher._ensure_local_model(model)
        assert isinstance(result, tuple) and len(result) == 2
        assert result == (False, False)
