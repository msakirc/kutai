"""intake #73 — dispatcher must reload an already-loaded local model when its
loaded context window is smaller than the task needs.

llama-server fixes n_ctx at load and cannot grow it at runtime, so reusing an
under-sized loaded instance silently overflows the prompt (Qwen3.5-9B loaded at
ctx 5786 under RAM pressure, reused for a ~14k-token analyst prompt). The fix
adds a loaded-ctx check to `_ensure_local_model`'s reload decision.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _fake_manager(*, loaded_ctx: int, current: str = "qwen3-9b") -> MagicMock:
    mgr = MagicMock()
    mgr.is_loaded = True
    mgr.current_model = current
    mgr._thinking_enabled = False
    mgr._vision_enabled = False
    mgr.loaded_context_length = loaded_ctx
    mgr.ensure_model = AsyncMock(return_value=True)
    mgr.keep_alive = MagicMock()
    return mgr


def _model(name: str = "qwen3-9b") -> MagicMock:
    m = MagicMock(thinking_model=False, has_vision=False, is_local=True)
    m.name = name
    return m


@pytest.mark.asyncio
async def test_reload_when_loaded_ctx_below_task_need():
    """loaded ctx 5786 < needed 14162 -> reload (ensure_model called)."""
    from src.core.llm_dispatcher import LLMDispatcher

    mgr = _fake_manager(loaded_ctx=5786)
    with patch("src.models.local_model_manager.get_local_manager", return_value=mgr):
        d = LLMDispatcher()
        ok, _ = await d._ensure_local_model(_model(), estimated_context=14162)

    assert ok is True
    mgr.ensure_model.assert_awaited_once()
    # The reload must carry the task's context need as the min_context floor.
    assert mgr.ensure_model.await_args.kwargs["min_context"] == 14162
    mgr.keep_alive.assert_not_called()


@pytest.mark.asyncio
async def test_reuse_when_loaded_ctx_sufficient():
    """loaded ctx 20000 >= needed 14162 -> reuse, no reload."""
    from src.core.llm_dispatcher import LLMDispatcher

    mgr = _fake_manager(loaded_ctx=20000)
    with patch("src.models.local_model_manager.get_local_manager", return_value=mgr):
        d = LLMDispatcher()
        ok, swapped = await d._ensure_local_model(_model(), estimated_context=14162)

    assert ok is True and swapped is False
    mgr.ensure_model.assert_not_called()
    mgr.keep_alive.assert_called_once()


@pytest.mark.asyncio
async def test_no_estimated_context_keeps_old_reuse_behaviour():
    """estimated_context=0 (unknown need) must not force a reload."""
    from src.core.llm_dispatcher import LLMDispatcher

    mgr = _fake_manager(loaded_ctx=5786)
    with patch("src.models.local_model_manager.get_local_manager", return_value=mgr):
        d = LLMDispatcher()
        ok, swapped = await d._ensure_local_model(_model(), estimated_context=0)

    assert ok is True and swapped is False
    mgr.ensure_model.assert_not_called()


def test_loaded_context_length_property():
    """Manager exposes the loaded n_ctx; 0 when nothing is healthy/loaded."""
    from src.models.local_model_manager import LocalModelManager

    mgr = object.__new__(LocalModelManager)  # skip heavy __init__
    mgr._dallama = MagicMock()
    mgr._dallama.status = MagicMock(model_name="qwen3-9b", healthy=True, context_length=5786)
    assert mgr.loaded_context_length == 5786

    mgr._dallama.status = MagicMock(model_name=None, healthy=False, context_length=0)
    assert mgr.loaded_context_length == 0
