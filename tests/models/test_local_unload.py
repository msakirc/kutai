"""LocalModelManager.unload() — minimal mode must free VRAM.

The bug: selecting load mode `minimal` (or `/game`) paused new local inference
but never stopped the resident llama-server, so VRAM stayed occupied. unload()
stops the server (frees VRAM) while keeping DaLLaMa's watchdog/idle loops alive
so the next ensure_model reloads normally.
"""
from __future__ import annotations
import asyncio
import pytest

from src.models.local_model_manager import LocalModelManager, ModelRuntimeState


class _FakeServer:
    def __init__(self, alive: bool):
        self._alive = alive
        self.stopped = False

    def is_alive(self) -> bool:
        return self._alive

    async def stop(self) -> None:
        self.stopped = True
        self._alive = False


class _FakeSwap:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.intentional_unload = False
        self.locked_during_stop = False


class _FakeDaLLaMa:
    def __init__(self, alive: bool):
        self._server = _FakeServer(alive)
        self._swap = _FakeSwap()
        self._current_config = object()  # pretend a model is loaded


def _make_manager(alive: bool) -> LocalModelManager:
    mgr = LocalModelManager.__new__(LocalModelManager)  # skip real DaLLaMa boot
    mgr._dallama = _FakeDaLLaMa(alive)
    mgr._started = True
    mgr.runtime_state = ModelRuntimeState(
        model_name="m", thinking_enabled=True, context_length=8192,
        gpu_layers=33, measured_tps=10.0,
    )
    mgr._thinking_enabled = True
    mgr._vision_enabled = True
    mgr._nerd_herd = None
    return mgr


def test_unload_stops_server_and_clears_state():
    mgr = _make_manager(alive=True)
    freed = asyncio.run(mgr.unload(reason="load_mode_minimal"))
    assert freed is True
    assert mgr._dallama._server.stopped is True
    assert mgr._dallama._current_config is None
    assert mgr.runtime_state is None
    assert mgr._thinking_enabled is False
    assert mgr._vision_enabled is False
    # intentional_unload flag must be reset after the stop
    assert mgr._dallama._swap.intentional_unload is False


def test_unload_noop_when_nothing_loaded():
    mgr = _make_manager(alive=False)
    freed = asyncio.run(mgr.unload())
    assert freed is False
    assert mgr._dallama._server.stopped is False


def test_unload_noop_when_not_started():
    mgr = _make_manager(alive=True)
    mgr._started = False
    freed = asyncio.run(mgr.unload())
    assert freed is False
    assert mgr._dallama._server.stopped is False
