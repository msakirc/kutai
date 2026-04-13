# load_manager.py
"""Load manager shim — delegates to nerd_herd.LoadManager.

All import paths preserved:
    from src.infra.load_manager import get_load_mode, set_load_mode, ...
"""
from __future__ import annotations

import asyncio

from nerd_herd.load import LOAD_MODES, VRAM_BUDGETS, DESCRIPTIONS  # noqa: F401


def _nh():
    from src.app.run import get_nerd_herd
    return get_nerd_herd()


async def get_load_mode() -> str:
    return _nh().get_load_mode()


async def set_load_mode(mode: str, source: str = "user") -> str:
    return _nh().set_load_mode(mode, source)


async def enable_auto_management():
    _nh().enable_auto_management()


def is_local_inference_allowed() -> bool:
    return _nh().is_local_inference_allowed()


def is_auto_managed() -> bool:
    return _nh()._load.is_auto_managed()


def get_vram_budget_fraction() -> float:
    return _nh().get_vram_budget_fraction()


def suggest_mode_for_external_usage(ext_frac: float) -> str:
    from nerd_herd.load import LoadManager
    return LoadManager.suggest_mode_for_external_usage(ext_frac)


async def run_gpu_autodetect_loop(notify_fn=None):
    """Shim: start NerdHerd's auto-detect and block until cancelled."""
    await _nh().start_auto_detect(notify_fn)
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await _nh()._load.stop_auto_detect()
