# load_manager.py
"""Load manager shim — delegates to nerd_herd.LoadManager.

All import paths preserved:
    from src.infra.load_manager import get_load_mode, set_load_mode, ...
"""
from __future__ import annotations

import asyncio

from nerd_herd.load import LOAD_MODES, VRAM_BUDGETS, DESCRIPTIONS  # noqa: F401


def _nh():
    """Get NerdHerd instance, or None if not yet initialized."""
    try:
        from src.app.run import get_nerd_herd
        return get_nerd_herd()
    except Exception:
        return None


async def get_load_mode() -> str:
    nh = _nh()
    return nh.get_load_mode() if nh else "full"


async def set_load_mode(mode: str, source: str = "user") -> str:
    nh = _nh()
    if nh is None:
        return "NerdHerd not initialized"
    return nh.set_load_mode(mode, source)


async def enable_auto_management():
    nh = _nh()
    if nh:
        nh.enable_auto_management()


def is_local_inference_allowed() -> bool:
    nh = _nh()
    return nh.is_local_inference_allowed() if nh else True


def is_auto_managed() -> bool:
    nh = _nh()
    return nh._load.is_auto_managed() if nh else True


def get_vram_budget_fraction() -> float:
    nh = _nh()
    return nh.get_vram_budget_fraction() if nh else 1.0


def suggest_mode_for_external_usage(ext_frac: float) -> str:
    from nerd_herd.load import LoadManager
    return LoadManager.suggest_mode_for_external_usage(ext_frac)


async def run_gpu_autodetect_loop(notify_fn=None):
    """Shim: start NerdHerd's auto-detect and block until cancelled."""
    nh = _nh()
    if nh is None:
        return
    await nh.start_auto_detect(notify_fn)
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await nh._load.stop_auto_detect()
