# load_manager.py
"""Load manager shim — delegates to NerdHerdClient.

All import paths preserved:
    from src.infra.load_manager import get_load_mode, set_load_mode, ...
"""
from __future__ import annotations

import asyncio

from nerd_herd.load import LOAD_MODES, VRAM_BUDGETS, DESCRIPTIONS  # noqa: F401


def _nh():
    """Get NerdHerdClient instance, or None if not yet initialized."""
    try:
        from src.app.run import get_nerd_herd
        return get_nerd_herd()
    except Exception:
        return None


async def get_load_mode() -> str:
    nh = _nh()
    if nh is None:
        return "full"
    return await nh.get_load_mode()


async def set_load_mode(mode: str, source: str = "user") -> str:
    nh = _nh()
    if nh is None:
        return "NerdHerd not connected"
    return await nh.set_load_mode(mode, source)


async def enable_auto_management():
    nh = _nh()
    if nh:
        await nh.enable_auto_management()


def is_local_inference_allowed() -> bool:
    """Sync — returns True as safe default. Use is_local_inference_allowed_async() in async code."""
    return True


async def is_local_inference_allowed_async() -> bool:
    nh = _nh()
    if nh is None:
        return True
    return await nh.is_local_inference_allowed()


def is_auto_managed() -> bool:
    """Sync — returns True as safe default. Use is_auto_managed_async() in async code."""
    return True


async def is_auto_managed_async() -> bool:
    nh = _nh()
    if nh is None:
        return True
    return await nh.is_auto_managed()


def get_vram_budget_fraction() -> float:
    """Sync — returns 1.0 as safe default. Use get_vram_budget_fraction_async() in async code."""
    return 1.0


async def get_vram_budget_fraction_async() -> float:
    nh = _nh()
    if nh is None:
        return 1.0
    return await nh.get_vram_budget_fraction()


def suggest_mode_for_external_usage(ext_frac: float) -> str:
    from nerd_herd.load import LoadManager
    return LoadManager.suggest_mode_for_external_usage(ext_frac)


async def run_gpu_autodetect_loop(notify_fn=None):
    """No-op — auto-detect runs in the NerdHerd sidecar now."""
    pass
