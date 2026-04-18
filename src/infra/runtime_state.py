# runtime_state.py
"""Singleton runtime state — health operations delegate to NerdHerd.

Import and read flags as before:
    from src.infra.runtime_state import runtime_state
    if runtime_state["telegram_available"]:
        ...
"""
import asyncio
from datetime import datetime, timezone

_pending_degraded_tasks: set[asyncio.Task] = set()

runtime_state: dict = {
    "sandbox_available":     False,
    "telegram_available":    False,
    "llm_available":         False,
    "degraded_capabilities": [],
    "boot_time":             datetime.now(tz=timezone.utc).isoformat(),
    "models_loaded":         [],
}


def mark_degraded(capability: str) -> None:
    if capability not in runtime_state["degraded_capabilities"]:
        runtime_state["degraded_capabilities"].append(capability)
    try:
        from nerd_herd.client import get_default
        nh = get_default()
        if nh is not None:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(nh.mark_degraded(capability))
                _pending_degraded_tasks.add(task)
                task.add_done_callback(_pending_degraded_tasks.discard)
            except RuntimeError:
                pass
    except Exception:
        pass


def is_degraded(capability: str) -> bool:
    return capability in runtime_state["degraded_capabilities"]
