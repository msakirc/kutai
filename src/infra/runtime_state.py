# runtime_state.py
"""Singleton runtime state — health operations delegate to NerdHerd.

Import and read flags as before:
    from src.infra.runtime_state import runtime_state
    if runtime_state["telegram_available"]:
        ...
"""
from datetime import datetime, timezone

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
        from src.app.run import get_nerd_herd
        nh = get_nerd_herd()
        if nh is not None:
            nh.mark_degraded(capability)
    except Exception:
        pass


def is_degraded(capability: str) -> bool:
    return capability in runtime_state["degraded_capabilities"]
