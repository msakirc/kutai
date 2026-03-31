# runtime_state.py
"""
Singleton runtime state populated during startup health checks.

Import and read flags before using optional services:
    from src.infra.runtime_state import runtime_state
    if runtime_state["web_search_available"]:
        ...
"""

from datetime import datetime, timezone

runtime_state: dict = {
    "web_search_available":  False,
    "sandbox_available":     False,
    "telegram_available":    False,
    "frontail_available":    False,
    "llm_available":         False,
    "degraded_capabilities": [],   # list[str] of capability names that are down
    "boot_time":             datetime.now(tz=timezone.utc).isoformat(),
    "models_loaded":         [],   # populated by model_registry at startup
}


def mark_degraded(capability: str) -> None:
    if capability not in runtime_state["degraded_capabilities"]:
        runtime_state["degraded_capabilities"].append(capability)


def is_degraded(capability: str) -> bool:
    return capability in runtime_state["degraded_capabilities"]
