# introspection.py
"""Read-only queries about the currently loaded local model.

Evicted from ``LLMDispatcher`` (Modularization Finish Plan Phase 4) — these
are introspection helpers, not part of the dispatcher's load→call loop. The
home for any future "what is loaded right now?" query.

All functions swallow errors and return a safe default — callers treat a
missing answer as "unknown", never as a hard failure.
"""

from __future__ import annotations


def get_loaded_litellm_name() -> str | None:
    """Return the currently loaded local model's litellm_name, or None."""
    try:
        from src.models.local_model_manager import get_local_manager
        from src.models.model_registry import get_registry
        manager = get_local_manager()
        if not manager.current_model:
            return None
        registry = get_registry()
        info = registry.get(manager.current_model)
        return info.litellm_name if info else None
    except Exception:
        return None
