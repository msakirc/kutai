"""Transitional shim kept for set_orchestrator compatibility.

Task 7 deletes this file entirely once Orchestrator stops calling
set_orchestrator(self) in its __init__.
"""
from __future__ import annotations

from typing import Any

_ORCH_INSTANCE: Any = None


def set_orchestrator(instance: Any) -> None:
    global _ORCH_INSTANCE
    _ORCH_INSTANCE = instance


def get_orchestrator() -> Any:
    if _ORCH_INSTANCE is None:
        raise RuntimeError("orchestrator not registered")
    return _ORCH_INSTANCE
