"""PromptStore port — versioned prompt overrides live behind this Protocol.

The app (src) injects a concrete DB-backed adapter via set_store(). With no
store set, get_active() returns None and the in-package YAML seed is used.
The leaf imports no DB and no src.
"""
from __future__ import annotations
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class PromptStore(Protocol):
    async def get_active(self, key: str) -> Optional[str]: ...
    async def save_version(self, key: str, text: str, notes: str = "", activate: bool = False) -> int: ...
    async def record_quality(self, key: str, score: float) -> None: ...
    async def list_versions(self, key: str) -> list[dict]: ...


_store: Optional[PromptStore] = None


def set_store(store: Optional[PromptStore]) -> None:
    global _store
    _store = store


async def get_active(key: str) -> Optional[str]:
    if _store is None:
        return None
    return await _store.get_active(key)


async def record_quality(key: str, score: float) -> None:
    if _store is None:
        return
    await _store.record_quality(key, score)
