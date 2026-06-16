"""Injected service hooks — the engine's ONLY sanctioned way to call UP into
app-level services (sandbox/shell, vector store, dead-letter quarantine)
without importing ``src.*``. Keeps ``dabidabi`` dependency-free on the app.

App startup wires real implementations via ``register(...)`` (see
``src/infra/db_hooks.py::wire``). Each registered impl is a thin wrapper that
lazy-imports the real ``src.*`` module on first call, so wiring stays cheap
and import-cycle-free. When a hook is unset (CLIs, tests, fresh DB) the engine
degrades to the SAME best-effort no-op the old ``try: from src.* import ...``
blocks produced on ImportError.

All hooks are ``None`` until registered. Engine call sites must null-check.
"""
from typing import Any, Awaitable, Callable, Optional

# mission_id -> awaitable; impl owns the SANDBOX_MODE gate (engine no longer
# knows about sandbox modes).
ensure_mission_container: Optional[Callable[[int], Awaitable[Any]]] = None

# Vector store (ChromaDB) — all best-effort; unset → embedding/recall skipped.
embed_and_store: Optional[Callable[..., Awaitable[Any]]] = None
vector_query: Optional[Callable[..., Awaitable[list]]] = None
purge_mission_chroma: Optional[Callable[[int], Awaitable[int]]] = None

# Dead-letter quarantine (startup poison-task recovery). Unset → quarantine
# skipped + warn-logged by the caller, same as the old ImportError fallback.
quarantine_task: Optional[Callable[..., Awaitable[Any]]] = None

_HOOK_NAMES = frozenset({
    "ensure_mission_container", "embed_and_store", "vector_query",
    "purge_mission_chroma", "quarantine_task",
})


def register(**kwargs: Callable) -> None:
    """Set one or more hooks by name. Unknown names raise (typo guard)."""
    g = globals()
    for name, fn in kwargs.items():
        if name not in _HOOK_NAMES:
            raise KeyError(f"unknown dabidabi hook: {name!r}")
        g[name] = fn


def reset() -> None:
    """Clear all hooks (test isolation)."""
    g = globals()
    for name in _HOOK_NAMES:
        g[name] = None
