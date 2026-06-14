"""Single source of truth for the llama-server endpoint.

Fail-loud replacement for the scattered
``int(os.environ.get("LLAMA_SERVER_PORT", "8080"))`` idiom. The silent 8080
fallback caused the 2026-06-14 split-brain orphan: a process spawned without
``LLAMA_SERVER_PORT`` in its environment bound 8080 while the rest of the
stack used 8081, and nothing raised.

Resolution order:
1. Read ``LLAMA_SERVER_PORT`` from the environment.
2. If absent, load ``.env`` once (so a process that skipped ``load_dotenv``
   still reads the same source of truth) and re-read.
3. If still absent or non-numeric, raise — never guess a port.
"""
from __future__ import annotations

import os

_HOST = "127.0.0.1"


def _ensure_dotenv_loaded() -> None:
    """Best-effort ``.env`` load. Idempotent; safe if python-dotenv is absent."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def resolve_llama_port() -> int:
    """Return the configured llama-server port, or raise.

    Raises ``RuntimeError`` if ``LLAMA_SERVER_PORT`` is unset (after a .env
    load attempt) or not a valid integer. Refusing to default prevents a
    misconfigured process from silently diverging onto port 8080.
    """
    raw = os.environ.get("LLAMA_SERVER_PORT")
    if raw is None:
        _ensure_dotenv_loaded()
        raw = os.environ.get("LLAMA_SERVER_PORT")
    if raw is None:
        raise RuntimeError(
            "LLAMA_SERVER_PORT is not set and no .env provided it. Refusing to "
            "guess a port (the silent 8080 default caused the 2026-06-14 "
            "wrong-port orphan). Set LLAMA_SERVER_PORT in the environment or .env."
        )
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"LLAMA_SERVER_PORT={raw!r} is not a valid integer port."
        ) from exc


def resolve_llama_url(path: str = "") -> str:
    """Return ``http://127.0.0.1:<port>`` (+ optional path) for llama-server."""
    base = f"http://{_HOST}:{resolve_llama_port()}"
    if not path:
        return base
    return base + (path if path.startswith("/") else "/" + path)
