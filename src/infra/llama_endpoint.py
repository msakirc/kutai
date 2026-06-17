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


def _dotenv_port() -> str | None:
    """``LLAMA_SERVER_PORT`` as written in the ``.env`` FILE (None if absent
    or python-dotenv is unavailable).

    Reads the file directly via ``dotenv_values`` — it does NOT mutate
    ``os.environ`` — so the resolver can treat ``.env`` as authoritative
    regardless of ``load_dotenv``'s env-wins precedence.
    """
    try:
        from dotenv import dotenv_values, find_dotenv
    except Exception:
        return None
    try:
        path = find_dotenv(usecwd=True)
        if not path:
            return None
        return dotenv_values(path).get("LLAMA_SERVER_PORT")
    except Exception:
        return None


def _warn_port_mismatch(env_raw: str, file_raw: str) -> None:
    msg = (
        "LLAMA_SERVER_PORT mismatch: process env=%s but .env=%s — using .env. "
        "Likely a STALE inherited env (a long-lived wrapper launched before a "
        ".env edit, whose restarted children inherit the old value); relaunch "
        "the wrapper from a clean shell to clear it."
    )
    try:
        from src.infra.logging_config import get_logger
        get_logger("infra.llama_endpoint").warning(msg, env_raw, file_raw)
    except Exception:
        import logging
        logging.getLogger("infra.llama_endpoint").warning(msg, env_raw, file_raw)


def resolve_llama_port() -> int:
    """Return the configured llama-server port, or raise.

    ``.env`` is the deployment source of truth. A stale ``LLAMA_SERVER_PORT``
    inherited in the process environment (a long-lived wrapper launched before
    a ``.env`` edit — its orchestrator children inherit the old value) must NOT
    silently win: that caused the 2026-06-17 Expo:8081 collision (the wrapper
    kept 8081 across orchestrator restarts while ``.env`` already said 8090).
    On a mismatch between the inherited env and ``.env`` we prefer ``.env`` and
    warn loudly.

    Raises ``RuntimeError`` if neither source provides a valid integer port —
    never guess (the silent 8080 default caused the 2026-06-14 wrong-port orphan).
    """
    env_raw = os.environ.get("LLAMA_SERVER_PORT")
    file_raw = _dotenv_port()

    if env_raw is not None and file_raw is not None and str(env_raw) != str(file_raw):
        _warn_port_mismatch(env_raw, file_raw)
        raw = file_raw
    else:
        raw = env_raw if env_raw is not None else file_raw

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
