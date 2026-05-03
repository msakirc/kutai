"""Continuation registry for Beckman on_complete handlers.

Usage::

    from general_beckman.continuations import register, dispatch_on_complete

    register("agent.resume", my_handler)          # idempotent
    await dispatch_on_complete("agent.resume", task_id, result)  # swallows errors
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from src.infra.logging_config import get_logger

_log = get_logger("beckman.continuations")

# name → async callable(task_id: int, result: dict) -> None
_HANDLERS: dict[str, Callable[[int, dict], Awaitable[None]]] = {}


def register(name: str, handler: Callable[[int, dict], Awaitable[None]]) -> None:
    """Register an on_complete handler. Idempotent — overrides existing entry."""
    _HANDLERS[name] = handler


async def dispatch_on_complete(name: str, task_id: int, result: dict) -> None:
    """Look up and invoke the named handler.

    Logs and swallows any error raised by the handler so the pump stays alive.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        _log.warning("dispatch_on_complete: no handler registered", name=name, task_id=task_id)
        return
    try:
        await handler(task_id, result)
    except Exception as exc:
        _log.error(
            "on_complete handler raised",
            name=name,
            task_id=task_id,
            error=str(exc),
        )
