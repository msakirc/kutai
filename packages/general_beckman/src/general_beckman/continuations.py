"""Durable continuation substrate for Beckman (CPS SP1).

A continuation = "when child task T reaches a terminal state, run a named
handler with saved parent state". The row lives in the `continuations` table
(written atomically with the child in add_task), so it survives restarts.

Handler signature: ``async (task_id: int, result: dict, state: dict) -> None``
where ``task_id`` is the CHILD id and ``state`` is the parent state passed as
``cont_state`` at enqueue time (``{}`` when none).
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

from src.infra.logging_config import get_logger

_log = get_logger("beckman.continuations")

# Default expiry for a pending continuation whose child never reaches terminal
# AND is no longer alive. Replaces the opaque 600s await_inline block.
CONTINUATION_TTL_SECONDS = 3600

# name → async callable(task_id: int, result: dict, state: dict) -> None
_HANDLERS: dict[str, Callable[[int, dict, dict], Awaitable[None]]] = {}


def register_resume(name: str, handler) -> None:
    """Register a resume / on_error handler. Idempotent — overrides existing."""
    _HANDLERS[name] = handler


# Back-compat alias: existing callers (analytics_digest, classify_signals) use
# ``register``. Keep it pointing at the same registry.
register = register_resume


async def dispatch_on_complete(name: str, task_id: int, result: dict,
                               state: dict | None = None) -> None:
    """Look up and invoke the named handler (3-arg). Swallows handler errors."""
    handler = _HANDLERS.get(name)
    if handler is None:
        _log.warning("dispatch: no handler registered", name=name, task_id=task_id)
        return
    try:
        await handler(task_id, result, state or {})
    except Exception as exc:  # noqa: BLE001
        _log.error("continuation handler raised", name=name, task_id=task_id,
                   error=str(exc))


async def claim_for_fire(child_task_id: int) -> dict | None:
    """Atomically claim the pending continuation for a child (CAS).

    Returns ``{resume_name, on_error_name, state}`` if THIS caller won the
    claim (flipped pending→fired), else ``None`` (no row, or already fired).
    Claim happens BEFORE handler dispatch so a re-entrant on_task_finished
    can never double-fire.
    """
    from src.infra.db import get_db
    db = await get_db()
    upd = await db.execute(
        "UPDATE continuations SET status='fired' "
        "WHERE child_task_id=? AND status='pending'",
        (child_task_id,),
    )
    await db.commit()
    if upd.rowcount != 1:
        return None
    cur = await db.execute(
        "SELECT resume_name, on_error_name, state_json "
        "FROM continuations WHERE child_task_id=?",
        (child_task_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    try:
        state = json.loads(row[2]) if row[2] else {}
    except Exception:
        state = {}
    return {"resume_name": row[0], "on_error_name": row[1], "state": state}


async def fire_for_task(child_task_id: int, result: dict, raw_status: str) -> bool:
    """Fire the continuation for a terminal child. Returns True if it claimed.

    Status mapping (raw agent status, BEFORE route_result rewrite):
      - 'needs_clarification' → leave pending, do NOT claim (not terminal).
      - 'failed'              → dispatch on_error if set, else claimed no-op.
      - anything else (incl. 'completed', and a grade that graded as
        {passed: false}) → dispatch resume.
    Handler dispatch is detached; the claim (CAS) is synchronous.
    """
    if raw_status == "needs_clarification":
        return False
    claim = await claim_for_fire(child_task_id)
    if claim is None:
        return False
    state = claim["state"]
    if raw_status == "failed":
        name = claim["on_error_name"]
        if name:
            asyncio.create_task(dispatch_on_complete(name, child_task_id, result, state))
        else:
            _log.info("continuation: failed child, no on_error — no-op",
                      child_task_id=child_task_id)
    else:
        name = claim["resume_name"]
        if name:
            asyncio.create_task(dispatch_on_complete(name, child_task_id, result, state))
    return True
