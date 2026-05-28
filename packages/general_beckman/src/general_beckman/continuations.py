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

    Handler-presence pre-check: if the row designates a handler that is NOT
    registered in this process, the row is LEFT PENDING (no claim, no fire)
    and a warning is logged — so a restart that fixes the missing import
    can recover via reconcile. Without this, a missing handler would still
    consume the row (CAS → 'fired') and the continuation would be lost.
    """
    if raw_status == "needs_clarification":
        return False

    # Peek at the handler names (no claim yet).
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT resume_name, on_error_name FROM continuations "
        "WHERE child_task_id=? AND status='pending'",
        (child_task_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return False
    resume_name, on_error_name = row[0], row[1]

    # Pick which name we would dispatch.
    if raw_status == "failed":
        target = on_error_name
        is_error_path = True
    else:
        target = resume_name
        is_error_path = False

    # Handler-presence check. Empty string = "no resume name" (on_error-only
    # continuation); that path simply claims + no-ops. Only warn when a
    # NAMED target is required but missing from the registry.
    if target and target not in _HANDLERS:
        _log.warning(
            "continuation handler not registered in this process — "
            "leaving row pending for recovery on next reconcile",
            child_task_id=child_task_id,
            handler=target,
            path="on_error" if is_error_path else "resume",
        )
        return False

    # Now claim atomically.
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


# ──────────────────────────────────────────────────────────────────────────
# Modules that register continuation handlers AT IMPORT TIME via a module-
# level `register_continuations()` call. register_startup_handlers() imports
# each one so the in-memory _HANDLERS registry is populated before reconcile
# runs at startup.
#
# ⚠️  CONTRACT: every NEW continuation handler module MUST be added here,
# or its handler will be ABSENT after restart and continuation rows that
# reference it will stay pending until the import is fixed. Forgetting to
# add a module is a silent correctness bug. SP3 / SP4 PRs that introduce
# new resume / on_error handlers MUST include an entry below.
#
# For DYNAMIC additions (tests / lazy plugins), use
# register_continuations_module(name) below — but for production
# restart-recovery, the static tuple is what reconcile sees.
# ──────────────────────────────────────────────────────────────────────────
_HANDLER_MODULES: list[str] = [
    "mr_roboto.executors.analytics_digest",
    "mr_roboto.executors.classify_signals",
]


def register_continuations_module(name: str) -> None:
    """Dynamically register an additional continuation-bearing module path.

    For tests + lazy plugins. The named module will be imported by the
    next register_startup_handlers() call. For PRODUCTION restart-
    recovery you STILL need a static entry in _HANDLER_MODULES — this
    function only affects the current process.
    """
    if name not in _HANDLER_MODULES:
        _HANDLER_MODULES.append(name)


def register_startup_handlers() -> None:
    """Import known continuation-bearing modules so their register() fires.

    The in-memory _HANDLERS registry is empty after restart; reconcile
    must run AFTER this. A missing handler at fire time leaves the
    continuation row pending (see fire_for_task), so a subsequent
    restart that fixes the import can still recover — but operators
    must SEE the import failure, hence WARNING (not DEBUG).
    """
    import importlib
    for mod in _HANDLER_MODULES:
        try:
            m = importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "continuation handler module failed to import — "
                "rows referencing its handlers will stay pending until "
                "the import is fixed and the next restart/reconcile",
                module=mod, error=str(exc),
            )
            continue
        reg = getattr(m, "register_continuations", None)
        if not callable(reg):
            _log.warning(
                "continuation handler module has no register_continuations() — "
                "handlers in this module will be absent on restart",
                module=mod,
            )
            continue
        try:
            reg()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "register_continuations() raised — handlers in this module "
                "may be partially registered",
                module=mod, error=str(exc),
            )


async def reconcile_continuations(ttl_seconds: int = CONTINUATION_TTL_SECONDS) -> None:
    """Startup/periodic recovery pass over pending continuations.

    For each pending row:
      - child terminal (completed/failed) → reconstruct result from tasks.result
        and fire (closes the down-while-child-finished gap);
      - else, if past TTL AND child is not alive (no in_flight entry) → expire
        (fire on_error if set, else log). A still-alive long-runner is left
        pending — no premature abandon.

    Result reconstruction: the persisted tasks.result is JSON-decoded at the
    TOP level only and passed to the handler as-is. Handlers that need nested
    decoding (e.g. raw_dispatch envelopes where content is itself a JSON
    string) must do that decoding inside the handler.
    """
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT child_task_id FROM continuations WHERE status='pending'"
    )
    pending_ids = [r[0] for r in await cur.fetchall()]

    for cid in pending_ids:
        try:
            tcur = await db.execute("SELECT status, result FROM tasks WHERE id=?", (cid,))
            trow = await tcur.fetchone()
            if trow is None:
                continue
            tstatus, tresult = trow[0], trow[1]

            if tstatus in ("completed", "failed"):
                res: dict = {}
                if tresult:
                    try:
                        parsed = json.loads(tresult) if isinstance(tresult, str) else tresult
                        res = dict(parsed) if isinstance(parsed, dict) else {"result": parsed}
                    except Exception:
                        res = {"result": tresult}
                res.setdefault("status", tstatus)
                await fire_for_task(cid, res, tstatus)
                continue

            # Not terminal — TTL + alive check.
            ecur = await db.execute(
                "SELECT 1 FROM continuations WHERE child_task_id=? "
                "AND datetime(created_at, '+' || ? || ' seconds') < datetime('now')",
                (cid, ttl_seconds),
            )
            if await ecur.fetchone() is None:
                continue  # not yet expired

            alive = False
            try:
                from src.core.in_flight import in_flight_snapshot
                alive = any(getattr(e, "task_id", None) == cid for e in in_flight_snapshot())
            except Exception:
                alive = False
            if alive:
                continue  # long-runner — leave pending

            claim = await claim_for_fire(cid)
            if claim is None:
                continue
            name = claim["on_error_name"]
            if name:
                asyncio.create_task(dispatch_on_complete(
                    name, cid,
                    {"status": "failed", "error": "continuation TTL expired"},
                    claim["state"],
                ))
            else:
                _log.warning("continuation expired (no on_error)", child_task_id=cid)
        except Exception as _row_exc:  # noqa: BLE001
            _log.warning(
                "reconcile row failed",
                child_task_id=cid,
                error=str(_row_exc),
            )
