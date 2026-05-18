# src/security/_audit_context.py
"""ContextVar-backed audit context for credential vault access.

Callers (HttpIntegration, vendor_call, expander, …) wrap a credential read
in :func:`audit_context` so the audit logger can attach mission/task/agent/
model metadata to the resulting `credential_access_log` row without
threading kwargs through every layer.

Example::

    async with audit_context(mission_id=42, task_id=101, agent="executor"):
        creds = await get_credential("github")
"""

from __future__ import annotations

import contextvars
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class AuditCtx:
    mission_id: int | None = None
    task_id: int | None = None
    agent: str | None = None
    model_id: str | None = None


_EMPTY = AuditCtx()

_current: contextvars.ContextVar[AuditCtx] = contextvars.ContextVar(
    "kutay_credential_audit_ctx", default=_EMPTY
)


def current() -> AuditCtx:
    """Return the AuditCtx attached to the current async task (or empty)."""
    return _current.get()


@asynccontextmanager
async def audit_context(
    *,
    mission_id: int | None = None,
    task_id: int | None = None,
    agent: str | None = None,
    model_id: str | None = None,
):
    """Async context manager that sets the audit context for its block.

    Nested calls merge: any field passed here overrides the outer value;
    unspecified fields fall through to the outer context.
    """
    outer = _current.get()
    merged = AuditCtx(
        mission_id=mission_id if mission_id is not None else outer.mission_id,
        task_id=task_id if task_id is not None else outer.task_id,
        agent=agent if agent is not None else outer.agent,
        model_id=model_id if model_id is not None else outer.model_id,
    )
    token = _current.set(merged)
    try:
        yield merged
    finally:
        _current.reset(token)


def reset() -> None:
    """Clear the audit context for the current task (test helper)."""
    _current.set(_EMPTY)
