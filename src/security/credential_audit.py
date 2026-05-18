# src/security/credential_audit.py
"""Audit logger for the credential vault (T2C).

Writes one row to ``credential_access_log`` per get/store/rotate/delete on
the vault. The mission/task/agent/model context flows in via
:mod:`src.security._audit_context` so callers don't have to thread kwargs
through every layer.

Failures here NEVER raise — audit logging is best-effort and must not block
a credential read or write. Errors are surfaced via the standard logger.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from src.infra.logging_config import get_logger

from . import _audit_context

logger = get_logger("security.credential_audit")

Action = Literal["read", "write", "rotate", "delete"]


async def log_access(
    service_name: str,
    action: Action,
    success: bool,
    *,
    mission_id: int | None = None,
    task_id: int | None = None,
    agent: str | None = None,
    model_id: str | None = None,
    scope: str | None = None,
    error: str | None = None,
) -> None:
    """Append a row to ``credential_access_log``.

    Any kwarg left as ``None`` is filled from the ambient ``audit_context``
    so most callers only need ``(service_name, action, success)``.
    """
    ctx = _audit_context.current()
    mission_id = mission_id if mission_id is not None else ctx.mission_id
    task_id = task_id if task_id is not None else ctx.task_id
    agent = agent if agent is not None else ctx.agent
    model_id = model_id if model_id is not None else ctx.model_id

    accessed_at = datetime.now(timezone.utc).isoformat()
    try:
        from ..infra.db import get_db

        db = await get_db()
        await db.execute(
            "INSERT INTO credential_access_log ("
            " service_name, mission_id, task_id, agent, model_id,"
            " action, scope, success, error, accessed_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                service_name,
                mission_id,
                task_id,
                agent,
                model_id,
                action,
                scope,
                1 if success else 0,
                error,
                accessed_at,
            ),
        )
        await db.commit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            f"credential audit log failed for {service_name}/{action}: {exc}",
            service=service_name,
            action=action,
            error=str(exc),
        )


async def recent_events(
    service_name: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return the latest ``limit`` rows, newest first.

    When *service_name* is given, scope to that service.
    """
    from ..infra.db import get_db

    db = await get_db()
    if service_name:
        cur = await db.execute(
            "SELECT service_name, mission_id, task_id, agent, model_id,"
            " action, scope, success, error, accessed_at "
            "FROM credential_access_log "
            "WHERE service_name = ? "
            "ORDER BY id DESC LIMIT ?",
            (service_name, limit),
        )
    else:
        cur = await db.execute(
            "SELECT service_name, mission_id, task_id, agent, model_id,"
            " action, scope, success, error, accessed_at "
            "FROM credential_access_log "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    rows = await cur.fetchall()
    cols = [
        "service_name", "mission_id", "task_id", "agent", "model_id",
        "action", "scope", "success", "error", "accessed_at",
    ]
    out: list[dict] = []
    for row in rows:
        if isinstance(row, tuple):
            out.append(dict(zip(cols, row)))
        else:
            out.append({k: row[k] for k in cols})
    return out
