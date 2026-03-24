# infra/audit.py
"""
Phase 8.4 — Audit Trail

Append-only audit log stored in DB. Records tool executions, model calls,
state transitions, file modifications, and human approvals.
"""
from __future__ import annotations
from typing import Any, Optional

from .logging_config import get_logger
from .db import get_db

logger = get_logger("infra.audit")

# Actor types
ACTOR_AGENT  = "agent"
ACTOR_SYSTEM = "system"
ACTOR_HUMAN  = "human"

# Action types
ACTION_TOOL_EXEC       = "tool_exec"
ACTION_MODEL_CALL      = "model_call"
ACTION_STATE_CHANGE    = "state_change"
ACTION_FILE_MODIFY     = "file_modify"
ACTION_HUMAN_APPROVE   = "human_approve"
ACTION_MISSION_CREATE  = "mission_create"
ACTION_MISSION_COMPLETE = "mission_complete"
ACTION_TASK_CREATE     = "task_create"
ACTION_TASK_COMPLETE   = "task_complete"


async def _ensure_table(db) -> None:
    """Ensure audit_log table exists."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT,
            details TEXT,
            task_id INTEGER,
            mission_id INTEGER
        )
    """)
    try:
        await db.execute("ALTER TABLE audit_log RENAME COLUMN goal_id TO mission_id")
    except Exception:
        pass
    try:
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_task ON audit_log(task_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_mission ON audit_log(mission_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor)")
        await db.commit()
    except Exception:
        pass


async def audit(
    actor: str,
    action: str,
    target: str = "",
    details: str = "",
    task_id: Optional[int] = None,
    mission_id: Optional[int] = None,
) -> None:
    """Append an audit log entry. Silently ignores failures."""
    try:
        db = await get_db()
        await _ensure_table(db)
        await db.execute(
            """INSERT INTO audit_log (actor, action, target, details, task_id, mission_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (actor, action, target, details[:2000] if details else "", task_id, mission_id),
        )
        await db.commit()
    except Exception as exc:
        logger.debug(f"Audit log write failed (non-critical): {exc}")


async def get_audit_log(
    task_id: Optional[int] = None,
    mission_id: Optional[int] = None,
    actor: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Query audit log with optional filters."""
    try:
        db = await get_db()
        await _ensure_table(db)

        conditions = []
        params: list[Any] = []
        if task_id is not None:
            conditions.append("task_id = ?")
            params.append(task_id)
        if mission_id is not None:
            conditions.append("mission_id = ?")
            params.append(mission_id)
        if actor:
            conditions.append("actor = ?")
            params.append(actor)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        cursor = await db.execute(
            f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []


def format_audit_log(entries: list[dict]) -> str:
    """Format audit log entries as a readable string."""
    if not entries:
        return "_No audit entries found._"

    lines = []
    for e in reversed(entries):  # oldest first
        ts = str(e.get("timestamp", ""))[:16]
        actor = e.get("actor", "?")
        action = e.get("action", "?")
        target = e.get("target", "")
        details = e.get("details", "")[:100]
        line = f"`{ts}` [{actor}] {action}"
        if target:
            line += f" → {target}"
        if details:
            line += f": {details}"
        lines.append(line)
    return "\n".join(lines)
