"""Mission lifecycle state transitions + audit log."""
from __future__ import annotations

import logging
from dabidabi import get_db

logger = logging.getLogger(__name__)


async def _transition(
    mission_id: int,
    from_states: tuple[str, ...],
    to_state: str,
    reason: str,
    triggered_by: str,
) -> bool:
    """Atomic transition. Returns True if state changed, False if no-op."""
    db = await get_db()
    placeholders = ",".join("?" * len(from_states))
    cur = await db.execute(
        f"UPDATE missions SET lifecycle_state = ? "
        f"WHERE id = ? AND lifecycle_state IN ({placeholders})",
        (to_state, mission_id, *from_states),
    )
    changed = cur.rowcount > 0
    if changed:
        await db.execute(
            "INSERT INTO mission_lifecycle_log (mission_id, from_state, to_state, reason, triggered_by) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                mission_id,
                from_states[0] if len(from_states) == 1 else "any",
                to_state,
                reason,
                triggered_by,
            ),
        )
    await db.commit()
    return changed


async def emit_pause(mission_id: int, reason: str, triggered_by: str = "auto") -> bool:
    return await _transition(mission_id, ("active",), "paused", reason, triggered_by)


async def emit_resume(mission_id: int, reason: str = "founder", triggered_by: str = "founder") -> bool:
    return await _transition(mission_id, ("paused",), "active", reason, triggered_by)


async def emit_kill(mission_id: int, reason: str = "founder", triggered_by: str = "founder") -> bool:
    return await _transition(mission_id, ("active", "paused"), "killed", reason, triggered_by)


async def emit_complete(mission_id: int, reason: str = "all_tasks_done", triggered_by: str = "auto") -> bool:
    return await _transition(mission_id, ("active",), "completed", reason, triggered_by)


async def dlq_cascade_check(mission_id: int) -> bool:
    """If the last 3 consecutive completed tasks for mission failed, pause.

    Resets on any successful completion.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT status FROM tasks WHERE mission_id = ? AND completed_at IS NOT NULL "
        "ORDER BY completed_at DESC LIMIT 3",
        (mission_id,),
    )
    rows = await cur.fetchall()
    if len(rows) < 3:
        return False
    if all(r[0] == "failed" for r in rows):
        return await emit_pause(mission_id, "dlq_cascade", "auto:dlq")
    return False
