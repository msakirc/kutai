"""Z7 B9 — audit_completeness_check posthook handler.

Asserts that every external send (vendor_call with reversibility != 'full')
produced an `external_comms_log` row within the audit window. Gaps mean a
publish/send/upload happened without an immutable audit record — a
compliance + post-incident-review hole.

Handler contract
----------------
``handle(task, result) -> dict``

Returns:

- ``{"status": "ok"}``                       — no audit gaps
- ``{"status": "skip", "reason": "..."}``     — audit subsystem unavailable
- ``{"status": "warning", "reason": "...", "gaps": [...]}`` — gaps found
  (registry default_severity is "warning" — advisory, never blocks a mission)
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.audit_completeness_check")

# external sends should land an audit row within this many minutes
AUDIT_WINDOW_MINUTES = 5


async def handle(task: dict, result: dict) -> dict:
    """Check for external sends missing an audit-log row."""
    try:
        from mr_roboto.audit_log import pending_audit_gaps
    except Exception as exc:  # audit subsystem not wired
        logger.debug("audit_completeness_check skipped: %s", exc)
        return {"status": "skip", "reason": f"audit_log unavailable: {exc}"}

    try:
        gaps = await pending_audit_gaps(window_minutes=AUDIT_WINDOW_MINUTES)
    except Exception as exc:
        logger.warning("audit_completeness_check failed: %s", exc)
        return {"status": "skip", "reason": f"gap scan failed: {exc}"}

    if not gaps:
        return {"status": "ok"}

    logger.warning(
        "audit_completeness_check found %d external send(s) with no audit row",
        len(gaps),
        task_id=task.get("id"),
        mission_id=task.get("mission_id"),
    )
    return {
        "status": "warning",
        "reason": (
            f"{len(gaps)} external send(s) lack an external_comms_log row "
            f"within {AUDIT_WINDOW_MINUTES}min"
        ),
        "gaps": gaps,
    }
