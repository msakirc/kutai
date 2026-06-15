"""Z9 T2B — ``arm_analytics_digest`` mechanical executor.

Routed via ``mr_roboto.run`` when ``payload["action"] == "arm_analytics_digest"``.
Wired as the final mechanical step of i2p Phase 14 (launch). Arms a weekly
``analytics_digest`` cron for the mission so post-launch the founder receives
a recurring growth digest.

Idempotency / restart-proofing
------------------------------
``mission_cron`` schedules are runtime-only (``asyncio`` tasks); they do not
survive an orchestrator restart on their own. The durable source of truth is
``missions.cursor['cron']`` — a list of ``{action, interval_seconds}`` entries
that ``orchestrator._rebind_ongoing`` replays via ``arm_from_cursor`` on boot.

This executor therefore does BOTH:

1. Appends the ``analytics_digest`` schedule to ``mission.cursor['cron']``
   (durable). The append is guarded — re-running the step (or re-launching
   the mission) never produces a duplicate entry.
2. Calls ``mission_cron.arm()`` directly (live arming for the current
   process, which is itself idempotent at the slot level).

This mirrors how Z8 ongoing-mission cadences self-arm: cursor entry is the
canonical record, ``arm`` is the live binding.
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.arm_analytics_digest")

_DIGEST_ACTION = "analytics_digest"
_WEEKLY_SECONDS = 604800


def _parse_cursor(raw: Any) -> dict:
    """Decode a missions.cursor value into a dict (cursor is opaque JSON)."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            return loaded if isinstance(loaded, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _has_digest_entry(cron_entries: list) -> bool:
    """True when an analytics_digest schedule is already in the cron list."""
    for entry in cron_entries:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action") or entry.get("task_type")
        if action == _DIGEST_ACTION:
            return True
    return False


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Arm the weekly analytics_digest cron for this mission. Never raises."""
    mission_id = task.get("mission_id")
    try:
        mission_id_int = int(mission_id) if mission_id is not None else None
    except (TypeError, ValueError):
        mission_id_int = None

    if mission_id_int is None:
        return {"ok": False, "reason": "no_mission_id"}

    cursor_written = False
    already_armed = False

    # 1. Durable: append the schedule to missions.cursor['cron'] (idempotent).
    try:
        from dabidabi import get_db

        db = await get_db()
        cur = await db.execute(
            "SELECT cursor FROM missions WHERE id = ?", (mission_id_int,)
        )
        row = await cur.fetchone()
        cursor = _parse_cursor(row[0] if row else None)

        cron_entries = cursor.get("cron")
        if not isinstance(cron_entries, list):
            cron_entries = []

        if _has_digest_entry(cron_entries):
            already_armed = True
        else:
            cron_entries.append(
                {"action": _DIGEST_ACTION, "interval_seconds": _WEEKLY_SECONDS}
            )
            cursor["cron"] = cron_entries
            from general_beckman import update_mission_fields as _umf
            await _umf(mission_id_int, cursor=json.dumps(cursor))
            cursor_written = True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "arm_analytics_digest cursor write failed",
            mission_id=mission_id_int,
            error=str(exc),
        )

    # 2. Live: arm the cron for the current process (arm() is idempotent).
    try:
        from general_beckman.mission_cron import arm

        await arm(mission_id_int, _DIGEST_ACTION, _WEEKLY_SECONDS)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "arm_analytics_digest live arm failed",
            mission_id=mission_id_int,
            error=str(exc),
        )
        return {
            "ok": False,
            "reason": "arm_failed",
            "error": str(exc),
            "cursor_written": cursor_written,
        }

    logger.info(
        "arm_analytics_digest complete",
        mission_id=mission_id_int,
        cursor_written=cursor_written,
        already_armed=already_armed,
    )
    return {
        "ok": True,
        "mission_id": mission_id_int,
        "action": _DIGEST_ACTION,
        "interval_seconds": _WEEKLY_SECONDS,
        "cursor_written": cursor_written,
        "already_armed": already_armed,
    }


__all__ = ["run"]
