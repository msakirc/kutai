"""Z8 T5-prep — cron scheduler for ongoing missions.

Per-mission cron registers ``arm(mission_id, action, interval_seconds)``
which spawns a loop that enqueues a mechanical task on Beckman's
ongoing lane every ``interval_seconds``. ``disarm`` cancels the loop.

This replaces the vestigial ``scheduled_tasks`` flow for ongoing-mission
cadences: cron config now lives on ``missions.cursor`` (key ``cron``)
as a list of ``{"action": str, "interval_seconds": int}`` entries. On
orchestrator boot, ``arm_from_cursor(mission)`` reads the cursor and
arms each registered schedule.

The legacy ``general_beckman.cron`` (``fire_due``) still services the
``scheduled_tasks`` table — it remains in place until that table is
fully retired.
"""
from __future__ import annotations

import asyncio
from typing import Any

from yazbunu import get_logger

logger = get_logger("beckman.mission_cron")

# mission_id -> {action_key: asyncio.Task}
_TASKS: dict[int, dict[str, "asyncio.Task[Any]"]] = {}


def _action_key(action: str, interval_seconds: int) -> str:
    return f"{action}@{int(interval_seconds)}"


async def arm(mission_id: int, action: str, interval_seconds: int) -> None:
    """Arm a cron schedule for a mission.

    If the (action, interval) combination is already armed for this
    mission, the existing task is cancelled and replaced. Safe to call
    repeatedly (idempotent at the slot level).
    """
    key = _action_key(action, interval_seconds)
    slot = _TASKS.setdefault(mission_id, {})
    existing = slot.get(key)
    if existing is not None and not existing.done():
        existing.cancel()
    slot[key] = asyncio.create_task(
        _loop(mission_id, action, int(interval_seconds))
    )
    logger.debug(
        "mission_cron.arm: mid=%s action=%s interval=%ss",
        mission_id, action, interval_seconds,
    )


async def disarm(mission_id: int) -> int:
    """Cancel every armed schedule for a mission. Returns count cancelled."""
    slot = _TASKS.pop(mission_id, None)
    if not slot:
        return 0
    cancelled = 0
    for t in slot.values():
        if not t.done():
            t.cancel()
            cancelled += 1
    if cancelled:
        logger.debug(
            "mission_cron.disarm: mid=%s cancelled=%s",
            mission_id, cancelled,
        )
    return cancelled


def is_armed(mission_id: int, action: str | None = None) -> bool:
    """Test helper: is the mission (and optionally specific action) armed?"""
    slot = _TASKS.get(mission_id)
    if not slot:
        return False
    if action is None:
        return any(not t.done() for t in slot.values())
    return any(
        k.startswith(f"{action}@") and not t.done() for k, t in slot.items()
    )


async def _loop(mission_id: int, action: str, interval_seconds: int) -> None:
    """Background loop: enqueue mechanical task every interval_seconds."""
    from general_beckman import enqueue
    from general_beckman.lanes import LANE_ONGOING
    from general_beckman.apply import _mechanical_context

    while True:
        try:
            await enqueue(
                {
                    "title": f"cron: {action} (mid={mission_id})",
                    "description": "",
                    "agent_type": "mechanical",
                    "context": _mechanical_context(action, mission_id=mission_id),
                    "depends_on": [],
                    "mission_id": mission_id,
                },
                lane=LANE_ONGOING,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(
                "mission_cron.enqueue failed mid=%s action=%s: %s",
                mission_id, action, e,
            )
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise


async def arm_from_cursor(mission: Any) -> int:
    """Read ``mission.cursor['cron']`` and arm each schedule entry.

    ``mission.cursor`` may be a dict (post-T1C ResumedMission) or absent.
    Each entry must be a dict with keys ``action`` and
    ``interval_seconds``. Bad entries are logged and skipped — one bad
    schedule must not block the rest.

    Returns the number of schedules armed.
    """
    cursor = getattr(mission, "cursor", None) or {}
    if not isinstance(cursor, dict):
        return 0
    entries = cursor.get("cron") or []
    if not isinstance(entries, list):
        logger.warning(
            "mission_cron.arm_from_cursor: cursor.cron is not a list for mid=%s",
            getattr(mission, "id", None),
        )
        return 0

    armed = 0
    mid = getattr(mission, "id", None)
    if mid is None:
        return 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action") or entry.get("task_type")
        interval = entry.get("interval_seconds") or entry.get("interval")
        if not action or not interval:
            logger.warning(
                "mission_cron.arm_from_cursor: bad entry mid=%s entry=%r",
                mid, entry,
            )
            continue
        try:
            await arm(int(mid), str(action), int(interval))
            armed += 1
        except Exception as e:
            logger.warning(
                "mission_cron.arm_from_cursor: arm failed mid=%s action=%s: %s",
                mid, action, e,
            )
    return armed


def _reset_for_tests() -> None:
    """Test-only: cancel everything and clear the registry."""
    for slot in _TASKS.values():
        for t in slot.values():
            if not t.done():
                t.cancel()
    _TASKS.clear()
