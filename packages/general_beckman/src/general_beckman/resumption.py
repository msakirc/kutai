"""Z8 T1C — orchestrator resumption + revocation for ongoing missions.

A *resumable* mission is one with ``kind='ongoing'`` and
``lifecycle_state='active'`` and ``revoked_at IS NULL``. On orchestrator
boot, ``find_resumable()`` returns the list of survivors; the
orchestrator logs them and lets handler-side code (webhook listener,
cron scheduler) replay from each mission's cursor.

The cursor is opaque JSON owned by the handler. ``update_cursor()``
persists progress (last webhook event id, last cron fire ts, etc.) so a
crash mid-flight resumes at the right point.

``revoke()`` is called by the ``/stop_ops`` Telegram command (T1D); it
transitions ``lifecycle_state`` → ``'revoked'`` and stamps
``revoked_at`` so subsequent ``find_resumable()`` calls skip the
mission.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ResumedMission:
    id: int
    title: str
    cursor: dict[str, Any]


async def find_resumable() -> list[ResumedMission]:
    """Return every ``kind='ongoing' AND lifecycle_state='active'`` mission
    that has not been revoked.

    Cursor JSON is parsed into a dict; an empty/NULL cursor returns
    ``{}`` so callers can always ``.get(...)``.
    """
    from src.infra.db import get_db
    out: list[ResumedMission] = []
    db = await get_db()
    cur = await db.execute(
        "SELECT id, title, cursor FROM missions "
        "WHERE kind='ongoing' AND lifecycle_state='active' "
        "AND revoked_at IS NULL"
    )
    rows = await cur.fetchall()
    for row in rows:
        raw_cursor = row[2]
        if raw_cursor:
            try:
                parsed = json.loads(raw_cursor)
                if not isinstance(parsed, dict):
                    parsed = {}
            except (json.JSONDecodeError, TypeError):
                parsed = {}
        else:
            parsed = {}
        out.append(ResumedMission(id=row[0], title=row[1] or "", cursor=parsed))
    return out


async def update_cursor(mission_id: int, cursor: dict) -> None:
    """Persist a handler-owned cursor blob onto the mission row.

    The cursor is opaque to Beckman; callers may store webhook event ids,
    last cron fire ts, "pages consumed" counters, etc. No schema is
    enforced — just JSON-encodable.
    """
    from src.infra.db import get_db
    db = await get_db()
    await db.execute(
        "UPDATE missions SET cursor=? WHERE id=?",
        (json.dumps(cursor), mission_id),
    )
    await db.commit()


async def revoke(mission_id: int) -> bool:
    """Mark an ongoing mission revoked. No-op for oneshot / missing rows.

    Returns ``True`` if exactly one row transitioned; ``False`` otherwise
    (caller surfaces "not ongoing or not found" to the founder).
    """
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "UPDATE missions SET lifecycle_state='revoked', "
        "revoked_at=datetime('now') "
        "WHERE id=? AND kind='ongoing' AND lifecycle_state != 'revoked'",
        (mission_id,),
    )
    await db.commit()
    # aiosqlite cursor exposes rowcount on the underlying cursor object.
    try:
        rc = cur.rowcount
    except AttributeError:
        rc = -1
    transitioned = rc == 1
    if transitioned:
        # Z8 T5-prep — tear down any cron schedules armed for this mission.
        try:
            from general_beckman.mission_cron import disarm
            await disarm(mission_id)
        except Exception:
            pass
    return transitioned
