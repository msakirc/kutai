"""yalayut_usage telemetry writes.

flash() emits one row per artifact it considered: exposed entries carry
their exposure_class + bind_args_json; conflict-losers (a same-slot,
same-kind collision outranked by a sibling) carry conflict_loser=1.
Telemetry never raises — a logging failure must not break dispatch.
"""
from __future__ import annotations

import json

from src.infra.logging_config import get_logger

logger = get_logger("intersect.telemetry")


async def record_usage(
    *,
    task_id: str,
    exposed: list[dict],
    conflict_losers: list[dict],
) -> None:
    """Write yalayut_usage rows for one flash() call. Never raises."""
    try:
        from dabidabi import get_db
        db = await get_db()
        for app in exposed:
            await db.execute(
                "INSERT INTO yalayut_usage "
                "(artifact_id, task_id, exposure_class, bind_args_json, "
                " exposed, called, succeeded, conflict_loser, occurred_at) "
                "VALUES (?, ?, ?, ?, 1, 0, 0, 0, datetime('now'))",
                (
                    app.get("artifact_id"),
                    str(task_id),
                    app.get("exposure_class"),
                    json.dumps(app["bind_args"], ensure_ascii=False)
                    if app.get("bind_args") is not None else None,
                ),
            )
        for loser in conflict_losers:
            await db.execute(
                "INSERT INTO yalayut_usage "
                "(artifact_id, task_id, exposure_class, exposed, called, "
                " succeeded, conflict_loser, occurred_at) "
                "VALUES (?, ?, ?, 0, 0, 0, 1, datetime('now'))",
                (
                    loser.get("artifact_id"),
                    str(task_id),
                    loser.get("exposure_class"),
                ),
            )
        await db.commit()
    except Exception as exc:
        logger.debug("yalayut_usage write failed: %s", exc)
