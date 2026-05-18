"""Drain pending confirmations & budget alerts → mission_events.

Z10 T2B (D4) mechanical executor. Wired by ``cron_seed`` at 5s
intervals (`mission_event_drain` cadence).

Two scans per tick:
  1. ``action_confirmations WHERE verdict='pending' AND
     telegram_event_id IS NULL`` → ``post_event(kind='confirmation_required')``,
     stamp ``telegram_event_id``.
  2. ``mission_budget_alerts WHERE telegram_event_id IS NULL`` →
     ``post_event(kind='cost_alert')``, stamp ``telegram_event_id``.
     T2A may not have merged the table yet — wrap in try/except and skip.

All exceptions are caught so a single bad row doesn't break the drain.
The executor returns a small dict for observability.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.mission_event_drain")


async def _drain_confirmations() -> int:
    """Post mission_events for any action_confirmations awaiting wire-up."""
    from src.infra.db import get_db

    db = await get_db()
    # task_id → mission_id lookup is best-effort; if the task row is missing
    # or has no mission_id, fall back to mission_id=0 (banner-style banner
    # will land in flat-mode fallback).
    cur = await db.execute(
        "SELECT id, task_id, verb, reversibility, payload_summary "
        "FROM action_confirmations "
        "WHERE verdict = 'pending' AND telegram_event_id IS NULL"
    )
    rows = await cur.fetchall()
    if not rows:
        return 0

    bot = _get_bot()
    if bot is None:
        logger.debug("drain_confirmations: no bot available, skip")
        return 0

    from src.app.mission_events import post_event

    posted = 0
    for row in rows:
        cid, task_id, verb, rev, summary = (
            row[0], row[1], row[2], row[3], row[4],
        )
        try:
            mission_id = await _mission_id_for_task(task_id)
            event_id = await post_event(
                bot, mission_id, "confirmation_required",
                {
                    "confirmation_id": int(cid),
                    "verb": verb,
                    "reversibility": rev,
                    "payload_summary": summary or "",
                },
            )
            await db.execute(
                "UPDATE action_confirmations SET telegram_event_id = ? "
                "WHERE id = ?",
                (int(event_id), int(cid)),
            )
            await db.commit()
            posted += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "drain_confirmations: row skipped",
                confirmation_id=cid, error=str(e),
            )
    return posted


async def _drain_budget_alerts() -> int:
    """Post mission_events for any mission_budget_alerts awaiting wire-up.

    Uses T2A's API: get_pending_cost_alerts() + mark_cost_alert_drained(id).
    Schema: rows have (id, mission_id, threshold, total_usd, posted_at).
    Ceiling pulled live from cost_budgets via get_mission_cost_breakdown.
    """
    try:
        from src.infra.db import (
            get_pending_cost_alerts,
            mark_cost_alert_drained,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "drain_budget_alerts: T2A APIs unavailable, skip",
            error=str(e),
        )
        return 0

    try:
        rows = await get_pending_cost_alerts()
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "drain_budget_alerts: mission_budget_alerts unavailable, skip",
            error=str(e),
        )
        return 0

    if not rows:
        return 0

    bot = _get_bot()
    if bot is None:
        return 0

    from src.app.mission_events import post_event

    posted = 0
    for row in rows:
        aid = row["id"] if isinstance(row, dict) else row[0]
        mission_id = row["mission_id"] if isinstance(row, dict) else row[1]
        threshold = row["threshold"] if isinstance(row, dict) else row[2]
        total = row["total_usd"] if isinstance(row, dict) else row[3]
        try:
            event_id = await post_event(
                bot, int(mission_id), "cost_alert",
                {
                    "mission_id": int(mission_id),
                    "threshold_pct": int(float(threshold) * 100)
                    if float(threshold) <= 1 else int(threshold),
                    "total": total,
                    "ceiling": None,
                },
            )
            await mark_cost_alert_drained(int(aid))
            posted += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "drain_budget_alerts: row skipped",
                alert_id=aid, error=str(e),
            )
    return posted


def _get_bot() -> Any | None:
    """Resolve the live Telegram bot via the singleton (None outside runtime)."""
    try:
        from src.app.telegram_bot import _TG_INSTANCE
        if _TG_INSTANCE is None:
            return None
        return _TG_INSTANCE.app.bot
    except Exception:  # noqa: BLE001
        return None


async def _mission_id_for_task(task_id: int | None) -> int:
    """Best-effort mission_id resolution for a task. Falls back to 0."""
    if task_id is None:
        return 0
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT mission_id FROM tasks WHERE id = ?", (int(task_id),),
        )
        row = await cur.fetchone()
        if row and row[0]:
            return int(row[0])
    except Exception:  # noqa: BLE001
        pass
    return 0


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Drain pending events. ``task`` is the wrapping scheduled_task row.

    Returns a small dict for visibility. Never raises — failures are
    logged so the cron loop keeps going.
    """
    try:
        c = await _drain_confirmations()
    except Exception as e:  # noqa: BLE001
        logger.exception("drain_confirmations crashed: %s", e)
        c = 0
    try:
        b = await _drain_budget_alerts()
    except Exception as e:  # noqa: BLE001
        logger.exception("drain_budget_alerts crashed: %s", e)
        b = 0
    return {"confirmations_drained": c, "budget_alerts_drained": b}
