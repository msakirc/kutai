"""Scheduled-tasks processor.

Reads due rows from scheduled_tasks, dispatches marker payloads internally
(sweep, benchmark refresh, nerd_herd health) and inserts concrete task
rows for non-marker payloads.
"""
from __future__ import annotations

import json
from datetime import timedelta

from src.infra.logging_config import get_logger
from src.infra.times import utc_now, to_db

from general_beckman.apply import _mechanical_context
from general_beckman.cron_seed import seed_internal_cadences
from general_beckman.sweep import sweep_queue

logger = get_logger("beckman.cron")


async def fire_due() -> None:
    """Fire every scheduled_tasks row whose next_run is due.

    Called from beckman.next_task(). Idempotent per row via last_run/next_run
    advancement.
    """
    from src.infra.db import get_due_scheduled_tasks

    await seed_internal_cadences()
    rows = await get_due_scheduled_tasks()
    now = utc_now()
    for row in rows:
        try:
            payload = _parse_payload(row.get("context"))
            marker = payload.get("_marker")
            if marker == "sweep":
                await sweep_queue()
            elif marker == "benchmark_refresh":
                await _refresh_benchmarks_if_stale()
            elif marker == "nerd_herd_health":
                await _nerd_herd_health_alert()
            else:
                await _insert_scheduled_task(row, payload)
            await _advance_schedule(row, now)
        except Exception as e:
            logger.warning("cron fire failed",
                           sched_id=row.get("id"),
                           title=row.get("title"),
                           error=str(e))


def _parse_payload(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


async def _insert_scheduled_task(row: dict, payload: dict) -> None:
    from src.infra.db import add_task

    executor = payload.get("_executor")
    if executor:
        extra = {k: v for k, v in payload.items() if k != "_executor"}
        await add_task(
            title=row.get("title", "scheduled"),
            description=row.get("description", ""),
            agent_type="mechanical",
            context=_mechanical_context(executor, **extra),
            depends_on=[],
        )
    else:
        # user-scheduled row with an agent_type — insert as that agent.
        await add_task(
            title=row.get("title", "scheduled"),
            description=row.get("description", ""),
            agent_type=row.get("agent_type", "executor"),
            context=payload,
            depends_on=[],
        )


async def _advance_schedule(row: dict, now) -> None:
    from src.infra.db import update_scheduled_task

    interval = row.get("interval_seconds")
    cron_expr = row.get("cron_expression")
    if interval:
        next_run = to_db(now + timedelta(seconds=int(interval)))
    elif cron_expr:
        # Reuse add_scheduled_task's inline parser via croniter if available,
        # else advance by 1h as a conservative fallback.
        try:
            from croniter import croniter
            next_run = to_db(croniter(cron_expr, now).get_next(type(now)))
        except Exception:
            next_run = to_db(now + timedelta(hours=1))
    else:
        next_run = to_db(now + timedelta(hours=1))
    await update_scheduled_task(row["id"], last_run=to_db(now), next_run=next_run)


async def _refresh_benchmarks_if_stale() -> None:
    try:
        import fatih_hoca
        hoca = getattr(fatih_hoca, "refresh_benchmarks_if_stale", None)
        if hoca is not None:
            await hoca()
    except Exception as e:
        logger.debug("hoca benchmark refresh skipped", error=str(e))


async def _nerd_herd_health_alert() -> None:
    from src.infra.db import add_task

    try:
        import nerd_herd
        summary = getattr(nerd_herd, "health_summary", None)
        if summary is None:
            return
        report = await summary() if callable(summary) else summary
        if not report or not report.get("alerts"):
            return
        await add_task(
            title="Notify: resource health",
            description="",
            agent_type="mechanical",
            context=_mechanical_context(
                "notify_user",
                message="\n".join(f"\u2022 {a}" for a in report["alerts"]),
            ),
            depends_on=[],
        )
    except Exception as e:
        logger.debug("nerd_herd health alert skipped", error=str(e))
