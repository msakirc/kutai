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
            elif marker == "btable_rollup":
                await _btable_rollup()
            elif marker == "file_locks_sweep":
                await _file_locks_sweep()
            elif marker == "mission_budget_alerts":
                await _mission_budget_alerts()
            elif marker == "mission_pacing_check":
                await _mission_pacing_check()
            elif marker == "confidence_calibration_recompute":
                await _confidence_calibration_recompute()
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


async def _file_locks_sweep() -> None:
    """Z10 T1A — release orphan file_locks rows.

    Defensive: file_locks lives in src.infra.db, which is the same DB this
    cron pump talks to. Failures are logged at warning, never crash the
    pump.
    """
    try:
        from src.infra.db import sweep_file_locks
        n = await sweep_file_locks()
        if n:
            logger.info("file_locks_sweep released orphans", count=n)
    except Exception as e:
        logger.warning("file_locks_sweep failed", error=str(e))


async def _mission_budget_alerts() -> None:
    """Z10 T2A — write mission_budget_alerts rows at threshold breaches.

    Idempotent via UNIQUE(mission_id, threshold). T2B drains.
    """
    try:
        from src.infra.db import check_and_write_mission_budget_alerts
        n = await check_and_write_mission_budget_alerts()
        if n:
            logger.info("mission_budget_alerts wrote rows", count=n)
    except Exception as e:
        logger.warning("mission_budget_alerts failed", error=str(e))


async def _mission_pacing_check() -> None:
    """Z10 T3A — pacing tradeoff prompt at 75% burn + 25% scope remaining.

    For every mission with status in {active, processing} and a non-NULL
    ``time_budget_hours``, compute pacing. If ``tradeoff_due`` is True
    and no row exists for ``(mission_id, today)`` in
    ``mission_tradeoff_prompts``, build a 30%-cut suggestion and post
    a single ``[asking]`` event via ``post_event``. UNIQUE index keeps
    it idempotent across cron ticks.
    """
    try:
        from src.infra.mission_pacing_cron import check_and_post_tradeoff_prompts
        n = await check_and_post_tradeoff_prompts()
        if n:
            logger.info("mission_pacing_check posted prompts", count=n)
    except Exception as e:
        logger.warning("mission_pacing_check failed", error=str(e))


async def _confidence_calibration_recompute() -> None:
    """Z10 T4B — aggregate confidence_outcomes → reliability_scores.

    After the rollup, push the matrix into the coulson prompt-builder
    cache so freshly-tuned scores influence the next prompt assembly
    without waiting for a process restart.
    """
    try:
        from src.infra.db import recompute_reliability_scores
        n = await recompute_reliability_scores()
        logger.info("confidence_calibration_recompute wrote rows", rows=n)
        try:
            from coulson.context import refresh_calibration_cache
            await refresh_calibration_cache()
        except Exception as e:
            logger.debug("calibration cache refresh failed", error=str(e))
    except Exception as e:
        logger.warning("confidence_calibration_recompute failed", error=str(e))


async def _btable_rollup() -> None:
    try:
        from general_beckman.btable_rollup import run_rollup
        n = await run_rollup()
        logger.info("btable_rollup complete", rows_written=n)
    except Exception as e:
        logger.warning("btable_rollup failed", error=str(e))


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
