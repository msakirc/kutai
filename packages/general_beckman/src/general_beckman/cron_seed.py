"""Internal-cadence seeder for General Beckman.

Seeds a fixed set of internal scheduled tasks (kind='internal') into the
scheduled_tasks table on startup. Uses upsert-by-(title, kind) semantics so
repeated calls are idempotent. A module-level flag avoids redundant DB round-
trips when the process is long-running.
"""
from __future__ import annotations

import asyncio
import json
import os

from src.infra.logging_config import get_logger

logger = get_logger("beckman.cron_seed")

# Canonical internal cadences.  Nothing outside this module should hardcode
# these — downstream consumers query by kind='internal'.
INTERNAL_CADENCES: list[dict] = [
    {
        "title": "beckman_sweep",
        "description": "Periodic Beckman queue sweep",
        "interval_seconds": 300,
        "payload": {"_marker": "sweep"},
    },
    {
        "title": "hoca_benchmark_refresh",
        "description": "Refresh Fatih Hoca benchmark cache",
        "interval_seconds": 300,
        "payload": {"_marker": "benchmark_refresh"},
    },
    {
        "title": "todo_reminder",
        "description": "Remind user of pending todos",
        "interval_seconds": 7200,
        "payload": {"_executor": "todo_reminder"},
    },
    # daily_digest + api_discovery cadences intentionally omitted —
    # their mr_roboto handlers were dropped in the Phase 2b Beckman refactor
    # and never re-implemented. Re-add here once mr_roboto gains matching
    # action handlers; until then, seeding them just fills DLQ with
    # `unknown mechanical action` rows every day.
    {
        "title": "price_watch_check",
        "description": "Re-scrape watched products and notify on price drops",
        "interval_seconds": 86400,
        "payload": {"_executor": "price_watch_check"},
    },
    {
        "title": "nerd_herd_health_alert",
        "description": "Alert on Nerd Herd health anomalies",
        "interval_seconds": 600,
        "payload": {"_marker": "nerd_herd_health"},
    },
    {
        "title": "cloud_refresh",
        "description": "Re-run cloud provider /models discovery + bench refresh",
        "interval_seconds": 21600,  # 6h
        "payload": {"_executor": "cloud_refresh"},
    },
    {
        "title": "kdv_persist",
        "description": "Persist KDV rate-limit state (adapted limits, 429 history, daily counters) to kutai.db",
        "interval_seconds": 60,
        "payload": {"_executor": "kdv_persist"},
    },
    {
        "title": "btable_rollup",
        "description": "Aggregate model_call_tokens into step_token_stats percentiles (14-day window)",
        "interval_seconds": 3600,  # hourly
        "payload": {"_marker": "btable_rollup"},
    },
    {
        "title": "monitoring_check",
        "description": "URL uptime and GitHub repo poll; alerts via notify_user sub-tasks",
        "interval_seconds": int(os.getenv("MONITOR_INTERVAL", "300")),
        "payload": {"_executor": "monitoring_check"},
    },
    {
        "title": "vector_maint_wal",
        "description": "ChromaDB WAL checkpoint to release write-ahead log bloat",
        "interval_seconds": 1800,
        "payload": {"_executor": "vector_maint_wal"},
    },
    {
        "title": "vector_maint_snapshot",
        "description": "ChromaDB directory snapshot for crash recovery (daily)",
        "interval_seconds": 86400,
        "payload": {"_executor": "vector_maint_snapshot"},
    },
    # Z1 Tier 7A (B12) — quarterly bash-audit. Cron: first of Jan/Apr/Jul/Oct
    # at 09:00. cron_expression beats interval_seconds because quarterly
    # intervals don't fit 86400-second arithmetic cleanly across leap years
    # and DST boundaries.
    {
        "title": "bash_audit",
        "description": "sade_kalsin scaffolding audit (quarterly): what does each layer do that bash + Claude can't?",
        "cron_expression": "0 9 1 jan,apr,jul,oct *",
        "payload": {"_executor": "run_bash_audit"},
    },
]

# Fast-path: once seeded in this process, skip DB round-trips on subsequent calls.
_seeded: bool = False

# Serialises concurrent seed attempts so two near-simultaneous next_task()
# callers can't both pass the SELECT-before-INSERT check and create duplicate
# rows. Harmless under the serial pump today, but cheap insurance.
_seed_lock: asyncio.Lock = asyncio.Lock()


async def seed_internal_cadences() -> None:
    """Upsert all INTERNAL_CADENCES rows into scheduled_tasks.

    Safe to call multiple times — skips rows that already exist by
    (title, kind='internal').  Sets the module-level ``_seeded`` flag only
    after a successful pass so a crash mid-seed allows retry.

    Newly inserted rows have next_run set to now + interval_seconds so they
    don't fire immediately on first tick (avoids spurious task insertion in
    tests and on fresh deployments).
    """
    global _seeded
    if _seeded:
        return

    from datetime import timedelta
    from src.infra.db import get_db  # lazy to avoid circular import at module load
    from src.infra.times import utc_now, to_db

    async with _seed_lock:
        # Re-check inside the lock: another coroutine may have finished seeding
        # while we were waiting to acquire it.
        if _seeded:
            return

        db = await get_db()
        now = utc_now()
        for cadence in INTERNAL_CADENCES:
            cursor = await db.execute(
                "SELECT id FROM scheduled_tasks WHERE title = ? AND kind = 'internal'",
                (cadence["title"],),
            )
            existing = await cursor.fetchone()
            if existing:
                logger.debug("cron_seed: skipping existing row", title=cadence["title"])
                continue

            interval = cadence.get("interval_seconds")
            cron_expr = cadence.get("cron_expression")
            if interval:
                first_run = to_db(now + timedelta(seconds=interval))
            elif cron_expr:
                # croniter is optional; fall back to "fire in 1h" if absent
                # so cron's _advance_schedule can compute the next real slot
                # on the first tick.
                try:
                    from croniter import croniter
                    first_run = to_db(croniter(cron_expr, now).get_next(type(now)))
                except Exception:
                    first_run = to_db(now + timedelta(hours=1))
            else:
                first_run = to_db(now + timedelta(hours=1))
            await db.execute(
                """INSERT INTO scheduled_tasks
                   (title, description, interval_seconds, cron_expression, kind, context, enabled, next_run)
                   VALUES (?, ?, ?, ?, 'internal', ?, 1, ?)""",
                (
                    cadence["title"],
                    cadence["description"],
                    interval,
                    cron_expr,
                    json.dumps(cadence["payload"]),
                    first_run,
                ),
            )
            logger.info("cron_seed: inserted internal cadence", title=cadence["title"])

        await db.commit()
        _seeded = True
        logger.info("cron_seed: all internal cadences seeded", count=len(INTERNAL_CADENCES))
