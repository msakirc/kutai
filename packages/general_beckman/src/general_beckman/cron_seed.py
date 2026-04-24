"""Internal-cadence seeder for General Beckman.

Seeds a fixed set of internal scheduled tasks (kind='internal') into the
scheduled_tasks table on startup. Uses upsert-by-(title, kind) semantics so
repeated calls are idempotent. A module-level flag avoids redundant DB round-
trips when the process is long-running.
"""
from __future__ import annotations

import asyncio
import json

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
    # their salako handlers were dropped in the Phase 2b Beckman refactor
    # and never re-implemented. Re-add here once salako gains matching
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

            first_run = to_db(now + timedelta(seconds=cadence["interval_seconds"]))
            await db.execute(
                """INSERT INTO scheduled_tasks
                   (title, description, interval_seconds, kind, context, enabled, next_run)
                   VALUES (?, ?, ?, 'internal', ?, 1, ?)""",
                (
                    cadence["title"],
                    cadence["description"],
                    cadence["interval_seconds"],
                    json.dumps(cadence["payload"]),
                    first_run,
                ),
            )
            logger.info("cron_seed: inserted internal cadence", title=cadence["title"])

        await db.commit()
        _seeded = True
        logger.info("cron_seed: all internal cadences seeded", count=len(INTERNAL_CADENCES))
