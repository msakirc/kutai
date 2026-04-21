"""Derive current QueueProfile from queue tables and push to nerd_herd.

Fire-and-forget: any exception is swallowed so callers (enqueue /
on_task_finished / sweep) never break if nerd_herd or the DB is wonky.
"""
from __future__ import annotations

import os

import aiosqlite

from nerd_herd.types import QueueProfile


# Aligned with src/infra/db.py::get_ready_tasks predicate:
#   status='pending' AND next_retry_at passed.
# Dependency-resolution (recursive walk of depends_on) is deliberately
# omitted: it would turn each push into an O(n) scan, and QueueProfile
# is a coarse signal — one-push-cycle of staleness on blocked tasks is
# acceptable. Consumers must not rely on total_ready_count being the
# exact dispatchable count.
#
# Difficulty lives in the context JSON blob either as
# context.classification.difficulty (orchestrator-classified) or
# context.difficulty (workflow expander). COALESCE both.
_QUERY = """
    SELECT
        SUM(
            CASE WHEN COALESCE(
                CAST(json_extract(context, '$.classification.difficulty') AS INTEGER),
                CAST(json_extract(context, '$.difficulty') AS INTEGER),
                0
            ) >= 7 THEN 1 ELSE 0 END
        ) AS hard,
        COUNT(*) AS total
    FROM tasks
    WHERE status = ?
      AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))
"""


async def build_and_push(db_path: str | None = None) -> None:
    db_path = db_path or os.environ.get("DB_PATH", "kutai.db")
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(_QUERY, ("pending",)) as cur:
                row = await cur.fetchone()
    except Exception:
        return  # fire-and-forget; never breaks callers
    hard = int(row[0] or 0) if row else 0
    total = int(row[1] or 0) if row else 0
    profile = QueueProfile(hard_tasks_count=hard, total_ready_count=total)
    try:
        import nerd_herd
        nerd_herd.push_queue_profile(profile)
    except Exception:
        return
