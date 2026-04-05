# dead_letter.py
"""
Dead-letter queue for permanently failed tasks.

When a task exhausts all retries and the error_recovery agent also fails
(or the task is non-recoverable), it enters the dead-letter queue.

The DLQ:
- Quarantines tasks so they don't block downstream work
- Notifies via Telegram
- Provides `/dlq` command to inspect / retry / discard
- Auto-pauses a workflow mission if too many tasks land here

Integration with existing systems:
- BackpressureQueue handles *transient* model call failures (rate limits)
- _spawn_error_recovery handles *individual* task failures (bugs, bad prompts)
- DeadLetterQueue handles *permanent* failures that survive both layers
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("infra.dead_letter")

# If this many tasks from the same mission enter the DLQ, pause the mission
MISSION_DLQ_THRESHOLD = 3


async def _ensure_dlq_table() -> None:
    """Create the dead_letter_tasks table if it doesn't exist."""
    from src.infra.db import get_db

    db = await get_db()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS dead_letter_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            mission_id INTEGER,
            error TEXT,
            error_category TEXT DEFAULT 'unknown',
            original_agent TEXT,
            retry_count INTEGER DEFAULT 0,
            quarantined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            resolution TEXT,
            UNIQUE(task_id)
        )
    """)
    await db.commit()


async def quarantine_task(
    task_id: int,
    mission_id: Optional[int],
    error: str,
    error_category: str = "unknown",
    original_agent: str = "executor",
    retry_count: int = 0,
) -> int:
    """Move a permanently-failed task into the dead-letter queue.

    Returns the DLQ entry ID.
    """
    from src.infra.db import get_db

    await _ensure_dlq_table()
    db = await get_db()

    try:
        cursor = await db.execute(
            """INSERT OR REPLACE INTO dead_letter_tasks
               (task_id, mission_id, error, error_category, original_agent,
                retry_count, quarantined_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id, mission_id,
                error[:2000],  # cap error text
                _classify_error(error, error_category),
                original_agent,
                retry_count,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        await db.commit()
        dlq_id = cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to quarantine task #{task_id}: {e}")
        raise

    logger.warning(
        f"[DLQ] Task #{task_id} quarantined (mission={mission_id}, "
        f"category={error_category})"
    )

    # Check if mission should be auto-paused
    if mission_id:
        await _check_mission_health(mission_id)

    return dlq_id


async def _check_mission_health(mission_id: int) -> None:
    """If too many tasks from this mission are in the DLQ, pause the mission."""
    from src.infra.db import get_db, update_mission

    db = await get_db()
    cursor = await db.execute(
        """SELECT COUNT(*) FROM dead_letter_tasks
           WHERE mission_id = ? AND resolved_at IS NULL""",
        (mission_id,),
    )
    row = await cursor.fetchone()
    count = row[0] if row else 0

    if count >= MISSION_DLQ_THRESHOLD:
        logger.warning(
            f"[DLQ] Mission #{mission_id} has {count} quarantined tasks — "
            f"auto-pausing mission"
        )
        try:
            await update_mission(mission_id, status="paused")
        except Exception as e:
            logger.error(f"[DLQ] Failed to pause mission #{mission_id}: {e}")

        # Notify via Telegram
        try:
            from src.app.telegram_bot import get_bot
            bot = get_bot()
            if bot:
                await bot.send_notification(
                    f"Mission #{mission_id} auto-paused: {count} tasks "
                    f"in dead-letter queue (threshold={MISSION_DLQ_THRESHOLD})"
                )
        except Exception:
            pass


def _classify_error(error: str, provided_category: str) -> str:
    """Auto-classify an error if no category was explicitly provided."""
    if provided_category != "unknown":
        return provided_category

    error_lower = error.lower()

    if any(k in error_lower for k in ("rate limit", "429", "quota")):
        return "rate_limit"
    if any(k in error_lower for k in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(k in error_lower for k in ("budget", "cost limit", "spending")):
        return "budget_exceeded"
    if any(k in error_lower for k in ("auth", "401", "403", "credential", "token")):
        return "auth_failure"
    if any(k in error_lower for k in ("syntax", "parse", "json", "invalid")):
        return "parse_error"
    if any(k in error_lower for k in ("not found", "404", "missing file")):
        return "not_found"
    if any(k in error_lower for k in ("connection", "network", "dns")):
        return "network_error"

    return "unknown"


async def get_dlq_tasks(
    mission_id: Optional[int] = None,
    unresolved_only: bool = True,
) -> list[dict]:
    """List dead-letter tasks, optionally filtered by mission."""
    from src.infra.db import get_db

    await _ensure_dlq_table()
    db = await get_db()

    query = "SELECT * FROM dead_letter_tasks WHERE 1=1"
    params: list = []

    if mission_id is not None:
        query += " AND mission_id = ?"
        params.append(mission_id)
    if unresolved_only:
        query += " AND resolved_at IS NULL"

    query += " ORDER BY quarantined_at DESC"
    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()

    return [dict(r) for r in rows]


async def resolve_dlq_task(
    task_id: int,
    resolution: str = "manual",
) -> bool:
    """Mark a DLQ entry as resolved.

    resolution: "retry" | "skip" | "manual" | "discarded"
    """
    from src.infra.db import get_db

    await _ensure_dlq_table()
    db = await get_db()

    cursor = await db.execute(
        """UPDATE dead_letter_tasks
           SET resolved_at = ?, resolution = ?
           WHERE task_id = ? AND resolved_at IS NULL""",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), resolution, task_id),
    )
    await db.commit()
    return cursor.rowcount > 0


async def retry_dlq_task(task_id: int) -> bool:
    """Re-queue a dead-letter task for another attempt.

    Phase-aware: restores to the phase where the task failed.
    Resets exclusions and backoff, preserves attempt counters.
    """
    import json
    from src.infra.db import get_task, update_task

    await resolve_dlq_task(task_id, resolution="retry")

    task = await get_task(task_id)
    if not task:
        logger.warning(f"[DLQ] Task #{task_id} not found for retry")
        return False

    # Phase-aware status
    failed_phase = task.get("failed_in_phase")
    new_status = "ungraded" if failed_phase == "grading" else "pending"

    # Reset exclusions and backoff in context
    ctx = {}
    try:
        ctx = json.loads(task.get("context") or "{}")
    except (json.JSONDecodeError, TypeError):
        pass

    ctx["last_avail_delay"] = 0
    ctx["excluded_models"] = []
    ctx["grade_excluded_models"] = []
    # Keep generating_model (prevents self-grading)

    await update_task(
        task_id,
        status=new_status,
        next_retry_at=None,
        retry_reason=None,
        context=json.dumps(ctx),
    )
    logger.info(f"[DLQ] Task #{task_id} re-queued → {new_status} (phase={failed_phase})")
    return True


async def get_dlq_summary() -> dict:
    """Get aggregate stats about the dead-letter queue."""
    from src.infra.db import get_db

    await _ensure_dlq_table()
    db = await get_db()

    cursor = await db.execute(
        """SELECT
             COUNT(*) as total,
             SUM(CASE WHEN resolved_at IS NULL THEN 1 ELSE 0 END) as unresolved,
             SUM(CASE WHEN resolved_at IS NOT NULL THEN 1 ELSE 0 END) as resolved
           FROM dead_letter_tasks"""
    )
    row = await cursor.fetchone()

    # Category breakdown for unresolved
    cursor2 = await db.execute(
        """SELECT error_category, COUNT(*) as cnt
           FROM dead_letter_tasks
           WHERE resolved_at IS NULL
           GROUP BY error_category
           ORDER BY cnt DESC"""
    )
    categories = {r[0]: r[1] for r in await cursor2.fetchall()}

    return {
        "total": row[0] or 0,
        "unresolved": row[1] or 0,
        "resolved": row[2] or 0,
        "categories": categories,
    }
