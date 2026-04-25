# dead_letter.py
"""
Dead-letter queue for permanently failed tasks.

When a task exhausts all retries (worker attempts, infrastructure resets,
or grading attempts), it enters the dead-letter queue.

The DLQ:
- Quarantines tasks so they don't block downstream work
- Notifies via Telegram
- Provides `/dlq` command to inspect / retry / discard
- Auto-pauses a workflow mission if too many tasks land here
- Feeds the DLQ Analyst for pattern detection and proactive alerts

Integration with existing systems:
- RetryContext handles in-flight failure recovery (model rotation, difficulty bumps)
- BackpressureQueue handles transient model call failures (rate limits)
- DLQAnalyst detects cross-task failure patterns after quarantine
"""

from __future__ import annotations

from typing import Optional

from .times import db_now

from src.infra.logging_config import get_logger

logger = get_logger("infra.dead_letter")

from src.infra.dlq_analyst import DLQAnalyst

_analyst = DLQAnalyst()

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
            attempts_snapshot INTEGER DEFAULT 0,
            quarantined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            resolution TEXT,
            UNIQUE(task_id)
        )
    """)
    await db.commit()

    # Rename column if old schema
    try:
        cursor = await db.execute("PRAGMA table_info(dead_letter_tasks)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "retry_count" in cols and "attempts_snapshot" not in cols:
            await db.execute(
                "ALTER TABLE dead_letter_tasks RENAME COLUMN retry_count TO attempts_snapshot"
            )
            await db.commit()
    except Exception:
        pass


async def quarantine_task(
    task_id: int,
    mission_id: Optional[int],
    error: str,
    error_category: str = "unknown",
    original_agent: str = "executor",
    attempts_snapshot: int = 0,
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
                attempts_snapshot, quarantined_at, resolved_at, resolution)
               VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
            (
                task_id, mission_id,
                error[:2000],  # cap error text
                _classify_error(error, error_category),
                original_agent,
                attempts_snapshot,
                db_now(),
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

    # Run DLQ pattern analysis
    try:
        await _run_pattern_analysis(task_id, error_category)
    except Exception as e:
        logger.debug(f"[DLQ] Pattern analysis failed (non-critical): {e}")

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
            from src.app.telegram_bot import get_telegram
            bot = get_telegram()
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
        (db_now(), resolution, task_id),
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

    # Mechanical side-effect tasks (clarify, workflow_advance, git_commit,
    # summarize, ...) retain a broken payload from the upstream LLM step
    # that spawned them. Retrying the MECHANICAL row just re-runs the
    # broken executor — the fix must go to the PARENT that produced the
    # payload. Redirect the retry to the parent, cancel this row.
    parent_id = task.get("parent_task_id")
    if task.get("agent_type") == "mechanical" and parent_id:
        parent = await get_task(parent_id)
        if parent and parent.get("status") in (
            "waiting_human", "failed", "ungraded", "processing",
        ):
            logger.info(
                "[DLQ] Redirecting mechanical retry #%d → parent #%d",
                task_id, parent_id,
            )
            await update_task(
                task_id, status="cancelled",
                error=f"redirected to parent #{parent_id} on DLQ retry",
            )
            # Recurse on parent (will hit the normal retry path below).
            return await retry_dlq_task(parent_id) if await _in_dlq(parent_id) else await _plain_retry(parent)
    return await _plain_retry(task)


async def _in_dlq(task_id: int) -> bool:
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM dead_letter_tasks WHERE task_id = ? AND resolved_at IS NULL",
        (task_id,),
    )
    return (await cur.fetchone()) is not None


async def _plain_retry(task: dict) -> bool:
    """Normal pending-reset retry for a single task row."""
    import json
    from src.infra.db import update_task
    task_id = task["id"]

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
    ctx["failed_models"] = []
    ctx["excluded_models"] = []
    ctx["grade_excluded_models"] = []
    # Drop retry feedback from the prior lifecycle — DLQ retry is a fresh
    # start. Without this, the agent's first post-DLQ prompt replayed
    # "your last output failed: <stale schema error from N attempts ago>".
    ctx.pop("_schema_error", None)
    ctx.pop("_prev_output", None)
    ctx.pop("_schema_error_for_attempt", None)
    # Keep generating_model (prevents self-grading)

    # Reset checkpoint iteration counter so the agent doesn't immediately
    # exhaust on resume, but keep the checkpoint data (tool results, messages)
    # so previous work isn't lost.
    try:
        from src.infra.db import load_task_checkpoint, save_task_checkpoint
        cp = await load_task_checkpoint(task_id)
        if cp:
            cp["iteration"] = 0
            cp["format_corrections"] = 0
            cp["consecutive_tool_failures"] = 0
            await save_task_checkpoint(task_id, cp)
    except Exception:
        pass

    await update_task(
        task_id,
        status=new_status,
        worker_attempts=0,
        next_retry_at=None,
        retry_reason=None,
        context=json.dumps(ctx),
    )

    # Reset downstream tasks that were cascade-failed due to this task's DLQ.
    # Without this, dependents stay permanently failed even after DLQ retry.
    try:
        from src.infra.db import get_db
        db = await get_db()
        cascade_cursor = await db.execute(
            """UPDATE tasks SET status = 'pending', error = NULL,
                   started_at = NULL, completed_at = NULL, worker_attempts = 0
               WHERE status = 'failed'
                 AND error = 'All dependencies failed'
                 AND depends_on LIKE ?""",
            (f"%{task_id}%",),
        )
        cascade_count = cascade_cursor.rowcount
        if cascade_count > 0:
            await db.commit()
            logger.info(f"[DLQ] Reset {cascade_count} cascade-failed dependents of task #{task_id}")
    except Exception as e:
        logger.debug(f"[DLQ] Cascade reset failed: {e}")

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


async def _run_pattern_analysis(task_id: int, error_category: str) -> None:
    """Check for failure patterns and send Telegram alert if detected."""
    from src.infra.db import get_db

    db = await get_db()
    await _ensure_dlq_table()

    # Fetch recent unresolved DLQ entries within the window
    cursor = await db.execute(
        """SELECT task_id, mission_id, error, error_category, original_agent,
                  quarantined_at
           FROM dead_letter_tasks
           WHERE resolved_at IS NULL
             AND quarantined_at >= datetime('now', ?)
           ORDER BY quarantined_at DESC""",
        (f"-{DLQAnalyst.WINDOW_HOURS} hours",),
    )
    rows = await cursor.fetchall()
    entries = [dict(r) for r in rows]

    if len(entries) < 3:
        return

    patterns = _analyst.detect_patterns(entries)

    for pattern in patterns:
        key = pattern["group_key"]
        if _analyst.is_deduped(key):
            continue

        # Run diagnostic check
        diagnostic = await _analyst.run_diagnostic(key, pattern["entries"])
        pattern["diagnostic"] = diagnostic

        # Format and send alert
        message = _analyst.format_alert(pattern)
        task_ids = [e["task_id"] for e in pattern["entries"]]
        await _send_dlq_alert(message, key, task_ids)
        _analyst.record_alert(key)


async def _send_dlq_alert(message: str, pattern_key: str, task_ids: list[int]) -> None:
    """Send a DLQ pattern alert via Telegram with inline action buttons."""
    try:
        from src.app.telegram_bot import get_telegram
        bot = get_telegram()
        if not bot:
            return

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        # Encode task IDs as comma-separated in callback data
        ids_str = ",".join(str(t) for t in task_ids[:20])  # cap at 20
        buttons = [
            [
                InlineKeyboardButton(
                    f"Retry All ({len(task_ids)})",
                    callback_data=f"dlqa:retry:{ids_str}",
                ),
                InlineKeyboardButton(
                    f"Drop All ({len(task_ids)})",
                    callback_data=f"dlqa:drop:{ids_str}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Pause Similar",
                    callback_data=f"dlqa:pause:{pattern_key}",
                ),
            ],
        ]
        markup = InlineKeyboardMarkup(buttons)
        await bot.send_notification(message, reply_markup=markup)
    except Exception as e:
        logger.debug(f"[DLQ] Failed to send pattern alert: {e}")
