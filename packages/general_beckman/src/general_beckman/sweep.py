"""Queue hygiene sweep — port of watchdog.check_stuck_tasks.

Renamed to sweep_queue(); telegram= parameter dropped.
All Telegram notifications replaced with mechanical salako notify_user tasks.

Preserves all 7 numbered sections from the original:
  1. Tasks stuck in "processing" > 5 min
  2. Ungraded tasks stuck > 30 min (safety net)
  3. Tasks blocked by ALL failed deps → cascade failure
  4. Parent tasks with all children done → rollup
  5. Pending tasks with overdue next_retry_at → clear gate
  6. waiting_human escalation tiers (4h nudge, 24h, 48h, 72h cancel)
  7. Workflow-level timeout check
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.infra.logging_config import get_logger
from src.infra.times import utc_now, db_now, to_db, from_db
from general_beckman.task_context import parse_context

logger = get_logger("beckman.sweep")


async def _notify(message: str) -> None:
    """Insert a mechanical notify_user task instead of sending Telegram inline."""
    from src.infra.db import add_task
    from general_beckman.apply import _mechanical_context
    await add_task(
        title="Notify: stuck-task sweep",
        description="",
        agent_type="mechanical",
        context=_mechanical_context("notify_user", message=message),
        depends_on=[],
    )


async def sweep_queue() -> None:
    """Task-level recovery: stuck, ungraded, dep cascade, subtasks,
    overdue retry gates, waiting_human escalation, workflow timeouts."""
    from src.infra.db import get_db, update_task, update_mission

    db = await get_db()

    # 1. Tasks stuck in "processing" for more than 5 minutes
    cursor = await db.execute(
        """SELECT id, title, worker_attempts, infra_resets, max_worker_attempts FROM tasks
           WHERE status = 'processing'
           AND started_at < datetime('now', '-5 minutes')"""
    )
    stuck = [dict(row) for row in await cursor.fetchall()]
    from general_beckman.apply import _dlq_write
    for task in stuck:
        infra_resets = (task.get("infra_resets") or 0) + 1
        if infra_resets >= 3:
            logger.warning(
                f"[Sweep] Task #{task['id']} stuck in processing "
                f"and exhausted infra resets ({infra_resets}/3), "
                f"routing to DLQ"
            )
            await db.execute(
                "UPDATE tasks SET infra_resets = ? WHERE id = ?",
                (infra_resets, task["id"])
            )
            await db.commit()
            # Route to DLQ so the user sees + can /dlq retry, and
            # dependents get the cascade-reset when retry runs.
            # Silent status=failed was the hole that let mission 46
            # freeze with 2 stuck tasks hidden from /dlq.
            fresh = dict(task)
            fresh["infra_resets"] = infra_resets
            fresh["failed_in_phase"] = "infrastructure"
            await _dlq_write(
                fresh,
                error="Stuck in processing — infra resets exhausted (sweep)",
                category="infrastructure",
                attempts=int(task.get("worker_attempts") or 0),
            )
        else:
            logger.warning(
                f"[Sweep] Task #{task['id']} stuck in processing, "
                f"infra-reset {infra_resets}/3"
            )
            await db.execute(
                "UPDATE tasks SET status = 'pending', "
                "infra_resets = ?, retry_reason = 'infrastructure' WHERE id = ?",
                (infra_resets, task["id"])
            )
    if stuck:
        await db.commit()

    # 2. Ungraded tasks stuck for > 30 min — safety net
    #    Use worker_completed_at from context (set by base.py on entering ungraded).
    #    Falls back to started_at if worker_completed_at is missing.
    cursor_ung = await db.execute(
        "SELECT id, context, started_at FROM tasks WHERE status = 'ungraded'"
    )
    all_ungraded = [dict(row) for row in await cursor_ung.fetchall()]
    stuck_ungraded = []
    for task in all_ungraded:
        ctx = parse_context(task)
        ref_time_str = ctx.get("worker_completed_at") or task.get("started_at")
        if not ref_time_str:
            continue
        try:
            ref_dt = from_db(str(ref_time_str))
            if (utc_now() - ref_dt).total_seconds() > 1800:
                stuck_ungraded.append(task)
        except (ValueError, TypeError):
            continue

    for task in stuck_ungraded:
        # Post-hook regime: an 'ungraded' row with _pending_posthooks is
        # legitimately awaiting a grader/summarizer verdict. Promoting it
        # silently turns an uncertain verdict into a pass. Skip it —
        # the post-hook DLQ cascade (apply._dlq_write) handles terminal
        # failure when a post-hook task exhausts retries.
        ctx = parse_context(task)
        if ctx.get("_pending_posthooks"):
            logger.debug(
                f"[Sweep] Stuck ungraded #{task['id']} has pending posthooks "
                f"{ctx['_pending_posthooks']}; leaving for verdict path"
            )
            continue
        await db.execute(
            "UPDATE tasks SET status = 'completed', "
            "completed_at = ? WHERE id = ?",
            (db_now(), task["id"]),
        )
        logger.warning(f"[Sweep] Stuck ungraded #{task['id']} promoted to completed (safety net)")
    if stuck_ungraded:
        await db.commit()

    # 3. Tasks blocked by ALL failed deps → cascade failure
    cursor2 = await db.execute(
        "SELECT id, title, depends_on FROM tasks "
        "WHERE status = 'pending' AND depends_on != '[]'"
    )
    blocked = [dict(row) for row in await cursor2.fetchall()]
    for task in blocked:
        try:
            deps = json.loads(task.get("depends_on", "[]"))
        except (json.JSONDecodeError, TypeError):
            deps = []
        if not deps:
            continue

        placeholders = ",".join("?" * len(deps))
        # Count non-skipped deps that are failed
        fail_cursor = await db.execute(
            f"SELECT COUNT(*) FROM tasks WHERE id IN ({placeholders}) AND status = 'failed'",
            deps
        )
        failed_count = (await fail_cursor.fetchone())[0]

        if failed_count == 0:
            continue

        # Count deps still in progress (not terminal)
        pending_cursor = await db.execute(
            f"SELECT COUNT(*) FROM tasks WHERE id IN ({placeholders}) AND status NOT IN ('completed', 'failed', 'cancelled', 'skipped')",
            deps
        )
        still_pending = (await pending_cursor.fetchone())[0]

        if still_pending > 0:
            continue  # some deps still running, don't cascade yet

        # All deps are terminal. Count non-skipped ones.
        total_cursor = await db.execute(
            f"SELECT COUNT(*) FROM tasks WHERE id IN ({placeholders}) AND status NOT IN ('skipped')",
            deps
        )
        total_non_skipped = (await total_cursor.fetchone())[0]

        if failed_count == total_non_skipped and total_non_skipped > 0:
            # Don't cascade if any failed dep is in DLQ (recoverable).
            # The human may retry it via /dlq retry.
            try:
                dlq_cursor = await db.execute(
                    f"""SELECT COUNT(*) FROM dead_letter_tasks
                        WHERE task_id IN ({placeholders})
                        AND resolved_at IS NULL""",
                    deps
                )
                dlq_count = (await dlq_cursor.fetchone())[0]
            except Exception:
                dlq_count = 0
            if dlq_count > 0:
                continue  # dep is in DLQ, don't cascade yet

            logger.warning(
                f"[Sweep] Task #{task['id']} all deps failed, routing to DLQ"
            )
            task_for_dlq = dict(task)
            task_for_dlq["failed_in_phase"] = "worker"
            await _dlq_write(
                task_for_dlq,
                error="All dependencies failed",
                category="dependency",
                attempts=int(task.get("worker_attempts") or 0),
            )
    if blocked:
        await db.commit()

    # 4. Parent tasks with all children done
    cursor3 = await db.execute(
        "SELECT id, title FROM tasks WHERE status = 'waiting_subtasks'"
    )
    waiting = [dict(row) for row in await cursor3.fetchall()]
    for task in waiting:
        child_cursor = await db.execute(
            """SELECT COUNT(*) as total,
               SUM(CASE WHEN status IN (
                   'completed','failed','cancelled','skipped'
               ) THEN 1 ELSE 0 END) as done,
               SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count
               FROM tasks WHERE parent_task_id = ?""",
            (task["id"],),
        )
        row = await child_cursor.fetchone()
        if row and row["total"] > 0 and row["total"] == row["done"]:
            if row["completed_count"] > 0:
                logger.info(f"[Sweep] Task #{task['id']} all subtasks done, marking complete")
                await db.execute(
                    "UPDATE tasks SET status = 'completed', "
                    "completed_at = ? WHERE id = ?",
                    (db_now(), task["id"]),
                )
            else:
                logger.warning(f"[Sweep] Task #{task['id']} all subtasks failed, routing to DLQ")
                task_for_dlq = dict(task)
                task_for_dlq["failed_in_phase"] = "worker"
                await _dlq_write(
                    task_for_dlq,
                    error="All subtasks failed",
                    category="subtasks",
                    attempts=int(task.get("worker_attempts") or 0),
                )
    if waiting:
        await db.commit()

    # 5. Pending tasks with next_retry_at far in the past
    cursor_overdue = await db.execute(
        """SELECT id FROM tasks
           WHERE status = 'pending'
           AND next_retry_at < datetime('now', '-1 hour')"""
    )
    overdue = [dict(row) for row in await cursor_overdue.fetchall()]
    for task in overdue:
        await db.execute(
            "UPDATE tasks SET next_retry_at = NULL WHERE id = ?",
            (task["id"],),
        )
    if overdue:
        await db.commit()
        logger.info(f"[Sweep] Cleared overdue next_retry_at for {len(overdue)} task(s)")

    # 6. Escalation tiers for tasks stuck in waiting_human
    #    Uses started_at as the baseline timestamp (set when task
    #    began processing, before entering waiting_human).
    #    We compute the threshold in Python with isoformat() so the
    #    string comparison matches the format used when storing
    #    started_at (which also uses db_now() format).
    threshold_24h = to_db(
        utc_now() - timedelta(hours=24)
    )

    # Tier 0: 4-hour gentle nudge (no escalation count increment)
    threshold_4h = to_db(
        utc_now() - timedelta(hours=4)
    )
    cursor_nudge = await db.execute(
        """SELECT id, title, context FROM tasks
           WHERE status = 'waiting_human'
           AND started_at < ?
           AND started_at >= ?""",
        (threshold_4h, threshold_24h),
    )
    nudge_tasks = [dict(row) for row in await cursor_nudge.fetchall()]
    for task in nudge_tasks:
        task_ctx = parse_context(task)
        if not task_ctx.get("nudge_sent"):
            task_ctx["nudge_sent"] = True
            await update_task(task["id"], context=json.dumps(task_ctx))
            await _notify(
                f"\U0001f4ac Gentle reminder: Task #{task['id']} needs your input.\n"
                f"*{task['title']}*"
            )

    cursor_clar = await db.execute(
        """SELECT id, title, context, started_at FROM tasks
           WHERE status = 'waiting_human'
           AND started_at < ?""",
        (threshold_24h,),
    )
    stale = [dict(row) for row in await cursor_clar.fetchall()]
    for task in stale:
        # Parse escalation_count from task context
        task_ctx = parse_context(task)
        escalation_count = task_ctx.get("escalation_count", 0)
        tid = task["id"]
        ttitle = task["title"]

        # Calculate hours since started_at
        try:
            started = from_db(task["started_at"])
        except (ValueError, TypeError):
            started = datetime.min.replace(tzinfo=timezone.utc)
        hours_waiting = (
            utc_now() - started
        ).total_seconds() / 3600

        if escalation_count == 0 and hours_waiting >= 24:
            # Tier 1: 24h reminder
            task_ctx["escalation_count"] = 1
            await update_task(
                tid, context=json.dumps(task_ctx),
            )
            logger.info(
                f"[Sweep] Task #{tid} escalation tier 1 (24h)"
            )
            await _notify(
                f"\u23f0 Task #{tid} has been waiting for "
                f"clarification for 24h.\n*{ttitle}*"
            )
        elif escalation_count == 1 and hours_waiting >= 48:
            # Tier 2: 48h urgent
            task_ctx["escalation_count"] = 2
            await update_task(
                tid, context=json.dumps(task_ctx),
            )
            logger.info(
                f"[Sweep] Task #{tid} escalation tier 2 (48h)"
            )
            await _notify(
                f"\U0001f6a8 *URGENT:* Task #{tid} needs your input!\n"
                f"*{ttitle}*\n\n"
                f"_This task will be cancelled in 24h if no "
                f"response is received._"
            )
        elif escalation_count >= 2 and hours_waiting >= 72:
            # Tier 3: 72h cancel
            task_ctx["escalation_count"] = 3
            logger.warning(
                f"[Sweep] Task #{tid} escalation tier 3 "
                f"(72h), cancelling"
            )
            await update_task(
                tid, status="cancelled",
                error="No clarification received within 72h",
                context=json.dumps(task_ctx),
            )
            await _notify(
                f"\u274c Task #{tid} cancelled — no clarification "
                f"received after 72h.\n*{ttitle}*"
            )

    # Workflow-level wall-clock timeout killed 2026-04-22 (queue-gated
    # missions could be paused while simply waiting on admission).

    await db.commit()

    try:
        from general_beckman.queue_profile_push import build_and_push
        await build_and_push()
    except Exception as e:
        logger.debug(f"[Sweep] queue_profile push failed: {e}")
