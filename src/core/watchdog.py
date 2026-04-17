"""Watchdog functions: periodic maintenance called from orchestrator's main loop.

Previously all inside orchestrator.watchdog(). Split by concern so each is
individually testable and extractable later.

Two top-level concerns:
  - check_stuck_tasks: all task-level recovery (stuck, ungraded, dep cascade,
    waiting_subtasks, overdue retry, waiting_human escalation, workflow timeout)
  - check_resources: resource-level recovery (llama-server, GPU, circuit
    breakers, rate limits, credential expiry)
"""

import json
from datetime import datetime, timedelta, timezone

from src.infra.logging_config import get_logger
from ..infra.db import get_db, update_task, update_mission
from ..infra.times import utc_now, db_now, to_db, from_db
from .task_context import parse_context
from .router import get_kdv

logger = get_logger("core.watchdog")


async def check_stuck_tasks(telegram=None):
    """Task-level recovery: stuck, ungraded, dep cascade, subtasks,
    overdue retry gates, waiting_human escalation, workflow timeouts."""
    db = await get_db()

    # 1. Tasks stuck in "processing" for more than 5 minutes
    cursor = await db.execute(
        """SELECT id, title, worker_attempts, infra_resets, max_worker_attempts FROM tasks
           WHERE status = 'processing'
           AND started_at < datetime('now', '-5 minutes')"""
    )
    stuck = [dict(row) for row in await cursor.fetchall()]
    for task in stuck:
        infra_resets = (task.get("infra_resets") or 0) + 1
        if infra_resets >= 3:
            logger.warning(
                f"[Watchdog] Task #{task['id']} stuck in processing "
                f"and exhausted infra resets ({infra_resets}/3), "
                f"marking failed"
            )
            await db.execute(
                "UPDATE tasks SET status = 'failed', "
                "error = 'Stuck in processing — infra resets exhausted (watchdog)', "
                "failed_in_phase = 'infrastructure', "
                "infra_resets = ? "
                "WHERE id = ?",
                (infra_resets, task["id"])
            )
        else:
            logger.warning(
                f"[Watchdog] Task #{task['id']} stuck in processing, "
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
        await db.execute(
            "UPDATE tasks SET status = 'completed', "
            "completed_at = ? WHERE id = ?",
            (db_now(), task["id"]),
        )
        logger.warning(f"[Watchdog] Stuck ungraded #{task['id']} promoted to completed (safety net)")
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
                f"[Watchdog] Task #{task['id']} all deps failed, cascading failure"
            )
            await db.execute(
                "UPDATE tasks SET status = 'failed', "
                "error = 'All dependencies failed', failed_in_phase = 'worker' "
                "WHERE id = ?",
                (task["id"],)
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
                logger.info(f"[Watchdog] Task #{task['id']} all subtasks done, marking complete")
                await db.execute(
                    "UPDATE tasks SET status = 'completed', "
                    "completed_at = ? WHERE id = ?",
                    (db_now(), task["id"]),
                )
            else:
                logger.warning(f"[Watchdog] Task #{task['id']} all subtasks failed, marking failed")
                await db.execute(
                    "UPDATE tasks SET status = 'failed', "
                    "error = 'All subtasks failed', failed_in_phase = 'worker' "
                    "WHERE id = ?",
                    (task["id"],)
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
        logger.info(f"[Watchdog] Cleared overdue next_retry_at for {len(overdue)} task(s)")

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
            await telegram.send_notification(
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
                f"[Watchdog] Task #{tid} escalation tier 1 (24h)"
            )
            await telegram.send_notification(
                f"⏰ Task #{tid} has been waiting for "
                f"clarification for 24h.\n*{ttitle}*"
            )
        elif escalation_count == 1 and hours_waiting >= 48:
            # Tier 2: 48h urgent
            task_ctx["escalation_count"] = 2
            await update_task(
                tid, context=json.dumps(task_ctx),
            )
            logger.info(
                f"[Watchdog] Task #{tid} escalation tier 2 (48h)"
            )
            await telegram.send_notification(
                f"🚨 *URGENT:* Task #{tid} needs your input!\n"
                f"*{ttitle}*\n\n"
                f"_This task will be cancelled in 24h if no "
                f"response is received._"
            )
        elif escalation_count >= 2 and hours_waiting >= 72:
            # Tier 3: 72h cancel
            task_ctx["escalation_count"] = 3
            logger.warning(
                f"[Watchdog] Task #{tid} escalation tier 3 "
                f"(72h), cancelling"
            )
            await update_task(
                tid, status="cancelled",
                error="No clarification received within 72h",
                context=json.dumps(task_ctx),
            )
            await telegram.send_notification(
                f"❌ Task #{tid} cancelled — no clarification "
                f"received after 72h.\n*{ttitle}*"
            )

    # 7. Workflow-level timeout check — pause workflows running too long
    try:
        mission_cursor = await db.execute(
            """SELECT id, title, context, created_at FROM missions
               WHERE status = 'active'"""
        )
        active_missions = [dict(row) for row in await mission_cursor.fetchall()]
        for mission in active_missions:
            raw_gctx = mission.get("context", "{}")
            if isinstance(raw_gctx, str):
                try:
                    gctx = json.loads(raw_gctx)
                except (json.JSONDecodeError, TypeError):
                    gctx = {}
            else:
                gctx = raw_gctx or {}

            timeout_hours = gctx.get("workflow_timeout_hours")
            if not timeout_hours:
                continue

            try:
                created = from_db(mission["created_at"])
            except (ValueError, TypeError):
                continue

            elapsed_hours = (utc_now() - created).total_seconds() / 3600
            if elapsed_hours > timeout_hours:
                logger.warning(
                    "[Watchdog] Mission #%d exceeded timeout (%dh > %dh), pausing",
                    mission["id"], int(elapsed_hours), timeout_hours,
                )
                await update_mission(mission["id"], status="paused")
                await telegram.send_notification(
                    f"⏱️ *Workflow timeout*: Mission #{mission['id']} paused after "
                    f"{int(elapsed_hours)}h (limit: {timeout_hours}h).\n"
                    f"*{mission['title']}*\nUse /resume to continue."
                )
    except Exception as e:
        logger.warning(f"[Watchdog] Workflow timeout check failed: {e}")

    await db.commit()


async def check_resources(telegram=None):
    """Resource-level recovery: llama-server health, GPU, circuit breakers,
    rate limits, credential expiry."""
    resource_issues: list[str] = []

    # 4. Check llama-server health
    # DaLLaMa's HealthWatchdog handles crash recovery internally.
    # We only report status for the resource summary.
    try:
        from ..models.local_model_manager import get_local_manager

        manager = get_local_manager()
        if manager.current_model and not manager.is_loaded:
            resource_issues.append(
                f"llama-server unhealthy (model: {manager.current_model})"
            )
    except Exception as e:
        logger.warning(f"[Watchdog] Local model check failed: {e}")

    # 5. Check GPU health
    try:
        from ..models.gpu_monitor import get_gpu_monitor

        gpu_state = get_gpu_monitor().get_state()

        if gpu_state.gpu.available:
            # Thermal throttling
            if gpu_state.gpu.is_throttling:
                resource_issues.append(
                    f"GPU thermal throttling! "
                    f"Temp: {gpu_state.gpu.temperature_c}°C"
                )
                logger.warning(
                    f"[Watchdog] 🌡️ GPU at {gpu_state.gpu.temperature_c}°C "
                    f"— thermal throttling detected"
                )

            # VRAM nearly full (>95%) without a model loaded
            # This suggests a leak or external process consuming VRAM
            from ..models.local_model_manager import get_local_manager
            mgr = get_local_manager()
            if (
                gpu_state.gpu.vram_usage_pct > 95
                and not mgr.is_loaded
            ):
                resource_issues.append(
                    f"VRAM nearly full ({gpu_state.gpu.vram_usage_pct:.0f}%) "
                    f"but no model loaded — possible leak"
                )
                logger.warning(
                    f"[Watchdog] VRAM at "
                    f"{gpu_state.gpu.vram_used_mb}/"
                    f"{gpu_state.gpu.vram_total_mb}MB "
                    f"with no model loaded"
                )

        # Low RAM
        if gpu_state.ram_available_mb < 2048:
            resource_issues.append(
                f"Low RAM: {gpu_state.ram_available_mb}MB available"
            )
            logger.warning(
                f"[Watchdog] Low RAM: "
                f"{gpu_state.ram_available_mb}MB available"
            )

    except Exception as e:
        logger.warning(f"[Watchdog] GPU health check failed: {e}")

    # 6. Check local model status
    try:
        from ..models.local_model_manager import get_local_manager

        mgr = get_local_manager()
        mgr_status = mgr.get_status()
        if not mgr_status.get("healthy", True) and mgr_status.get("loaded_model"):
            resource_issues.append("Local model unhealthy")
            logger.warning("[Watchdog] Local model unhealthy")

    except Exception as e:
        logger.warning(f"[Watchdog] GPU scheduler check failed: {e}")

    # 7. Check circuit breakers — are ALL cloud providers down?
    try:
        kdv_status = get_kdv().status
        degraded_providers = [
            p for p, prov_status in kdv_status.items()
            if prov_status.circuit_breaker_open
        ]
        if degraded_providers:
            resource_issues.append(
                f"Degraded providers: {', '.join(degraded_providers)}"
            )
            logger.warning(
                f"[Watchdog] Circuit breakers tripped: "
                f"{degraded_providers}"
            )

            # Check if ALL providers are degraded
            from ..models.model_registry import get_registry
            registry = get_registry()
            all_cloud_providers = set(
                m.provider for m in registry.cloud_models()
            )
            if all_cloud_providers and all_cloud_providers.issubset(
                set(degraded_providers)
            ):
                resource_issues.append(
                    "⚠️ ALL cloud providers are degraded! "
                    "Only local inference available."
                )

    except Exception as e:
        logger.warning(f"[Watchdog] Circuit breaker check failed: {e}")

    # 8. Restore rate limits that were adaptively reduced
    try:
        get_kdv().restore_limits()
    except Exception as e:
        logger.warning(f"[Watchdog] Rate limit restore failed: {e}")

    # 9. Check for expiring credentials (warn 24h before expiry)
    try:
        from ..security.credential_store import list_credentials, get_credential

        services = await list_credentials()
        for svc in services:
            cred = await get_credential(svc)
            if cred is None:
                # Already expired — get_credential returns None
                resource_issues.append(
                    f"🔑 Credential '{svc}' has expired. Refresh with /credential add."
                )
                await telegram.send_notification(
                    f"🔑 *Credential expired*: `{svc}`\n"
                    f"Use /credential add to refresh."
                )
    except Exception as e:
        logger.warning(f"[Watchdog] Credential expiry check failed: {e}")

    # ── Alert on resource issues ──
    if resource_issues:
        issues_text = "\n".join(f"  • {i}" for i in resource_issues)
        logger.warning(
            f"[Watchdog] {len(resource_issues)} resource issue(s):\n"
            f"{issues_text}"
        )

        # Only send Telegram alert for serious issues
        serious = [
            i for i in resource_issues
            if any(kw in i.lower() for kw in [
                "crashed", "failed to restart", "overloaded",
                "all cloud", "thermal", "low ram",
            ])
        ]
        if serious:
            try:
                await telegram.send_notification(
                    f"🚨 *Watchdog Alert*\n\n"
                    + "\n".join(f"• {i}" for i in serious)
                )
            except Exception:
                pass
