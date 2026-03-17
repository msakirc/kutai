# orchestrator.py
import asyncio
import dataclasses
import json
import os
import signal
from ..app.config import DB_PATH, MAX_CONTEXT_CHAIN_LENGTH, TASK_PRIORITY
from datetime import datetime, timedelta
from ..infra.db import (
    init_db, get_db, close_db, get_ready_tasks, update_task, add_task,
    claim_task, add_subtasks_atomically, log_conversation,
    get_active_goals, get_tasks_for_goal, update_goal, get_daily_stats,
    store_memory, compute_task_hash,
    get_due_scheduled_tasks, update_scheduled_task,
    cancel_task, get_task,
    save_workspace_snapshot, release_task_locks, release_goal_locks,
)
from src.infra.logging_config import get_logger
from .router import _circuit_breakers
from ..agents import get_agent
from ..tools import execute_tool
from ..tools.workspace import (
    get_file_tree,
    get_goal_workspace,
    get_goal_workspace_relative,
    compute_workspace_hashes,
)
from ..tools.git_ops import (
    git_commit, ensure_git_repo,
    create_goal_branch, get_current_branch, get_commit_sha,
)
from ..app.telegram_bot import TelegramInterface

logger = get_logger("core.orchestrator")


    # Default timeouts per agent type (seconds).  Override via
    # tasks.timeout_seconds column for per-task control.
AGENT_TIMEOUTS: dict[str, int] = {
    "planner":        120,
    "architect":      180,
    "coder":          300,
    "implementer":    300,
    "fixer":          240,
    "test_generator": 180,
    "reviewer":       120,
    "visual_reviewer":120,
    "researcher":     180,
    "analyst":        240,
    "writer":         180,
    "summarizer":     120,
    "assistant":      120,
    "executor":       180,
    "error_recovery": 240,
    "pipeline":       600,
    "workflow":       900,  # 15 min — workflow steps can be lengthy
}

# Maximum number of independent tasks to run concurrently.
MAX_CONCURRENT_TASKS: int = int(os.getenv("MAX_CONCURRENT_TASKS", "3"))


def _compute_max_concurrent(tasks: list[dict]) -> int:
    """Compute how many tasks to run concurrently based on task characteristics.

    - Base = MAX_CONCURRENT_TASKS (default 3)
    - Multiple independent goals: +2 per additional goal (up to 8 total)
    - Same goal, phase_8 feature implementations: allow up to 5
    - Hard cap at 8 to avoid overwhelming API rate limits
    """
    if not tasks:
        return MAX_CONCURRENT_TASKS

    # Gather goal_ids from tasks
    goal_ids: set[int] = set()
    for t in tasks:
        ctx = t.get("context", {})
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError):
                ctx = {}
        gid = ctx.get("goal_id") or t.get("goal_id")
        if gid is not None:
            goal_ids.add(gid)

    base = MAX_CONCURRENT_TASKS
    num_goals = len(goal_ids)

    if num_goals > 1:
        # Allow +2 per additional goal beyond the first
        limit = base + 2 * (num_goals - 1)
        return min(limit, 8)

    # Single goal (or no goal info) — check for phase_8 feature implementations
    for t in tasks:
        ctx = t.get("context", {})
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError):
                ctx = {}
        wp = ctx.get("workflow_phase", "")
        step_id = ctx.get("template_step_id", "")
        if wp == "phase_8" or (isinstance(step_id, str) and step_id.startswith("feat.")):
            return min(5, 8)

    return min(base, 8)


class Orchestrator:
    def __init__(self, shutdown_event=None):
        self.telegram = TelegramInterface(self)
        self.running = False
        self.cycle_count = 0
        self.last_digest = datetime.now()
        self.last_scheduler_check = datetime.min
        self.last_decay_check = datetime.min
        self.shutdown_event = shutdown_event or asyncio.Event()
        self._current_task_future = None
        self._model_manager_tasks: list[asyncio.Task] = []


    # ─── NEW: Context Chaining ───────────────────────────────────────────

    async def _inject_chain_context(self, task: dict) -> dict:
        """
        Before executing a task, inject results from completed sibling tasks
        (prior steps in the same goal) and a workspace snapshot into its context.
        This is how step 5 knows what steps 1-4 produced.
        """
        task_context = {}
        raw_context = task.get("context", "{}")
        if isinstance(raw_context, str):
            try:
                task_context = json.loads(raw_context)
            except (json.JSONDecodeError, TypeError):
                task_context = {}
        elif isinstance(raw_context, dict):
            task_context = raw_context

        # ── Gather completed sibling tasks (same parent or same goal) ──
        parent_id = task.get("parent_task_id")
        goal_id = task.get("goal_id")

        prior_steps = []

        if parent_id:
            # Get all completed siblings under the same parent
            db = await get_db()
            cursor = await db.execute(
                """SELECT id, title, result, agent_type, status
                   FROM tasks
                   WHERE parent_task_id = ?
                     AND status = 'completed'
                     AND id != ?
                   ORDER BY completed_at ASC""",
                (parent_id, task["id"])
            )
            siblings = [dict(row) for row in await cursor.fetchall()]

            for sib in siblings:
                result_text = sib.get("result", "")
                # Truncate individual results but keep them meaningful
                if len(result_text) > 1500:
                    result_text = result_text[:1500] + "\n... [truncated]"
                prior_steps.append({
                    "title": sib["title"],
                    "agent_type": sib.get("agent_type", "?"),
                    "status": sib["status"],
                    "result": result_text,
                })

        # Trim total prior context to fit within budget
        total_chars = sum(len(s.get("result", "")) for s in prior_steps)
        while total_chars > MAX_CONTEXT_CHAIN_LENGTH and prior_steps:
            # Remove the oldest/longest results first
            longest_idx = max(range(len(prior_steps)),
                              key=lambda i: len(prior_steps[i].get("result", "")))
            old_result = prior_steps[longest_idx]["result"]
            prior_steps[longest_idx]["result"] = old_result[:500] + "\n... [heavily truncated]"
            total_chars = sum(len(s.get("result", "")) for s in prior_steps)

        if prior_steps:
            task_context["prior_steps"] = prior_steps

        # ── Workspace snapshot (Phase 6: per-goal workspace) ──
        # For coder/reviewer/writer agents, include the file tree
        # from the goal's isolated workspace if available.
        agent_type = task.get("agent_type", "executor")
        goal_id = task.get("goal_id")
        if agent_type in ("coder", "reviewer", "writer", "planner"):
            try:
                # Use per-goal workspace if goal_id exists
                tree_path = (
                    get_goal_workspace_relative(goal_id)
                    if goal_id else ""
                )
                tree = await get_file_tree(path=tree_path, max_depth=3)
                if tree and "File not found" not in tree and len(tree.split("\n")) > 1:
                    task_context["workspace_snapshot"] = tree
                    if goal_id:
                        task_context["workspace_path"] = (
                            get_goal_workspace_relative(goal_id)
                        )
            except Exception as e:
                logger.debug(f"Could not get workspace snapshot: {e}")

        # Write context back to task dict
        task["context"] = json.dumps(task_context)
        return task

    # ─── NEW: Auto-commit after coder tasks ─────────────────────────────

    async def _auto_commit(self, task: dict, result: dict):
        """Auto-commit workspace changes after a successful coder task."""
        try:
            # Use goal-specific workspace path if available
            goal_id = task.get("goal_id")
            repo_path = (
                get_goal_workspace_relative(goal_id) if goal_id else ""
            )
            await ensure_git_repo(repo_path)
            commit_msg = f"Task #{task['id']}: {task.get('title', 'untitled')[:60]}"
            commit_result = await git_commit(commit_msg, path=repo_path)
            if "Nothing to commit" not in commit_result:
                logger.info(f"[Task #{task['id']}] Auto-committed: {commit_msg}")
        except Exception as e:
            logger.debug(f"Auto-commit skipped: {e}")

    # ─── Watchdog ────────────────────────────────────────────

    async def watchdog(self):
        """
        Detect and fix stuck states at BOTH task and resource level.

        Task-level:
          - Tasks stuck in processing
          - Tasks blocked by failed dependencies
          - Goals with all children done but still waiting

        Resource-level:
          - Crashed llama-server → auto-restart
          - GPU OOM / thermal throttle → pause local, route to cloud
          - All cloud providers rate-limited → log warning, wait
          - Backpressure queue overload → alert via Telegram
        """
        db = await get_db()

        # ═══════════════════════════════════════════════════════════
        #  TASK-LEVEL RECOVERY (existing logic, preserved)
        # ═══════════════════════════════════════════════════════════

        # 1. Tasks stuck in "processing" for more than 5 minutes
        cursor = await db.execute(
            """SELECT id, title FROM tasks
               WHERE status = 'processing'
               AND started_at < datetime('now', '-5 minutes')"""
        )
        stuck = [dict(row) for row in await cursor.fetchall()]
        for task in stuck:
            logger.warning(
                f"[Watchdog] Task #{task['id']} stuck in processing, "
                f"resetting"
            )
            await db.execute(
                "UPDATE tasks SET status = 'pending', "
                "retry_count = retry_count + 1 WHERE id = ?",
                (task["id"],)
            )

        # 2. Tasks blocked by FAILED dependencies
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
            failed_cursor = await db.execute(
                f"SELECT COUNT(*) FROM tasks "
                f"WHERE id IN ({placeholders}) AND status = 'failed'",
                deps,
            )
            failed_count = (await failed_cursor.fetchone())[0]

            if failed_count > 0:
                logger.warning(
                    f"[Watchdog] Task #{task['id']} has failed deps, "
                    f"clearing"
                )
                await db.execute(
                    "UPDATE tasks SET depends_on = '[]' WHERE id = ?",
                    (task["id"],),
                )

        # 3. Goals with all children done but parent still waiting
        cursor3 = await db.execute(
            "SELECT id, title FROM tasks "
            "WHERE status = 'waiting_subtasks'"
        )
        waiting = [dict(row) for row in await cursor3.fetchall()]
        for task in waiting:
            child_cursor = await db.execute(
                """SELECT COUNT(*) as total,
                   SUM(CASE WHEN status IN (
                       'completed','failed','rejected','cancelled'
                   ) THEN 1 ELSE 0 END) as done
                   FROM tasks WHERE parent_task_id = ?""",
                (task["id"],),
            )
            row = await child_cursor.fetchone()
            if row and row["total"] > 0 and row["total"] == row["done"]:
                logger.info(
                    f"[Watchdog] Task #{task['id']} all subtasks done, "
                    f"marking complete"
                )
                await db.execute(
                    "UPDATE tasks SET status = 'completed', "
                    "completed_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), task["id"]),
                )

        # 4. Escalation tiers for tasks stuck in needs_clarification
        #    Uses started_at as the baseline timestamp (set when task
        #    began processing, before entering needs_clarification).
        #    We compute the threshold in Python with isoformat() so the
        #    string comparison matches the format used when storing
        #    started_at (which also uses datetime.now().isoformat()).
        threshold_24h = (
            datetime.now() - timedelta(hours=24)
        ).isoformat()

        # Tier 0: 4-hour gentle nudge (no escalation count increment)
        threshold_4h = (
            datetime.now() - timedelta(hours=4)
        ).isoformat()
        cursor_nudge = await db.execute(
            """SELECT id, title, context FROM tasks
               WHERE status = 'needs_clarification'
               AND started_at < ?
               AND started_at >= ?""",
            (threshold_4h, threshold_24h),
        )
        nudge_tasks = [dict(row) for row in await cursor_nudge.fetchall()]
        for task in nudge_tasks:
            raw_ctx = task.get("context", "{}")
            if isinstance(raw_ctx, str):
                try:
                    task_ctx = json.loads(raw_ctx)
                except (json.JSONDecodeError, TypeError):
                    task_ctx = {}
            else:
                task_ctx = raw_ctx if isinstance(raw_ctx, dict) else {}

            if not task_ctx.get("nudge_sent"):
                task_ctx["nudge_sent"] = True
                await update_task(task["id"], context=json.dumps(task_ctx))
                await self.telegram.send_notification(
                    f"\U0001f4ac Gentle reminder: Task #{task['id']} needs your input.\n"
                    f"*{task['title']}*"
                )

        cursor_clar = await db.execute(
            """SELECT id, title, context, started_at FROM tasks
               WHERE status = 'needs_clarification'
               AND started_at < ?""",
            (threshold_24h,),
        )
        stale = [dict(row) for row in await cursor_clar.fetchall()]
        for task in stale:
            # Parse escalation_count from task context
            raw_ctx = task.get("context", "{}")
            if isinstance(raw_ctx, str):
                try:
                    task_ctx = json.loads(raw_ctx)
                except (json.JSONDecodeError, TypeError):
                    task_ctx = {}
            else:
                task_ctx = raw_ctx if isinstance(raw_ctx, dict) else {}

            escalation_count = task_ctx.get("escalation_count", 0)
            tid = task["id"]
            ttitle = task["title"]

            # Calculate hours since started_at
            try:
                started = datetime.fromisoformat(task["started_at"])
            except (ValueError, TypeError):
                started = datetime.min
            hours_waiting = (
                datetime.now() - started
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
                await self.telegram.send_notification(
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
                await self.telegram.send_notification(
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
                await self.telegram.send_notification(
                    f"❌ Task #{tid} cancelled — no clarification "
                    f"received after 72h.\n*{ttitle}*"
                )

        # 5. Workflow-level timeout check — pause workflows running too long
        try:
            goal_cursor = await db.execute(
                """SELECT id, title, context, created_at FROM goals
                   WHERE status = 'active'"""
            )
            active_goals = [dict(row) for row in await goal_cursor.fetchall()]
            for goal in active_goals:
                raw_gctx = goal.get("context", "{}")
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
                    created = datetime.fromisoformat(goal["created_at"])
                except (ValueError, TypeError):
                    continue

                elapsed_hours = (datetime.now() - created).total_seconds() / 3600
                if elapsed_hours > timeout_hours:
                    logger.warning(
                        "[Watchdog] Goal #%d exceeded timeout (%dh > %dh), pausing",
                        goal["id"], int(elapsed_hours), timeout_hours,
                    )
                    await update_goal(goal["id"], status="paused")
                    await self.telegram.send_notification(
                        f"⏱️ *Workflow timeout*: Goal #{goal['id']} paused after "
                        f"{int(elapsed_hours)}h (limit: {timeout_hours}h).\n"
                        f"*{goal['title']}*\nUse /resume to continue."
                    )
        except Exception as e:
            logger.warning(f"[Watchdog] Workflow timeout check failed: {e}")

        await db.commit()

        # ═══════════════════════════════════════════════════════════
        #  RESOURCE-LEVEL RECOVERY
        # ═══════════════════════════════════════════════════════════

        resource_issues: list[str] = []

        # 4. Check llama-server health
        try:
            from ..models.local_model_manager import get_local_manager

            manager = get_local_manager()
            if manager.current_model:
                # Server should be running
                if manager.process is None or manager.process.poll() is not None:
                    resource_issues.append(
                        f"llama-server crashed (model: {manager.current_model})"
                    )
                    logger.error(
                        f"[Watchdog] llama-server process died! "
                        f"Attempting restart of {manager.current_model}"
                    )
                    model_name = manager.current_model
                    manager.process = None
                    manager.current_model = None

                    # Attempt restart
                    success = await manager.ensure_model(
                        model_name, reason="watchdog crash recovery",
                    )
                    if success:
                        logger.info(
                            f"[Watchdog] ✅ llama-server recovered: "
                            f"{model_name}"
                        )
                    else:
                        resource_issues.append(
                            f"Failed to restart llama-server for "
                            f"{model_name}"
                        )
        except Exception as e:
            logger.warning(f"[Watchdog] Local model check failed: {e}")

        # 5. Check GPU health
        try:
            from ..models.gpu_monitor import get_gpu_monitor

            gpu_state = get_gpu_monitor().get_state()

            if gpu_state.gpu.available:
                # Thermal throttling
                if gpu_state.gpu.is_thermal_throttling:
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

        # 6. Check GPU scheduler queue depth
        try:
            from ..models.gpu_scheduler import get_gpu_scheduler

            sched = get_gpu_scheduler()
            sched_status = sched.get_status()

            if sched_status["queue_depth"] > 5:
                resource_issues.append(
                    f"GPU scheduler queue deep: "
                    f"{sched_status['queue_depth']} waiting"
                )
                logger.warning(
                    f"[Watchdog] GPU queue depth: "
                    f"{sched_status['queue_depth']} tasks waiting"
                )

        except Exception as e:
            logger.warning(f"[Watchdog] GPU scheduler check failed: {e}")

        # 7. Check backpressure queue
        try:
            from ..infra.backpressure import get_backpressure_queue

            bp = get_backpressure_queue()
            bp_status = bp.get_status()

            if bp_status["queue_depth"] > 10:
                resource_issues.append(
                    f"Backpressure queue overloaded: "
                    f"{bp_status['queue_depth']} calls waiting"
                )
                logger.warning(
                    f"[Watchdog] Backpressure queue: "
                    f"{bp_status['queue_depth']} calls, "
                    f"{bp_status['total_expired']} expired"
                )

        except Exception as e:
            logger.warning(f"[Watchdog] Backpressure check failed: {e}")

        # 8. Check circuit breakers — are ALL cloud providers down?
        try:
            degraded_providers = [
                p for p, cb in _circuit_breakers.items()
                if cb.is_degraded
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

        # 9. Restore rate limits that were adaptively reduced
        try:
            from ..models.rate_limiter import get_rate_limit_manager
            get_rate_limit_manager().restore_limits()
        except Exception as e:
            logger.warning(f"[Watchdog] Rate limit restore failed: {e}")

        # 10. Check for expiring credentials (warn 24h before expiry)
        try:
            from ..security.credential_store import list_credentials, get_credential
            from datetime import timezone

            services = await list_credentials()
            for svc in services:
                cred = await get_credential(svc)
                if cred is None:
                    # Already expired — get_credential returns None
                    resource_issues.append(
                        f"🔑 Credential '{svc}' has expired. Refresh with /credential add."
                    )
                    await self.telegram.send_notification(
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
                    await self.telegram.send_notification(
                        f"🚨 *Watchdog Alert*\n\n"
                        + "\n".join(f"• {i}" for i in serious)
                    )
                except Exception:
                    pass

    # ─── Cron Scheduler ──────────────────────────────────────────────────

    async def check_scheduled_tasks(self):
        """Check for due scheduled tasks and create task instances.

        Runs every 60s alongside the main loop.
        """
        try:
            due = await get_due_scheduled_tasks()
            if not due:
                return

            for sched in due:
                sched_id = sched["id"]
                title = sched["title"]
                logger.info(
                    f"[Scheduler] Triggering scheduled task #{sched_id}: "
                    f"'{title}'"
                )
                sched_ctx = (
                    json.loads(sched.get("context", "{}"))
                    if isinstance(sched.get("context"), str)
                    else sched.get("context", {})
                )
                task_id = await add_task(
                    title=title,
                    description=sched.get("description", ""),
                    agent_type=sched.get("agent_type", "executor"),
                    tier=sched.get("tier", "cheap"),
                    goal_id=sched_ctx.get("goal_id"),
                    context=sched_ctx,
                )
                if task_id:
                    logger.info(
                        f"[Scheduler] Created task #{task_id} from "
                        f"schedule #{sched_id}"
                    )

                # Update last_run and compute next_run
                now = datetime.now()
                next_run = self._compute_next_run(
                    sched.get("cron_expression", "0 * * * *"), now
                )
                await update_scheduled_task(
                    sched_id,
                    last_run=now.isoformat(),
                    next_run=next_run.isoformat() if next_run else None,
                )

        except Exception as e:
            logger.error(f"[Scheduler] Error checking schedules: {e}")

    @staticmethod
    def _compute_next_run(
        cron_expr: str, after: datetime
    ) -> datetime | None:
        """Simple cron parser supporting: minute hour day month weekday.

        Examples: "0 * * * *" (hourly), "30 9 * * *" (daily 9:30),
                  "0 0 * * 1" (Monday midnight).
        Returns the next datetime after *after*, or None on parse failure.
        """
        try:
            parts = cron_expr.strip().split()
            if len(parts) != 5:
                return None

            minute, hour, day, month, weekday = parts

            # Simple: advance by fixed intervals for common patterns
            from datetime import timedelta

            if minute != "*" and hour == "*":
                # Every hour at minute M
                m = int(minute)
                candidate = after.replace(
                    minute=m, second=0, microsecond=0
                )
                if candidate <= after:
                    candidate += timedelta(hours=1)
                return candidate

            if minute != "*" and hour != "*":
                # Daily at H:M
                m, h = int(minute), int(hour)
                candidate = after.replace(
                    hour=h, minute=m, second=0, microsecond=0
                )
                if candidate <= after:
                    candidate += timedelta(days=1)
                return candidate

            # Fallback: every hour from now
            return after + timedelta(hours=1)
        except Exception:
            return None

    # ─── Core Task Processing ────────────────────────────────────────────

    async def process_task(self, task: dict):
        """Process a single task through the appropriate agent with context injection."""
        task_id = task["id"]
        title = task["title"]
        agent_type = task.get("agent_type", "executor")

        logger.info("task received", task_id=task_id, title=title, agent_type=agent_type)

        task_ctx = {}  # initialized here so except handler can safely access it
        try:
            # Atomic claim — if another worker grabbed it first, skip
            claimed = await claim_task(task_id)
            if not claimed:
                logger.info("task already claimed", task_id=task_id)
                return

            # ── Check for cancellation before starting ──
            fresh = await get_task(task_id)
            if fresh and fresh.get("status") == "cancelled":
                logger.info("task cancelled before execution", task_id=task_id)
                return

            # ── Inject context from prior steps + workspace snapshot ──
            task = await self._inject_chain_context(task)

            # ── Classify task if not already classified ──
            task_ctx = task.get("context", "{}")
            if isinstance(task_ctx, str):
                try:
                    task_ctx = json.loads(task_ctx)
                except (json.JSONDecodeError, TypeError):
                    task_ctx = {}
            if not isinstance(task_ctx, dict):
                task_ctx = {}

            if "classification" not in task_ctx:
                from .task_classifier import classify_task as classify
                classification = await classify(
                    task["title"], task.get("description", ""),
                )
                task_ctx["classification"] = dataclasses.asdict(classification)
                if classification.confidence >= 0.7:
                    task["agent_type"] = classification.agent_type
                    agent_type = classification.agent_type
                task["context"] = json.dumps(task_ctx)

            # ── Workflow step pre-hook: inject artifact context ──
            from ..workflows.engine.hooks import (
                pre_execute_workflow_step,
                post_execute_workflow_step,
                is_workflow_step,
            )
            if is_workflow_step(task_ctx):
                task = await pre_execute_workflow_step(task)

                # Check if this step should delegate to CodingPipeline
                from ..workflows.engine.pipeline_bridge import should_delegate_to_pipeline
                template_step_id = task_ctx.get("template_step_id", "")
                if should_delegate_to_pipeline(template_step_id, agent_type):
                    agent_type = "pipeline"
                    task["agent_type"] = "pipeline"
                    logger.info("workflow step delegated to pipeline", task_id=task_id)

            # ── Human approval gate ──
            if task_ctx.get("human_gate"):
                try:
                    logger.info("human approval gate triggered", task_id=task_id)
                    approved = await self.telegram.request_approval(
                        task_id,
                        task.get("title", ""),
                        task.get("description", "")[:200],
                        tier=task.get("tier", "auto"),
                        goal_id=task.get("goal_id"),
                    )
                    if not approved:
                        logger.info("human gate rejected", task_id=task_id)
                        await update_task(task_id, status="paused")
                        return
                    logger.info("human gate approved", task_id=task_id)
                except Exception as e:
                    logger.error("human gate error", task_id=task_id, error=str(e))

            # ── Phase 6: Snapshot workspace before coder/pipeline tasks ──
            goal_id = task.get("goal_id")
            if goal_id and agent_type in ("coder", "pipeline", "implementer", "fixer"):
                try:
                    ws_path = get_goal_workspace(goal_id)
                    hashes = compute_workspace_hashes(ws_path)
                    repo_path = get_goal_workspace_relative(goal_id)
                    sha = await get_commit_sha(path=repo_path)
                    branch = await get_current_branch(path=repo_path)
                    await save_workspace_snapshot(
                        goal_id=goal_id,
                        file_hashes=hashes,
                        task_id=task_id,
                        branch_name=branch,
                        commit_sha=sha,
                    )
                except Exception as e:
                    logger.debug(f"[Task #{task_id}] Snapshot skipped: {e}")

            # ── Determine timeout ──
            timeout_seconds = (
                task.get("timeout_seconds")
                or AGENT_TIMEOUTS.get(agent_type, 180)
            )

            if agent_type == "pipeline":
                from ..workflows.pipeline import CodingPipeline
                pipeline = CodingPipeline()
                logger.info("delegating to pipeline", task_id=task_id)
                coro = pipeline.run(task)
            else:
                agent = get_agent(agent_type)
                logger.info(
                    "agent dispatched",
                    task_id=task_id,
                    agent_name=agent.name,
                    agent_type=agent_type,
                    tier=task.get('tier', 'auto'),
                    timeout_seconds=timeout_seconds,
                )
                coro = agent.execute(task)

            # Wrap with timeout
            try:
                result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                timeout_err = f"TimeoutError: Task timed out after {timeout_seconds}s"
                logger.error("task timeout", task_id=task_id, timeout_seconds=timeout_seconds, error=timeout_err)
                await update_task(
                    task_id, status="failed", error=timeout_err
                )
                await self.telegram.send_error(task_id, title, timeout_err)

                # Spawn error recovery for timeouts too
                await self._spawn_error_recovery(task, timeout_err)
                return

            status = result.get("status", "completed")

            logger.info("result received", task_id=task_id, status=status)

            # Auto-commit after successful coder tasks
            if status == "completed" and agent_type == "coder":
                await self._auto_commit(task, result)

            if status == "completed":
                # Extract structured pipeline artifacts before the post-hook
                if agent_type == "pipeline" and is_workflow_step(task_ctx):
                    try:
                        from ..workflows.engine.pipeline_artifacts import extract_pipeline_artifacts
                        from ..workflows.engine.hooks import get_artifact_store

                        ws_path = None
                        if task.get("goal_id"):
                            try:
                                ws_path = get_goal_workspace(task["goal_id"])
                            except Exception:
                                pass

                        extra_artifacts = await extract_pipeline_artifacts(task, result, ws_path)
                        if extra_artifacts:
                            store = get_artifact_store()
                            goal_id = task.get("goal_id")
                            for name, content in extra_artifacts.items():
                                await store.store(goal_id, name, content)
                            logger.info(f"[Task #{task_id}] Stored {len(extra_artifacts)} pipeline artifacts")
                    except Exception as e:
                        logger.debug(f"[Task #{task_id}] Pipeline artifact extraction failed: {e}")

                # Workflow step post-hook: store output artifacts
                if is_workflow_step(task_ctx):
                    await post_execute_workflow_step(task, result)
                await self._handle_complete(task, result)
            elif status == "needs_subtasks":
                await self._handle_subtasks(task, result)
            elif status == "needs_clarification":
                await self._handle_clarification(task, result)
            elif status == "needs_review":
                await self._handle_review(task, result)
            else:
                logger.warning("unknown task status", task_id=task_id, status=status)
                await self._handle_complete(task, result)

            # ── Phase 6: Release file locks held by this task ──
            try:
                await release_task_locks(task_id)
            except Exception:
                pass

        except Exception as e:
            logger.exception("task failed", task_id=task_id, error_type=type(e).__name__, error=str(e))
            # Release locks on failure too
            try:
                await release_task_locks(task_id)
            except Exception:
                pass
            retry_count = task.get("retry_count", 0)
            max_retries = task.get("max_retries", 3)

            if retry_count < max_retries:
                await update_task(task_id, status="pending",
                                  retry_count=retry_count + 1,
                                  error=f"{type(e).__name__}: {str(e)[:200]}")
                logger.info("task will retry", task_id=task_id, retry_count=retry_count + 1, max_retries=max_retries)
            else:
                error_str = f"{type(e).__name__}: {str(e)[:500]}"
                # Classify the error for analytics and DLQ routing
                try:
                    from ..infra.dead_letter import _classify_error
                    error_cat = _classify_error(error_str, "unknown")
                except Exception:
                    error_cat = "unknown"
                await update_task(
                    task_id, status="failed", error=error_str,
                    error_category=error_cat,
                )
                await self.telegram.send_error(task_id, title, error_str)

                # ── Fix #9: Workflow step failure notification ──
                if task_ctx.get("is_workflow_step"):
                    try:
                        wf_phase = task_ctx.get("workflow_phase", "?")
                        await self.telegram.send_notification(
                            f"Workflow step failed: #{task_id}\n"
                            f"_{task.get('title', '')[:60]}_\n"
                            f"Phase: {wf_phase}"
                        )
                    except Exception:
                        pass

                # Phase 11.2: Store failed task in episodic memory
                try:
                    from ..memory.episodic import store_task_result
                    await store_task_result(
                        task=task, result=error_str, model="unknown",
                        cost=0.0, duration=0.0, success=False,
                    )
                except Exception:
                    pass

                # ── Spawn error recovery agent ──
                await self._spawn_error_recovery(task, error_str)

                # ── Dead-letter queue: quarantine permanently failed tasks ──
                try:
                    from ..infra.dead_letter import quarantine_task
                    await quarantine_task(
                        task_id=task_id,
                        goal_id=task.get("goal_id"),
                        error=error_str,
                        original_agent=task.get("agent_type", "executor"),
                        retry_count=max_retries,
                    )
                except Exception as dlq_err:
                    logger.error("dlq quarantine failed", task_id=task_id, error=str(dlq_err))

    # ─── Result Handlers ─────────────────────────────────────────────────

    async def _handle_complete(self, task, result):
        task_id = task["id"]
        result_text = result.get("result", "No result")
        model = result.get("model", "unknown")
        cost = result.get("cost", 0)
        iterations = result.get("iterations", 1)

        # Parse task context for workflow-aware handling
        task_ctx = task.get("context", "{}")
        if isinstance(task_ctx, str):
            try:
                task_ctx = json.loads(task_ctx)
            except (json.JSONDecodeError, ValueError):
                task_ctx = {}
        if not isinstance(task_ctx, dict):
            task_ctx = {}

        await update_task(
            task_id, status="completed", result=result_text,
            completed_at=datetime.now().isoformat()
        )

        if task.get("goal_id"):
            await self._check_goal_completion(task["goal_id"])

        # ── Fix #8: Goal cost accumulator ──
        if task.get("goal_id") and cost > 0:
            try:
                from ..collaboration.blackboard import read_blackboard, write_blackboard
                goal_id = task["goal_id"]
                current = await read_blackboard(goal_id, "cost_tracking")
                if not isinstance(current, dict):
                    current = {"total_cost": 0.0, "task_count": 0, "by_phase": {}}
                current["total_cost"] = current.get("total_cost", 0.0) + cost
                current["task_count"] = current.get("task_count", 0) + 1
                # Track by phase
                phase = task_ctx.get("workflow_phase", "unknown") if isinstance(task_ctx, dict) else "unknown"
                phase_costs = current.get("by_phase", {})
                phase_costs[phase] = phase_costs.get(phase, 0.0) + cost
                current["by_phase"] = phase_costs
                await write_blackboard(goal_id, "cost_tracking", current)
                # Budget warning at milestones
                if current["total_cost"] > 0:
                    for threshold in [1.0, 5.0, 10.0]:
                        prev = current["total_cost"] - cost
                        if prev < threshold <= current["total_cost"]:
                            await self.telegram.send_notification(
                                f"Goal #{goal_id} cost milestone: ${current['total_cost']:.2f}\n"
                                f"({current['task_count']} tasks completed)"
                            )
                            break
            except Exception as e:
                logger.debug(f"Cost tracking update failed: {e}")

        # Notify for top-level tasks or multi-iteration tasks
                # Always notify for interactive (critical priority) tasks
        # Skip only background subtasks from goal decomposition
        is_interactive = task.get("priority", 5) >= TASK_PRIORITY.get("critical", 10)
        is_goal_subtask = task.get("goal_id") and task.get("parent_task_id")

        if is_interactive or not is_goal_subtask:
            await self.telegram.send_result(task_id, task["title"],
                                            result_text, model, cost)
        elif iterations > 3:
            await self.telegram.send_notification(
                f"🔧 Task #{task_id} completed after {iterations} iterations\n"
                f"_{task['title'][:60]}_"
            )

        logger.info("task completed", task_id=task_id, model=model, cost=cost, iterations=iterations)

        # ── Fix #9: Workflow phase completion notification ──
        if task_ctx.get("is_workflow_step") and task.get("goal_id"):
            try:
                from ..workflows.engine.status import compute_phase_progress
                goal_id = task["goal_id"]
                workflow_phase = task_ctx.get("workflow_phase", "")
                all_tasks = await get_tasks_for_goal(goal_id)
                progress = compute_phase_progress(all_tasks)
                current_phase = progress.get(workflow_phase, {})
                if (current_phase.get("completed", 0) == current_phase.get("total", 0)
                        and current_phase.get("total", 0) > 0):
                    total_phases = len(progress)
                    completed_phases = sum(
                        1 for p in progress.values()
                        if p.get("completed", 0) == p.get("total", 0)
                    )
                    phase_name = current_phase.get("name", workflow_phase)
                    await self.telegram.send_notification(
                        f"Phase '{phase_name}' complete for goal #{goal_id}\n"
                        f"Progress: {completed_phases}/{total_phases} phases"
                    )
            except Exception as e:
                logger.debug(f"Workflow progress notification failed: {e}")

        # ── Phase 11.2: Store episodic memory ──
        try:
            from ..memory.episodic import store_task_result
            await store_task_result(
                task=task, result=result_text, model=model,
                cost=cost, duration=0.0, success=True,
            )
        except Exception as e:
            logger.debug(f"Episodic memory store failed (non-critical): {e}")

        # ── Phase 11.7: Record implicit acceptance feedback ──
        try:
            from ..memory.preferences import record_feedback
            await record_feedback(task, "accepted")
        except Exception as e:
            logger.debug(f"Preference feedback failed (non-critical): {e}")

        # ── Error recovery: extract and store diagnosis ──
        agent_type = task.get("agent_type", "")
        if agent_type == "error_recovery":
            try:
                await self._process_recovery_result(task, result_text)
            except Exception as e:
                logger.warning(f"[Task #{task_id}] Recovery result processing failed: {e}")

    async def _handle_subtasks(self, task, result):
        task_id = task["id"]
        goal_id = task.get("goal_id")
        subtasks = result.get("subtasks", [])

        if not subtasks:
            await self._handle_complete(task, result)
            return

        MAX_SUBTASKS = 8
        if len(subtasks) > MAX_SUBTASKS:
            dropped_titles = [
                s.get("title", "?")[:60] for s in subtasks[MAX_SUBTASKS:]
            ]
            logger.warning(
                f"[Task #{task_id}] Planner created {len(subtasks)} subtasks, "
                f"capping at {MAX_SUBTASKS}. Dropped: {dropped_titles}"
            )
            try:
                await self.telegram.send_notification(
                    f"⚠️ *Subtask cap* (Task #{task_id})\n"
                    f"Created {len(subtasks)}, keeping {MAX_SUBTASKS}.\n"
                    f"Dropped: {', '.join(dropped_titles)}"
                )
            except Exception:
                pass
            subtasks = subtasks[:MAX_SUBTASKS]

        # ── Phase 13.2: Plan verification ──
        try:
            from ..collaboration.plan_verification import verify_plan
            issues = verify_plan(subtasks, goal_budget=10.0)
            if issues:
                logger.warning(
                    f"[Task #{task_id}] Plan verification found "
                    f"{len(issues)} issue(s): " + "; ".join(issues)
                )
                await self.telegram.send_notification(
                    f"⚠️ *Plan Issues (Task #{task_id})*\n"
                    + "\n".join(f"  • {i}" for i in issues)
                )
        except Exception as e:
            logger.debug(f"Plan verification failed (non-critical): {e}")

        # Pre-process subtasks: resolve dep_step references.
        # We need a two-pass approach because depends_on_step references
        # IDs that are created during insertion.
        processed: list[dict] = []
        for i, st in enumerate(subtasks):
            processed.append({
                "title": st.get("title", f"Subtask {i+1}")[:80],
                "description": st.get("description", "")[:2000],
                "agent_type": st.get("agent_type", "executor"),
                "tier": st.get("tier", "auto"),
                "priority": st.get("priority", task.get("priority", 5)),
                "depends_on": [],  # resolved after creation
                "_dep_step": st.get("depends_on_step"),
            })

        # Classify each subtask for difficulty and agent routing
        from .task_classifier import classify_task as classify
        for st in processed:
            cls = await classify(st["title"], st.get("description", ""))
            st["difficulty"] = cls.difficulty
            if cls.confidence > 0.8 and cls.agent_type != st["agent_type"]:
                st["agent_type"] = cls.agent_type

        # Use transactional batch insert
        plan_summary = result.get("plan_summary", f"Created {len(subtasks)} subtasks")
        created_ids = await add_subtasks_atomically(
            parent_task_id=task_id,
            subtasks=processed,
            goal_id=goal_id,
            parent_status="waiting_subtasks",
            parent_result=plan_summary,
        )

        # Post-process: wire up depends_on_step references now that we
        # have the created IDs.  This requires individual updates for
        # tasks that reference earlier siblings.
        for i, st in enumerate(processed):
            dep_step = st.get("_dep_step")
            if (
                dep_step is not None
                and isinstance(dep_step, int)
                and 0 <= dep_step < len(created_ids)
                and created_ids[dep_step] > 0
                and created_ids[i] > 0
            ):
                await update_task(
                    created_ids[i],
                    depends_on=json.dumps([created_ids[dep_step]])
                )

        subtask_list = "\n".join(
            f"  {i+1}. [{st.get('agent_type', '?')}] {st.get('title', '?')[:50]}"
            for i, st in enumerate(subtasks)
        )
        await self.telegram.send_notification(
            f"📋 *Plan — Task #{task_id}*\n"
            f"**{task['title'][:60]}**\n\n"
            f"{plan_summary}\n\n"
            f"{subtask_list}\n\n"
            f"_Working autonomously..._"
        )

        logger.info(f"[Task #{task_id}] Decomposed into {len(subtasks)} subtasks")

    async def _handle_clarification(self, task, result):
        task_id = task["id"]
        question = result.get("clarification", "Need more information")
        await update_task(task_id, status="needs_clarification")
        await self.telegram.request_clarification(task_id, task["title"], question)
        logger.info(f"[Task #{task_id}] Asking human for clarification")

    async def _handle_review(self, task, result):
        task_id = task["id"]
        content = result.get("result", "")
        review_note = result.get("review_note", "Agent requested human review")

        review_task_id = await add_task(
            title=f"Review: {task['title'][:40]}",
            description=f"Review this output:\n\n{content}\n\nNote: {review_note}",
            goal_id=task.get("goal_id"),
            parent_task_id=task_id,
            agent_type="reviewer",
            tier="medium",
            depends_on=[task_id]
        )

        await update_task(task_id, status="completed", result=content,
                          completed_at=datetime.now().isoformat())
        if review_task_id:
            logger.info(f"[Task #{task_id}] Sent to reviewer (Task #{review_task_id})")
        else:
            logger.info(f"[Task #{task_id}] Review task deduped, skipping")

    # ─── Error Recovery ─────────────────────────────────────────────────

    async def _spawn_error_recovery(self, failed_task: dict, error_str: str):
        """Spawn an error_recovery agent to diagnose and learn from a failed task.

        Skips if:
          - The failed task is itself an error_recovery task (prevent loops)
          - The failed task is low-priority background work (not worth diagnosing)
        """
        task_id = failed_task["id"]
        agent_type = failed_task.get("agent_type", "")

        # Never spawn recovery for recovery tasks — prevent infinite loops
        if agent_type == "error_recovery":
            logger.debug(f"[Task #{task_id}] Skipping error recovery for error_recovery task")
            return

        # Skip for very low priority tasks (background noise)
        if failed_task.get("priority", 5) <= 1:
            logger.debug(f"[Task #{task_id}] Skipping error recovery for low-priority task")
            return

        title = failed_task.get("title", "Unknown")
        description = failed_task.get("description", "")
        goal_id = failed_task.get("goal_id")

        recovery_description = (
            f"## Failed Task Diagnosis\n\n"
            f"**Original Task:** {title}\n"
            f"**Agent:** {agent_type}\n"
            f"**Task ID:** {task_id}\n"
            f"**Retries exhausted:** {failed_task.get('max_retries', 3)}\n\n"
            f"**Error:**\n```\n{error_str}\n```\n\n"
            f"**Task Description:**\n{description[:1000]}\n\n"
            f"Diagnose the root cause of this failure. If you can fix the "
            f"underlying issue (missing file, bad config, etc.), do so. "
            f"Report what went wrong and how to prevent it."
        )

        # Use a stable title for dedup — error text varies between retries
        # but we only want ONE recovery task per failed task
        stable_title = f"Error recovery: task#{task_id}"

        try:
            recovery_task_id = await add_task(
                title=stable_title,
                description=recovery_description,
                goal_id=goal_id,
                parent_task_id=task_id,
                agent_type="error_recovery",
                tier="medium",
                priority=max(failed_task.get("priority", 5), 7),
                context={
                    "failed_task_id": task_id,
                    "failed_agent_type": agent_type,
                    "error": error_str,
                    "original_title": title,
                },
            )
            if recovery_task_id:
                logger.info(
                    f"[Task #{task_id}] Spawned error recovery → Task #{recovery_task_id}"
                )
            else:
                logger.info(f"[Task #{task_id}] Error recovery task deduped, skipping")
        except Exception as e:
            logger.warning(f"[Task #{task_id}] Failed to spawn error recovery: {e}")

    async def _process_recovery_result(self, task: dict, result_text: str):
        """Extract diagnosis from error recovery result and store in episodic memory."""
        ctx = task.get("context")
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError):
                ctx = {}
        ctx = ctx or {}

        failed_task_id = ctx.get("failed_task_id")
        failed_agent_type = ctx.get("failed_agent_type", "unknown")
        original_title = ctx.get("original_title", task.get("title", ""))
        error_str = ctx.get("error", "unknown error")

        # Parse structured fields from the recovery report
        root_cause = ""
        category = ""
        fix_applied = ""

        for line in result_text.split("\n"):
            line_lower = line.strip().lower()
            if line_lower.startswith("**root cause:**"):
                root_cause = line.split(":", 1)[-1].strip().strip("*")
            elif line_lower.startswith("**category:**"):
                category = line.split(":", 1)[-1].strip().strip("*")
            elif line_lower.startswith("**fix applied:**"):
                fix_applied = line.split(":", 1)[-1].strip().strip("*")

        # Fallback: use the full result if parsing didn't find fields
        if not root_cause:
            root_cause = result_text[:300]

        # Build a concise error signature
        error_sig = error_str.split("\n")[0][:150]
        if category:
            error_sig = f"[{category}] {error_sig}"

        # Store the recovery pattern for future tasks
        try:
            from ..memory.episodic import store_error_recovery
            await store_error_recovery(
                task={
                    "id": failed_task_id or task["id"],
                    "title": original_title,
                    "agent_type": failed_agent_type,
                },
                error_signature=error_sig,
                root_cause=root_cause,
                fix_applied=fix_applied or "No fix applied — diagnosis only",
                prevention_hint=f"Agent: {failed_agent_type}. {root_cause[:100]}",
            )
            logger.info(
                f"[ErrorRecovery] Stored pattern for task #{failed_task_id}: "
                f"{error_sig[:80]}"
            )
        except Exception as e:
            logger.warning(f"[ErrorRecovery] Failed to store recovery pattern: {e}")

        # Notify about the recovery outcome
        fixed = bool(fix_applied and "no fix" not in fix_applied.lower())
        emoji = "🔧" if fixed else "🔍"
        try:
            await self.telegram.send_notification(
                f"{emoji} *Error Recovery — Task #{failed_task_id}*\n\n"
                f"**Task:** {original_title[:60]}\n"
                f"**Root Cause:** {root_cause[:150]}\n"
                f"**Fix:** {fix_applied[:150] if fix_applied else 'Diagnosis only'}"
            )
        except Exception:
            pass

    # ─── Goal Completion ─────────────────────────────────────────────────

    async def _check_goal_completion(self, goal_id):
        """Check if all tasks for a goal are done."""
        tasks = await get_tasks_for_goal(goal_id)
        if not tasks:
            return

        statuses = [t["status"] for t in tasks]
        pending = [s for s in statuses if s not in ("completed", "failed", "rejected")]

        if not pending:
            completed = [t for t in tasks if t["status"] == "completed"]
            failed = [t for t in tasks if t["status"] == "failed"]

            await update_goal(goal_id, status="completed",
                              completed_at=datetime.now().isoformat())

            # Phase 6: Release all locks held by this goal
            try:
                await release_goal_locks(goal_id)
            except Exception:
                pass

            results_summary = "\n".join(
                f"• {t['title']}: {(t.get('result') or '')[:100]}"
                for t in completed[-10:]
            )

            await self.telegram.send_notification(
                f"🎯 *Goal Completed!*\n\n"
                f"Tasks: {len(completed)} completed, {len(failed)} failed\n\n"
                f"Results:\n{results_summary}"
            )

    # ─── Goal Planning ───────────────────────────────────────────────────

    async def plan_goal(self, goal_id: int, title: str, description: str):
        """Create initial planning task for a new goal."""
        # ── Phase 6: Set up per-goal workspace + branch ──
        try:
            goal_ws = get_goal_workspace(goal_id)
            await ensure_git_repo(get_goal_workspace_relative(goal_id))
            branch = await create_goal_branch(
                goal_id, title,
                path=get_goal_workspace_relative(goal_id),
            )
            if not branch.startswith("❌"):
                logger.info(
                    f"[Goal #{goal_id}] Created workspace + branch: {branch}"
                )
        except Exception as e:
            logger.debug(f"[Goal #{goal_id}] Workspace setup skipped: {e}")

        await add_task(
            title=f"Plan: {title[:40]}",
            description=f"Create an execution plan for this goal:\n\n{title}\n\n{description}",
            goal_id=goal_id,
            agent_type="planner",
            priority=TASK_PRIORITY["high"],
        )

    # ─── Daily Digest ────────────────────────────────────────────────────

    async def daily_digest(self):
        """Send daily status summary."""
        stats = await get_daily_stats()
        goals = await get_active_goals()

        goals_text = "\n".join(f"  • {g['title']}" for g in goals[:5]) or "  None"

        await self.telegram.send_notification(
            f"📊 *Daily Digest*\n\n"
            f"**Tasks today:**\n"
            f"  ✅ Completed: {stats['completed']}\n"
            f"  ⏳ Pending: {stats['pending']}\n"
            f"  ⚙️ Processing: {stats['processing']}\n"
            f"  ❌ Failed: {stats['failed']}\n"
            f"  💰 Cost today: ${stats['today_cost']:.4f}\n\n"
            f"**Active goals:**\n{goals_text}"
        )

    # ─── Main Loop ───────────────────────────────────────────────────────

    async def run_loop(self):
        """Main autonomous work loop."""
        self.running = True
        logger.info("🚀 Autonomous orchestrator started")

        # Ensure workspace and git are ready
        try:
            import os
            os.makedirs("workspace", exist_ok=True)
            await ensure_git_repo()
        except Exception as e:
            logger.warning(f"Workspace/git init: {e}")

        while self.running and not self.shutdown_event.is_set():
            try:
                self.cycle_count += 1

                if self.cycle_count % 10 == 0:
                    await self.watchdog()

                # ── Cron scheduler check (every 60s) ──
                sched_elapsed = (
                    datetime.now() - self.last_scheduler_check
                ).total_seconds()
                if sched_elapsed >= 60:
                    await self.check_scheduled_tasks()
                    self.last_scheduler_check = datetime.now()

                # Get a generous batch, then compute how many to actually run
                candidate_tasks = await get_ready_tasks(limit=8)
                max_concurrent = _compute_max_concurrent(candidate_tasks)
                tasks = candidate_tasks[:max_concurrent]

                if tasks:
                    task_names = [
                        f"#{t['id']}({t.get('agent_type','?')})"
                        for t in tasks
                    ]
                    logger.info(
                        f"[Cycle {self.cycle_count}] "
                        f"Processing {len(tasks)} task(s): {task_names}"
                    )

                    # Partition tasks: at most 1 local-model task runs at a time
                    if len(tasks) > 1:
                        local_tasks = []
                        cloud_tasks = []
                        for t in tasks:
                            ctx = t.get("context", "{}")
                            if isinstance(ctx, str):
                                try:
                                    ctx = json.loads(ctx)
                                except (json.JSONDecodeError, TypeError):
                                    ctx = {}
                            cls = ctx.get("classification", {}) if isinstance(ctx, dict) else {}
                            if cls.get("local_only", False):
                                local_tasks.append(t)
                            else:
                                cloud_tasks.append(t)

                        # Run at most 1 local task concurrently with cloud tasks
                        batch = cloud_tasks + local_tasks[:1]
                        deferred = local_tasks[1:]
                    else:
                        batch = tasks
                        deferred = []

                    if len(batch) == 1:
                        # Single task — run directly
                        t = batch[0]
                        try:
                            self._current_task_future = asyncio.ensure_future(
                                self.process_task(t)
                            )
                            await self._current_task_future
                            self._current_task_future = None
                        except Exception as e:
                            self._current_task_future = None
                            logger.error(
                                f"Task #{t['id']} error: {e}",
                                exc_info=True,
                            )
                    else:
                        # Multiple tasks — run concurrently
                        futures = [
                            asyncio.ensure_future(self.process_task(t))
                            for t in batch
                        ]
                        results = await asyncio.gather(
                            *futures, return_exceptions=True
                        )
                        for t, res in zip(batch, results):
                            if isinstance(res, Exception):
                                logger.error(
                                    f"Task #{t['id']} error: {res}",
                                    exc_info=True,
                                )

                    # Run deferred local tasks sequentially
                    for t in deferred:
                        try:
                            await self.process_task(t)
                        except Exception as e:
                            logger.error(
                                f"Task #{t['id']} error: {e}",
                                exc_info=True,
                            )

                    await asyncio.sleep(2)
                else:
                    if self.cycle_count % 20 == 0:
                        logger.info(f"[Cycle {self.cycle_count}] Idle")
                    await asyncio.sleep(3)

                hours_since_digest = (datetime.now() - self.last_digest).total_seconds() / 3600
                if hours_since_digest >= 24:
                    await self.daily_digest()
                    self.last_digest = datetime.now()

                # ── Phase 11.6: Memory decay (weekly) ──
                days_since_decay = (
                    datetime.now() - self.last_decay_check
                ).total_seconds() / 86400
                if days_since_decay >= 7:
                    try:
                        from ..memory.decay import run_decay_cycle
                        await run_decay_cycle()
                    except Exception as e:
                        logger.debug(f"Memory decay failed (non-critical): {e}")
                    self.last_decay_check = datetime.now()

            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def start(self):
        await init_db()

        # ── Start background infrastructure ──
        from ..models.local_model_manager import get_local_manager
        from ..infra.backpressure import get_backpressure_queue

        manager = get_local_manager()
        bp_queue = get_backpressure_queue()

        self._background_tasks: list[asyncio.Task] = [
            asyncio.create_task(manager.run_idle_unloader()),
            asyncio.create_task(manager.run_health_watchdog()),
            asyncio.create_task(bp_queue.run_processor()),
        ]

        async with self.telegram.app:
            await self.telegram.app.start()
            await self.telegram.app.updater.start_polling()

            logger.info(
                "✅ System online — Telegram + Orchestrator + "
                "GPU Scheduler + Backpressure Queue running"
            )

            try:
                await self.run_loop()
            finally:
                # ── Graceful shutdown ──
                if self.shutdown_event.is_set():
                    logger.info("🛑 Graceful shutdown initiated...")
                    self.running = False

                    # Stop background tasks
                    bp_queue.stop()
                    for t in self._background_tasks:
                        t.cancel()

                    # Wait for current task to finish
                    if (
                        self._current_task_future
                        and not self._current_task_future.done()
                    ):
                        logger.info(
                            "⏳ Waiting for current task to complete "
                            "(60s timeout)..."
                        )
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(self._current_task_future),
                                timeout=60,
                            )
                            logger.info("✅ Current task completed cleanly")
                        except asyncio.TimeoutError:
                            logger.warning(
                                "⚠️ Shutdown timeout — task abandoned"
                            )
                        except Exception as e:
                            logger.warning(
                                f"⚠️ Task error during shutdown: {e}"
                            )

                    logger.info("👋 Orchestrator stopped")

                await close_db()
                await self.telegram.app.updater.stop()
                await self.telegram.app.stop()
