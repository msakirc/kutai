# orchestrator.py
import asyncio
import dataclasses
import json
import os
import signal
from ..app.config import DB_PATH, MAX_CONTEXT_CHAIN_LENGTH, TASK_PRIORITY
from datetime import datetime, timedelta, timezone
from ..infra.db import (
    init_db, get_db, close_db, get_ready_tasks, update_task, add_task,
    claim_task, add_subtasks_atomically, log_conversation,
    get_active_missions, get_tasks_for_mission, update_mission, get_daily_stats,
    store_memory, compute_task_hash,
    get_due_scheduled_tasks, update_scheduled_task,
    cancel_task, get_task, get_mission,
    save_workspace_snapshot, release_task_locks, release_mission_locks,
)
from src.infra.logging_config import get_logger
from .router import _circuit_breakers
from ..agents import get_agent
from ..tools import execute_tool
from ..tools.workspace import (
    get_file_tree,
    get_mission_workspace,
    get_mission_workspace_relative,
    compute_workspace_hashes,
)
from ..tools.git_ops import (
    git_commit, ensure_git_repo,
    create_mission_branch, get_current_branch, get_commit_sha,
)
from ..app.telegram_bot import TelegramInterface

logger = get_logger("core.orchestrator")

# SQLite-compatible datetime format (no 'T', no timezone offset).
# Must match what datetime('now') returns for <= comparisons to work.
_DB_DT_FMT = "%Y-%m-%d %H:%M:%S"


    # Default timeouts per agent type (seconds).  Override via
    # tasks.timeout_seconds column for per-task control.
AGENT_TIMEOUTS: dict[str, int] = {
    "planner":        300,
    "architect":      180,
    "coder":          300,
    "implementer":    300,
    "fixer":          240,
    "test_generator": 180,
    "reviewer":       120,
    "visual_reviewer":120,
    "researcher":     300,
    "analyst":        240,
    "writer":         180,
    "summarizer":     120,
    "assistant":      120,
    "executor":       180,
    "error_recovery": 240,
    "pipeline":       600,
    "workflow":       900,  # 15 min — workflow steps can be lengthy
    "shopping_advisor":    600,
    "product_researcher":  300,
    "deal_analyst":        240,
    "shopping_clarifier":  120,
}

# Maximum number of independent tasks to run concurrently.
MAX_CONCURRENT_TASKS: int = int(os.getenv("MAX_CONCURRENT_TASKS", "3"))


def _compute_max_concurrent(tasks: list[dict]) -> int:
    """Compute how many tasks to run concurrently based on task characteristics.

    - Base = MAX_CONCURRENT_TASKS (default 3)
    - Multiple independent missions: +2 per additional mission (up to 8 total)
    - Same mission, phase_8 feature implementations: allow up to 5
    - Hard cap at 8 to avoid overwhelming API rate limits
    """
    if not tasks:
        return MAX_CONCURRENT_TASKS

    # Gather mission_ids from tasks
    mission_ids: set[int] = set()
    for t in tasks:
        ctx = t.get("context", {})
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError):
                ctx = {}
        mid = ctx.get("mission_id") or t.get("mission_id")
        if mid is not None:
            mission_ids.add(mid)

    base = MAX_CONCURRENT_TASKS
    num_missions = len(mission_ids)

    if num_missions > 1:
        # Allow +2 per additional mission beyond the first
        limit = base + 2 * (num_missions - 1)
        return min(limit, 8)

    # Single mission (or no mission info) — check for phase_8 feature implementations
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


def _reorder_by_model_affinity(tasks: list[dict]) -> list[dict]:
    """Reorder tasks to prefer those matching the currently loaded local model.

    Boost is per-model (what's loaded), reducing swaps by batching compatible
    work. Max boost is +0.9, so a 2+ priority gap is NEVER overridden.

    Returns a new sorted list (does not mutate input).
    """
    if not tasks or len(tasks) <= 1:
        return tasks

    try:
        from src.models.local_model_manager import get_local_manager
        from src.models.model_registry import get_registry
        from src.models.capabilities import (
            score_model_for_task, TASK_PROFILES,
            TaskRequirements as CapTaskReqs,
        )
        from src.core.router import CAPABILITY_TO_TASK, AGENT_REQUIREMENTS

        manager = get_local_manager()
        if not manager.current_model:
            return tasks  # nothing loaded, can't optimize

        registry = get_registry()
        model_info = registry.get(manager.current_model)
        if not model_info:
            return tasks

        def _sort_key(task: dict):
            priority = task.get("priority", 5)
            ctx = task.get("context", {})
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except (json.JSONDecodeError, TypeError):
                    ctx = {}
            cls = ctx.get("classification", {})
            agent_type = task.get("agent_type", cls.get("agent_type", "executor"))
            difficulty = max(1, min(10, int(cls.get("difficulty", 5))))

            # Resolve task key
            task_key = agent_type
            if task_key in CAPABILITY_TO_TASK:
                task_key = CAPABILITY_TO_TASK[task_key]
            template = AGENT_REQUIREMENTS.get(agent_type)
            if template:
                task_key = template.task or task_key

            # Quick capability check: can loaded model handle this task?
            fit = 0.0
            if task_key in TASK_PROFILES:
                cap_reqs = CapTaskReqs(
                    task_name=task_key,
                    needs_function_calling=cls.get("needs_tools", False),
                    needs_vision=cls.get("needs_vision", False),
                )
                min_score = max(0.0, (difficulty - 1) * 0.47)
                cap_score = score_model_for_task(
                    model_info.capabilities,
                    model_info.operational_dict(),
                    cap_reqs,
                )
                if cap_score >= min_score and cap_score > 0:
                    # Normalize fit to 0.0-1.0 range (cap_score is 0-10)
                    fit = min(1.0, cap_score / 10.0)

            # Boost by fit * 0.9, so max boost < 1 priority level
            effective_priority = priority + (fit * 0.9)
            # Sort descending by effective priority, then FIFO by created_at
            return (-effective_priority, task.get("created_at", ""))

        return sorted(tasks, key=_sort_key)

    except Exception as e:
        logger.debug(f"Model affinity reorder failed: {e}")
        return tasks


def _parse_task_difficulty(task: dict) -> int:
    """Extract difficulty from a task's classification context.

    Falls back to 5 (moderate) if not classified yet.
    """
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    cls = ctx.get("classification", {})
    return max(1, min(10, int(cls.get("difficulty", 5))))


class Orchestrator:
    def __init__(self, shutdown_event=None):
        self.telegram = TelegramInterface(self)
        self.running = False
        self._shutting_down = False
        self.cycle_count = 0
        self.last_digest = datetime.now()
        self.last_scheduler_check = datetime.min
        self.last_decay_check = datetime.min
        self.shutdown_event = shutdown_event or asyncio.Event()
        self.requested_exit_code: int | None = None  # Set by /kutai_restart (42) or /kutai_stop (0)
        self._current_task_future = None
        self._running_futures: list[asyncio.Task] = []
        self._model_manager_tasks: list[asyncio.Task] = []


    # ─── NEW: Context Chaining ───────────────────────────────────────────

    async def _inject_chain_context(self, task: dict) -> dict:
        """
        Before executing a task, inject results from completed sibling tasks
        (prior steps in the same mission) and a workspace snapshot into its context.
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

        # ── Gather completed sibling tasks (same parent or same mission) ──
        parent_id = task.get("parent_task_id")
        mission_id = task.get("mission_id")

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

        # ── Workspace snapshot (Phase 6: per-mission workspace) ──
        # For coder/reviewer/writer agents, include the file tree
        # from the mission's isolated workspace if available.
        agent_type = task.get("agent_type", "executor")
        mission_id = task.get("mission_id")
        if agent_type in ("coder", "reviewer", "writer", "planner"):
            try:
                # Use per-mission workspace if mission_id exists
                tree_path = (
                    get_mission_workspace_relative(mission_id)
                    if mission_id else ""
                )
                tree = await get_file_tree(path=tree_path, max_depth=3)
                if tree and "File not found" not in tree and len(tree.split("\n")) > 1:
                    task_context["workspace_snapshot"] = tree
                    if mission_id:
                        task_context["workspace_path"] = (
                            get_mission_workspace_relative(mission_id)
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
            # Use mission-specific workspace path if available
            mission_id = task.get("mission_id")
            repo_path = (
                get_mission_workspace_relative(mission_id) if mission_id else ""
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
          - Missions with all children done but still waiting

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
            """SELECT id, title, retry_count, max_retries FROM tasks
               WHERE status = 'processing'
               AND started_at < datetime('now', '-5 minutes')"""
        )
        stuck = [dict(row) for row in await cursor.fetchall()]
        for task in stuck:
            retry_count = task.get("retry_count", 0) or 0
            max_retries = task.get("max_retries", 3) or 3
            if retry_count >= max_retries:
                logger.warning(
                    f"[Watchdog] Task #{task['id']} stuck in processing "
                    f"and exhausted retries ({retry_count}/{max_retries}), "
                    f"marking failed"
                )
                await db.execute(
                    "UPDATE tasks SET status = 'failed', "
                    "error = 'Stuck in processing — retries exhausted (watchdog)' "
                    "WHERE id = ?",
                    (task["id"],)
                )
            else:
                logger.warning(
                    f"[Watchdog] Task #{task['id']} stuck in processing, "
                    f"resetting (retry {retry_count + 1}/{max_retries})"
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

        # 3. Missions with all children done but parent still waiting
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
                    created = datetime.fromisoformat(mission["created_at"])
                except (ValueError, TypeError):
                    continue

                elapsed_hours = (datetime.now() - created).total_seconds() / 3600
                if elapsed_hours > timeout_hours:
                    logger.warning(
                        "[Watchdog] Mission #%d exceeded timeout (%dh > %dh), pausing",
                        mission["id"], int(elapsed_hours), timeout_hours,
                    )
                    await update_mission(mission["id"], status="paused")
                    await self.telegram.send_notification(
                        f"⏱️ *Workflow timeout*: Mission #{mission['id']} paused after "
                        f"{int(elapsed_hours)}h (limit: {timeout_hours}h).\n"
                        f"*{mission['title']}*\nUse /resume to continue."
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

                # Special handling: todo reminders create suggestion tasks first
                if sched_ctx.get("type") == "todo_reminder":
                    try:
                        await self._start_todo_suggestions()
                    except Exception as e:
                        logger.error(f"[Scheduler] Todo suggestion creation failed: {e}")
                        # Fallback: send reminder without suggestions
                        try:
                            from src.app.reminders import send_todo_reminder
                            if self.telegram:
                                await send_todo_reminder(self.telegram)
                        except Exception:
                            pass
                    # Update last_run / next_run and skip task creation
                    now = datetime.now(timezone.utc)
                    next_run = self._compute_next_run(
                        sched.get("cron_expression", "0 * * * *"), now
                    )
                    await update_scheduled_task(
                        sched_id,
                        last_run=now.strftime(_DB_DT_FMT),
                        next_run=next_run.strftime(_DB_DT_FMT) if next_run else None,
                    )
                    continue

                task_id = await add_task(
                    title=title,
                    description=sched.get("description", ""),
                    agent_type=sched.get("agent_type", "executor"),
                    tier=sched.get("tier", "cheap"),
                    mission_id=sched_ctx.get("mission_id"),
                    context=sched_ctx,
                )
                if task_id:
                    logger.info(
                        f"[Scheduler] Created task #{task_id} from "
                        f"schedule #{sched_id}"
                    )

                # Update last_run and compute next_run
                now = datetime.now(timezone.utc)
                next_run = self._compute_next_run(
                    sched.get("cron_expression", "0 * * * *"), now
                )
                await update_scheduled_task(
                    sched_id,
                    last_run=now.strftime(_DB_DT_FMT),
                    next_run=next_run.strftime(_DB_DT_FMT) if next_run else None,
                )

        except Exception as e:
            logger.error(f"[Scheduler] Error checking schedules: {e}")

    async def _start_todo_suggestions(self):
        """Create one suggestion task per pending todo item."""
        from src.infra.db import get_todos, add_task, get_db
        todos = await get_todos(status="pending")
        if not todos:
            return

        # Skip todos that already had a failed/cancelled suggestion attempt
        # in the last 4 hours — prevents infinite retry loops.
        db = await get_db()
        recently_attempted: set[int] = set()
        for todo in todos:
            cursor = await db.execute(
                """SELECT id FROM tasks
                   WHERE title = ?
                     AND status IN ('failed', 'cancelled')
                     AND created_at > strftime('%Y-%m-%d %H:%M:%S',
                                               datetime('now', '-4 hours'))
                   LIMIT 1""",
                (f"Suggest action for: {todo['title'][:50]}",),
            )
            if await cursor.fetchone():
                recently_attempted.add(todo["id"])

        eligible = [t for t in todos if t["id"] not in recently_attempted]
        if not eligible and todos:
            # All todos had recent failed attempts — just send reminder
            logger.info(
                f"[Todo] All {len(todos)} todos had recent failed suggestion "
                f"attempts, sending reminder without suggestions"
            )
            from src.app.reminders import send_todo_reminder
            if self.telegram:
                await send_todo_reminder(self.telegram)
            return

        batch_id = f"todo_suggest_{int(datetime.now(timezone.utc).timestamp())}"
        task_ids = []

        for todo in eligible:
            task_id = await add_task(
                title=f"Suggest action for: {todo['title'][:50]}",
                description=(
                    f"The user has this todo item: \"{todo['title']}\"\n"
                    f"Description: {todo.get('description') or '(none)'}\n\n"
                    f"Suggest ONE concrete, actionable way you (an AI assistant) could help. "
                    f"Be creative — even mundane items like 'buy milk' could mean "
                    f"price comparison or online ordering. "
                    f"If you genuinely cannot help, just say 'no suggestion'. "
                    f"Reply with ONLY the suggestion, one sentence, no preamble."
                ),
                agent_type="assistant",
                tier="auto",
                priority=3,  # low — background work
                context={
                    "local_only": True,
                    "prefer_quality": True,
                    "silent": True,
                    "todo_suggest_batch": batch_id,
                    "todo_id": todo["id"],
                    "todo_count": len(todos),
                },
            )
            if task_id:
                task_ids.append(task_id)

        if task_ids:
            logger.info(
                f"[Todo] Created {len(task_ids)} suggestion tasks, batch={batch_id}"
            )
        else:
            # All deduped or failed — send reminder without suggestions
            from src.app.reminders import send_todo_reminder
            if self.telegram:
                await send_todo_reminder(self.telegram)

    async def _check_todo_suggestions_complete(self, task_ctx):
        """Check if all suggestion tasks in this batch are done. If so, send reminder."""
        batch_id = task_ctx["todo_suggest_batch"]
        todo_count = task_ctx.get("todo_count", 0)

        # Query all tasks in this batch
        db = await get_db()
        cursor = await db.execute(
            """SELECT id, status, result, context FROM tasks
               WHERE context LIKE ?""",
            (f'%"{batch_id}"%',),
        )
        rows = [dict(r) for r in await cursor.fetchall()]

        # Check if all are terminal (completed or failed)
        terminal = [r for r in rows if r["status"] in ("completed", "failed")]
        if len(terminal) < len(rows):
            return  # Still waiting

        # Collect suggestions
        suggestions = {}
        for row in rows:
            ctx = row.get("context", "{}")
            if isinstance(ctx, str):
                ctx = json.loads(ctx)
            todo_id = ctx.get("todo_id")
            result = row.get("result", "")
            if todo_id and result and "no suggestion" not in result.lower():
                # Clean up the suggestion text
                suggestion = result.strip().strip('"').strip("'")
                if len(suggestion) > 5:  # Skip empty/trivial
                    suggestions[todo_id] = suggestion

        # Send reminder with suggestions
        from src.app.reminders import send_todo_reminder
        if self.telegram:
            await send_todo_reminder(self.telegram, suggestions=suggestions)

        logger.info(
            f"[Todo] Reminder sent with {len(suggestions)}/{len(rows)} suggestions, "
            f"batch={batch_id}"
        )

    async def _check_stale_todo_batches(self):
        """Watchdog: if a suggestion batch has been pending > 5 min, send reminder without it."""
        try:
            db = await get_db()
            cursor = await db.execute(
                """SELECT id, context FROM tasks
                   WHERE status NOT IN ('completed', 'failed', 'cancelled')
                     AND context LIKE '%todo_suggest_batch%'"""
            )
            rows = [dict(r) for r in await cursor.fetchall()]
            if not rows:
                return

            # Group by batch_id and check staleness
            batches: dict[str, list[dict]] = {}
            for row in rows:
                ctx = row.get("context", "{}")
                if isinstance(ctx, str):
                    ctx = json.loads(ctx)
                bid = ctx.get("todo_suggest_batch", "")
                if bid:
                    batches.setdefault(bid, []).append(row)

            now_ts = datetime.now(timezone.utc).timestamp()
            for bid, tasks in batches.items():
                # Extract timestamp from batch_id: "todo_suggest_<unix_ts>"
                try:
                    batch_ts = int(bid.split("_")[-1])
                except (ValueError, IndexError):
                    continue
                age_seconds = now_ts - batch_ts
                if age_seconds > 300:  # 5 minutes
                    logger.warning(
                        f"[Todo] Stale suggestion batch {bid} ({age_seconds:.0f}s old), "
                        f"cancelling {len(tasks)} stuck tasks and sending reminder"
                    )
                    # Cancel stuck tasks
                    for t in tasks:
                        await cancel_task(t["id"])
                    # Send reminder without suggestions
                    from src.app.reminders import send_todo_reminder
                    if self.telegram:
                        await send_todo_reminder(self.telegram)
        except Exception as e:
            logger.error(f"[Todo] Stale batch check failed: {e}")

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
                m = int(minute)
                # Handle comma-separated hours (e.g., "9,11,13,15,17,19,21")
                if "," in hour:
                    hours = sorted(int(h) for h in hour.split(","))
                    # Find next hour that's still in the future today
                    for h in hours:
                        candidate = after.replace(
                            hour=h, minute=m, second=0, microsecond=0
                        )
                        if candidate > after:
                            return candidate
                    # All hours passed today — first hour tomorrow
                    candidate = after.replace(
                        hour=hours[0], minute=m, second=0, microsecond=0
                    )
                    return candidate + timedelta(days=1)
                # Single hour: daily at H:M
                h = int(hour)
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
                # Only let the classifier set agent_type when the task was
                # created with the default ("executor").  Command handlers
                # (e.g. /shop) explicitly set agent_type at creation time;
                # the classifier must not overwrite those.
                if classification.confidence >= 0.7 and agent_type == "executor":
                    task["agent_type"] = classification.agent_type
                    agent_type = classification.agent_type
                if classification.agent_type == "shopping_advisor" and classification.shopping_sub_intent:
                    task_ctx["shopping_workflow"] = classification.shopping_sub_intent
                task["context"] = json.dumps(task_ctx)

            # ── Shopping intent detection fallback ──
            # If the LLM classifier didn't confidently pick shopping_advisor,
            # use regex-based detection from dispatch as a fallback.
            if agent_type not in ("shopping_advisor", "product_researcher",
                                  "deal_analyst", "shopping_clarifier"):
                from ..workflows.engine.dispatch import (
                    should_start_shopping_workflow,
                )
                shopping_wf = should_start_shopping_workflow(title)
                if shopping_wf:
                    agent_type = "shopping_advisor"
                    task["agent_type"] = "shopping_advisor"
                    task_ctx["shopping_workflow"] = shopping_wf
                    task["context"] = json.dumps(task_ctx)
                    logger.info(
                        "shopping intent detected via dispatch fallback",
                        task_id=task_id,
                        shopping_workflow=shopping_wf,
                    )

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
                        mission_id=task.get("mission_id"),
                    )
                    if not approved:
                        logger.info("human gate rejected", task_id=task_id)
                        await update_task(task_id, status="paused")
                        return
                    logger.info("human gate approved", task_id=task_id)
                except Exception as e:
                    logger.error("human gate error", task_id=task_id, error=str(e))

            # ── Phase 14.2: Risk assessment gate ──
            try:
                from ..security.risk_assessor import assess_risk, format_risk_assessment
                risk = assess_risk(
                    task_title=task.get("title", ""),
                    task_description=task.get("description", ""),
                )
                if risk["needs_approval"] and not task_ctx.get("human_gate"):
                    logger.info(
                        "risk gate triggered",
                        task_id=task_id,
                        risk_score=risk["score"],
                        factors=risk["risk_factors"],
                    )
                    approved = await self.telegram.request_approval(
                        task_id,
                        task.get("title", ""),
                        format_risk_assessment(risk),
                        tier=task.get("tier", "auto"),
                        mission_id=task.get("mission_id"),
                    )
                    if not approved:
                        logger.info("risk gate rejected", task_id=task_id)
                        await update_task(task_id, status="paused")
                        return
            except Exception as e:
                logger.debug(f"Risk assessment skipped: {e}")

            # ── Phase 6: Snapshot workspace before coder/pipeline tasks ──
            mission_id = task.get("mission_id")
            if mission_id and agent_type in ("coder", "pipeline", "implementer", "fixer"):
                try:
                    ws_path = get_mission_workspace(mission_id)
                    hashes = compute_workspace_hashes(ws_path)
                    repo_path = get_mission_workspace_relative(mission_id)
                    sha = await get_commit_sha(path=repo_path)
                    branch = await get_current_branch(path=repo_path)
                    await save_workspace_snapshot(
                        mission_id=mission_id,
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
                # Phase 4.6: Wire progress streaming
                async def _progress_cb(tid, iteration, max_iter, summary):
                    if self.telegram:
                        msg = (
                            f"\U0001f504 *Task #{tid}* — iteration {iteration}/{max_iter}\n"
                            f"{summary[:200]}"
                        )
                        try:
                            await self.telegram.send_notification(msg)
                        except Exception:
                            pass

                coro = agent.execute(task, progress_callback=_progress_cb)

            # Wrap with timeout
            try:
                result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                timeout_err = f"TimeoutError: Task timed out after {timeout_seconds}s"
                logger.error("task timeout", task_id=task_id, timeout_seconds=timeout_seconds, error=timeout_err)

                # Try to recover last checkpoint result before failing
                try:
                    from src.infra.db import load_task_checkpoint
                    checkpoint = await load_task_checkpoint(task_id)
                    if checkpoint:
                        last_messages = checkpoint.get("messages", [])
                        # Look for the last assistant message with a tool result
                        for msg in reversed(last_messages):
                            if msg.get("role") == "user" and "Tool Result" in msg.get("content", ""):
                                logger.info(f"[Task #{task_id}] Timeout recovery: using checkpoint from iteration {checkpoint.get('iteration', '?')}")
                                result_text = f"(Partial result from iteration {checkpoint.get('iteration', '?')} before timeout)\n\n{msg['content'][:3000]}"
                                await update_task(task_id, status="completed", result=result_text)
                                await self.telegram.send_result(task_id, title, result_text, "timeout-recovery", 0)
                                return
                except Exception as recovery_err:
                    logger.debug(f"[Task #{task_id}] Checkpoint recovery failed: {recovery_err}")

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
                        if task.get("mission_id"):
                            try:
                                ws_path = get_mission_workspace(task["mission_id"])
                            except Exception:
                                pass

                        extra_artifacts = await extract_pipeline_artifacts(task, result, ws_path)
                        if extra_artifacts:
                            store = get_artifact_store()
                            mission_id = task.get("mission_id")
                            for name, content in extra_artifacts.items():
                                await store.store(mission_id, name, content)
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
                # Silent/background tasks must not ask the user for clarification
                task_ctx = task.get("context") or {}
                if isinstance(task_ctx, str):
                    import json as _json
                    try:
                        task_ctx = _json.loads(task_ctx)
                    except (ValueError, TypeError):
                        task_ctx = {}
                if task_ctx.get("silent"):
                    logger.info(f"[Task #{task_id}] Suppressed clarification (silent task)")
                    await update_task(task_id, status="failed",
                                      error="Insufficient info (silent task, no clarification)")
                else:
                    await self._handle_clarification(task, result)
            elif status == "needs_review":
                await self._handle_review(task, result)
            elif status == "failed":
                error_str = result.get("error", result.get("result", "Unknown error"))
                logger.error("agent returned failure", task_id=task_id,
                             error=error_str[:200])
                await update_task(task_id, status="failed",
                                  error=error_str[:500])
                await self.telegram.send_error(task_id, title, error_str)
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
                        mission_id=task.get("mission_id"),
                        error=error_str,
                        original_agent=task.get("agent_type", "executor"),
                        retry_count=max_retries,
                    )
                except Exception as dlq_err:
                    logger.error("dlq quarantine failed", task_id=task_id, error=str(dlq_err))

                # Record failure for model health monitoring
                try:
                    from ..infra.db import update_model_stats
                    agent_type = task.get("agent_type", "executor")
                    model = result.get("model", "unknown") if isinstance(result, dict) else "unknown"
                    await update_model_stats(
                        model=model,
                        agent_type=agent_type,
                        success=False,
                        cost=0,
                        latency_ms=0,
                        grade=0.0,
                    )
                except Exception as _e:
                    logger.debug("update_model_stats failed", error=str(_e))

                # Phase 9.1: Record failed metrics
                try:
                    from src.infra.metrics import record_task_failed
                    model = result.get("model", "") if isinstance(result, dict) else ""
                    record_task_failed(model=model)
                except Exception:
                    pass

    # ─── Result Handlers ─────────────────────────────────────────────────

    async def _handle_complete(self, task, result):
        task_id = task["id"]
        result_text = result.get("result", "No result")

        # Guard: if result_text is raw JSON with an embedded "result" field,
        # extract just the human-readable part (prevents showing raw JSON to users).
        if isinstance(result_text, str) and result_text.lstrip().startswith("{"):
            try:
                _parsed_result = json.loads(result_text)
                if isinstance(_parsed_result, dict) and "result" in _parsed_result:
                    result_text = _parsed_result["result"]
            except (json.JSONDecodeError, ValueError):
                pass
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
            completed_at=datetime.now().isoformat(),
            cost=cost,
        )

        if task.get("mission_id"):
            await self._check_mission_completion(task["mission_id"])

        # ── Fix #8: Mission cost accumulator ──
        if task.get("mission_id") and cost > 0:
            try:
                from ..collaboration.blackboard import read_blackboard, write_blackboard
                mission_id = task["mission_id"]
                current = await read_blackboard(mission_id, "cost_tracking")
                if not isinstance(current, dict):
                    current = {"total_cost": 0.0, "task_count": 0, "by_phase": {}}
                current["total_cost"] = current.get("total_cost", 0.0) + cost
                current["task_count"] = current.get("task_count", 0) + 1
                # Track by phase
                phase = task_ctx.get("workflow_phase", "unknown") if isinstance(task_ctx, dict) else "unknown"
                phase_costs = current.get("by_phase", {})
                phase_costs[phase] = phase_costs.get(phase, 0.0) + cost
                current["by_phase"] = phase_costs
                await write_blackboard(mission_id, "cost_tracking", current)
                # Budget warning at milestones
                if current["total_cost"] > 0:
                    for threshold in [1.0, 5.0, 10.0]:
                        prev = current["total_cost"] - cost
                        if prev < threshold <= current["total_cost"]:
                            await self.telegram.send_notification(
                                f"Mission #{mission_id} cost milestone: ${current['total_cost']:.2f}\n"
                                f"({current['task_count']} tasks completed)"
                            )
                            break
            except Exception as e:
                logger.debug(f"Cost tracking update failed: {e}")

        # Notify for top-level tasks or multi-iteration tasks
        # Always notify for interactive (critical priority) tasks
        # Skip only background subtasks from mission decomposition
        # Silent tasks (e.g., todo suggestions) skip Telegram notification entirely
        task_ctx_raw = task.get("context", "{}")
        if isinstance(task_ctx_raw, str):
            try:
                task_ctx_parsed = json.loads(task_ctx_raw)
            except (json.JSONDecodeError, ValueError, TypeError):
                task_ctx_parsed = {}
        else:
            task_ctx_parsed = task_ctx_raw
        if not isinstance(task_ctx_parsed, dict):
            task_ctx_parsed = {}
        is_interactive = task.get("priority", 5) >= TASK_PRIORITY.get("critical", 10)
        is_mission_subtask = task.get("mission_id") and task.get("parent_task_id")

        if task_ctx_parsed.get("silent"):
            logger.info("task completed (silent)", task_id=task_id, model=model, cost=cost)
        elif is_interactive or not is_mission_subtask:
            await self.telegram.send_result(task_id, task["title"],
                                            result_text, model, cost)
        elif iterations > 3:
            await self.telegram.send_notification(
                f"🔧 Task #{task_id} completed after {iterations} iterations\n"
                f"_{task['title'][:60]}_"
            )

        logger.info("task completed", task_id=task_id, model=model, cost=cost, iterations=iterations)

        # Check if this is a todo suggestion task — trigger reminder when all done
        task_ctx = task.get("context", {})
        if isinstance(task_ctx, str):
            import json as _json_ctx
            task_ctx = _json_ctx.loads(task_ctx)
        if task_ctx.get("todo_suggest_batch"):
            await self._check_todo_suggestions_complete(task_ctx)

        # Phase 13.2: Extract skill from successful multi-iteration tasks
        if iterations >= 3 and cost > 0:
            try:
                from ..memory.skills import add_skill, record_skill_outcome
                agent_type = task.get("agent_type", "executor")
                title = task.get("title", "")
                desc = task.get("description", "")[:200]
                words = [w for w in title.lower().split() if len(w) > 3 and w.isalpha()]
                if len(words) >= 2:
                    trigger = "|".join(words[:5])
                    skill_name = f"auto:{agent_type}:{title[:40]}"
                    await add_skill(
                        name=skill_name,
                        description=f"Learned from task #{task_id}: {title}",
                        trigger_pattern=trigger,
                        tool_sequence=f"agent={agent_type}, iterations={iterations}, model={model}",
                        examples=desc,
                    )
                    # Record success so find_relevant_skills can discover it
                    await record_skill_outcome(skill_name, success=True)
            except Exception:
                pass

        # ── Fix #9: Workflow phase completion notification ──
        if task_ctx.get("is_workflow_step") and task.get("mission_id"):
            try:
                from ..workflows.engine.status import compute_phase_progress
                mission_id = task["mission_id"]
                workflow_phase = task_ctx.get("workflow_phase", "")
                all_tasks = await get_tasks_for_mission(mission_id)
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
                        f"Phase '{phase_name}' complete for mission #{mission_id}\n"
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

        # Record model performance for health monitoring and routing
        try:
            from ..infra.db import update_model_stats
            agent_type = task.get("agent_type", "executor")
            grade = result.get("grade", 3.0)  # agents may include a self-grade
            await update_model_stats(
                model=model,
                agent_type=agent_type,
                success=True,
                cost=cost,
                latency_ms=int(iterations * 2000),  # rough proxy: iterations * avg_ms
                grade=grade,
            )
        except Exception as _e:
            logger.debug("update_model_stats failed", error=str(_e))

        # Phase 9.1: Record metrics
        try:
            from src.infra.metrics import record_task_complete, record_queue_depth
            cost = result.get("cost", 0.0) if isinstance(result, dict) else 0.0
            model = result.get("model", "") if isinstance(result, dict) else ""
            record_task_complete(model=model, cost=cost)
        except Exception:
            pass

    async def _handle_subtasks(self, task, result):
        task_id = task["id"]
        mission_id = task.get("mission_id")
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
            issues = verify_plan(subtasks, mission_budget=10.0)
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
            mission_id=mission_id,
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
            mission_id=task.get("mission_id"),
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
        mission_id = failed_task.get("mission_id")

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

        # If the failure was a timeout, the local model is likely the problem.
        # Route error recovery to cloud/fast models to avoid repeating the timeout.
        is_timeout = "timed out" in error_str.lower() or "timeout" in error_str.lower()

        # Find which model the failed task used — exclude it from recovery routing
        failed_model = None
        try:
            from ..infra.db import get_last_model_for_task
            failed_model = await get_last_model_for_task(task_id)
            if failed_model:
                logger.info(
                    f"[Task #{task_id}] Error recovery will exclude model: {failed_model}"
                )
        except Exception as e:
            logger.debug(f"[Task #{task_id}] Could not look up failed model: {e}")

        try:
            error_ctx = {
                "failed_task_id": task_id,
                "failed_agent_type": agent_type,
                "error": error_str,
                "original_title": title,
                "prefer_speed": is_timeout,
                "exclude_models": [failed_model] if failed_model else [],
            }

            # Inherit clarification context from failed task
            failed_ctx = failed_task.get("context", {})
            if isinstance(failed_ctx, str):
                try:
                    failed_ctx = json.loads(failed_ctx)
                except (json.JSONDecodeError, TypeError):
                    failed_ctx = {}
            if failed_ctx.get("user_clarification"):
                error_ctx["user_clarification"] = failed_ctx["user_clarification"]
            if failed_ctx.get("clarification_history"):
                error_ctx["clarification_history"] = failed_ctx["clarification_history"]

            recovery_task_id = await add_task(
                title=stable_title,
                description=recovery_description,
                mission_id=mission_id,
                parent_task_id=task_id,
                agent_type="error_recovery",
                tier="medium",
                priority=max(failed_task.get("priority", 5), 7),
                context=error_ctx,
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

    # ─── Mission Completion ───────────────────────────────────────────────

    async def _check_mission_completion(self, mission_id):
        """Check if all tasks for a mission are done."""
        tasks = await get_tasks_for_mission(mission_id)
        if not tasks:
            return

        statuses = [t["status"] for t in tasks]
        pending = [s for s in statuses if s not in ("completed", "failed", "rejected")]

        if not pending:
            completed = [t for t in tasks if t["status"] == "completed"]
            failed = [t for t in tasks if t["status"] == "failed"]

            await update_mission(mission_id, status="completed",
                              completed_at=datetime.now().isoformat())

            # Phase 6: Release all locks held by this mission
            try:
                await release_mission_locks(mission_id)
            except Exception:
                pass

            results_summary = "\n".join(
                f"• {t['title']}: {(t.get('result') or '')[:100]}"
                for t in completed[-10:]
            )

            # Phase 7.4: Generate and save mission completion summary
            total_cost = 0.0
            elapsed_str = ""
            try:
                from src.infra.progress import add_note, NOTE_MILESTONE
                await add_note(
                    content=f"Mission completed: {len(completed)} tasks done, {len(failed)} failed",
                    note_type=NOTE_MILESTONE,
                    mission_id=mission_id,
                )
                mission_info = await get_mission(mission_id)
                mission_title = mission_info.get('title', 'Unknown') if mission_info else 'Unknown'
                mission_created = mission_info.get('created_at', '') if mission_info else ''
                now = datetime.now()

                # Calculate elapsed time
                elapsed_str = ""
                if mission_created:
                    try:
                        from datetime import datetime as _dt
                        created_dt = _dt.fromisoformat(str(mission_created))
                        delta = now - created_dt
                        hours, rem = divmod(int(delta.total_seconds()), 3600)
                        minutes = rem // 60
                        if hours > 0:
                            elapsed_str = f"{hours}h {minutes}m"
                        else:
                            elapsed_str = f"{minutes}m"
                    except Exception:
                        pass

                # Fetch cost data from blackboard
                total_cost = 0.0
                phase_costs: dict = {}
                try:
                    from ..collaboration.blackboard import read_blackboard
                    cost_data = await read_blackboard(mission_id, "cost_tracking")
                    if isinstance(cost_data, dict):
                        total_cost = cost_data.get("total_cost", 0.0)
                        phase_costs = cost_data.get("by_phase", {})
                except Exception:
                    pass

                summary_lines = [
                    f"# Mission #{mission_id} Summary",
                    f"**Title:** {mission_title}",
                    f"**Completed:** {now.isoformat()[:19]}",
                    f"**Tasks:** {len(completed)} completed, {len(failed)} failed",
                ]
                if elapsed_str:
                    summary_lines.append(f"**Duration:** {elapsed_str}")
                if total_cost > 0:
                    summary_lines.append(f"**Total Cost:** ${total_cost:.4f}")
                summary_lines += ["", "## Results"]

                for t in completed[-20:]:
                    result_text = (t.get('result') or '')[:300]
                    summary_lines.append(f"### {t['title']}\n{result_text}\n")

                # Per-phase cost breakdown
                if phase_costs:
                    summary_lines.append("## Cost Breakdown by Phase")
                    for phase_name, phase_cost in sorted(phase_costs.items(), key=lambda x: -x[1]):
                        summary_lines.append(f"- **{phase_name}:** ${phase_cost:.4f}")
                summary_content = "\n".join(summary_lines)
                import os
                results_dir = os.path.join("workspace", "results")
                os.makedirs(results_dir, exist_ok=True)
                summary_path = os.path.join(results_dir, f"mission_{mission_id}_summary.md")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary_content)
                logger.info(f"[Mission #{mission_id}] Completion summary saved to {summary_path}")
            except Exception as exc:
                logger.debug(f"[Mission #{mission_id}] Summary generation failed: {exc}")

            cost_line = f"\nCost: ${total_cost:.4f}" if total_cost > 0 else ""
            time_line = f"\nDuration: {elapsed_str}" if elapsed_str else ""
            await self.telegram.send_notification(
                f"🎯 *Mission Completed!*\n\n"
                f"Tasks: {len(completed)} completed, {len(failed)} failed"
                f"{cost_line}{time_line}\n\n"
                f"Results:\n{results_summary}"
            )

    # ─── Mission Planning ─────────────────────────────────────────────────

    async def plan_mission(self, mission_id: int, title: str, description: str):
        """Create initial planning task for a new mission."""
        # ── Phase 6: Set up per-mission workspace + branch ──
        try:
            mission_ws = get_mission_workspace(mission_id)
            await ensure_git_repo(get_mission_workspace_relative(mission_id))
            branch = await create_mission_branch(
                mission_id, title,
                path=get_mission_workspace_relative(mission_id),
            )
            if not branch.startswith("❌"):
                logger.info(
                    f"[Mission #{mission_id}] Created workspace + branch: {branch}"
                )
        except Exception as e:
            logger.debug(f"[Mission #{mission_id}] Workspace setup skipped: {e}")

        await add_task(
            title=f"Plan: {title[:40]}",
            description=f"Create an execution plan for this mission:\n\n{title}\n\n{description}",
            mission_id=mission_id,
            agent_type="planner",
            priority=TASK_PRIORITY["high"],
        )

    # ─── Daily Digest ────────────────────────────────────────────────────

    async def daily_digest(self):
        """Phase 14.1: Enhanced morning briefing with overnight results and system health."""
        stats = await get_daily_stats()
        missions = await get_active_missions()

        missions_text = "\n".join(f"  - {g['title']}" for g in missions[:5]) or "  None"

        # Gather additional intelligence
        pending_approvals = 0
        try:
            from ..infra.db import get_db
            db = await get_db()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'awaiting_approval'"
            )
            row = await cursor.fetchone()
            pending_approvals = row[0] if row else 0
        except Exception:
            pass

        # System health indicators
        health_lines = []
        try:
            from ..infra.load_manager import get_load_mode
            mode = await get_load_mode()
            health_lines.append(f"  GPU: {mode} mode")
        except Exception:
            pass
        try:
            from ..infra.metrics import get_all_counters
            counters = get_all_counters()
            queue = int(counters.get("queue_depth", 0))
            health_lines.append(f"  Queue: {queue} tasks")
        except Exception:
            pass

        health_text = "\n".join(health_lines) if health_lines else "  All systems normal"
        approval_line = f"\n*Pending approvals:* {pending_approvals}" if pending_approvals else ""

        await self.telegram.send_notification(
            f"*Morning Briefing*\n\n"
            f"*Tasks (last 24h):*\n"
            f"  Completed: {stats['completed']}\n"
            f"  Pending: {stats['pending']}\n"
            f"  Processing: {stats['processing']}\n"
            f"  Failed: {stats['failed']}\n"
            f"  Cost: ${stats['today_cost']:.4f}\n\n"
            f"*Active missions:*\n{missions_text}"
            f"{approval_line}\n\n"
            f"*System health:*\n{health_text}"
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
                # ── Graceful shutdown: stop accepting new tasks ──
                if self._shutting_down:
                    logger.info(
                        "Shutdown flag set — draining running tasks, "
                        "no new tasks will be accepted"
                    )
                    break

                self.cycle_count += 1

                if self.cycle_count % 10 == 0:
                    await self.watchdog()

                # ── Cron scheduler check (every 60s) ──
                sched_elapsed = (
                    datetime.now() - self.last_scheduler_check
                ).total_seconds()
                if sched_elapsed >= 60:
                    await self.check_scheduled_tasks()
                    await self._check_stale_todo_batches()
                    self.last_scheduler_check = datetime.now()

                # Get a generous batch, then compute how many to actually run
                candidate_tasks = await get_ready_tasks(limit=8)

                # ── Model-aware task reordering ──
                # Boost tasks that match the currently loaded model to reduce swaps.
                # Max boost is +0.9 priority, so a 2+ priority gap is never overridden.
                candidate_tasks = _reorder_by_model_affinity(candidate_tasks)

                max_concurrent = _compute_max_concurrent(candidate_tasks)
                tasks = candidate_tasks[:max_concurrent]

                # ── Quota planner: forward-looking queue scan ──
                # Build a full QueueProfile from upcoming tasks so the planner
                # can reserve cloud quota for tasks that genuinely need it
                # (vision, thinking, hard difficulty).
                try:
                    from src.models.quota_planner import get_quota_planner, QueueProfile
                    _qp = get_quota_planner()
                    # Use the full candidate batch (up to 8) for the scan.
                    # For deeper lookahead, fetch more if we have tasks.
                    _lookahead = candidate_tasks
                    if len(candidate_tasks) >= 6:
                        # Queue is busy — peek further ahead
                        _lookahead = await get_ready_tasks(limit=30)
                    if _lookahead:
                        _profile = QueueProfile()
                        _profile.total_tasks = len(_lookahead)
                        for _t in _lookahead:
                            _d = _parse_task_difficulty(_t)
                            _profile.max_difficulty = max(_profile.max_difficulty, _d)
                            if _d >= 7:
                                _profile.hard_tasks_count += 1
                            # Extract capability needs from classification context
                            _ctx = _t.get("context", {})
                            if isinstance(_ctx, str):
                                try:
                                    import json as _json
                                    _ctx = _json.loads(_ctx)
                                except Exception:
                                    _ctx = {}
                            _cls = _ctx.get("classification", {})
                            if _cls.get("needs_vision", False):
                                _profile.needs_vision_count += 1
                                _profile.cloud_only_count += 1  # vision = cloud only
                            if _cls.get("needs_tools", False):
                                _profile.needs_tools_count += 1
                            if _cls.get("needs_thinking", False):
                                _profile.needs_thinking_count += 1
                        _qp.set_queue_profile(_profile)
                        _qp.recalculate()
                except Exception as _qp_err:
                    logger.debug(f"Quota planner scan failed: {_qp_err}")

                # Drain deferred grade queue if it's getting full
                try:
                    from src.core.llm_dispatcher import get_dispatcher
                    await get_dispatcher().drain_grades_if_full()
                except Exception as _gd_err:
                    logger.debug(f"Grade queue drain check failed: {_gd_err}")

                # Update queue depth metric for Prometheus/Grafana
                try:
                    from src.infra.metrics import record_queue_depth
                    record_queue_depth(len(candidate_tasks))
                except Exception:
                    pass

                # ── Proactive GPU loading ──
                # If GPU is idle and queue has work, load the best-fit model
                # BEFORE tasks start. Local inference is free — don't waste GPU.
                if candidate_tasks:
                    try:
                        from src.core.llm_dispatcher import get_dispatcher
                        await get_dispatcher().ensure_gpu_utilized(candidate_tasks)
                    except Exception as _gpu_err:
                        logger.debug(f"Proactive GPU load failed: {_gpu_err}")

                if tasks:
                    task_names = [
                        f"#{t['id']}({t.get('agent_type','?')})"
                        for t in tasks
                    ]
                    logger.info(
                        f"[Cycle {self.cycle_count}] "
                        f"Processing {len(tasks)} task(s): {task_names}"
                    )

                    # Partition tasks: only tasks that are guaranteed to NOT
                    # need llama-server can run in parallel.  Since the router
                    # decides local vs cloud at call time, we conservatively
                    # treat any task that *might* use the local model as local.
                    # Only agent_types that never touch the LLM are "safe".
                    _CLOUD_SAFE_AGENTS = {
                        "assistant",  # casual replies (quick LLM but low priority)
                    }
                    if len(tasks) > 1:
                        local_tasks = []
                        cloud_tasks = []
                        for t in tasks:
                            if t.get("agent_type") in _CLOUD_SAFE_AGENTS:
                                cloud_tasks.append(t)
                            else:
                                local_tasks.append(t)

                        # Run at most 1 local-model task; cloud-safe tasks can
                        # run alongside it without causing model swap storms.
                        batch = cloud_tasks + local_tasks[:1]
                        deferred = local_tasks[1:]
                    else:
                        batch = tasks
                        deferred = []

                    # Helper: wait for futures but break out if shutdown requested
                    shutdown_fut = asyncio.ensure_future(self.shutdown_event.wait())

                    if len(batch) == 1:
                        # Single task — run directly
                        t = batch[0]
                        try:
                            self._current_task_future = asyncio.ensure_future(
                                self.process_task(t)
                            )
                            self._running_futures = [self._current_task_future]
                            done, _ = await asyncio.wait(
                                [self._current_task_future, shutdown_fut],
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if self._current_task_future in done:
                                # Propagate any exception
                                self._current_task_future.result()
                            self._current_task_future = None
                            self._running_futures = []
                        except Exception as e:
                            self._current_task_future = None
                            self._running_futures = []
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
                        self._running_futures = list(futures)
                        await asyncio.wait(
                            futures + [shutdown_fut],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        # Collect results from completed futures
                        for t_item, f in zip(batch, futures):
                            if f.done() and f.exception():
                                logger.error(
                                    f"Task #{t_item['id']} error: {f.exception()}",
                                    exc_info=True,
                                )
                        self._running_futures = []

                    # Cancel the shutdown waiter if it didn't fire
                    if not shutdown_fut.done():
                        shutdown_fut.cancel()

                    # Break immediately if shutdown was requested
                    if self.shutdown_event.is_set():
                        break

                    # Run deferred local tasks sequentially
                    for t in deferred:
                        if self.shutdown_event.is_set():
                            break
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

                    # Drain deferred grades during idle (uses cloud if needed)
                    try:
                        from src.core.llm_dispatcher import get_dispatcher
                        await get_dispatcher().drain_grades_if_idle()
                    except Exception as _gd_err:
                        logger.debug(f"Idle grade drain failed: {_gd_err}")

                    # Use shutdown-aware sleep instead of plain asyncio.sleep
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(), timeout=3
                        )
                        break  # shutdown requested during idle
                    except asyncio.TimeoutError:
                        pass  # normal idle cycle

                # Phase 14.1: Time-based morning briefing (default 9:00 local)
                now = datetime.now()
                briefing_hour = int(os.environ.get("BRIEFING_HOUR", "9"))
                if (now.hour == briefing_hour
                        and now.date() > self.last_digest.date()):
                    await self.daily_digest()
                    self.last_digest = now

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

                # Phase 9.3: Check alerts periodically
                try:
                    from src.infra.alerting import check_alerts
                    await check_alerts()
                except Exception:
                    pass

                # Phase 9.1: Persist in-memory metrics hourly
                try:
                    from src.infra.metrics import maybe_persist
                    await maybe_persist()
                except Exception:
                    pass

                # Phase 2.4: Auto-tune model capability scores
                try:
                    from src.models.auto_tuner import maybe_run_tuning
                    await maybe_run_tuning()
                except Exception as e:
                    logger.debug(f"Auto-tuning cycle failed: {e}")

                # Phase 13.4: Weekly self-improvement analysis
                try:
                    if not hasattr(self, '_last_improvement_check'):
                        self._last_improvement_check = datetime.now()
                    elif (datetime.now() - self._last_improvement_check).total_seconds() > 604800:  # 7 days
                        from src.memory.self_improvement import (
                            analyze_and_propose, format_proposals_for_telegram
                        )
                        proposals = await analyze_and_propose()
                        if proposals:
                            msg = format_proposals_for_telegram(proposals)
                            await self.telegram.send_notification(msg)
                        self._last_improvement_check = datetime.now()
                except Exception as e:
                    logger.debug(f"Self-improvement check failed: {e}")

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

        # Phase 13.1: Seed prompt versions from hardcoded prompts on first run
        try:
            from ..memory.prompt_versions import seed_from_agents
            seeded = await seed_from_agents()
            if seeded:
                logger.info(f"Seeded {seeded} prompt versions from hardcoded agents")
        except Exception as e:
            logger.debug(f"Prompt seeding skipped: {e}")

        # Initialise shopping DB schemas (cache, request_tracker, memory)
        try:
            from ..shopping.cache import init_cache_db
            from ..shopping.request_tracker import init_request_db
            from ..shopping.memory import init_memory_db
            await init_cache_db()
            await init_request_db()
            await init_memory_db()
            logger.info("Shopping DB schemas initialised")
        except Exception as e:
            logger.warning(f"Shopping DB init failed (non-fatal): {e}")

        async with self.telegram.app:
            await self.telegram.app.start()
            await self.telegram.app.updater.start_polling()
            await self.telegram.set_bot_commands()

            logger.info(
                "✅ System online — Telegram + Orchestrator + "
                "GPU Scheduler + Backpressure Queue running"
            )

            # Send persistent keyboard on startup so buttons are always visible
            try:
                from ..app.config import TELEGRAM_ADMIN_CHAT_ID
                from ..app.telegram_bot import REPLY_KEYBOARD
                if TELEGRAM_ADMIN_CHAT_ID:
                    await self.telegram.app.bot.send_message(
                        chat_id=TELEGRAM_ADMIN_CHAT_ID,
                        text="✅ KutAI online. Buttons ready.",
                        reply_markup=REPLY_KEYBOARD,
                    )
            except Exception as e:
                logger.debug(f"Startup keyboard send failed: {e}")

            try:
                await self.run_loop()
            finally:
                # ── Graceful shutdown ──
                if self.shutdown_event.is_set():
                    logger.info("Graceful shutdown initiated...")
                    self._shutting_down = True
                    self.running = False

                    # Stop background tasks
                    bp_queue.stop()
                    for t in self._background_tasks:
                        t.cancel()

                    # Collect all in-flight task futures
                    active_futures = [
                        f for f in self._running_futures
                        if f and not f.done()
                    ]
                    if self._current_task_future and not self._current_task_future.done():
                        active_futures.append(self._current_task_future)

                    if active_futures:
                        logger.info(
                            f"Waiting for {len(active_futures)} running "
                            f"task(s) to complete (30s timeout)..."
                        )
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(
                                    *[asyncio.shield(f) for f in active_futures],
                                    return_exceptions=True,
                                ),
                                timeout=30,
                            )
                            logger.info("All running tasks completed cleanly")
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Shutdown timeout (30s) — "
                                f"{sum(1 for f in active_futures if not f.done())} "
                                "task(s) abandoned"
                            )
                        except Exception as e:
                            logger.warning(f"Task error during shutdown: {e}")

                    # Release all file locks so they aren't stuck on next startup.
                    try:
                        db = await get_db()
                        await db.execute("DELETE FROM file_locks")
                        await db.commit()
                        logger.info("Released all file locks")
                    except Exception as e:
                        logger.warning(f"Lock release failed: {e}")

                    # Persist in-memory metrics before exiting
                    try:
                        from src.infra.metrics import persist_metrics
                        await persist_metrics()
                        logger.info("Final metrics snapshot persisted")
                    except Exception:
                        pass

                    # Stop llama-server if running
                    try:
                        await manager._stop_server()
                        logger.info("llama-server stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping llama-server: {e}")

                    logger.info("Graceful shutdown complete")
                else:
                    # Non-graceful exit (crash, code 42 restart, etc.)
                    # Still need to stop llama-server to prevent orphans
                    try:
                        await manager._stop_server()
                        logger.info("llama-server stopped (non-graceful exit)")
                    except Exception as e:
                        logger.warning(f"Error stopping llama-server: {e}")

                await close_db()
                await self.telegram.app.updater.stop()
                await self.telegram.app.stop()
