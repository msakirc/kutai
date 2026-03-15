# orchestrator.py
import asyncio
import json
import logging
import signal
from ..app.config import DB_PATH, MAX_CONTEXT_CHAIN_LENGTH, TASK_PRIORITY
from datetime import datetime
from ..infra.db import (
    init_db, get_db, close_db, get_ready_tasks, update_task, add_task,
    claim_task, add_subtasks_atomically, log_conversation,
    get_active_goals, get_tasks_for_goal, update_goal, get_daily_stats,
    store_memory, compute_task_hash,
    get_due_scheduled_tasks, update_scheduled_task,
    cancel_task, get_task,
    save_workspace_snapshot, release_task_locks, release_goal_locks,
)
from .router import classify_task, _circuit_breakers
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    # Default timeouts per agent type (seconds).  Override via
    # tasks.timeout_seconds column for per-task control.
AGENT_TIMEOUTS: dict[str, int] = {
    "planner":    120,
    "coder":      300,
    "researcher": 180,
    "reviewer":   120,
    "executor":   180,
    "pipeline":   600,
    "error_recovery": 240,
}

# Maximum number of independent tasks to run concurrently.
MAX_CONCURRENT_TASKS: int = 2


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
            logger.debug(f"[Watchdog] Local model check failed: {e}")

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
            logger.debug(f"[Watchdog] GPU health check failed: {e}")

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
            logger.debug(f"[Watchdog] GPU scheduler check failed: {e}")

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
            logger.debug(f"[Watchdog] Backpressure check failed: {e}")

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
            logger.debug(f"[Watchdog] Circuit breaker check failed: {e}")

        # 9. Restore rate limits that were adaptively reduced
        try:
            from ..models.rate_limiter import get_rate_limit_manager
            get_rate_limit_manager().restore_limits()
        except Exception as e:
            logger.debug(f"[Watchdog] Rate limit restore failed: {e}")

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
                task_id = await add_task(
                    title=title,
                    description=sched.get("description", ""),
                    agent_type=sched.get("agent_type", "executor"),
                    context=json.loads(sched.get("context", "{}"))
                    if isinstance(sched.get("context"), str)
                    else sched.get("context", {}),
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

        logger.info(f"[Task #{task_id}] Starting: '{title}' (agent: {agent_type})")

        try:
            # Atomic claim — if another worker grabbed it first, skip
            claimed = await claim_task(task_id)
            if not claimed:
                logger.info(f"[Task #{task_id}] Already claimed by another worker, skipping")
                return

            # ── Check for cancellation before starting ──
            fresh = await get_task(task_id)
            if fresh and fresh.get("status") == "cancelled":
                logger.info(f"[Task #{task_id}] Cancelled before execution, skipping")
                return

            # ── Inject context from prior steps + workspace snapshot ──
            task = await self._inject_chain_context(task)

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
                logger.info(f"[Task #{task_id}] Delegating to CodingPipeline")
                coro = pipeline.run(task)
            else:
                agent = get_agent(agent_type)
                logger.info(
                    f"[Task #{task_id}] Agent '{agent.name}' executing "
                    f"(tier: {task.get('tier', 'auto')}, "
                    f"timeout: {timeout_seconds}s)"
                )
                coro = agent.execute(task)

            # Wrap with timeout
            try:
                result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                timeout_err = f"TimeoutError: Task timed out after {timeout_seconds}s"
                logger.error(f"[Task #{task_id}] TIMEOUT after {timeout_seconds}s")
                await update_task(
                    task_id, status="failed", error=timeout_err
                )
                await self.telegram.send_error(task_id, title, timeout_err)

                # Spawn error recovery for timeouts too
                await self._spawn_error_recovery(task, timeout_err)
                return

            status = result.get("status", "completed")

            logger.info(f"[Task #{task_id}] Agent returned status: '{status}'")

            # Auto-commit after successful coder tasks
            if status == "completed" and agent_type == "coder":
                await self._auto_commit(task, result)

            if status == "completed":
                await self._handle_complete(task, result)
            elif status == "needs_subtasks":
                await self._handle_subtasks(task, result)
            elif status == "needs_clarification":
                await self._handle_clarification(task, result)
            elif status == "needs_review":
                await self._handle_review(task, result)
            else:
                logger.warning(f"[Task #{task_id}] Unknown status '{status}', treating as complete")
                await self._handle_complete(task, result)

            # ── Phase 6: Release file locks held by this task ──
            try:
                await release_task_locks(task_id)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[Task #{task_id}] FAILED: {type(e).__name__}: {e}", exc_info=True)
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
                logger.info(f"[Task #{task_id}] Will retry ({retry_count + 1}/{max_retries})")
            else:
                error_str = f"{type(e).__name__}: {str(e)[:500]}"
                await update_task(task_id, status="failed", error=error_str)
                await self.telegram.send_error(task_id, title, error_str)

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

    # ─── Result Handlers ─────────────────────────────────────────────────

    async def _handle_complete(self, task, result):
        task_id = task["id"]
        result_text = result.get("result", "No result")
        model = result.get("model", "unknown")
        cost = result.get("cost", 0)
        iterations = result.get("iterations", 1)

        await update_task(
            task_id, status="completed", result=result_text,
            completed_at=datetime.now().isoformat()
        )

        if task.get("goal_id"):
            await self._check_goal_completion(task["goal_id"])

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

        logger.info(f"[Task #{task_id}] ✅ Complete via {model} "
                     f"(${cost:.4f}, {iterations} iter)")

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
            logger.warning(
                f"[Task #{task_id}] Planner created {len(subtasks)} subtasks, "
                f"capping at {MAX_SUBTASKS}"
            )
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

        try:
            recovery_task_id = await add_task(
                title=f"Error recovery: {title[:50]}",
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

                tasks = await get_ready_tasks(limit=MAX_CONCURRENT_TASKS)

                if tasks:
                    task_names = [
                        f"#{t['id']}({t.get('agent_type','?')})"
                        for t in tasks
                    ]
                    logger.info(
                        f"[Cycle {self.cycle_count}] "
                        f"Processing {len(tasks)} task(s): {task_names}"
                    )

                    if len(tasks) == 1:
                        # Single task — run directly
                        t = tasks[0]
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
                            for t in tasks
                        ]
                        results = await asyncio.gather(
                            *futures, return_exceptions=True
                        )
                        for t, res in zip(tasks, results):
                            if isinstance(res, Exception):
                                logger.error(
                                    f"Task #{t['id']} error: {res}",
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
