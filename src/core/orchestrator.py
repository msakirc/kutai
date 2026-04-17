# orchestrator.py
import asyncio
import dataclasses
import json
import os
import re
import signal
import time
from pathlib import Path
from ..app.config import DB_PATH, MAX_CONTEXT_CHAIN_LENGTH, TASK_PRIORITY
from datetime import datetime, timedelta, timezone
from ..infra.times import utc_now, db_now, to_db, from_db, DB_FMT
from ..infra.db import (
    init_db, get_db, close_db, get_ready_tasks, update_task, add_task,
    claim_task, add_subtasks_atomically, log_conversation,
    get_active_missions, get_tasks_for_mission, update_mission, get_daily_stats,
    store_memory, compute_task_hash,
    get_due_scheduled_tasks, update_scheduled_task,
    cancel_task, get_task, get_mission,
    release_task_locks, release_mission_locks,
)
from src.infra.logging_config import get_logger
from .router import ModelCallFailed, get_kdv
from .task_context import parse_context, set_context
from .task_gates import run_gates
from .mechanical.workspace_snapshot import snapshot_workspace
from .decisions import Cancel as GateCancel
from .result_router import (
    route_result, Complete, SpawnSubtasks, RequestClarification,
    RequestReview, Exhausted, Failed as FailedAction,
)
from ..agents import get_agent
from ..tools import execute_tool
from ..tools.workspace import (
    get_file_tree,
    get_mission_workspace,
    get_mission_workspace_relative,
)
from ..tools.git_ops import (
    git_commit, ensure_git_repo,
    create_mission_branch,
)
from ..app.telegram_bot import TelegramInterface

logger = get_logger("core.orchestrator")

# DB_FMT removed — use DB_FMT from src.infra.times instead


    # Default timeouts per agent type (seconds).  Override via
    # tasks.timeout_seconds column for per-task control.
    # Timeouts include model load time (up to 60s for first swap).
    # Don't set below 180s for any agent that triggers LLM calls.
AGENT_TIMEOUTS: dict[str, int] = {
    "planner":        420,  # was 300 — workflow steps read 3+ artifacts + produce output
    "architect":      420,  # was 300 — same: multi-artifact reads + design output
    "coder":          420,  # was 300
    "implementer":    420,  # was 300
    "fixer":          420,  # was 300
    "test_generator": 300,  # was 240
    "reviewer":       300,  # was 180 — reviews read multiple artifacts
    "visual_reviewer":180,
    "researcher":     420,  # was 300 — web search + artifact reads + synthesis
    "analyst":        420,  # was 300
    "writer":         420,  # was 300 — report tasks read many artifacts + long output
    "summarizer":     180,
    "assistant":      180,
    "executor":       300,  # was 240
    "pipeline":       600,
    "workflow":       900,
    "shopping_advisor":    600,
    "product_researcher":  300,
    "deal_analyst":        240,
    "shopping_pipeline":   60,   # mechanical Python steps — no LLM
    "shopping_clarifier":  120,
}

async def _check_internet() -> bool:
    """Quick internet connectivity check (3 s timeout)."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.head(
                "https://duckduckgo.com",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as r:
                return r.status < 500
    except Exception:
        return False


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
                # Handle double-encoded JSON strings
                if isinstance(ctx, str):
                    ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError):
                ctx = {}
        if not isinstance(ctx, dict):
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

        # Vision batching: when vision is loaded (expensive — 876MB mmproj),
        # boost all vision tasks so they run together before the model
        # unloads and frees VRAM. Avoids repeated vision swap toggles.
        vision_loaded = manager._vision_enabled

        def _sort_key(task: dict):
            priority = task.get("_effective_priority", task.get("priority", 5))
            ctx = parse_context(task)
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

            # Zero affinity for tasks that would reject the loaded model
            if _should_defer_for_loaded_model(task, manager.current_model or ""):
                fit = 0.0

            # Boost by fit * 0.9, so max boost < 1 priority level
            effective_priority = priority + (fit * 0.9)

            # Vision batching: boost vision tasks when mmproj is loaded
            task_needs_vision = cls.get("needs_vision", False)
            if vision_loaded and task_needs_vision:
                effective_priority += 0.8  # strong boost to batch with other vision tasks
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
    ctx = parse_context(task)
    cls = ctx.get("classification", {})
    return max(1, min(10, int(cls.get("difficulty", 5))))


def _should_defer_for_loaded_model(task: dict, loaded_model: str) -> bool:
    """Check if this task would reject the currently loaded model."""
    worker_attempts = task.get("worker_attempts", task.get("attempts", 0)) or 0
    if worker_attempts < 3:
        return False
    ctx = parse_context(task)
    return loaded_model in ctx.get("failed_models", [])


class Orchestrator:
    def __init__(self, shutdown_event=None):
        self.telegram = TelegramInterface(self)
        self.running = False
        self._shutting_down = False
        self.cycle_count = 0
        from ..infra.times import turkey_now
        self.last_digest = turkey_now()
        self.last_scheduler_check = datetime.min.replace(tzinfo=timezone.utc)
        self.last_decay_check = datetime.min.replace(tzinfo=timezone.utc)
        self.shutdown_event = shutdown_event or asyncio.Event()
        self.requested_exit_code: int | None = None  # Set via yasar_usta.EXIT_RESTART or EXIT_STOP
        self._current_task_future = None
        self._running_futures: list[asyncio.Task] = []
        self._model_manager_tasks: list[asyncio.Task] = []
        self.paused_patterns: set[str] = set()

        from src.app.scheduled_jobs import ScheduledJobs
        self.scheduled_jobs = ScheduledJobs(telegram=self.telegram)


    # ─── NEW: Context Chaining ───────────────────────────────────────────

    async def _inject_chain_context(self, task: dict) -> dict:
        """
        Before executing a task, inject results from completed sibling tasks
        (prior steps in the same mission) and a workspace snapshot into its context.
        This is how step 5 knows what steps 1-4 produced.
        """
        task_context = parse_context(task)

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
        task = set_context(task, task_context)
        return task

    # ─── NEW: Auto-commit after coder tasks ─────────────────────────────

    async def _auto_commit(self, task: dict, result: dict):
        # Dormant in Phase 1; live copy in src/core/mechanical/git_commit.py.
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

    # ─── Independent stuck-task watchdog ────────────────────

    async def _run_stuck_task_watchdog(self):
        """Independent periodic check for tasks stuck in 'processing'.

        Runs as a separate asyncio task so it fires even when the main
        loop is blocked (e.g. all concurrent tasks hung on a dead
        llama-server).  Uses per-task timeouts + grace period instead
        of a hardcoded 5-minute threshold.
        """
        GRACE_SECONDS = 60  # buffer for graceful shutdown / slow finalization
        INTERVAL = 30       # check every 30 seconds

        while not self.shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self.shutdown_event.wait(), timeout=INTERVAL
                )
                break  # shutdown requested
            except asyncio.TimeoutError:
                pass  # normal tick

            try:
                db = await get_db()
                cursor = await db.execute(
                    """SELECT id, title, agent_type, timeout_seconds,
                              started_at, infra_resets
                       FROM tasks WHERE status = 'processing'"""
                )
                rows = [dict(r) for r in await cursor.fetchall()]
                if not rows:
                    continue

                now = utc_now()
                reset_tasks = []

                for task in rows:
                    started = task.get("started_at")
                    if not started:
                        continue
                    try:
                        started_dt = from_db(str(started))
                    except (ValueError, TypeError):
                        continue

                    timeout_sec = (
                        task.get("timeout_seconds")
                        or AGENT_TIMEOUTS.get(task.get("agent_type", "executor"), 240)
                    )
                    elapsed = (now - started_dt).total_seconds()
                    if elapsed > timeout_sec + GRACE_SECONDS:
                        reset_tasks.append((task, elapsed, timeout_sec))

                for task, elapsed, timeout_sec in reset_tasks:
                    infra_resets = (task.get("infra_resets") or 0) + 1
                    if infra_resets >= 3:
                        logger.warning(
                            "stuck task exhausted infra resets",
                            task_id=task["id"],
                            elapsed=int(elapsed),
                            timeout=timeout_sec,
                            resets=infra_resets,
                        )
                        await db.execute(
                            "UPDATE tasks SET status = 'failed', "
                            "error = 'Stuck in processing — infra resets "
                            "exhausted (stuck-task watchdog)', "
                            "failed_in_phase = 'infrastructure', "
                            "infra_resets = ? WHERE id = ?",
                            (infra_resets, task["id"]),
                        )
                        try:
                            from src.infra.dead_letter import quarantine_task
                            await quarantine_task(
                                task_id=task["id"],
                                mission_id=None,
                                error=f"Stuck in processing {int(elapsed)}s "
                                      f"(timeout={timeout_sec}s, "
                                      f"infra resets exhausted)",
                                error_category="infrastructure",
                                original_agent=task.get("agent_type", "executor"),
                            )
                        except Exception:
                            pass
                    else:
                        logger.warning(
                            "stuck task reset to pending",
                            task_id=task["id"],
                            elapsed=int(elapsed),
                            timeout=timeout_sec,
                            infra_reset=f"{infra_resets}/3",
                        )
                        await db.execute(
                            "UPDATE tasks SET status = 'pending', "
                            "infra_resets = ?, "
                            "retry_reason = 'infrastructure' "
                            "WHERE id = ?",
                            (infra_resets, task["id"]),
                        )

                if reset_tasks:
                    await db.commit()
                    # Cancel hung futures in the main loop so they don't
                    # keep consuming resources after we reset the task.
                    for f in getattr(self, "_running_futures", []):
                        if not f.done():
                            f.cancel()

            except Exception as e:
                logger.debug(f"Stuck-task watchdog error: {e}")

    # ─── Watchdog (inline, runs every 10 cycles) ─────────────

    async def watchdog(self):
        """Periodic maintenance. Delegates to src/core/watchdog module."""
        from src.core import watchdog as wd
        try:
            await wd.check_stuck_tasks(telegram=self.telegram)
        except Exception as e:
            logger.error("watchdog stuck_tasks error: %s", e)
        try:
            await wd.check_resources(telegram=self.telegram)
        except Exception as e:
            logger.error("watchdog resources error: %s", e)

    # ─── Core Task Processing ────────────────────────────────────────────

    async def _prepare(self, task: dict) -> "tuple[dict, str, int] | None":
        """Run all pre-dispatch work.

        Returns (prepared_task, agent_type, timeout_seconds) or None if the task
        should not proceed (already claimed, cancelled, fast-resolved, gate-blocked,
        or waiting on internet).
        """
        task_id = task["id"]
        title = task["title"]
        agent_type = task.get("agent_type", "executor")

        # Atomic claim — if another worker grabbed it first, skip
        claimed = await claim_task(task_id)
        if not claimed:
            logger.info("task already claimed", task_id=task_id)
            return None

        # ── Check for cancellation before starting ──
        fresh = await get_task(task_id)
        if fresh and fresh.get("status") == "cancelled":
            logger.info("task cancelled before execution", task_id=task_id)
            return None

        # ── Inject context from prior steps + workspace snapshot ──
        task = await self._inject_chain_context(task)

        # ── Classify task if not already classified ──
        task_ctx = parse_context(task)

        if "classification" not in task_ctx and agent_type == "executor":
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
            task = set_context(task, task_ctx)

        # ── Layer 0: Fast-path resolution via API registry ──
        # Skip for pipeline agent types — they have their own execution.
        _skip_fast_path = agent_type in ("pipeline", "shopping_pipeline")
        try:
            from ..core.fast_resolver import try_resolve
            fast_result = None if _skip_fast_path else await try_resolve(task)
            if fast_result:
                logger.info("task resolved via fast-path", task_id=task_id)
                await update_task(task_id, status="completed", result=fast_result,
                                  completed_at=db_now())
                if self.telegram and task.get("chat_id"):
                    await self.telegram.send_notification(fast_result)
                return None
        except Exception as exc:
            logger.debug("fast-path check failed (continuing to agent): %s", exc)

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
                task = set_context(task, task_ctx)
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
            template_step_id = task_ctx.get("workflow_step_id", "")
            if should_delegate_to_pipeline(template_step_id, agent_type):
                agent_type = "pipeline"
                task["agent_type"] = "pipeline"
                logger.info("workflow step delegated to pipeline", task_id=task_id)

        # ── Layer 1: Enrich context with pre-fetched API data ──
        # Skip for pipeline agent types — they don't need API enrichment.
        try:
            from ..core.fast_resolver import enrich_context
            enrichment = None if _skip_fast_path else await enrich_context(task)
            if enrichment:
                task_ctx["api_enrichment"] = enrichment
                task = set_context(task, task_ctx)
                logger.info("task enriched with API data", task_id=task_id)
        except Exception as exc:
            logger.debug("context enrichment failed (non-critical): %s", exc)

        # ── Human approval + risk gates ──
        gate_decision = await run_gates(
            task=task,
            task_ctx=task_ctx,
            approval_fn=self.telegram.request_approval,
        )
        if isinstance(gate_decision, GateCancel):
            logger.info("gate rejected", task_id=task_id, reason=gate_decision.reason)
            await update_task(task_id, status="cancelled")
            return None

        # ── Phase 6: Snapshot workspace before coder/pipeline tasks ──
        mission_id = task.get("mission_id")
        if mission_id and agent_type in ("coder", "pipeline", "implementer", "fixer"):
            ws_path = get_mission_workspace(mission_id)
            repo_path = get_mission_workspace_relative(mission_id)
            await snapshot_workspace(
                mission_id=mission_id,
                task_id=task_id,
                workspace_path=ws_path,
                repo_path=repo_path,
            )

        # ── Internet connectivity pre-check for web-dependent tasks ──
        classification = task_ctx.get("classification", {})
        if classification.get("search_depth", "none") != "none":
            if not await _check_internet():
                logger.warning(
                    "internet unreachable, deferring web task",
                    task_id=task_id,
                )
                await update_task(
                    task_id, started_at=None, status="pending",
                )
                return None

        # ── Determine timeout ──
        timeout_seconds = (
            task.get("timeout_seconds")
            or AGENT_TIMEOUTS.get(agent_type, 240)
        )

        return task, agent_type, timeout_seconds

    async def process_task(self, task: dict):
        """Process a single task through the appropriate agent with context injection."""
        task_id = task["id"]
        title = task["title"]
        agent_type = task.get("agent_type", "executor")

        logger.info("task received", task_id=task_id, title=title, agent_type=agent_type)

        task_ctx = {}  # initialized here so except handler can safely access it
        result = None  # initialized here so except handler can safely access it
        try:
            prepared = await self._prepare(task)
            if prepared is None:
                return
            task, agent_type, timeout_seconds = prepared
            title = task["title"]

            # Make workflow hook functions available for the dispatch and
            # result-handling code below (they were lazily imported inside the
            # old process_task prefix; _prepare owns that import now).
            from ..workflows.engine.hooks import post_execute_workflow_step, is_workflow_step

            if agent_type == "pipeline":
                from ..workflows.pipeline import CodingPipeline
                pipeline = CodingPipeline()
                logger.info("delegating to pipeline", task_id=task_id)
                coro = pipeline.run(task)
            elif agent_type == "shopping_pipeline":
                from ..workflows.shopping.pipeline import ShoppingPipeline
                pipeline = ShoppingPipeline()
                logger.info("delegating to shopping pipeline", task_id=task_id)
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
                _task_start_time = time.time()

                _attempt_num = (task.get("worker_attempts") or 0) + 1

                async def _progress_cb(tid, iteration, max_iter, summary):
                    if self.telegram:
                        elapsed = int(time.time() - _task_start_time)
                        attempt_tag = f" | attempt {_attempt_num}" if _attempt_num > 1 else ""
                        msg = (
                            f"\U0001f504 *Task #{tid}* — iteration {iteration}/{max_iter} ({elapsed}s elapsed{attempt_tag})\n"
                            f"{summary[:200]}"
                        )
                        try:
                            await self.telegram.send_notification(msg)
                        except Exception:
                            pass

                # Send "task started" notification
                try:
                    task_ctx = parse_context(task)
                    if not task_ctx.get("silent"):
                        chat_id = task_ctx.get("chat_id")
                        if chat_id and hasattr(self, 'telegram') and self.telegram:
                            await self.telegram.app.bot.send_message(
                                chat_id=chat_id,
                                text=f"\U0001f680 Task #{task['id']} assigned to {agent_type}, starting...",
                            )
                except Exception:
                    pass

                agent._task_timeout = timeout_seconds
                coro = agent.execute(task, progress_callback=_progress_cb)

            # Wrap with timeout
            try:
                result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                timeout_err = f"TimeoutError: Task timed out after {timeout_seconds}s"
                logger.error("task timeout", task_id=task_id, timeout_seconds=timeout_seconds, error=timeout_err)

                # Try to recover partial results from the last checkpoint
                # before marking the task as failed.
                partial_result = None
                try:
                    from src.infra.db import load_task_checkpoint
                    checkpoint = await load_task_checkpoint(task_id)
                    if checkpoint:
                        last_messages = checkpoint.get("messages", [])
                        iter_num = checkpoint.get("iteration", "?")
                        # Strategy 1: look for assistant final_answer that
                        # was produced but not yet returned (agent timed out
                        # between producing the answer and returning it).
                        for msg in reversed(last_messages):
                            if msg.get("role") == "assistant" and msg.get("content"):
                                c = msg["content"]
                                # Quick check for final_answer JSON
                                if "final_answer" in c and len(c) > 100:
                                    try:
                                        _p = json.loads(c)
                                        partial_result = _p.get("result", c)[:8000]
                                    except (json.JSONDecodeError, TypeError):
                                        partial_result = c[:8000]
                                    break
                        # Strategy 2: last substantial assistant message
                        # (the LLM's reasoning or partial answer — far more
                        # useful than raw tool output).
                        if not partial_result:
                            for msg in reversed(last_messages):
                                if msg.get("role") == "assistant" and len(msg.get("content", "")) > 100:
                                    partial_result = msg["content"][:8000]
                                    break
                        # Strategy 3: last tool result (user message echoing
                        # a tool's output — last resort, often just a search
                        # cache snippet that's not useful as a result).
                        if not partial_result:
                            for msg in reversed(last_messages):
                                if msg.get("role") == "user" and "Tool Result" in msg.get("content", ""):
                                    partial_result = msg["content"][:8000]
                                    break

                        if partial_result:
                            logger.info(f"[Task #{task_id}] Timeout recovery: using checkpoint from iteration {iter_num}")
                            result_text = f"(Partial result from iteration {iter_num} before timeout)\n\n{partial_result}"

                            # For workflow steps, don't mark partial results
                            # as completed — they bypass the post-hook and
                            # poison downstream tasks with garbage.  Let them
                            # fail and go through normal retry/DLQ.
                            task_ctx = parse_context(task)

                            if is_workflow_step(task_ctx):
                                logger.warning(
                                    f"[Task #{task_id}] Timeout recovery: "
                                    f"workflow step — failing instead of "
                                    f"completing with partial result"
                                )
                                # Fall through to the failed path below
                            else:
                                task_ctx["partial"] = True
                                await update_task(
                                    task_id, status="completed",
                                    result=result_text,
                                    context=json.dumps(task_ctx),
                                )
                                await self.telegram.send_result(task_id, title, result_text, "timeout-recovery", 0,
                                                                mission_id=task.get("mission_id"))
                                return
                except Exception as recovery_err:
                    logger.debug(f"[Task #{task_id}] Checkpoint recovery failed: {recovery_err}")

                # Roll back the checkpoint iteration counter so the retry
                # gets more than 1 shot.  Keep messages intact — the partial
                # output from the interrupted generation is valuable (often
                # 6-7k of 8k chars done) and the next attempt can finish it
                # quickly instead of regenerating from scratch.
                try:
                    from src.infra.db import load_task_checkpoint, save_task_checkpoint
                    cp = await load_task_checkpoint(task_id)
                    if cp:
                        old_iter = cp.get("iteration", 0)
                        new_iter = max(old_iter - 2, 0)
                        cp["iteration"] = new_iter
                        await save_task_checkpoint(task_id, cp)
                        logger.info(
                            f"[Task #{task_id}] Timeout: rolled back checkpoint "
                            f"iteration {old_iter}→{new_iter}, "
                            f"{len(cp.get('messages', []))} messages preserved"
                        )
                except Exception:
                    pass

                # Use the same retry/DLQ pipeline as other failure paths
                task_ctx = parse_context(task)

                # Inject partial output so next attempt can continue
                if partial_result:
                    task_ctx["_prev_output"] = partial_result[:6000]
                    task_ctx["_timeout_hint"] = (
                        "Your previous attempt timed out while generating "
                        "a large output. Your partial work is shown in the "
                        "context. CONTINUE from where you left off — do NOT "
                        "start over. If the output is too large for one "
                        "write_file call, break it into sections."
                    )

                from src.core.retry import RetryContext
                retry_ctx = RetryContext.from_task(task)
                decision = retry_ctx.record_failure("timeout")

                # ── Bonus attempt: if terminal but task made real progress,
                # grant one more try instead of DLQ.  The timeout is a safety
                # net, not a quality judgment — if the agent was productive
                # (wrote files, completed iterations), let it finish.
                _MAX_BONUS_ATTEMPTS = 2  # hard cap on bonus attempts per task lifetime
                if decision.action == "terminal":
                    bonus_granted = False
                    bonus_count = task_ctx.get("_bonus_count", 0)
                    if bonus_count < _MAX_BONUS_ATTEMPTS:
                        try:
                            progress = await self._assess_timeout_progress(
                                task_id, task_ctx
                            )
                            if progress >= 0.5:
                                bonus_granted = True
                                task_ctx["_bonus_count"] = bonus_count + 1
                                retry_ctx.max_worker_attempts += 1
                                decision = retry_ctx.record_failure("timeout")
                                logger.info(
                                    f"[Task #{task_id}] Bonus attempt granted "
                                    f"({bonus_count + 1}/{_MAX_BONUS_ATTEMPTS}, "
                                    f"progress={progress:.0%})"
                                )
                        except Exception:
                            pass

                if decision.action == "terminal":
                    task_ctx.update(retry_ctx.to_context_patch())
                    await update_task(
                        task_id, status="failed",
                        error=timeout_err,
                        context=json.dumps(task_ctx),
                        **retry_ctx.to_db_fields(),
                    )
                    try:
                        from src.infra.dead_letter import quarantine_task
                        await quarantine_task(
                            task_id=task_id,
                            mission_id=task.get("mission_id"),
                            error=f"Timeout after {retry_ctx.worker_attempts} worker attempts: {timeout_err}",
                            error_category="timeout",
                            original_agent=agent_type,
                            attempts_snapshot=retry_ctx.worker_attempts,
                        )
                    except Exception:
                        pass
                    await self.telegram.send_notification(
                        f"❌ Task #{task_id} timeout → DLQ\n"
                        f"**{title[:60]}**\n"
                        f"Failed {retry_ctx.worker_attempts} worker attempts"
                    )
                else:
                    next_retry = None
                    if decision.action == "delayed":
                        next_retry = to_db(
                            utc_now() + timedelta(seconds=decision.delay_seconds)
                        )
                    retry_ctx.next_retry_at = next_retry
                    task_ctx.update(retry_ctx.to_context_patch())
                    await update_task(
                        task_id, status="pending",
                        error=timeout_err,
                        context=json.dumps(task_ctx),
                        **retry_ctx.to_db_fields(),
                    )
                    await self.telegram.send_error(task_id, title,
                        f"{timeout_err} (worker-retry {retry_ctx.worker_attempts}/{retry_ctx.max_worker_attempts})")
                return  # timeout fully handled above — don't fall through

            status = result.get("status", "completed")

            logger.info("result received", task_id=task_id, status=status)

            # PHASE 1 DISCONNECTED: auto-commit moved to src/core/mechanical/git_commit.py.
            # Next i2p workflow refactor will re-wire this as an explicit workflow step
            # or agent tool. Do not delete — code preserved in the mechanical module.
            # if status == "completed" and agent_type == "coder":
            #     await self._auto_commit(task, result)

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
                    # Re-read context from DB — the post-hook may have
                    # stored _schema_error or other fields.  Without this,
                    # the retry update below overwrites them with the stale
                    # task_ctx snapshot from the start of process_task.
                    try:
                        _fresh = await get_task(task_id)
                        if _fresh:
                            _fc = _fresh.get("context", "{}")
                            if isinstance(_fc, str):
                                _fc = json.loads(_fc)
                            if isinstance(_fc, dict):
                                task_ctx.update(_fc)
                    except Exception:
                        pass
                    # Post-hook may override status
                    if result.get("status") == "needs_clarification":
                        if not await self._validate_clarification(
                            task_id, task, task_ctx, result
                        ):
                            # Validation failed — fall through to retry
                            result["status"] = "failed"
                        else:
                            await self._handle_clarification(task, result)
                            return
                    if result.get("status") == "failed":
                        error_msg = result.get("error", "Disguised failure detected")
                        from src.core.retry import RetryContext
                        retry_ctx = RetryContext.from_task(task)
                        decision = retry_ctx.record_failure("quality", model=result.get("model", ""))

                        # Bonus attempt for quality failures with real progress
                        _MAX_BONUS = 2
                        if decision.action == "terminal":
                            bonus_count = task_ctx.get("_bonus_count", 0)
                            if bonus_count < _MAX_BONUS:
                                try:
                                    progress = await self._assess_timeout_progress(
                                        task_id, task_ctx
                                    )
                                    if progress >= 0.5:
                                        task_ctx["_bonus_count"] = bonus_count + 1
                                        retry_ctx.max_worker_attempts += 1
                                        decision = retry_ctx.record_failure("quality",
                                            model=result.get("model", ""))
                                        logger.info(
                                            f"[Task #{task_id}] Quality bonus attempt "
                                            f"({bonus_count + 1}/{_MAX_BONUS}, "
                                            f"progress={progress:.0%})"
                                        )
                                except Exception:
                                    pass

                        if decision.action == "terminal":
                            task_ctx.update(retry_ctx.to_context_patch())
                            await update_task(
                                task_id, status="failed",
                                error=f"Disguised failure exhausted: {error_msg[:300]}",
                                context=json.dumps(task_ctx),
                                **retry_ctx.to_db_fields(),
                            )
                            try:
                                from src.infra.dead_letter import quarantine_task
                                await quarantine_task(
                                    task_id=task_id,
                                    mission_id=task.get("mission_id"),
                                    error=f"Disguised failure after {retry_ctx.worker_attempts} attempts: {error_msg[:300]}",
                                    error_category="quality",
                                    original_agent=task.get("agent_type", "executor"),
                                    attempts_snapshot=retry_ctx.worker_attempts,
                                )
                            except Exception:
                                pass
                            await self.telegram.send_notification(
                                f"❌ Task #{task_id} disguised failure → DLQ\n"
                                f"**{task.get('title', '')[:60]}**\n"
                                f"Reason: {error_msg[:100]}"
                            )
                            logger.warning(
                                "disguised failure terminal",
                                task_id=task_id,
                                attempts=retry_ctx.worker_attempts,
                            )
                        else:
                            next_retry = None
                            if decision.action == "delayed":
                                next_retry = to_db(
                                    utc_now() + timedelta(seconds=decision.delay_seconds)
                                )
                            retry_ctx.next_retry_at = next_retry
                            task_ctx.update(retry_ctx.to_context_patch())
                            # Inject previous output so next attempt can continue
                            result_text = result.get("result", "")
                            if result_text:
                                task_ctx["_prev_output"] = str(result_text)[:6000]
                                task_ctx["_retry_hint"] = (
                                    "Your previous attempt's output failed quality checks. "
                                    "Your partial work is shown in context. Build on it — "
                                    "do NOT start over."
                                )
                            await update_task(
                                task_id, status="pending",
                                error=error_msg[:500],
                                context=json.dumps(task_ctx),
                                **retry_ctx.to_db_fields(),
                            )
                            logger.warning(
                                f"disguised failure, retrying"
                                f"{retry_ctx.worker_attempts}/{retry_ctx.max_worker_attempts}",
                                task_id=task_id,
                            )
                        return
                await self._handle_complete(task, result)
            elif status == "ungraded":
                # Task deferred to grading phase — store result, don't notify.
                # BUT: still run the workflow post-hook so artifacts are stored
                # and phase completion is tracked. Grading only affects the
                # task status (ungraded→completed vs ungraded→retry), not
                # whether the output artifacts should be persisted.
                if is_workflow_step(task_ctx):
                    await post_execute_workflow_step(task, result)
                    # Re-read context — post-hook may have stored _schema_error.
                    try:
                        _fresh = await get_task(task_id)
                        if _fresh:
                            _fc = _fresh.get("context", "{}")
                            if isinstance(_fc, str):
                                _fc = json.loads(_fc)
                            if isinstance(_fc, dict):
                                task_ctx.update(_fc)
                    except Exception:
                        pass
                    # Post-hook may override status
                    if result.get("status") == "needs_clarification":
                        if not await self._validate_clarification(
                            task_id, task, task_ctx, result
                        ):
                            result["status"] = "failed"
                        else:
                            await self._handle_clarification(task, result)
                            return
                    if result.get("status") == "failed":
                        error_msg = result.get("error", "Disguised failure detected")
                        from src.core.retry import RetryContext
                        retry_ctx = RetryContext.from_task(task)
                        decision = retry_ctx.record_failure("quality", model=result.get("model", ""))

                        # Bonus attempt for quality failures with real progress
                        _MAX_BONUS = 2
                        if decision.action == "terminal":
                            bonus_count = task_ctx.get("_bonus_count", 0)
                            if bonus_count < _MAX_BONUS:
                                try:
                                    progress = await self._assess_timeout_progress(
                                        task_id, task_ctx
                                    )
                                    if progress >= 0.5:
                                        task_ctx["_bonus_count"] = bonus_count + 1
                                        retry_ctx.max_worker_attempts += 1
                                        decision = retry_ctx.record_failure("quality",
                                            model=result.get("model", ""))
                                        logger.info(
                                            f"[Task #{task_id}] Quality bonus attempt "
                                            f"({bonus_count + 1}/{_MAX_BONUS}, "
                                            f"progress={progress:.0%})"
                                        )
                                except Exception:
                                    pass

                        if decision.action == "terminal":
                            task_ctx.update(retry_ctx.to_context_patch())
                            await update_task(
                                task_id, status="failed",
                                error=f"Disguised failure exhausted: {error_msg[:300]}",
                                context=json.dumps(task_ctx),
                                **retry_ctx.to_db_fields(),
                            )
                            try:
                                from src.infra.dead_letter import quarantine_task
                                await quarantine_task(
                                    task_id=task_id,
                                    mission_id=task.get("mission_id"),
                                    error=f"Disguised failure after {retry_ctx.worker_attempts} attempts: {error_msg[:300]}",
                                    error_category="quality",
                                    original_agent=task.get("agent_type", "executor"),
                                    attempts_snapshot=retry_ctx.worker_attempts,
                                )
                            except Exception:
                                pass
                            await self.telegram.send_notification(
                                f"❌ Task #{task_id} disguised failure → DLQ\n"
                                f"**{task.get('title', '')[:60]}**\n"
                                f"Reason: {error_msg[:100]}"
                            )
                        else:
                            next_retry = None
                            if decision.action == "delayed":
                                next_retry = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))
                            retry_ctx.next_retry_at = next_retry
                            task_ctx.update(retry_ctx.to_context_patch())
                            # Inject previous output so next attempt can continue
                            result_text = result.get("result", "")
                            if result_text:
                                task_ctx["_prev_output"] = str(result_text)[:6000]
                                task_ctx["_retry_hint"] = (
                                    "Your previous attempt's output failed quality checks. "
                                    "Your partial work is shown in context. Build on it — "
                                    "do NOT start over."
                                )
                            await update_task(
                                task_id, status="pending",
                                error=error_msg[:500],
                                context=json.dumps(task_ctx),
                                **retry_ctx.to_db_fields(),
                            )
                        return

                result_text = result.get("result", "No result")
                cost = result.get("cost", 0)
                await update_task(
                    task_id, status="ungraded", result=result_text, cost=cost,
                )
                logger.info("task ungraded (deferred grading)", task_id=task_id,
                            model=result.get("model", "unknown"))
            elif status == "pending":
                # Grade FAIL may have triggered retry or terminal DLQ.
                # Check actual DB state — apply_grade_result may have already
                # transitioned to 'failed' (terminal) since this status was set.
                db_task = await get_task(task_id)
                actual = db_task["status"] if db_task else "unknown"
                if actual == "failed":
                    logger.info("task grade-failed terminal (DLQ)", task_id=task_id)
                else:
                    logger.info("task grade-failed, retrying", task_id=task_id)
            elif status == "needs_subtasks":
                if is_workflow_step(task_ctx):
                    # Workflow steps must not decompose — they should produce
                    # their artifact directly.  Treat subtask plan as a
                    # quality failure so the task retries with a different model.
                    logger.warning(
                        f"[Task #{task_id}] Blocked subtask creation for "
                        f"workflow step — retrying"
                    )
                    from src.core.retry import RetryContext
                    retry_ctx = RetryContext.from_task(task)
                    retry_ctx.record_failure("quality", model=result.get("model", ""))
                    task_ctx.update(retry_ctx.to_context_patch())
                    await update_task(
                        task_id, status="pending",
                        error="Workflow step tried to decompose instead of producing artifact",
                        context=json.dumps(task_ctx),
                        **retry_ctx.to_db_fields(),
                    )
                else:
                    await self._handle_subtasks(task, result)
            elif status == "needs_clarification":
                # Silent/background tasks must not ask the user for clarification
                task_ctx = parse_context(task)
                if task_ctx.get("silent"):
                    logger.info(f"[Task #{task_id}] Suppressed clarification (silent task)")
                    await update_task(task_id, status="failed",
                                      error="Insufficient info (silent task, no clarification)",
                                      failed_in_phase="worker")
                elif task_ctx.get("may_need_clarification") is False:
                    # Workflow step declared it doesn't need clarification —
                    # agent is confused, retry with a different model.
                    logger.warning(
                        f"[Task #{task_id}] Suppressed clarification "
                        f"(may_need_clarification=false), retrying"
                    )
                    from src.core.retry import RetryContext
                    retry_ctx = RetryContext.from_task(task)
                    decision = retry_ctx.record_failure("quality")
                    if decision.action == "terminal":
                        await update_task(task_id, status="failed",
                                          error="Agent requested clarification on no-clarification step",
                                          **retry_ctx.to_db_fields())
                    else:
                        next_retry = None
                        if decision.action == "delayed":
                            next_retry = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))
                        retry_ctx.next_retry_at = next_retry
                        await update_task(task_id, status="pending",
                                          error="Suppressed clarification, retrying",
                                          **retry_ctx.to_db_fields())
                elif task_ctx.get("clarification_history"):
                    # Agent tried to clarify again despite having answers —
                    # treat as completed, using the Q&A exchange as the result
                    # so downstream artifacts capture the human's input.
                    logger.info(f"[Task #{task_id}] Suppressed repeat clarification "
                                f"(clarification_history already exists)")
                    history = task_ctx["clarification_history"]
                    # Build a readable Q&A result from the clarification exchange
                    qa_parts = []
                    for entry in history:
                        if isinstance(entry, dict):
                            q = entry.get("question", "")
                            a = entry.get("answer", "")
                        else:
                            # Legacy: plain string entries (answer only)
                            q, a = "", str(entry)
                        if q or a:
                            qa_parts.append(f"**Q:** {q}\n**A:** {a}")
                    qa_result = "\n\n".join(qa_parts) if qa_parts else task_ctx.get("user_clarification", "")
                    result["status"] = "completed"
                    result["result"] = qa_result or result.get("result", "")
                    # Run post-hook if this is a workflow step
                    if is_workflow_step(task_ctx):
                        await post_execute_workflow_step(task, result)
                    await self._handle_complete(task, result)
                else:
                    if is_workflow_step(task_ctx) and not await self._validate_clarification(
                        task_id, task, task_ctx, result
                    ):
                        # Validation failed — retry via the standard pipeline
                        from src.core.retry import RetryContext
                        retry_ctx = RetryContext.from_task(task)
                        decision = retry_ctx.record_failure("quality", model=result.get("model", ""))
                        if decision.action == "terminal":
                            task_ctx.update(retry_ctx.to_context_patch())
                            await update_task(
                                task_id, status="failed",
                                error=f"Clarification schema failed: {result.get('error', '')[:300]}",
                                context=json.dumps(task_ctx),
                                **retry_ctx.to_db_fields(),
                            )
                        else:
                            next_retry = None
                            if decision.action == "delayed":
                                next_retry = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))
                            retry_ctx.next_retry_at = next_retry
                            task_ctx.update(retry_ctx.to_context_patch())
                            await update_task(
                                task_id, status="pending",
                                error=f"Clarification schema failed: {result.get('error', '')[:200]}",
                                context=json.dumps(task_ctx),
                                **retry_ctx.to_db_fields(),
                            )
                    else:
                        await self._handle_clarification(task, result)
            elif status == "needs_review":
                await self._handle_review(task, result)
            elif status == "exhausted":
                exhaustion_reason = result.get("exhaustion_reason", "budget")
                exhaustion_guard_burns = result.get("guard_burns", 0)
                useful_iters = result.get("useful_iterations", 0)

                logger.warning(
                    "exhausted",
                    task_id=task_id,
                    reason=exhaustion_reason,
                    guards=exhaustion_guard_burns,
                    useful=useful_iters,
                )

                from src.core.retry import RetryContext
                retry_ctx = RetryContext.from_task(task)

                # Reason-aware retry
                if exhaustion_reason == "budget" and retry_ctx.worker_attempts < 2:
                    # First budget exhaustion — retry with more iterations
                    task_ctx["iteration_budget_boost"] = 1.5
                    retry_ctx.worker_attempts += 1
                    if result.get("model"):
                        if result["model"] not in retry_ctx.failed_models:
                            retry_ctx.failed_models.append(result["model"])
                    task_ctx.update(retry_ctx.to_context_patch())
                    await update_task(
                        task_id, status="pending",
                        context=json.dumps(task_ctx),
                        **retry_ctx.to_db_fields(),
                    )
                    logger.info(
                        "exhaustion budget retry",
                        task_id=task_id,
                        boost="1.5x",
                    )

                elif exhaustion_reason == "guards":
                    # Guards ate budget — suppress on retry
                    task_ctx["suppress_guards"] = True
                    retry_ctx.worker_attempts += 1
                    task_ctx.update(retry_ctx.to_context_patch())
                    await update_task(
                        task_id, status="pending",
                        context=json.dumps(task_ctx),
                        **retry_ctx.to_db_fields(),
                    )
                    logger.info(
                        "exhaustion guards retry",
                        task_id=task_id,
                        suppress_guards=True,
                    )

                else:
                    # tool_failures or repeated budget — standard quality retry
                    decision = retry_ctx.record_failure("quality", model=result.get("model", ""))
                    if decision.action == "terminal":
                        await update_task(
                            task_id, status="failed",
                            error=f"Exhausted after {retry_ctx.worker_attempts} attempts (reason={exhaustion_reason})",
                            **retry_ctx.to_db_fields(),
                        )
                        try:
                            from src.infra.dead_letter import quarantine_task
                            await quarantine_task(
                                task_id=task_id,
                                mission_id=task.get("mission_id"),
                                error=f"Iteration exhaustion ({exhaustion_reason})",
                                error_category="quality",
                                original_agent=task.get("agent_type", "executor"),
                                attempts_snapshot=retry_ctx.worker_attempts,
                            )
                        except Exception:
                            pass
                        await self.telegram.send_notification(
                            f"❌ Task #{task_id} exhaustion → DLQ\n"
                            f"**{title[:60]}**\n"
                            f"Reason: {exhaustion_reason}"
                        )
                    else:
                        next_retry = None
                        if decision.action == "delayed":
                            next_retry = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))
                        retry_ctx.next_retry_at = next_retry
                        retry_ctx.retry_reason = "quality"
                        task_ctx.update(retry_ctx.to_context_patch())
                        await update_task(
                            task_id, status="pending",
                            context=json.dumps(task_ctx),
                            **retry_ctx.to_db_fields(),
                        )

            elif status == "failed":
                error_str = result.get("error", result.get("result", "Unknown error"))
                from src.core.retry import RetryContext
                retry_ctx = RetryContext.from_task(task)
                decision = retry_ctx.record_failure("quality", model=result.get("model", ""))

                if decision.action == "terminal":
                    task_ctx.update(retry_ctx.to_context_patch())
                    await update_task(
                        task_id, status="failed",
                        error=error_str[:500],
                        context=json.dumps(task_ctx),
                        **retry_ctx.to_db_fields(),
                    )
                    logger.error("agent failure terminal", task_id=task_id,
                                 error=error_str[:200])
                    await self.telegram.send_error(task_id, title, error_str)
                    try:
                        from src.infra.dead_letter import quarantine_task
                        await quarantine_task(
                            task_id=task_id,
                            mission_id=task.get("mission_id"),
                            error=error_str[:500],
                            error_category="quality",
                            original_agent=task.get("agent_type", "executor"),
                            attempts_snapshot=retry_ctx.worker_attempts,
                        )
                    except Exception:
                        pass
                    try:
                        await self.telegram.send_notification(
                            f"🪦 Task #{task_id} → DLQ\n"
                            f"_{title[:60]}_\n"
                            f"Attempts: {retry_ctx.worker_attempts} | "
                            f"Reason: {error_str[:100]}"
                        )
                    except Exception:
                        pass
                else:
                    next_retry = None
                    if decision.action == "delayed":
                        next_retry = to_db(
                            utc_now() + timedelta(seconds=decision.delay_seconds)
                        )
                    retry_ctx.next_retry_at = next_retry
                    task_ctx.update(retry_ctx.to_context_patch())
                    await update_task(
                        task_id, status="pending",
                        error=error_str[:500],
                        context=json.dumps(task_ctx),
                        **retry_ctx.to_db_fields(),
                    )
                    logger.warning(f"agent failed, worker-retry {retry_ctx.worker_attempts}/{retry_ctx.max_worker_attempts}",
                                   task_id=task_id, error=error_str[:200])
            else:
                logger.warning("unknown task status", task_id=task_id, status=status)
                await self._handle_complete(task, result)

            # ── Phase 6: Release file locks held by this task ──
            try:
                await release_task_locks(task_id)
            except Exception:
                pass

        except ModelCallFailed as mcf:
            # ── Availability failure: all models exhausted ──
            # Use unified retry with backoff. Signal wakes (model_swap,
            # gpu_available, rate_limit_reset) can accelerate the retry.
            try:
                await release_task_locks(task_id)
            except Exception:
                pass

            # Read last_avail_delay from context for backoff progression
            _avail_ctx = task_ctx if isinstance(task_ctx, dict) else {}
            last_delay = _avail_ctx.get("last_avail_delay", 0)

            from src.core.retry import compute_retry_timing
            decision = compute_retry_timing("availability", last_avail_delay=last_delay)

            if decision.action == "terminal":
                await update_task(
                    task_id, status="failed",
                    error=str(mcf)[:500],
                    failed_in_phase="worker",
                    retry_reason="availability",
                )
                try:
                    from src.infra.dead_letter import quarantine_task
                    await quarantine_task(
                        task_id=task_id,
                        mission_id=task.get("mission_id"),
                        error=str(mcf)[:500],
                        error_category="availability",
                        original_agent=task.get("agent_type", "executor"),
                    )
                except Exception:
                    pass
                logger.warning(
                    "availability DLQ",
                    task_id=task_id,
                    error=str(mcf)[:200],
                )
            else:
                _avail_ctx["last_avail_delay"] = decision.delay_seconds
                next_retry = to_db(
                    utc_now() + timedelta(seconds=decision.delay_seconds)
                )
                await update_task(
                    task_id, status="pending",
                    error=str(mcf)[:500],
                    next_retry_at=next_retry,
                    retry_reason="availability",
                    context=json.dumps(_avail_ctx),
                )
                logger.warning(
                    "availability backoff",
                    task_id=task_id,
                    delay=decision.delay_seconds,
                )

        except Exception as e:
            logger.exception("task failed", task_id=task_id, error_type=type(e).__name__, error=str(e))
            # Release locks on failure too
            try:
                await release_task_locks(task_id)
            except Exception:
                pass
            error_str = f"{type(e).__name__}: {str(e)[:500]}"
            failed_model = result.get("model", "unknown") if isinstance(result, dict) else "unknown"

            from src.core.retry import RetryContext
            retry_ctx = RetryContext.from_task(task)
            decision = retry_ctx.record_failure("quality", model=failed_model)

            if decision.action == "terminal":
                try:
                    from ..infra.dead_letter import _classify_error
                    error_cat = _classify_error(error_str, "unknown")
                except Exception:
                    error_cat = "unknown"
                if isinstance(task_ctx, dict):
                    task_ctx.update(retry_ctx.to_context_patch())
                await update_task(
                    task_id, status="failed", error=error_str,
                    error_category=error_cat,
                    **retry_ctx.to_db_fields(),
                )
                await self.telegram.send_error(task_id, title, error_str)

                # Workflow step failure notification
                if isinstance(task_ctx, dict) and task_ctx.get("is_workflow_step"):
                    try:
                        wf_phase = task_ctx.get("workflow_phase", "?")
                        await self.telegram.send_notification(
                            f"Workflow step failed: #{task_id}\n"
                            f"_{task.get('title', '')[:60]}_\n"
                            f"Phase: {wf_phase}"
                        )
                    except Exception:
                        pass

                # Episodic memory
                try:
                    from ..memory.episodic import store_task_result
                    await store_task_result(
                        task=task, result=error_str, model="unknown",
                        cost=0.0, duration=0.0, success=False,
                    )
                except Exception:
                    pass

                # DLQ fallback
                try:
                    from ..infra.dead_letter import quarantine_task
                    await quarantine_task(
                        task_id=task_id,
                        mission_id=task.get("mission_id"),
                        error=error_str,
                        original_agent=task.get("agent_type", "executor"),
                        attempts_snapshot=retry_ctx.worker_attempts,
                    )
                except Exception as dlq_err:
                    logger.error("dlq quarantine failed", task_id=task_id, error=str(dlq_err))

                # Model health
                try:
                    from ..infra.db import update_model_stats
                    agent_type = task.get("agent_type", "executor")
                    model = result.get("model", "unknown") if isinstance(result, dict) else "unknown"
                    await update_model_stats(
                        model=model, agent_type=agent_type,
                        success=False, cost=0,
                    )
                except Exception:
                    pass
            else:
                next_retry = None
                if decision.action == "delayed":
                    next_retry = to_db(
                        utc_now() + timedelta(seconds=decision.delay_seconds)
                    )
                retry_ctx.next_retry_at = next_retry
                if isinstance(task_ctx, dict):
                    task_ctx.update(retry_ctx.to_context_patch())
                await update_task(
                    task_id, status="pending",
                    error=error_str,
                    **retry_ctx.to_db_fields(),
                )
                logger.info("task will retry", task_id=task_id,
                            attempts=retry_ctx.worker_attempts, max_attempts=retry_ctx.max_worker_attempts)

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
        task_ctx = parse_context(task)

        await update_task(
            task_id, status="completed", result=result_text,
            completed_at=db_now(),
            cost=cost,
        )

        # Clear checkpoint — task is fully done, no more retries possible
        try:
            from src.infra.db import clear_task_checkpoint
            await clear_task_checkpoint(task_id)
            logger.debug(f"Cleared checkpoint for completed task #{task_id}")
        except Exception:
            pass

        # Track skill A/B metrics
        try:
            injected_skills = task_ctx.get("injected_skills", [])
            agent_type = task.get("agent_type", "")
            started = task.get("started_at", "")
            completed_at_str = db_now()
            duration = 0.0
            if started:
                try:
                    t1 = from_db(started)
                    t2 = from_db(completed_at_str)
                    duration = (t2 - t1).total_seconds()
                except Exception:
                    pass
            from src.infra.db import record_skill_metric, record_no_skill_metric
            if injected_skills:
                for skill_name in injected_skills:
                    await record_skill_metric(
                        task_id=task_id, skill_name=skill_name,
                        succeeded=True, iterations=iterations,
                        agent_type=agent_type, duration=duration,
                    )
            else:
                await record_no_skill_metric(
                    task_id=task_id, succeeded=True,
                    iterations=iterations, agent_type=agent_type,
                    duration=duration,
                )
        except Exception:
            pass  # Non-critical

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
        task_ctx_parsed = task_ctx  # already parsed above
        is_interactive = task.get("priority", 5) >= TASK_PRIORITY.get("critical", 10)
        is_mission_subtask = task.get("mission_id") and task.get("parent_task_id")

        if task_ctx_parsed.get("silent"):
            logger.info("task completed (silent)", task_id=task_id, model=model, cost=cost)
        elif is_interactive or not is_mission_subtask:
            await self.telegram.send_result(task_id, task["title"],
                                            result_text, model, cost,
                                            mission_id=task.get("mission_id"))
        elif iterations > 3:
            await self.telegram.send_notification(
                f"🔧 Task #{task_id} completed after {iterations} iterations\n"
                f"_{task['title'][:60]}_"
            )

        logger.info("task completed", task_id=task_id, model=model, cost=cost, iterations=iterations)

        # Track injection success
        try:
            injected = task_ctx_parsed.get("injected_skills", [])
            if injected and result.get("status") != "ungraded":
                from ..memory.skills import record_injection_success
                await record_injection_success(injected)
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
        from dogru_mu_samet import assess as cq_assess
        for i, st in enumerate(subtasks):
            _st_title = st.get("title", f"Subtask {i+1}")[:80]
            _st_desc = st.get("description", "")[:2000]
            _st_cq = cq_assess(f"{_st_title} {_st_desc}")
            if _st_cq.is_degenerate:
                logger.warning(
                    f"[Orchestrator] Skipping degenerate subtask {i+1}: "
                    f"{_st_cq.summary}"
                )
                continue
            processed.append({
                "title": _st_title,
                "description": _st_desc,
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

    async def _assess_timeout_progress(
        self, task_id: int, task_ctx: dict
    ) -> float:
        """Estimate how much progress a timed-out task made (0.0–1.0).

        Checks: checkpoint iteration ratio, workspace files written,
        partial content length.  Used to decide bonus attempts.
        """
        score = 0.0

        # 1. Checkpoint iteration progress
        try:
            checkpoint = await load_task_checkpoint(task_id)
            if checkpoint:
                iteration = checkpoint.get("iteration", 0)
                max_iter = checkpoint.get("max_iterations", 7)
                if max_iter > 0:
                    score = max(score, iteration / max_iter)
                # Partial content from streaming
                msgs = checkpoint.get("messages", [])
                for msg in reversed(msgs):
                    if msg.get("role") == "assistant" and len(msg.get("content", "")) > 500:
                        score = max(score, 0.6)
                        break
        except Exception:
            pass

        # 2. Workspace files written by this task's mission
        # Agents may write to subdirectories (user_artifacts/, results/)
        # so walk the entire mission directory.
        mission_id = task_ctx.get("mission_id")
        output_names = task_ctx.get("output_artifacts", [])
        if mission_id and output_names:
            try:
                import os
                from ..tools.workspace import WORKSPACE_DIR
                mission_dir = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
                if os.path.isdir(mission_dir):
                    for root, _dirs, files in os.walk(mission_dir):
                        for fname in files:
                            stem = os.path.splitext(fname)[0]
                            if stem in output_names:
                                fpath = os.path.join(root, fname)
                                if os.path.getsize(fpath) > 200:
                                    score = max(score, 0.8)
                                    break
                        if score >= 0.8:
                            break
            except Exception:
                pass

        return score

    async def _validate_clarification(
        self, task_id, task, task_ctx, result
    ) -> bool:
        """Validate clarification content for triggers_clarification steps.

        Returns True if valid (or no schema to check), False if rejected.
        On rejection, sets result["error"] for the retry path.
        """
        artifact_schema = task_ctx.get("artifact_schema")
        if not artifact_schema or not task_ctx.get("triggers_clarification"):
            return True

        question_text = result.get("clarification", "")
        from src.workflows.engine.hooks import validate_artifact_schema
        is_valid, err = validate_artifact_schema(question_text, artifact_schema)
        if is_valid:
            return True

        logger.warning(
            f"[Task #{task_id}] Clarification rejected by schema: {err}"
        )
        result["error"] = f"Schema validation: {err}"

        # Store _schema_error so the retry prompt shows validation feedback
        try:
            new_ctx = dict(task_ctx)
            new_ctx["_schema_error"] = err
            await update_task(task_id, context=json.dumps(new_ctx))
        except Exception:
            pass
        return False

    async def _handle_clarification(self, task, result):
        task_id = task["id"]
        question = result.get("clarification", "Need more information")
        await update_task(task_id, status="waiting_human")
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
                          completed_at=db_now())
        if review_task_id:
            logger.info(f"[Task #{task_id}] Sent to reviewer (Task #{review_task_id})")
        else:
            logger.info(f"[Task #{task_id}] Review task deduped, skipping")

    # ─── Mission Completion ───────────────────────────────────────────────

    async def _check_mission_completion(self, mission_id):
        """Check if all tasks for a mission are done."""
        tasks = await get_tasks_for_mission(mission_id)
        if not tasks:
            return

        statuses = [t["status"] for t in tasks]
        pending = [s for s in statuses if s not in ("completed", "failed", "cancelled", "skipped")]

        if not pending:
            completed = [t for t in tasks if t["status"] == "completed"]
            failed = [t for t in tasks if t["status"] == "failed"]

            await update_mission(mission_id, status="completed",
                              completed_at=db_now())

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
                now = utc_now()

                # Calculate elapsed time
                elapsed_str = ""
                if mission_created:
                    try:
                        created_dt = from_db(str(mission_created))
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
                from src.tools.workspace import WORKSPACE_DIR as _ws_root
                results_dir = os.path.join(_ws_root, "results")
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

    # ─── Main Loop ───────────────────────────────────────────────────────

    async def run_loop(self):
        """Main autonomous work loop."""
        self.running = True
        logger.info("🚀 Autonomous orchestrator started")

        # Ensure workspace and git are ready
        try:
            import os
            from src.app.config import WORKSPACE_ROOT
            os.makedirs(WORKSPACE_ROOT, exist_ok=True)
            await ensure_git_repo()
        except Exception as e:
            logger.warning(f"Workspace/git init: {e}")

        _shutdown_signal = Path("logs") / "shutdown.signal"

        while self.running and not self.shutdown_event.is_set():
            try:
                # ── Check for external shutdown signal file ──
                if _shutdown_signal.exists():
                    try:
                        intent = _shutdown_signal.read_text().strip()
                        _shutdown_signal.unlink()
                        logger.info("External shutdown signal: %s", intent)
                        if intent == "restart":
                            self.requested_exit_code = 42
                        else:
                            self.requested_exit_code = 0
                        self.shutdown_event.set()
                    except Exception as e:
                        logger.warning("Failed to read shutdown signal: %s", e)

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
                    utc_now() - self.last_scheduler_check
                ).total_seconds()
                if sched_elapsed >= 60:
                    await self.scheduled_jobs.check_scheduled_tasks()
                    self.last_scheduler_check = utc_now()

                # Get a generous batch, then compute how many to actually run
                candidate_tasks = await get_ready_tasks(limit=8)

                # ── Age-based priority boost (starvation prevention) ──
                # +0.1 per hour waiting, max +1.0, so old tasks don't starve
                for _t in candidate_tasks:
                    _created = _t.get("created_at", "")
                    if _created:
                        try:
                            _age_h = (utc_now() - from_db(
                                _created
                            )).total_seconds() / 3600
                            _age_boost = min(_age_h * 0.1, 1.0)
                            _t["_effective_priority"] = _t.get("priority", 5) + _age_boost
                        except Exception:
                            _t["_effective_priority"] = _t.get("priority", 5)
                    else:
                        _t["_effective_priority"] = _t.get("priority", 5)

                # ── Skip tasks matching paused DLQ patterns ──
                if self.paused_patterns:
                    filtered = []
                    for _t in candidate_tasks:
                        if _t.get("error_category"):
                            pattern_key = f"category:{_t['error_category']}"
                            if pattern_key in self.paused_patterns:
                                logger.debug(f"[Task #{_t['id']}] Skipped — pattern {pattern_key} paused")
                                continue
                        filtered.append(_t)
                    candidate_tasks = filtered

                # ── Swap-aware: defer tasks that will reject the loaded model ──
                loaded_model = ""
                try:
                    from src.models.local_model_manager import get_local_manager
                    _mgr = get_local_manager()
                    loaded_model = getattr(_mgr, 'current_model', '') or ''
                except Exception:
                    pass

                if loaded_model and len(candidate_tasks) > 1:
                    runnable = [t for t in candidate_tasks if not _should_defer_for_loaded_model(t, loaded_model)]
                    deferred = [t for t in candidate_tasks if _should_defer_for_loaded_model(t, loaded_model)]
                    candidate_tasks = runnable or deferred  # fallback: run deferred if nothing else

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

                # Update queue depth metric for Prometheus/Grafana
                try:
                    from src.infra.metrics import record_queue_depth
                    record_queue_depth(len(candidate_tasks))
                except Exception:
                    pass


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

                # Drain overhead work — runs after every batch AND during
                # idle.  Without this, ungraded tasks and pending summaries
                # pile up while the queue has main work.
                try:
                    from src.core.grading import drain_ungraded_tasks
                    await drain_ungraded_tasks()
                except Exception as _gd_err:
                    logger.debug(f"Grade drain failed: {_gd_err}")

                try:
                    from src.workflows.engine.hooks import drain_pending_summaries
                    await drain_pending_summaries()
                except Exception as _sum_err:
                    logger.debug(f"Summary drain failed: {_sum_err}")

                if not tasks:
                    if self.cycle_count % 20 == 0:
                        logger.info(f"[Cycle {self.cycle_count}] Idle")

                    # Use shutdown-aware sleep instead of plain asyncio.sleep
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(), timeout=3
                        )
                        break  # shutdown requested during idle
                    except asyncio.TimeoutError:
                        pass  # normal idle cycle

                # Phase 14.1: Time-based morning briefing (default 9:00 Turkey local)
                from ..infra.times import turkey_now as _turkey_now
                now = _turkey_now()
                briefing_hour = int(os.environ.get("BRIEFING_HOUR", "9"))
                if (now.hour == briefing_hour
                        and now.date() > self.last_digest.date()):
                    await self.scheduled_jobs.tick_digest()
                    self.last_digest = now

                # ── Phase 11.6: Memory decay (weekly) ──
                days_since_decay = (
                    utc_now() - self.last_decay_check
                ).total_seconds() / 86400
                if days_since_decay >= 7:
                    try:
                        from ..memory.decay import run_decay_cycle
                        await run_decay_cycle()
                    except Exception as e:
                        logger.debug(f"Memory decay failed (non-critical): {e}")
                    self.last_decay_check = utc_now()

                # ── Prune old conversations (daily) ──
                if self.cycle_count == 1 or self.cycle_count % 8640 == 0:
                    try:
                        from src.infra.db import prune_old_conversations
                        pruned = await prune_old_conversations(30)
                        if pruned:
                            logger.info(f"Pruned {pruned} old conversations")
                    except Exception:
                        pass

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
                        self._last_improvement_check = utc_now()
                    elif (utc_now() - self._last_improvement_check).total_seconds() > 604800:  # 7 days
                        from src.memory.self_improvement import (
                            analyze_and_propose, format_proposals_for_telegram
                        )
                        proposals = await analyze_and_propose()
                        if proposals:
                            msg = format_proposals_for_telegram(proposals)
                            await self.telegram.send_notification(msg)
                        self._last_improvement_check = utc_now()
                except Exception as e:
                    logger.debug(f"Self-improvement check failed: {e}")

                # Heartbeat is now written by _heartbeat_loop() background task
                # so it stays alive even when the main loop blocks on LLM calls.

            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _heartbeat_loop(self):
        """Write heartbeat every 15s in a separate task.

        This runs independently of the main loop so the wrapper can detect
        when the orchestrator is truly hung (vs. just blocked on a long LLM call
        in the main loop).  If the event loop itself is blocked (e.g. sync I/O),
        this task also stops — which is the correct signal for 'hung'.
        """
        from yasar_usta import HeartbeatWriter
        writer = HeartbeatWriter(
            "logs/orchestrator.heartbeat", "logs/heartbeat", interval=15.0)
        await writer.run()

    async def _startup_recovery(self):
        """One-time recovery after restart: wake stuck/sleeping tasks.

        Runs once at boot before the main loop. Acts as a wake signal so
        that tasks stuck in retry backoff or interrupted mid-processing
        don't stay dormant until the watchdog slowly picks them up.
        """
        db = await get_db()
        summary: list[str] = []

        # 1. Reset tasks stuck in 'processing' back to 'pending'.
        #    These were interrupted by the restart and should be retried.
        cursor_proc = await db.execute(
            """SELECT id, title, infra_resets FROM tasks
               WHERE status = 'processing'"""
        )
        interrupted = [dict(row) for row in await cursor_proc.fetchall()]
        for task in interrupted:
            infra_resets = (task.get("infra_resets") or 0) + 1
            await db.execute(
                "UPDATE tasks SET status = 'pending', "
                "infra_resets = ?, retry_reason = 'infrastructure' WHERE id = ?",
                (infra_resets, task["id"]),
            )
        if interrupted:
            await db.commit()
            summary.append(
                f"Reset {len(interrupted)} interrupted processing task(s) "
                f"to pending"
            )

        # 2. Reset backoff context for availability-delayed tasks FIRST
        #    (must run before clearing next_retry_at, because
        #    accelerate_retries queries next_retry_at > datetime('now'))
        try:
            from ..infra.db import accelerate_retries
            woken = await accelerate_retries("startup")
            if woken:
                summary.append(
                    f"Accelerated {woken} availability-delayed task(s)"
                )
        except Exception as e:
            logger.debug(f"accelerate_retries on startup failed: {e}")

        # 3. Clear next_retry_at for remaining pending/ungraded tasks
        #    (infrastructure failures, etc.) so they retry immediately.
        cursor_retry = await db.execute(
            """SELECT id FROM tasks
               WHERE status IN ('pending', 'ungraded')
               AND next_retry_at IS NOT NULL
               AND next_retry_at > datetime('now')"""
        )
        delayed = [dict(row) for row in await cursor_retry.fetchall()]
        for task in delayed:
            await db.execute(
                "UPDATE tasks SET next_retry_at = NULL WHERE id = ?",
                (task["id"],),
            )
        if delayed:
            await db.commit()
            summary.append(
                f"Cleared retry backoff for {len(delayed)} delayed task(s)"
            )

        # 4. Release all stale file locks from the previous session
        try:
            await db.execute("DELETE FROM file_locks")
            await db.commit()
            summary.append("Released all stale file locks")
        except Exception:
            pass  # file_locks table may not exist yet

        if summary:
            msg = " | ".join(summary)
            logger.info(f"[Startup Recovery] {msg}")
        else:
            logger.info("[Startup Recovery] No stuck tasks found — clean start")

    async def start(self):
        await init_db()

        # ── Startup recovery: wake stuck/sleeping tasks ──
        try:
            await self._startup_recovery()
        except Exception as e:
            logger.warning(f"Startup recovery failed (non-fatal): {e}")

        # ── Start background infrastructure ──
        from ..models.local_model_manager import get_local_manager

        manager = get_local_manager()

        self._background_tasks: list[asyncio.Task] = [
            asyncio.create_task(manager.run_idle_unloader()),
            asyncio.create_task(manager.run_health_watchdog()),
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._run_stuck_task_watchdog()),
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
            await asyncio.gather(
                init_cache_db(),
                init_request_db(),
                init_memory_db(),
            )
            logger.info("Shopping DB schemas initialised")
        except Exception as e:
            logger.warning(f"Shopping DB init failed (non-fatal): {e}")

        async with self.telegram.app:
            await self.telegram.app.start()
            await self.telegram.app.updater.start_polling()
            await self.telegram.set_bot_commands()

            logger.info(
                "✅ System online — Telegram + Orchestrator + "
                "GPU Scheduler + Sleeping Queue running"
            )

            # Send persistent keyboard on startup so buttons are always visible
            try:
                from ..app.config import TELEGRAM_ADMIN_CHAT_ID
                from ..app.telegram_bot import REPLY_KEYBOARD
                if TELEGRAM_ADMIN_CHAT_ID:
                    await self.telegram.app.bot.send_message(
                        chat_id=TELEGRAM_ADMIN_CHAT_ID,
                        text="✅ Kutay online. Buttons ready.",
                        reply_markup=REPLY_KEYBOARD,
                    )
            except Exception as e:
                logger.debug(f"Startup keyboard send failed: {e}")

            # Restore pending clarification state from DB (re-ask pending questions)
            try:
                await self.telegram.restore_clarification_state()
            except Exception as e:
                logger.debug(f"Clarification state restore failed: {e}")

            try:
                await self.run_loop()
            finally:
                # ── Graceful shutdown ──
                if self.shutdown_event.is_set():
                    logger.info("Graceful shutdown initiated...")
                    self._shutting_down = True
                    self.running = False

                    # Stop background tasks
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
                            f"task(s) to complete (10s timeout)..."
                        )
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(
                                    *[asyncio.shield(f) for f in active_futures],
                                    return_exceptions=True,
                                ),
                                timeout=10,
                            )
                            logger.info("All running tasks completed cleanly")
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Shutdown timeout (10s) — "
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

                    # Flush model speed cache to disk
                    try:
                        from src.models.model_registry import get_registry
                        get_registry().flush_speed_cache()
                        logger.info("Model speed cache flushed")
                    except Exception:
                        pass

                    # Stop llama-server if running
                    try:
                        await manager.shutdown()
                        logger.info("llama-server stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping llama-server: {e}")

                    logger.info("Graceful shutdown complete")
                else:
                    # Non-graceful exit (crash, code 42 restart, etc.)
                    # Still need to stop llama-server to prevent orphans
                    try:
                        await manager.shutdown()
                        logger.info("llama-server stopped (non-graceful exit)")
                    except Exception as e:
                        logger.warning(f"Error stopping llama-server: {e}")

                # Skip WAL checkpoint on restarts (next instance uses WAL anyway)
                is_clean_stop = self.shutdown_event.is_set()
                await close_db(checkpoint=is_clean_stop)

                try:
                    await asyncio.wait_for(
                        self.telegram.app.updater.stop(), timeout=5
                    )
                except asyncio.TimeoutError:
                    logger.warning("Telegram updater.stop() timed out (5s)")
                try:
                    await asyncio.wait_for(
                        self.telegram.app.stop(), timeout=5
                    )
                except asyncio.TimeoutError:
                    logger.warning("Telegram app.stop() timed out (5s)")
