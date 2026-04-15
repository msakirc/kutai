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
    save_workspace_snapshot, release_task_locks, release_mission_locks,
)
from src.infra.logging_config import get_logger
from .router import ModelCallFailed, get_kdv
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
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    cls = ctx.get("classification", {})
    return max(1, min(10, int(cls.get("difficulty", 5))))


def _should_defer_for_loaded_model(task: dict, loaded_model: str) -> bool:
    """Check if this task would reject the currently loaded model."""
    worker_attempts = task.get("worker_attempts", task.get("attempts", 0)) or 0
    if worker_attempts < 3:
        return False
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            return False
    return loaded_model in ctx.get("failed_models", [])


_VALID_SUGGESTION_AGENTS = {"researcher", "shopping_advisor", "assistant", "coder"}

def _parse_todo_suggestions(raw: str, todo_count: int) -> list[dict]:
    """Parse LLM response into per-todo suggestions.

    Returns a list of length todo_count. Each element is:
      {"suggestion": str | None, "agent": str}

    Lenient parser: handles N. or N) prefixes, optional [agent] tags,
    markdown bold around tags, extra whitespace.
    """
    results = [{"suggestion": None, "agent": "researcher"} for _ in range(todo_count)]
    if not raw or not raw.strip():
        return results

    # Build a map: line_number → parsed content
    parsed_lines: dict[int, tuple[str, str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match: optional whitespace, number, . or ), rest
        m = re.match(r"(\d+)\s*[.)]\s*(.+)", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1  # 0-based
        text = m.group(2).strip()

        # Skip "no suggestion" variants
        if re.match(r"(?i)no\s+suggestion|n/a|none|-$", text):
            continue
        if len(text) < 6:
            continue

        # Extract [agent_type] — handle optional markdown bold: **[agent]**
        text = re.sub(r"^\*{1,2}\[", "[", text)
        text = re.sub(r"\]\*{1,2}", "]", text)
        agent_m = re.match(r"\[(\w+)\]\s*(.+)", text)
        if agent_m and agent_m.group(1).lower() in _VALID_SUGGESTION_AGENTS:
            agent = agent_m.group(1).lower()
            suggestion = agent_m.group(2).strip()
        else:
            agent = "researcher"
            suggestion = text

        if 0 <= idx < todo_count:
            parsed_lines[idx] = (suggestion, agent)

    for idx, (suggestion, agent) in parsed_lines.items():
        results[idx] = {"suggestion": suggestion, "agent": agent}

    return results


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
        """
        Detect and fix stuck states at BOTH task and resource level.

        Task-level:
          1. Tasks stuck in processing → retry or fail (attempts-based)
          2. Ungraded tasks stuck > 30 min → promote to completed
          3. Tasks with all deps failed → cascade failure
          4. Parent tasks with all children done → complete or fail
          5. Overdue next_retry_at → clear stale retry gates
          6. Waiting_human escalation (4h → 24h → 48h → 72h cancel)

        Resource-level:
          - Crashed llama-server → auto-restart
          - GPU OOM / thermal throttle → pause local, route to cloud
          - All cloud providers rate-limited → log warning, wait
          - Backpressure queue overload → alert via Telegram
        """
        db = await get_db()

        # ═══════════════════════════════════════════════════════════
        #  TASK-LEVEL RECOVERY
        # ═══════════════════════════════════════════════════════════

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
            raw_ctx = task.get("context", "{}")
            try:
                ctx = json.loads(raw_ctx) if isinstance(raw_ctx, str) else (raw_ctx or {})
            except (json.JSONDecodeError, TypeError):
                ctx = {}
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
               WHERE status = 'waiting_human'
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
            # ── API discovery (8:30am daily, catch-up if missed) ──
            try:
                await self._check_api_discovery()
            except Exception as exc:
                logger.debug("API discovery check failed: %s", exc)

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
                    now = utc_now()
                    next_run = self._compute_next_run(
                        sched.get("cron_expression", "0 * * * *"), now
                    )
                    await update_scheduled_task(
                        sched_id,
                        last_run=to_db(now),
                        next_run=to_db(next_run) if next_run else None,
                    )
                    continue

                # Special handling: one-shot reminders — send directly, then disable
                if sched_ctx.get("one_shot"):
                    reminder_text = sched_ctx.get("reminder_text", title)
                    try:
                        if self.telegram:
                            chat_id = sched_ctx.get("chat_id")
                            if chat_id:
                                await self.telegram.app.bot.send_message(
                                    chat_id=int(chat_id),
                                    text=f"⏰ *Hatırlatma*\n\n{reminder_text}",
                                    parse_mode="Markdown",
                                )
                            else:
                                await self.telegram.send_notification(
                                    f"⏰ *Hatırlatma*\n\n{reminder_text}"
                                )
                        logger.info(f"[Scheduler] One-shot reminder sent: {title}")
                    except Exception as e:
                        logger.error(f"[Scheduler] Reminder send failed: {e}")
                    # Disable — no next_run for one-shot
                    await update_scheduled_task(
                        sched_id,
                        last_run=db_now(),
                        next_run=None,
                        enabled=False,
                    )
                    continue

                # Special handling: price watch checker
                if sched_ctx.get("type") == "price_watch_check":
                    try:
                        from src.app.price_watch_checker import check_price_watches
                        summary = await check_price_watches(self.telegram)
                        logger.info(
                            f"[Scheduler] Price watch check complete: {summary}"
                        )
                    except Exception as e:
                        logger.error(f"[Scheduler] Price watch check failed: {e}")
                    now = utc_now()
                    next_run = self._compute_next_run(
                        sched.get("cron_expression", "0 * * * *"), now
                    )
                    await update_scheduled_task(
                        sched_id,
                        last_run=to_db(now),
                        next_run=to_db(next_run) if next_run else None,
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
                now = utc_now()
                next_run = self._compute_next_run(
                    sched.get("cron_expression", "0 * * * *"), now
                )
                await update_scheduled_task(
                    sched_id,
                    last_run=to_db(now),
                    next_run=to_db(next_run) if next_run else None,
                )

        except Exception as e:
            logger.error(f"[Scheduler] Error checking schedules: {e}")

    async def _check_api_discovery(self):
        """Run API discovery daily at 8:30am, with catch-up if missed."""
        now = utc_now()

        # Check DB for last discovery time (persists across restarts)
        last_discovery = None
        try:
            from src.infra.db import get_db
            db = await get_db()
            cur = await db.execute(
                "SELECT MAX(timestamp) FROM smart_search_log WHERE source = 'discovery'"
            )
            row = await cur.fetchone()
            if row and row[0]:
                last_discovery = from_db(row[0])
        except Exception:
            pass

        # Already ran today? Skip.
        if last_discovery and (now - last_discovery).total_seconds() < 86400:
            return

        from ..infra.times import turkey_now
        tr_now = turkey_now()
        in_window = tr_now.hour == 8 and 25 <= tr_now.minute <= 35
        overdue = (
            last_discovery is None
            or (now - last_discovery).total_seconds() > 36 * 3600
        )

        if not in_window and not overdue:
            return

        logger.info("Starting API discovery run")
        try:
            from src.tools.free_apis import discover_new_apis, build_keyword_index, seed_category_patterns
            from src.infra.db import log_smart_search

            new_count = await discover_new_apis("all")
            await build_keyword_index()
            await seed_category_patterns()

            # Record discovery run in DB so we don't re-run on restart
            await log_smart_search("discovery", layer=0, source="discovery", success=True, response_ms=0)

            if new_count > 0:
                logger.info("API discovery complete: %d new APIs", new_count)
                if hasattr(self, "_morning_brief_extras"):
                    self._morning_brief_extras.append(
                        f"Discovered {new_count} new APIs/MCP tools."
                    )
                if new_count >= 5 and self.telegram:
                    await self.telegram.send_notification(
                        f"API discovery: {new_count} new APIs added to registry."
                    )
            else:
                logger.info("API discovery complete: no new APIs found")
        except Exception as exc:
            logger.warning("API discovery failed: %s", exc)

    async def _start_todo_suggestions(self):
        """Generate AI suggestions for pending todos that don't have one yet.

        Only queries LLM for todos where suggestion IS NULL and suggestion_at IS NULL
        (never attempted). Todos with suggestion_at set but suggestion NULL were
        previously attempted and failed — skip them.
        """
        from src.infra.db import get_todos, update_todo
        from src.app.reminders import send_todo_reminder

        todos = await get_todos(status="pending")
        if not todos:
            return

        # Filter to todos that need suggestions (never attempted)
        need_suggestions = [
            t for t in todos
            if t.get("suggestion") is None and t.get("suggestion_at") is None
        ]

        if need_suggestions:
            await self._generate_suggestions(need_suggestions)

        # Always send the reminder (suggestions are read from DB by reminders.py)
        if self.telegram:
            await send_todo_reminder(self.telegram)

    async def _generate_suggestions(self, todos: list[dict]):
        """Call LLM to generate suggestions for given todos, persist results."""
        from src.infra.db import update_todo

        todo_lines = "\n".join(
            f"{i+1}. {t['title']}"
            + (f" (priority: {t.get('priority', 'normal')})" if t.get("priority") != "normal" else "")
            + (f" (notes: {t['description'][:80]})" if t.get("description") else "")
            for i, t in enumerate(todos[:10])
        )
        prompt = (
            f"The user has {len(todos)} pending todo item(s):\n\n"
            f"{todo_lines}\n\n"
            f"For each item, suggest ONE concrete, actionable way an AI assistant could help "
            f"(e.g. search, compare prices, book, remind, draft a message). "
            f"Be creative — even mundane tasks like 'buy milk' could mean price comparison or online ordering. "
            f"If you genuinely cannot help with an item, write 'no suggestion'.\n\n"
            f"Also pick the best agent type for each suggestion:\n"
            f"  researcher — web search, information gathering, fact-checking\n"
            f"  shopping_advisor — product search, price comparison, deal finding\n"
            f"  assistant — drafting messages, reminders, general help\n"
            f"  coder — writing code, scripts, technical tasks\n\n"
            f"Reply ONLY with a numbered list. Format: NUMBER. [agent_type] suggestion text\n"
            f"Example: 1. [researcher] Search for nearby tire shops and compare prices.\n"
            f"No preamble, no extra commentary."
        )

        now_str = db_now()

        try:
            from src.core.llm_dispatcher import get_dispatcher, CallCategory

            dispatcher = get_dispatcher()
            response = await asyncio.wait_for(
                dispatcher.request(
                    category=CallCategory.OVERHEAD,
                    task="assistant",
                    difficulty=2,
                    messages=[{"role": "user", "content": prompt}],
                    estimated_input_tokens=400,
                    estimated_output_tokens=150,
                    prefer_speed=True,
                    priority=2,
                ),
                timeout=45,
            )
            raw = (response.get("content") or "").strip()
            logger.info(f"[Todo] Suggestion LLM response ({len(raw)} chars)")

            parsed = _parse_todo_suggestions(raw, len(todos[:10]))

            for i, todo in enumerate(todos[:10]):
                entry = parsed[i]
                if entry["suggestion"]:
                    await update_todo(
                        todo["id"],
                        suggestion=entry["suggestion"],
                        suggestion_agent=entry["agent"],
                        suggestion_at=now_str,
                    )
                else:
                    # Mark as attempted-but-failed so we don't retry
                    await update_todo(todo["id"], suggestion_at=now_str)

            generated = sum(1 for p in parsed if p["suggestion"])
            logger.info(f"[Todo] Generated {generated}/{len(todos[:10])} suggestions")

        except asyncio.TimeoutError:
            logger.warning("[Todo] Suggestion LLM call timed out — marking todos as attempted")
            for todo in todos[:10]:
                await update_todo(todo["id"], suggestion_at=now_str)
        except Exception as exc:
            logger.warning(f"[Todo] Suggestion LLM call failed: {exc} — marking todos as attempted")
            for todo in todos[:10]:
                await update_todo(todo["id"], suggestion_at=now_str)

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
        result = None  # initialized here so except handler can safely access it
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
                task["context"] = json.dumps(task_ctx)

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
                    return
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
                    task["context"] = json.dumps(task_ctx)
                    logger.info("task enriched with API data", task_id=task_id)
            except Exception as exc:
                logger.debug("context enrichment failed (non-critical): %s", exc)

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
                        await update_task(task_id, status="cancelled")
                        return
                    logger.info("human gate approved", task_id=task_id)
                except Exception as e:
                    logger.error("human gate error", task_id=task_id, error=str(e))

            # ── Phase 14.2: Risk assessment gate ──
            # Skip for workflow steps — they are pre-defined in the workflow
            # JSON and approval would block the automated pipeline.  Workflow
            # human gates are handled separately via needs_clarification.
            is_workflow = task_ctx.get("is_workflow_step", False)
            try:
                from ..security.risk_assessor import assess_risk, format_risk_assessment
                risk = assess_risk(
                    task_title=task.get("title", ""),
                    task_description=task.get("description", ""),
                )
                if risk["needs_approval"] and not task_ctx.get("human_gate") and not is_workflow:
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
                        await update_task(task_id, status="cancelled")
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
                    return

            # ── Determine timeout ──
            timeout_seconds = (
                task.get("timeout_seconds")
                or AGENT_TIMEOUTS.get(agent_type, 240)
            )

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
                    task_ctx = json.loads(task.get("context", "{}"))
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
                            task_ctx = task.get("context", {})
                            if isinstance(task_ctx, str):
                                try:
                                    task_ctx = json.loads(task_ctx)
                                except (json.JSONDecodeError, TypeError):
                                    task_ctx = {}

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
                task_ctx = task.get("context", {})
                if isinstance(task_ctx, str):
                    try:
                        task_ctx = json.loads(task_ctx)
                    except (json.JSONDecodeError, TypeError):
                        task_ctx = {}

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
            completed_at=db_now(),
            cost=cost,
        )

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
        mission_id = task_ctx.get("mission_id")
        output_names = task_ctx.get("output_artifacts", [])
        if mission_id and output_names:
            try:
                import os
                from ..tools.workspace import WORKSPACE_DIR
                artifact_dir = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
                for name in output_names:
                    for ext in (".md", ".json", ".txt"):
                        fpath = os.path.join(artifact_dir, f"{name}{ext}")
                        if os.path.isfile(fpath) and os.path.getsize(fpath) > 200:
                            score = max(score, 0.8)
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
                    await self.check_scheduled_tasks()
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
                else:
                    if self.cycle_count % 20 == 0:
                        logger.info(f"[Cycle {self.cycle_count}] Idle")

                    # Grade ungraded tasks with loaded model (if compatible)
                    try:
                        from src.core.llm_dispatcher import get_dispatcher
                        from src.core.grading import drain_ungraded_tasks
                        dispatcher = get_dispatcher()
                        loaded = dispatcher._get_loaded_litellm_name()
                        if loaded:
                            await drain_ungraded_tasks(loaded)
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

                # Phase 14.1: Time-based morning briefing (default 9:00 Turkey local)
                from ..infra.times import turkey_now as _turkey_now
                now = _turkey_now()
                briefing_hour = int(os.environ.get("BRIEFING_HOUR", "9"))
                if (now.hour == briefing_hour
                        and now.date() > self.last_digest.date()):
                    await self.daily_digest()
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

        # 2. Clear next_retry_at for all pending/ungraded tasks so they
        #    retry immediately instead of sleeping through old backoff
        #    timers from the previous session.
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

        # 3. Accelerate availability-delayed tasks (resets backoff context)
        try:
            from ..infra.db import accelerate_retries
            woken = await accelerate_retries("startup")
            if woken:
                summary.append(
                    f"Accelerated {woken} availability-delayed task(s)"
                )
        except Exception as e:
            logger.debug(f"accelerate_retries on startup failed: {e}")

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
