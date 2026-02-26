# orchestrator.py
import asyncio
import json
import logging
import aiosqlite
from config import DB_PATH, MAX_CONTEXT_CHAIN_LENGTH, TASK_PRIORITY
from datetime import datetime
from db import (
    init_db, get_ready_tasks, update_task, add_task, log_conversation,
    get_active_goals, get_tasks_for_goal, update_goal, get_daily_stats,
    store_memory
)
from router import classify_task
from agents import get_agent, AGENT_REGISTRY
from tools import execute_tool
from tools.workspace import get_file_tree
from tools.git_ops import git_commit, ensure_git_repo
from telegram_bot import TelegramInterface
from router import MODEL_TIERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        self.telegram = TelegramInterface(self)
        self.running = False
        self.cycle_count = 0
        self.last_digest = datetime.now()

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
            async with aiosqlite.connect(DB_PATH) as db:
                db.row_factory = aiosqlite.Row
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

        # ── Workspace snapshot ──
        # For coder/reviewer/writer agents, include the current file tree
        agent_type = task.get("agent_type", "executor")
        if agent_type in ("coder", "reviewer", "writer", "planner"):
            try:
                tree = await get_file_tree(max_depth=3)
                # Only include if there's something interesting
                if tree and "File not found" not in tree and len(tree.split("\n")) > 1:
                    task_context["workspace_snapshot"] = tree
            except Exception as e:
                logger.debug(f"Could not get workspace snapshot: {e}")

        # Write context back to task dict
        task["context"] = json.dumps(task_context)
        return task

    # ─── NEW: Auto-commit after coder tasks ─────────────────────────────

    async def _auto_commit(self, task: dict, result: dict):
        """Auto-commit workspace changes after a successful coder task."""
        try:
            await ensure_git_repo()
            commit_msg = f"Task #{task['id']}: {task.get('title', 'untitled')[:60]}"
            commit_result = await git_commit(commit_msg)
            if "Nothing to commit" not in commit_result:
                logger.info(f"[Task #{task['id']}] Auto-committed: {commit_msg}")
        except Exception as e:
            logger.debug(f"Auto-commit skipped: {e}")

    # ─── Watchdog (unchanged) ────────────────────────────────────────────

    async def watchdog(self):
        """Detect and fix stuck states."""
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row

            # 1. Tasks stuck in "processing" for more than 5 minutes
            cursor = await db.execute(
                """SELECT id, title FROM tasks
                   WHERE status = 'processing'
                   AND started_at < datetime('now', '-5 minutes')"""
            )
            stuck = [dict(row) for row in await cursor.fetchall()]
            for task in stuck:
                logger.warning(f"[Watchdog] Task #{task['id']} stuck in processing, resetting")
                await db.execute(
                    "UPDATE tasks SET status = 'pending', retry_count = retry_count + 1 WHERE id = ?",
                    (task["id"],)
                )

            # 2. Tasks blocked by FAILED dependencies — unblock or fail them
            cursor2 = await db.execute(
                "SELECT id, title, depends_on FROM tasks WHERE status = 'pending' AND depends_on != '[]'"
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
                    f"SELECT COUNT(*) FROM tasks WHERE id IN ({placeholders}) AND status = 'failed'",
                    deps
                )
                failed_count = (await failed_cursor.fetchone())[0]

                if failed_count > 0:
                    logger.warning(
                        f"[Watchdog] Task #{task['id']} has failed dependencies, "
                        f"clearing deps so it can attempt to run"
                    )
                    await db.execute(
                        "UPDATE tasks SET depends_on = '[]' WHERE id = ?",
                        (task["id"],)
                    )

            # 3. Goals with "waiting_subtasks" parent but all children done
            cursor3 = await db.execute(
                "SELECT id, title FROM tasks WHERE status = 'waiting_subtasks'"
            )
            waiting = [dict(row) for row in await cursor3.fetchall()]
            for task in waiting:
                child_cursor = await db.execute(
                    """SELECT COUNT(*) as total,
                              SUM(CASE WHEN status IN ('completed','failed','rejected') THEN 1 ELSE 0 END) as done
                       FROM tasks WHERE parent_task_id = ?""",
                    (task["id"],)
                )
                row = await child_cursor.fetchone()
                if row and row["total"] > 0 and row["total"] == row["done"]:
                    logger.info(f"[Watchdog] Task #{task['id']} all subtasks done, marking complete")
                    await db.execute(
                        "UPDATE tasks SET status = 'completed', completed_at = ? WHERE id = ?",
                        (datetime.now().isoformat(), task["id"])
                    )

            await db.commit()

    # ─── Core Task Processing ────────────────────────────────────────────

    async def process_task(self, task: dict):
        """Process a single task through the appropriate agent with context injection."""
        task_id = task["id"]
        title = task["title"]
        agent_type = task.get("agent_type", "executor")

        logger.info(f"[Task #{task_id}] Starting: '{title}' (agent: {agent_type})")

        try:
            await update_task(task_id, status="processing",
                              started_at=datetime.now().isoformat())

            # ── Inject context from prior steps + workspace snapshot ──
            task = await self._inject_chain_context(task)

            agent = get_agent(agent_type)
            logger.info(f"[Task #{task_id}] Agent '{agent.name}' executing (tier: {task.get('tier', 'auto')})")

            result = await agent.execute(task)
            status = result.get("status", "complete")

            logger.info(f"[Task #{task_id}] Agent returned status: '{status}'")

            # Auto-commit after successful coder tasks
            if status == "complete" and agent_type == "coder":
                await self._auto_commit(task, result)

            if status == "complete":
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

        except Exception as e:
            logger.error(f"[Task #{task_id}] FAILED: {type(e).__name__}: {e}", exc_info=True)
            retry_count = task.get("retry_count", 0)
            max_retries = task.get("max_retries", 3)

            if retry_count < max_retries:
                await update_task(task_id, status="pending",
                                  retry_count=retry_count + 1,
                                  error=f"{type(e).__name__}: {str(e)[:200]}")
                logger.info(f"[Task #{task_id}] Will retry ({retry_count + 1}/{max_retries})")
            else:
                await update_task(task_id, status="failed",
                                  error=f"{type(e).__name__}: {str(e)[:500]}")
                await self.telegram.send_error(task_id, title, str(e)[:500])

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

        created_ids = []
        for i, st in enumerate(subtasks):
            depends_on = []
            dep_step = st.get("depends_on_step")
            if dep_step is not None and isinstance(dep_step, int) and 0 <= dep_step < len(created_ids):
                depends_on = [created_ids[dep_step]]

            requested_tier = st.get("tier", "auto")
            if requested_tier != "auto" and requested_tier not in MODEL_TIERS:
                tier_map = {"expensive": "medium", "medium": "cheap"}
                resolved = tier_map.get(requested_tier, "auto")
                if resolved not in MODEL_TIERS and resolved != "auto":
                    resolved = "auto"
                requested_tier = resolved

            sub_id = await add_task(
                title=st.get("title", f"Subtask {i+1}")[:80],
                description=st.get("description", "")[:2000],
                goal_id=goal_id,
                parent_task_id=task_id,
                agent_type=st.get("agent_type", "executor"),
                tier=requested_tier,
                priority=st.get("priority", task.get("priority", 5)),
                depends_on=depends_on
            )
            created_ids.append(sub_id)

        plan_summary = result.get("plan_summary", f"Created {len(subtasks)} subtasks")
        await update_task(task_id, status="waiting_subtasks", result=plan_summary)

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
        logger.info(f"[Task #{task_id}] Sent to reviewer (Task #{review_task_id})")

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
        await add_task(
            title=f"Plan: {title[:40]}",
            description=f"Create an execution plan for this goal:\n\n{title}\n\n{description}",
            goal_id=goal_id,
            agent_type="planner",
            tier="medium",
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

        while self.running:
            try:
                self.cycle_count += 1

                if self.cycle_count % 10 == 0:
                    await self.watchdog()

                tasks = await get_ready_tasks(limit=3)

                if tasks:
                    task_names = [f"#{t['id']}({t.get('agent_type','?')})" for t in tasks]
                    logger.info(
                        f"[Cycle {self.cycle_count}] "
                        f"Processing {len(tasks)} task(s): {task_names}"
                    )

                    for t in tasks:
                        try:
                            await self.process_task(t)
                            await asyncio.sleep(1)
                        except Exception as e:
                            logger.error(f"Task #{t['id']} error: {e}", exc_info=True)

                    await asyncio.sleep(2)
                else:
                    if self.cycle_count % 20 == 0:
                        logger.info(f"[Cycle {self.cycle_count}] Idle")
                    await asyncio.sleep(3)

                hours_since_digest = (datetime.now() - self.last_digest).total_seconds() / 3600
                if hours_since_digest >= 24:
                    await self.daily_digest()
                    self.last_digest = datetime.now()

            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def start(self):
        await init_db()

        async with self.telegram.app:
            await self.telegram.app.start()
            await self.telegram.app.updater.start_polling()

            logger.info("✅ System online — Telegram + Orchestrator running")

            try:
                await self.run_loop()
            finally:
                await self.telegram.app.updater.stop()
                await self.telegram.app.stop()
