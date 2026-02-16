# orchestrator.py
import asyncio
import json
import logging
import aiosqlite
from config import DB_PATH
from datetime import datetime
from db import (
    init_db, get_ready_tasks, update_task, add_task, log_conversation,
    get_active_goals, get_tasks_for_goal, update_goal, get_daily_stats,
    store_memory
)
from router import classify_task
from agents import get_agent, AGENT_REGISTRY
from tools import execute_tool
from telegram_bot import TelegramInterface
from config import MODEL_TIERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLANNING_PROMPT_CONTEXT = """Break this goal into actionable subtasks."""


class Orchestrator:
    def __init__(self):
        self.telegram = TelegramInterface(self)
        self.running = False
        self.cycle_count = 0

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

                # Check if ANY dependency has failed
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

    async def process_task(self, task: dict):
        """Process a single task through the appropriate agent."""
        task_id = task["id"]
        title = task["title"]
        agent_type = task.get("agent_type", "executor")

        logger.info(f"[Task #{task_id}] Starting: '{title}' (agent: {agent_type})")

        try:
            await update_task(task_id, status="processing",
                              started_at=datetime.now().isoformat())

            agent = get_agent(agent_type)
            logger.info(f"[Task #{task_id}] Agent '{agent.name}' executing with tier '{task.get('tier','auto')}'")

            result = await agent.execute(task)
            status = result.get("status", "complete")

            logger.info(f"[Task #{task_id}] Agent returned status: '{status}'")

            if status == "complete":
                await self._handle_complete(task, result)
            elif status == "needs_subtasks":
                await self._handle_subtasks(task, result)
            elif status == "needs_clarification":
                await self._handle_clarification(task, result)
            elif status == "needs_review":
                await self._handle_review(task, result)
            elif status == "needs_tool":
                await self._handle_tool_request(task, result)
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

    async def _handle_complete(self, task, result):
        task_id = task["id"]
        result_text = result.get("result", "No result")
        model = result.get("model", "unknown")
        cost = result.get("cost", 0)

        await update_task(
            task_id, status="completed", result=result_text,
            completed_at=datetime.now().isoformat()
        )

        # Check if all tasks for this goal are done
        if task.get("goal_id"):
            await self._check_goal_completion(task["goal_id"])

        # Only notify human for top-level tasks or important results
        if not task.get("parent_task_id"):
            await self.telegram.send_result(task_id, task["title"],
                                            result_text, model, cost)

        logger.info(f"[Task #{task_id}] ✅ Complete via {model} (${cost:.4f})")

    async def _handle_subtasks(self, task, result):
        task_id = task["id"]
        goal_id = task.get("goal_id")
        subtasks = result.get("subtasks", [])

        if not subtasks:
            await self._handle_complete(task, result)
            return

        # Limit subtasks to prevent explosion
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

            # Resolve tier: don't trust planner to assign tiers that exist
            requested_tier = st.get("tier", "auto")
            if requested_tier != "auto" and requested_tier not in MODEL_TIERS:
                # Map to best available
                tier_map = {"expensive": "medium", "medium": "cheap"}
                resolved = tier_map.get(requested_tier, "auto")
                if resolved not in MODEL_TIERS and resolved != "auto":
                    resolved = "auto"
                logger.info(
                    f"Subtask tier '{requested_tier}' unavailable, "
                    f"using '{resolved}'"
                )
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

        # Create a reviewer task
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

    async def _handle_tool_request(self, task: dict, result: dict):
        """Agent needs a tool — execute it and feed result back."""
        task_id = task["id"]
        tool_name = result.get("tool", "")
    
        # Track how deep the tool chain goes
        task_context = json.loads(task.get("context", "{}"))
        tool_depth = task_context.get("tool_depth", 0)
    
        MAX_TOOL_DEPTH = 3  # prevent infinite loops
    
        if tool_depth >= MAX_TOOL_DEPTH:
            logger.warning(
                f"[Task #{task_id}] Tool depth limit reached ({MAX_TOOL_DEPTH}). "
                f"Forcing completion."
            )
            await update_task(
                task_id, status="completed",
                result=f"Task stopped: exceeded maximum tool chain depth ({MAX_TOOL_DEPTH}). "
                       f"Last tool requested: {tool_name}",
                completed_at=datetime.now().isoformat()
            )
            return
    
        # Check if tool exists
        from tools import TOOL_REGISTRY
        if tool_name not in TOOL_REGISTRY:
            logger.warning(
                f"[Task #{task_id}] Unknown tool '{tool_name}' requested. "
                f"Completing with error."
            )
            # Don't create a follow-up — just tell the agent the tool doesn't exist
            # and let it complete with what it has
            await update_task(
                task_id, status="completed",
                result=f"Agent requested unavailable tool '{tool_name}'. "
                       f"Available tools: {list(TOOL_REGISTRY.keys())}. "
                       f"Task completed with partial result.",
                completed_at=datetime.now().isoformat()
            )
            return
    
        logger.info(f"[Task #{task_id}] Using tool: {tool_name}")
    
        # Build tool kwargs from result, excluding metadata
        tool_args = {k: v for k, v in result.items()
                     if k not in ("status", "tool", "model", "cost", "tier",
                                  "memories", "result")}
    
        tool_result = await execute_tool(tool_name, **tool_args)
    
        # Create follow-up task with incremented depth
        followup_context = {
            "tool_result": tool_result[:3000],  # Limit tool result size
            "original_task_id": task_id,
            "tool_depth": tool_depth + 1
        }
    
        followup_id = await add_task(
            title=f"Continue: {task['title'][:40]}",
            description=(
                f"Original task: {task['description'][:500]}\n\n"
                f"Tool '{tool_name}' returned:\n{tool_result[:2000]}\n\n"
                f"Now complete the original task using this information. "
                f"Do NOT request more tools unless absolutely necessary. "
                f"Provide your final answer."
            ),
            goal_id=task.get("goal_id"),
            parent_task_id=task.get("parent_task_id"),
            agent_type=task.get("agent_type", "executor"),
            tier=task.get("tier", "auto"),
            priority=task.get("priority", 5),
            context=followup_context
        )
    
        await update_task(
            task_id, status="completed",
            result=f"Used tool {tool_name}, continuing in Task #{followup_id}",
            completed_at=datetime.now().isoformat()
        )

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

            # Summarize goal results
            results_summary = "\n".join(
                f"• {t['title']}: {(t.get('result') or '')[:100]}"
                for t in completed[-10:]
            )

            await self.telegram.send_notification(
                f" *Goal Completed!*\n\n"
                f"Tasks: {len(completed)} completed, {len(failed)} failed\n\n"
                f"Results:\n{results_summary}"
            )

    async def plan_goal(self, goal_id: int, title: str, description: str):
        """Create initial planning task for a new goal."""
        await add_task(
            title=f"Plan: {title[:40]}",
            description=f"Create an execution plan for this goal:\n\n{title}\n\n{description}",
            goal_id=goal_id,
            agent_type="planner",
            tier="medium",
            priority=8
        )

    async def daily_digest(self):
        """Send daily status summary."""
        stats = await get_daily_stats()
        goals = await get_active_goals()

        goals_text = "\n".join(f"  • {g['title']}" for g in goals[:5]) or "  None"

        await self.telegram.send_notification(
            f" *Daily Digest*\n\n"
            f"**Tasks today:**\n"
            f"  ✅ Completed: {stats['completed']}\n"
            f"  ⏳ Pending: {stats['pending']}\n"
            f"   Processing: {stats['processing']}\n"
            f"  ❌ Failed: {stats['failed']}\n"
            f"   Cost today: ${stats['today_cost']:.4f}\n\n"
            f"**Active goals:**\n{goals_text}"
        )

    async def run_loop(self):
        """Main autonomous work loop."""
        self.running = True
        logger.info("🚀 Autonomous orchestrator started")

        while self.running:
            try:
                self.cycle_count += 1

                if self.cycle_count % 10 == 0:
                    await self.watchdog()

                # Fetch up to 5 ready tasks per cycle
                tasks = await get_ready_tasks(limit=5)

                if tasks:
                    task_names = [f"#{t['id']}({t.get('agent_type','?')})" for t in tasks]
                    logger.info(
                        f"[Cycle {self.cycle_count}] "
                        f"Processing {len(tasks)} task(s): {task_names}"
                    )

                    # Process tasks SEQUENTIALLY to avoid rate limits on free tier
                    for t in tasks:
                        try:
                            await self.process_task(t)
                            await asyncio.sleep(3)  # 3 second gap between tasks
                        except Exception as e:
                            logger.error(f"Task #{t['id']} error: {e}", exc_info=True)

                    await asyncio.sleep(5)  # extra breathing room after batch
                else:
                    if self.cycle_count % 20 == 0:
                        logger.info(f"[Cycle {self.cycle_count}] Idle")
                    await asyncio.sleep(15)

                if self.cycle_count % 500 == 0:
                    await self.daily_digest()

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
