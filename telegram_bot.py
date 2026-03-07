# telegram_bot.py
import asyncio
import logging
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID, TASK_PRIORITY
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from db import (add_task, add_goal, get_active_goals, get_ready_tasks,
                get_daily_stats, update_task, get_recent_completed_tasks,
                get_db, cancel_task, reprioritize_task, get_task_tree,
                get_task, get_budget, set_budget, get_model_stats,
                get_goal_locks)
from tools.workspace import (
    list_goal_workspaces, load_projects_config, get_project,
)


pending_clarifications = {}  # task_id -> asyncio.Event + response


class TelegramInterface:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self._setup_handlers()
        self._approval_events = {}
        self._clarification_events = {}
        self.user_last_task_id = {}


    def _setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("goal", self.cmd_add_goal))
        self.app.add_handler(CommandHandler("task", self.cmd_add_task))
        self.app.add_handler(CommandHandler("goals", self.cmd_list_goals))
        self.app.add_handler(CommandHandler("queue", self.cmd_view_queue))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("digest", self.cmd_digest))
        self.app.add_handler(CommandHandler("debug", self.cmd_debug))       # NEW
        self.app.add_handler(CommandHandler("reset", self.cmd_reset))       # NEW
        self.app.add_handler(CommandHandler("resetall", self.cmd_reset_all)) # NEW
        self.app.add_handler(CommandHandler("cancel", self.cmd_cancel))     # Phase 3
        self.app.add_handler(CommandHandler("priority", self.cmd_priority)) # Phase 3
        self.app.add_handler(CommandHandler("graph", self.cmd_graph))       # Phase 3
        self.app.add_handler(CommandHandler("budget", self.cmd_budget))     # Phase 4
        self.app.add_handler(CommandHandler("modelstats", self.cmd_model_stats))  # Phase 4
        self.app.add_handler(CommandHandler("workspace", self.cmd_workspace))  # Phase 6
        self.app.add_handler(CommandHandler("project", self.cmd_project))      # Phase 6
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.REPLY,
            self.handle_reply
        ))
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_message
        ))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            " *Autonomous AI Orchestrator*\n\n"
            "I work 24/7 and only bug you when needed.\n\n"
            "*Commands:*\n"
            "/goal <description> — Set a high-level goal\n"
            "/task <description> — Add a one-off task\n"
            "/goals — View active goals\n"
            "/queue — View pending tasks\n"
            "/status — System stats\n"
            "/cancel <id> — Cancel a task\n"
            "/priority <id> <1-10> — Reprioritize\n"
            "/graph <goal\\_id> — Show task dependency graph\n"
            "/budget [daily\\_limit] — View/set cost budget\n"
            "/modelstats — View model performance stats\n"
            "/workspace — View goal workspaces\n"
            "/project [name] — List/view projects\n"
            "/digest — Get daily digest now\n\n"
            "Or just send a message — I'll figure out what to do.",
            parse_mode="Markdown"
        )

    async def cmd_add_goal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text(
                "Usage: /goal <description>\n"
                "Example: /goal Research the top 5 competitors in the EV market "
                "and create a comparison report"
            )
            return

        description = " ".join(context.args)
        title = description[:80]
        goal_id = await add_goal(title=title, description=description, priority=7)

        # Trigger planning
        if self.orchestrator:
            await self.orchestrator.plan_goal(goal_id, title, description)

        await update.message.reply_text(
            f" *Goal #{goal_id} created*\n\n"
            f"{description}\n\n"
            f"_I'll create a plan and start working on it. "
            f"I'll update you on progress._",
            parse_mode="Markdown"
        )

    async def cmd_add_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text(
                "Usage: /task <description> [--model <litellm_name>]"
            )
            return
        raw_args = list(context.args)
        chat_id = update.message.chat_id
        parent_id = self.user_last_task_id.get(chat_id)

        # Phase 10.5: Parse --model flag
        model_override = None
        if "--model" in raw_args:
            idx = raw_args.index("--model")
            if idx + 1 < len(raw_args):
                model_override = raw_args[idx + 1]
                raw_args = raw_args[:idx] + raw_args[idx + 2:]
            else:
                raw_args = raw_args[:idx]

        description = " ".join(raw_args)

        task_id = await add_task(
            title=description[:50],
            description=description,
            tier="auto",
            parent_task_id=parent_id,
            priority=TASK_PRIORITY["critical"],
            context={"model_override": model_override} if model_override else None,
        )
        self.user_last_task_id[chat_id] = task_id
        pin_msg = f" (pinned: {model_override})" if model_override else ""
        await update.message.reply_text(f"✅ Task #{task_id} queued.{pin_msg}")


    async def cmd_list_goals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        goals = await get_active_goals()
        if not goals:
            await update.message.reply_text("No active goals. Use /goal to set one.")
            return
        msg = " *Active Goals:*\n\n"
        for g in goals:
            msg += f"#{g['id']} [P{g['priority']}] {g['title']}\n"
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def cmd_view_queue(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        tasks = await get_ready_tasks(limit=15)
        if not tasks:
            await update.message.reply_text(" No pending tasks. System is idle.")
            return
        msg = " *Task Queue:*\n\n"
        for t in tasks:
            agent = t.get('agent_type', '?')
            msg += f"#{t['id']} [{agent}|{t['tier']}] {t['title'][:50]}\n"
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = await get_daily_stats()
        await update.message.reply_text(
            f"⚙️ *System Status*\n\n"
            f"✅ Completed: {stats['completed']}\n"
            f"⏳ Pending: {stats['pending']}\n"
            f" Processing: {stats['processing']}\n"
            f"❌ Failed: {stats['failed']}\n"
            f" Cost today: ${stats['today_cost']:.4f}",
            parse_mode="Markdown"
        )

    async def cmd_digest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.orchestrator:
            await self.orchestrator.daily_digest()
        else:
            await update.message.reply_text("Orchestrator not connected.")

    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show ALL tasks with full status details."""
        db = await get_db()

        # All tasks
        cursor = await db.execute(
            """SELECT id, goal_id, parent_task_id, title, agent_type,
                      status, tier, depends_on, error, retry_count
               FROM tasks ORDER BY id"""
        )
        tasks = [dict(row) for row in await cursor.fetchall()]

        # All goals
        cursor2 = await db.execute("SELECT id, title, status FROM goals ORDER BY id")
        goals = [dict(row) for row in await cursor2.fetchall()]

        if not tasks and not goals:
            await update.message.reply_text("Database is empty.")
            return

        msg = "🔍 *FULL SYSTEM DEBUG*\n\n"

        if goals:
            msg += "*Goals:*\n"
            for g in goals:
                icon = "🎯" if g["status"] == "active" else "✅" if g["status"] == "completed" else "❌"
                msg += f"{icon} G#{g['id']} [{g['status']}] {g['title'][:40]}\n"
            msg += "\n"

        msg += "*All Tasks:*\n"
        status_icons = {
            "pending": "⏳",
            "processing": "🔄",
            "completed": "✅",
            "failed": "❌",
            "waiting_subtasks": "📋",
            "needs_clarification": "❓",
            "rejected": "🚫",
        }

        for t in tasks:
            icon = status_icons.get(t["status"], "❔")
            deps = t.get("depends_on", "[]")
            goal_tag = f" G#{t['goal_id']}" if t.get("goal_id") else ""
            parent_tag = f" ←#{t['parent_task_id']}" if t.get("parent_task_id") else ""
            dep_tag = f" deps:{deps}" if deps and deps != "[]" else ""
            err_tag = ""
            if t.get("error"):
                err_tag = f"\n    ⚠️ {t['error'][:60]}"
            retry_tag = f" r{t['retry_count']}" if t.get("retry_count", 0) > 0 else ""

            msg += (
                f"{icon} #{t['id']} [{t['status']}] [{t['agent_type']}|{t['tier']}]"
                f"{goal_tag}{parent_tag}{dep_tag}{retry_tag}\n"
                f"    {t['title'][:50]}{err_tag}\n"
            )

        # Split long messages (Telegram limit is 4096)
        if len(msg) > 4000:
            parts = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
            for part in parts:
                await update.message.reply_text(part, parse_mode="Markdown")
        else:
            await update.message.reply_text(msg, parse_mode="Markdown")


    async def cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Reset a specific stuck task back to pending."""
        if not context.args:
            await update.message.reply_text(
                "Usage:\n"
                "/reset <task_id> — Reset one task to pending\n"
                "/reset failed — Reset all failed tasks\n"
                "/reset stuck — Reset all processing tasks (stuck)"
            )
            return

        arg = context.args[0].lower()

        if arg == "failed":
            db = await get_db()
            cursor = await db.execute(
                """UPDATE tasks SET status = 'pending', retry_count = 0, error = NULL
                   WHERE status = 'failed'"""
            )
            count = cursor.rowcount
            await db.commit()
            await update.message.reply_text(f"♻️ Reset {count} failed task(s) to pending.")

        elif arg == "stuck":
            db = await get_db()
            cursor = await db.execute(
                """UPDATE tasks SET status = 'pending'
                   WHERE status = 'processing'"""
            )
            count = cursor.rowcount
            await db.commit()
            await update.message.reply_text(f"♻️ Reset {count} stuck task(s) to pending.")

        elif arg == "blocked":
            # Clear all dependency references so blocked tasks can run
            db = await get_db()
            cursor = await db.execute(
                """UPDATE tasks SET depends_on = '[]'
                   WHERE status = 'pending' AND depends_on != '[]'"""
            )
            count = cursor.rowcount
            await db.commit()
            await update.message.reply_text(
                f"♻️ Cleared dependencies on {count} blocked task(s). They'll run now."
            )

        else:
            try:
                task_id = int(arg)
                await update_task(task_id, status="pending", retry_count=0, error=None)
                await update.message.reply_text(f"♻️ Task #{task_id} reset to pending.")
            except ValueError:
                await update.message.reply_text("Invalid argument. Use a task ID, 'failed', 'stuck', or 'blocked'.")


    async def cmd_reset_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Nuclear option: wipe everything and start fresh."""
        keyboard = [[
            InlineKeyboardButton("☢️ Yes, wipe everything", callback_data="resetall_confirm"),
            InlineKeyboardButton("Cancel", callback_data="resetall_cancel"),
        ]]
        await update.message.reply_text(
            "⚠️ This will delete ALL goals, tasks, memory, and conversations.\n"
            "Are you sure?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # ─── Phase 3 Commands ──────────────────────────────────────────────

    async def cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cancel a task and its children."""
        if not context.args:
            await update.message.reply_text("Usage: /cancel <task_id>")
            return
        try:
            task_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Task ID must be a number.")
            return

        success = await cancel_task(task_id)
        if success:
            await update.message.reply_text(
                f"🚫 Task #{task_id} and its children cancelled."
            )
        else:
            await update.message.reply_text(
                f"Task #{task_id} not found or already finished."
            )

    async def cmd_priority(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Change task priority."""
        if len(context.args) < 2:
            await update.message.reply_text(
                "Usage: /priority <task_id> <1-10>"
            )
            return
        try:
            task_id = int(context.args[0])
            level = int(context.args[1])
            if not 1 <= level <= 10:
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Task ID and priority (1-10) must be numbers."
            )
            return

        success = await reprioritize_task(task_id, level)
        if success:
            await update.message.reply_text(
                f"✅ Task #{task_id} priority set to {level}."
            )
        else:
            await update.message.reply_text(
                f"Task #{task_id} not found or not pending/processing."
            )

    async def cmd_graph(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show text DAG of task dependencies for a goal."""
        if not context.args:
            await update.message.reply_text("Usage: /graph <goal_id>")
            return
        try:
            goal_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Goal ID must be a number.")
            return

        tasks = await get_task_tree(goal_id)
        if not tasks:
            await update.message.reply_text(
                f"No tasks found for goal #{goal_id}."
            )
            return

        # Build text DAG
        lines = [f"📊 *Task Graph — Goal #{goal_id}*\n"]
        status_icons = {
            "pending": "⏳", "processing": "⚙️",
            "completed": "✅", "failed": "❌",
            "cancelled": "🚫", "waiting_subtasks": "🔄",
            "needs_clarification": "❓", "needs_review": "👀",
        }

        # Build parent→children map
        by_parent: dict[int | None, list] = {}
        for t in tasks:
            pid = t.get("parent_task_id")
            by_parent.setdefault(pid, []).append(t)

        def _render(parent_id, indent=0):
            children = by_parent.get(parent_id, [])
            for t in children:
                icon = status_icons.get(t["status"], "❔")
                prefix = "  " * indent + ("├─ " if indent > 0 else "")
                agent = t.get("agent_type", "?")
                lines.append(
                    f"{prefix}{icon} #{t['id']} `{agent}` "
                    f"{t['title'][:40]}"
                )
                _render(t["id"], indent + 1)

        # Render root tasks (no parent) then their children
        _render(None)

        await update.message.reply_text(
            "\n".join(lines), parse_mode="Markdown"
        )

    async def cmd_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View or set daily cost budget."""
        if context.args:
            try:
                new_limit = float(context.args[0])
                await set_budget("daily", daily_limit=new_limit)
                await update.message.reply_text(
                    f"💰 Daily budget set to ${new_limit:.2f}"
                )
            except ValueError:
                await update.message.reply_text(
                    "Usage: /budget [daily_limit]\n"
                    "Example: /budget 1.50"
                )
            return

        budget = await get_budget("daily")
        if not budget:
            await update.message.reply_text(
                "💰 No daily budget set.\n"
                "Use /budget <amount> to set one.\n"
                "Example: /budget 1.00"
            )
            return

        today = budget.get("last_reset_date", "N/A")
        await update.message.reply_text(
            f"💰 *Cost Budget*\n\n"
            f"Daily limit: ${budget['daily_limit']:.4f}\n"
            f"Spent today: ${budget['spent_today']:.4f}\n"
            f"Spent total: ${budget['spent_total']:.4f}\n"
            f"Last reset: {today}",
            parse_mode="Markdown"
        )

    async def cmd_model_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show model performance statistics."""
        stats = await get_model_stats()
        if not stats:
            await update.message.reply_text("📊 No model stats yet.")
            return

        lines = ["📊 *Model Performance Stats*\n"]
        for s in stats[:15]:  # limit output
            model = s["model"].split("/")[-1][:20]
            lines.append(
                f"`{model}` ({s['agent_type']})\n"
                f"  Grade: {s['avg_grade']:.1f}/5 | "
                f"SR: {s['success_rate']*100:.0f}% | "
                f"Calls: {s['total_calls']} | "
                f"Cost: ${s['avg_cost']:.4f}"
            )

        await update.message.reply_text(
            "\n".join(lines), parse_mode="Markdown"
        )

    async def cmd_workspace(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show active goal workspaces."""
        workspaces = list_goal_workspaces()
        if not workspaces:
            await update.message.reply_text("📁 No goal workspaces active.")
            return

        lines = ["📁 *Goal Workspaces*\n"]
        for ws in workspaces:
            locks = await get_goal_locks(ws["goal_id"])
            lock_str = f" ({len(locks)} locks)" if locks else ""
            lines.append(
                f"  Goal #{ws['goal_id']}: "
                f"{ws['file_count']} files{lock_str}"
            )

        await update.message.reply_text(
            "\n".join(lines), parse_mode="Markdown"
        )

    async def cmd_project(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List or switch projects. /project [name]"""
        args = context.args
        projects = load_projects_config()

        if not args:
            if not projects:
                await update.message.reply_text(
                    "📂 No projects configured. "
                    "Add projects to `projects.json`."
                )
                return
            lines = ["📂 *Projects*\n"]
            for p in projects:
                lang = p.get("language", "?")
                lines.append(f"  `{p['name']}` — {lang} ({p.get('path', '?')})")
            await update.message.reply_text(
                "\n".join(lines), parse_mode="Markdown"
            )
            return

        name = args[0]
        project = get_project(name)
        if not project:
            await update.message.reply_text(f"❌ Project '{name}' not found.")
            return

        await update.message.reply_text(
            f"📂 *Project: {project['name']}*\n"
            f"Path: `{project.get('path', 'N/A')}`\n"
            f"Language: {project.get('language', 'N/A')}\n"
            f"Conventions: {project.get('conventions', 'N/A')}",
            parse_mode="Markdown",
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Smart handler: detect if it's a goal or simple task."""
        text = update.message.text
        chat_id = update.message.chat_id
        parent_id = self.user_last_task_id.get(chat_id)

        if len(text) > 200 or any(kw in text.lower() for kw in
            ["research", "create a", "build", "analyze", "develop",
             "write a report", "compare", "plan", "strategy"]):
            goal_id = await add_goal(title=text[:80], description=text, priority=5)
            if self.orchestrator:
                await self.orchestrator.plan_goal(goal_id, text[:80], text)
            await update.message.reply_text(
                f" Interpreted as Goal #{goal_id}. Planning now..."
            )
            self.user_last_task_id.pop(chat_id, None)
        else:
            task_id = await add_task(
                title=text[:50],
                description=text,
                tier="auto",
                parent_task_id=parent_id,
                priority=TASK_PRIORITY["critical"],
            )
            self.user_last_task_id[chat_id] = task_id
            await update.message.reply_text(f"✅ Task #{task_id} queued.")

    async def handle_reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle replies to clarification requests."""
        replied_to = update.message.reply_to_message
        if not replied_to:
            return

        # Check if this is a reply to a clarification
        text = replied_to.text or ""
        if "Clarification needed" in text:
            # Extract task ID from the message
            import re
            match = re.search(r"Task #(\d+)", text)
            if match:
                task_id = int(match.group(1))
                answer = update.message.text

                # Resume task with clarification
                await add_task(
                    title=f"Continue #{task_id} with clarification",
                    description=(
                        f"Continue the previous task with this clarification "
                        f"from the human:\n\n{answer}"
                    ),
                    goal_id=None,
                    parent_task_id=task_id,
                    tier="auto"
                )
                await update.message.reply_text(
                    f"↩️ Got it. Resuming task #{task_id} with your input."
                )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data

        if data == "resetall_confirm":
            db = await get_db()
            await db.execute("DELETE FROM conversations")
            await db.execute("DELETE FROM tasks")
            await db.execute("DELETE FROM goals")
            await db.execute("DELETE FROM memory")
            await db.commit()
            await query.edit_message_text("☢️ Everything wiped. Fresh start.")
            return
        elif data == "resetall_cancel":
            await query.edit_message_text("Cancelled.")
            return

        action, task_id_str = data.split("_", 1)
        task_id = int(task_id_str)

        if action == "approve":
            if task_id in self._approval_events:
                self._approval_events[task_id]["result"] = "approved"
                self._approval_events[task_id]["event"].set()
            await query.edit_message_text(f"✅ Task #{task_id} approved.")
        elif action == "reject":
            if task_id in self._approval_events:
                self._approval_events[task_id]["result"] = "rejected"
                self._approval_events[task_id]["event"].set()
            await update_task(task_id, status="rejected")
            await query.edit_message_text(f"❌ Task #{task_id} rejected.")

    # --- Outbound notifications ---

    async def send_notification(self, text: str):
        try:
            await self.app.bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=text,
                parse_mode="Markdown"
            )
        except Exception as e:
            # Retry without markdown if formatting fails
            try:
                await self.app.bot.send_message(
                    chat_id=TELEGRAM_ADMIN_CHAT_ID, text=text
                )
            except Exception:
                logging.error(f"Failed to send Telegram notification: {e}")

    async def send_result(self, task_id, title, result, model, cost):
        truncated = result[:3000] if len(result) > 3000 else result
        await self.send_notification(
            f"✅ *Task #{task_id} Complete*\n"
            f"**{title}**\n"
            f"Model: `{model}` | Cost: ${cost:.4f}\n\n"
            f"{truncated}"
        )

    async def send_error(self, task_id, title, error):
        await self.send_notification(
            f"❌ *Task #{task_id} Failed* (after retries)\n"
            f"**{title}**\n\nError: {error[:500]}"
        )

    async def request_clarification(self, task_id, title, question):
        await self.send_notification(
            f"❓ *Clarification needed — Task #{task_id}*\n"
            f"**{title}**\n\n"
            f"{question}\n\n"
            f"_Reply to this message with your answer._"
        )

    async def request_approval(self, task_id, title, plan, tier):
        keyboard = [[
            InlineKeyboardButton("✅ Approve", callback_data=f"approve_{task_id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"reject_{task_id}"),
        ]]
        event = asyncio.Event()
        self._approval_events[task_id] = {"event": event, "result": None}

        await self.app.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=f" *Approval Required — Task #{task_id}*\n\n"
                 f"**{title}**\nTier: {tier}\n\n{plan[:500]}",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

        try:
            await asyncio.wait_for(event.wait(), timeout=1800)
            return self._approval_events[task_id]["result"] == "approved"
        except asyncio.TimeoutError:
            await self.send_notification(f"⏰ Approval for #{task_id} timed out. Skipped.")
            return False
        finally:
            self._approval_events.pop(task_id, None)
