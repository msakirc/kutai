# telegram_bot.py
import asyncio
import aiosqlite
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID, DB_PATH
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from db import (add_task, add_goal, get_active_goals, get_ready_tasks,
                get_daily_stats, update_task)

pending_clarifications = {}  # task_id -> asyncio.Event + response


class TelegramInterface:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self._setup_handlers()
        self._approval_events = {}
        self._clarification_events = {}


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
            "/goal <description> — Set a high-level goal (I'll plan & execute)\n"
            "/task <description> — Add a one-off task\n"
            "/goals — View active goals\n"
            "/queue — View pending tasks\n"
            "/status — System stats\n"
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
            await update.message.reply_text("Usage: /task <description>")
            return
        description = " ".join(context.args)
        task_id = await add_task(
            title=description[:50], description=description, tier="auto"
        )
        await update.message.reply_text(f"✅ Task #{task_id} queued.")

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
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row

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
            async with aiosqlite.connect(DB_PATH) as db:
                cursor = await db.execute(
                    """UPDATE tasks SET status = 'pending', retry_count = 0, error = NULL
                       WHERE status = 'failed'"""
                )
                count = cursor.rowcount
                await db.commit()
            await update.message.reply_text(f"♻️ Reset {count} failed task(s) to pending.")

        elif arg == "stuck":
            async with aiosqlite.connect(DB_PATH) as db:
                cursor = await db.execute(
                    """UPDATE tasks SET status = 'pending'
                       WHERE status = 'processing'"""
                )
                count = cursor.rowcount
                await db.commit()
            await update.message.reply_text(f"♻️ Reset {count} stuck task(s) to pending.")

        elif arg == "blocked":
            # Clear all dependency references so blocked tasks can run
            async with aiosqlite.connect(DB_PATH) as db:
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

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Smart handler: detect if it's a goal or simple task."""
        text = update.message.text

        # If it's long or sounds like a project, treat as goal
        if len(text) > 200 or any(kw in text.lower() for kw in 
            ["research", "create a", "build", "analyze", "develop",
             "write a report", "compare", "plan", "strategy"]):
            goal_id = await add_goal(title=text[:80], description=text, priority=5)
            if self.orchestrator:
                await self.orchestrator.plan_goal(goal_id, text[:80], text)
            await update.message.reply_text(
                f" Interpreted as Goal #{goal_id}. Planning now..."
            )
        else:
            task_id = await add_task(title=text[:50], description=text, tier="auto")
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
            async with aiosqlite.connect(DB_PATH) as db:
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
