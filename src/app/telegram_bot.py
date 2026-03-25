# telegram_bot.py
import asyncio
from datetime import datetime
from pathlib import Path
from src.infra.logging_config import get_logger
from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID, TASK_PRIORITY

logger = get_logger("app.telegram_bot")
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

from ..infra.db import (add_task, add_mission, get_active_missions,
                get_ready_tasks, get_daily_stats, update_task, get_recent_completed_tasks,
                get_db, cancel_task, reprioritize_task, get_task_tree,
                get_task, get_mission, get_budget, set_budget, get_model_stats,
                get_mission_locks, get_tasks_for_mission,
                insert_approval_request, update_approval_status,
                add_todo, get_todos, get_todo, toggle_todo, delete_todo)
from ..memory.conversations import format_recent_context, find_followup_context, \
    store_exchange
from ..memory.ingest import ingest_document
from ..memory.preferences import record_feedback
from ..tools.workspace import list_mission_workspaces


pending_clarifications = {}  # task_id -> asyncio.Event + response

def _friendly_error(error: str) -> str:
    """Convert raw error strings into user-friendly messages."""
    e = error.lower()
    if "import" in e or "module" in e or "attribute" in e:
        return "Internal configuration error. Check logs for details."
    if "rate limit" in e or "429" in e:
        return "Rate limited by the AI provider. Will retry automatically."
    if "timeout" in e:
        return "The task timed out. Try again or simplify the request."
    if "no models available" in e or "no models matched" in e:
        return "No suitable AI model available right now. Try again later."
    if "connection" in e or "network" in e:
        return "Network error. Check connectivity and try again."
    if any(kw in e for kw in ["typeerror", "keyerror", "valueerror",
                               "nameerror", "indexerror", "traceback"]):
        return "Internal error. Check logs for details."
    if "json" in e or "parsing" in e or "decode" in e:
        return "Failed to process the response. Will retry."
    # Generic — show first 100 chars, strip tracebacks and Python internals
    first_line = error.split("\n")[0][:100]
    if any(kw in first_line.lower() for kw in ["object has no", "expected", "got an"]):
        return "Internal error. Check logs for details."
    return f"Something went wrong. Check logs for details."

# ─── Product/App Detection ────────────────────────────────────────────────
_PRODUCT_KEYWORDS = [
    "build ", "create ", "make ", "develop ", "design ",
    " app ", " app.", " application", " platform", " website", " webapp",
    " web app", " tool ", " saas", " service ", " system ",
    " bot ", " dashboard", " portal", " marketplace",
    " that allows", " that lets", " that enables", " where users",
    " for users to", " for people to",
]

def _looks_like_product_idea(description: str) -> bool:
    """Detect if a mission description is about building a product/app."""
    desc_lower = f" {description.lower()} "
    # Need at least one "build" verb AND one "product" noun
    has_verb = any(kw in desc_lower for kw in _PRODUCT_KEYWORDS[:5])
    has_noun = any(kw in desc_lower for kw in _PRODUCT_KEYWORDS[5:])
    # Or contains strong product phrases
    has_phrase = any(kw in desc_lower for kw in _PRODUCT_KEYWORDS[11:])
    return (has_verb and has_noun) or has_phrase

# ─── Interactive Menu Definitions ─────────────────────────────────────────
# Each category: (emoji_label, key, [(button_label, command, needs_arg, arg_prompt)])
MENU_CATEGORIES = [
    ("🎯 Missions", "missions", [
        ("🎯 New Mission", "mission", True, "Describe your mission:"),
        ("🔄 Workflow Mission", "mission_wf", True, "Describe the product/workflow idea:"),
        ("📋 List Missions", "missions", False, None),
        ("📬 Queue", "queue", False, None),
        ("🚫 Cancel", "cancel", True, "Which mission or task ID to cancel?"),
        ("🔢 Priority", "priority", True, "Enter: <id> <priority 1-10>"),
        ("⏸️ Pause", "pause", True, "Which mission ID to pause?"),
        ("▶️ Resume", "resume", True, "Which mission ID to resume?"),
        ("🌳 Graph", "graph", True, "Which mission ID to graph?"),
    ]),
    ("📊 Monitoring", "monitoring", [
        ("📊 Status", "status", False, None),
        ("📰 Digest", "digest", False, None),
        ("📈 Progress", "progress", True, "Which mission ID? (empty for all)"),
        ("📉 Metrics", "metrics", False, None),
        ("🔍 Audit", "audit", True, "Which task ID to audit?"),
        ("🔁 Replay", "replay", True, "Which task ID to replay?"),
        ("💰 Cost", "cost", True, "Which mission ID for cost breakdown?"),
        ("🤖 Model Stats", "modelstats", False, None),
        ("💵 Budget", "budget", True, "Enter daily budget limit (empty to view):"),
        ("🗂️ Workspaces", "workspace", False, None),
        ("📋 WF Status", "wfstatus", True, "Which mission ID for workflow status?"),
    ]),
    ("📝 Personal", "personal", [
        ("📝 Add Todo", "todo", True, "What do you need to remember?"),
        ("📋 My Todos", "todos", False, None),
        ("🗑️ Clear Done", "cleartodos", False, None),
    ]),
    ("🧠 Knowledge", "knowledge", [
        ("💾 Remember", "remember", True, "What should I remember?"),
        ("🔎 Recall", "recall", True, "What do you want to recall?"),
        ("📥 Ingest", "ingest", True, "Send a URL or file path to ingest:"),
        ("⭐ Feedback", "feedback", True, "Enter: <task_id> <good|bad|partial> [reason]"),
    ]),
    ("⚙️ System", "system", [
        ("🎚️ Autonomy", "autonomy", True, "Set level: low, medium, high, or paranoid"),
        ("🔑 Credential", "credential", True, "Credential key=value (or key to view):"),
        ("🎛️ Tune", "tune", True, "What setting to tune?"),
        ("🖥️ Load", "load", True, "Set load mode (minimal/normal/auto):"),
        ("🧪 Improve", "improve", False, None),
        ("📭 DLQ", "dlq", False, None),
    ]),
    ("🔴 Danger Zone", "danger", [
        ("🐛 Debug", "debug", False, None),
        ("♻️ Reset", "reset", True, "Reset what? (task ID, 'failed', 'stuck', 'blocked')"),
        ("☢️ Reset All", "resetall", False, None),
    ]),
]


def _build_category_keyboard() -> InlineKeyboardMarkup:
    """Build the top-level category selection keyboard."""
    rows = []
    for i in range(0, len(MENU_CATEGORIES), 2):
        row = []
        for cat_label, cat_key, _ in MENU_CATEGORIES[i:i+2]:
            row.append(InlineKeyboardButton(cat_label, callback_data=f"menu_cat:{cat_key}"))
        rows.append(row)
    return InlineKeyboardMarkup(rows)


def _build_command_keyboard(cat_key: str) -> InlineKeyboardMarkup:
    """Build the command buttons for a specific category."""
    for _, key, commands in MENU_CATEGORIES:
        if key == cat_key:
            rows = []
            row = []
            for btn_label, cmd, needs_arg, prompt in commands:
                cb = f"menu_ask:{cmd}" if needs_arg else f"menu_cmd:{cmd}"
                row.append(InlineKeyboardButton(btn_label, callback_data=cb))
                if len(row) >= 3:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu_back")])
            return InlineKeyboardMarkup(rows)
    return _build_category_keyboard()  # fallback


# Map command -> prompt for conversation flow
_CMD_ARG_PROMPTS: dict[str, str] = {}
for _, _, cmds in MENU_CATEGORIES:
    for _, cmd, needs_arg, prompt in cmds:
        if needs_arg and prompt:
            _CMD_ARG_PROMPTS[cmd] = prompt

# Map command string -> actual method name (where they differ)
_CMD_METHOD_MAP: dict[str, str] = {
    "mission": "cmd_mission",
    "mission_wf": "cmd_mission_workflow",
    "missions": "cmd_missions",
    "task": "cmd_add_task",
    "queue": "cmd_view_queue",
    "modelstats": "cmd_model_stats",
    "resetall": "cmd_reset_all",
    "cleartodos": "cmd_cleartodos",
}


class TelegramInterface:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self._setup_handlers()
        self._approval_events = {}
        self._clarification_events = {}
        self.user_last_task_id = {}
        # Explicit clarification tracking: chat_id → task_id
        self._pending_clarifications: dict[int, int] = {}
        # Conversation flow: chat_id → {"command": str} for button-initiated arg prompts
        self._pending_action: dict[int, dict] = {}

    def _resolve_cmd_handler(self, cmd: str):
        """Resolve a command string to its handler method."""
        method_name = _CMD_METHOD_MAP.get(cmd, f"cmd_{cmd}")
        return getattr(self, method_name, None)


    def _setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        # Mission commands (new unified interface)
        self.app.add_handler(CommandHandler("mission", self.cmd_mission))
        self.app.add_handler(CommandHandler("mish", self.cmd_mission))      # abbreviation
        self.app.add_handler(CommandHandler("missions", self.cmd_missions))
        self.app.add_handler(CommandHandler("task", self.cmd_add_task))
        self.app.add_handler(CommandHandler("queue", self.cmd_view_queue))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("digest", self.cmd_digest))
        self.app.add_handler(CommandHandler("debug", self.cmd_debug))
        self.app.add_handler(CommandHandler("reset", self.cmd_reset))
        self.app.add_handler(CommandHandler("resetall", self.cmd_reset_all))
        self.app.add_handler(CommandHandler("cancel", self.cmd_cancel))
        self.app.add_handler(CommandHandler("priority", self.cmd_priority))
        self.app.add_handler(CommandHandler("graph", self.cmd_graph))
        self.app.add_handler(CommandHandler("budget", self.cmd_budget))
        self.app.add_handler(CommandHandler("modelstats", self.cmd_model_stats))
        self.app.add_handler(CommandHandler("workspace", self.cmd_workspace))
        self.app.add_handler(CommandHandler("progress", self.cmd_progress))
        self.app.add_handler(CommandHandler("audit", self.cmd_audit))
        self.app.add_handler(CommandHandler("metrics", self.cmd_metrics))
        self.app.add_handler(CommandHandler("replay", self.cmd_replay))
        self.app.add_handler(CommandHandler("ingest", self.cmd_ingest))
        self.app.add_handler(CommandHandler("wfstatus", self.cmd_wfstatus))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("credential", self.cmd_credential))
        self.app.add_handler(CommandHandler("cost", self.cmd_cost))
        self.app.add_handler(CommandHandler("dlq", self.cmd_dlq))
        self.app.add_handler(CommandHandler("load", self.cmd_load))
        self.app.add_handler(CommandHandler("tune", self.cmd_tune))
        self.app.add_handler(CommandHandler("feedback", self.cmd_feedback))
        self.app.add_handler(CommandHandler("improve", self.cmd_improve))
        self.app.add_handler(CommandHandler("remember", self.cmd_remember))
        self.app.add_handler(CommandHandler("recall", self.cmd_recall))
        self.app.add_handler(CommandHandler("autonomy", self.cmd_autonomy))
        self.app.add_handler(CommandHandler("todo", self.cmd_todo))
        self.app.add_handler(CommandHandler("todos", self.cmd_todos))
        self.app.add_handler(CommandHandler("cleartodos", self.cmd_cleartodos))
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
            "🤖 *Autonomous AI Orchestrator*\n\n"
            "I work 24/7 and only bug you when needed.\n\n"
            "Tap a category below, or just send a message — "
            "I understand missions, tasks, bug reports, feature requests, "
            "status questions, feedback, and more.",
            parse_mode="Markdown",
            reply_markup=_build_category_keyboard()
        )

    # ─── Mission Commands ──────────────────────────────────────────────────────

    async def cmd_mission(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Create a new mission. /mission <description> or /mish <description>"""
        if not context.args:
            await update.message.reply_text(
                "Usage: /mission <description>\n"
                "       /mission --workflow <description>\n"
                "       /mish <description> (shorthand)\n\n"
                "Examples:\n"
                "  /mission Fix the login page bug\n"
                "  /mission --workflow Build an inventory management app"
            )
            return

        text_args = list(context.args)
        workflow = None

        if "--workflow" in text_args:
            text_args.remove("--workflow")
            workflow = "idea_to_product_v2"

        description = " ".join(text_args)
        if not description:
            await update.message.reply_text("Please provide a mission description.")
            return

        # Let the classifier decide if this needs a workflow
        if not workflow:
            classification = await self._classify_user_message(description)
            if classification.get("workflow") == "idea_to_product":
                workflow = "idea_to_product_v2"

        chat_id = update.message.chat_id

        if workflow:
            # Workflow mission — delegate to workflow runner
            try:
                from ..workflows.engine.runner import WorkflowRunner
                runner = WorkflowRunner(self.orchestrator)
                mission_id = await runner.start(
                    workflow_name=workflow,
                    initial_input={"idea": description, "product_name": description[:50]},
                    title=description[:80],
                )
                await update.message.reply_text(
                    f"🔄 Workflow mission #{mission_id} created!\n"
                    f"_{description[:60]}_\n\n"
                    f"Use /wfstatus {mission_id} to track progress.",
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.error("workflow mission failed", error=str(e))
                await update.message.reply_text(f"❌ {_friendly_error(str(e))}")
            return

        # Regular mission — create and plan
        mission_id = await add_mission(
            title=description[:80],
            description=description,
            priority=7,
        )

        if self.orchestrator:
            await self.orchestrator.plan_mission(mission_id, description[:80], description)

        await update.message.reply_text(
            f"🎯 Mission #{mission_id} created. Planning now...\n"
            f"_{description[:60]}_",
            parse_mode="Markdown",
        )
        self.user_last_task_id[chat_id] = None

    async def cmd_mission_workflow(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Create a workflow mission from menu button."""
        if not context.args:
            await update.message.reply_text("Describe your product/workflow idea:")
            return
        # Inject --workflow and delegate
        context.args = ["--workflow"] + list(context.args)
        await self.cmd_mission(update, context)

    async def cmd_missions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List active missions."""
        missions = await get_active_missions()
        if not missions:
            await update.message.reply_text("No active missions.")
            return

        lines = ["📋 *Active Missions:*\n"]
        for m in missions:
            mid = m["id"]
            title = m.get("title", "Untitled")[:50]
            workflow = m.get("workflow", "")
            # Count tasks
            try:
                tasks = await get_tasks_for_mission(mid)
                total = len(tasks)
                done = sum(1 for t in tasks if t.get("status") in ("completed", "skipped"))
                task_info = f" ({done}/{total} tasks)"
            except Exception:
                task_info = ""

            badge = "🔄" if workflow else "🎯"
            wf_tag = f" \\[{workflow}]" if workflow else ""
            lines.append(f"  {badge} #{mid} {title}{wf_tag}{task_info}")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

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
            """SELECT id, mission_id, parent_task_id, title, agent_type,
                      status, tier, depends_on, error, retry_count
               FROM tasks ORDER BY id"""
        )
        tasks = [dict(row) for row in await cursor.fetchall()]

        # All missions
        cursor2 = await db.execute("SELECT id, title, status FROM missions ORDER BY id")
        missions = [dict(row) for row in await cursor2.fetchall()]

        if not tasks and not missions:
            await update.message.reply_text("Database is empty.")
            return

        msg = "🔍 *FULL SYSTEM DEBUG*\n\n"

        if missions:
            msg += "*Missions:*\n"
            for g in missions:
                icon = "🎯" if g["status"] == "active" else "✅" if g["status"] == "completed" else "❌"
                msg += f"{icon} M#{g['id']} [{g['status']}] {g['title'][:40]}\n"
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
            mission_tag = f" M#{t['mission_id']}" if t.get("mission_id") else ""
            parent_tag = f" ←#{t['parent_task_id']}" if t.get("parent_task_id") else ""
            dep_tag = f" deps:{deps}" if deps and deps != "[]" else ""
            err_tag = ""
            if t.get("error"):
                err_tag = f"\n    ⚠️ {t['error'][:60]}"
            retry_tag = f" r{t['retry_count']}" if t.get("retry_count", 0) > 0 else ""

            msg += (
                f"{icon} #{t['id']} [{t['status']}] [{t['agent_type']}|{t['tier']}]"
                f"{mission_tag}{parent_tag}{dep_tag}{retry_tag}\n"
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
            "⚠️ This will delete ALL missions, tasks, memory, and conversations.\n"
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
        """Show text DAG of task dependencies for a mission."""
        if not context.args:
            await update.message.reply_text("Usage: /graph <mission_id>")
            return
        try:
            mission_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Mission ID must be a number.")
            return

        tasks = await get_task_tree(mission_id)
        if not tasks:
            await update.message.reply_text(
                f"No tasks found for mission #{mission_id}."
            )
            return

        # Build text DAG
        lines = [f"📊 *Task Graph — Mission #{mission_id}*\n"]
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

    async def cmd_cost(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show per-mission cost breakdown."""
        if not context.args:
            await update.message.reply_text("Usage: /cost <mission\\_id>",
                                            parse_mode="Markdown")
            return
        try:
            mission_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Mission ID must be a number.")
            return

        try:
            from ..collaboration.blackboard import read_blackboard
            cost_data = await read_blackboard(mission_id, "cost_tracking")
            if not isinstance(cost_data, dict):
                cost_data = {}
        except Exception:
            cost_data = {}

        if not cost_data or cost_data.get("total_cost", 0) == 0:
            await update.message.reply_text(f"No cost data for mission #{mission_id}")
            return

        import json as _json
        total = cost_data.get("total_cost", 0)
        count = cost_data.get("task_count", 0)
        by_phase = cost_data.get("by_phase", {})

        lines = [
            f"*Cost Report — Mission #{mission_id}*",
            f"Total: ${total:.4f} ({count} tasks)",
        ]
        if by_phase:
            lines.append("\nBy phase:")
            for phase, pcost in sorted(by_phase.items()):
                lines.append(f"  {phase}: ${pcost:.4f}")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

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
        """Show active mission workspaces."""
        workspaces = list_mission_workspaces()
        if not workspaces:
            await update.message.reply_text("📁 No mission workspaces active.")
            return

        lines = ["📁 *Mission Workspaces*\n"]
        for ws in workspaces:
            locks = await get_mission_locks(ws["mission_id"])
            lock_str = f" ({len(locks)} locks)" if locks else ""
            lines.append(
                f"  Mission #{ws['mission_id']}: "
                f"{ws['file_count']} files{lock_str}"
            )

        await update.message.reply_text(
            "\n".join(lines), parse_mode="Markdown"
        )

    async def cmd_progress(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show progress timeline for a mission or project. /progress [mission_id]"""
        args = context.args
        try:
            from src.infra.progress import get_notes, format_notes_timeline
            mission_id = int(args[0]) if args else None
            notes = await get_notes(mission_id=mission_id, limit=20)
            timeline = format_notes_timeline(notes)
            header = f"📊 *Progress Notes* (mission #{mission_id})" if mission_id else "📊 *Recent Progress Notes*"
            msg = f"{header}\n\n{timeline}"
            if len(msg) > 4000:
                msg = msg[:4000] + "\n... (truncated)"
            await update.message.reply_text(msg, parse_mode="Markdown")
        except (ValueError, IndexError):
            await update.message.reply_text("Usage: /progress [mission_id]")
        except Exception as e:
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def cmd_audit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show audit log for a task. /audit [task_id]"""
        args = context.args
        try:
            from src.infra.audit import get_audit_log, format_audit_log
            task_id = int(args[0]) if args else None
            entries = await get_audit_log(task_id=task_id, limit=30)
            log_text = format_audit_log(entries)
            header = f"🔍 *Audit Log* (task #{task_id})" if task_id else "🔍 *Recent Audit Log*"
            msg = f"{header}\n\n{log_text}"
            if len(msg) > 4000:
                msg = msg[:4000] + "\n... (truncated)"
            await update.message.reply_text(msg, parse_mode="Markdown")
        except (ValueError, IndexError):
            await update.message.reply_text("Usage: /audit [task_id]")
        except Exception as e:
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def cmd_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show system metrics summary. /metrics"""
        try:
            from src.infra.metrics import format_metrics_summary
            msg = format_metrics_summary()
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def cmd_replay(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Replay task execution trace. /replay <task_id>"""
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /replay <task_id>")
            return
        try:
            from src.infra.tracing import get_trace, format_trace
            task_id = int(args[0])
            trace = await get_trace(task_id)
            trace_text = format_trace(trace)
            msg = f"🔄 *Trace for Task #{task_id}*\n\n{trace_text}"
            if len(msg) > 4000:
                msg = msg[:4000] + "\n... (truncated)"
            await update.message.reply_text(msg, parse_mode="Markdown")
        except (ValueError, IndexError):
            await update.message.reply_text("Usage: /replay <task_id>")
        except Exception as e:
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def cmd_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Rate a completed task. /feedback <task_id> <good|bad|partial> [reason]"""
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("Usage: /feedback <task_id> <good|bad|partial> [reason]")
            return
        try:
            from src.memory.feedback import record_feedback
            task_id = int(args[0])
            rating = args[1].lower()
            if rating not in ("good", "bad", "partial"):
                await update.message.reply_text("Rating must be: good, bad, or partial")
                return
            reason = " ".join(args[2:]) if len(args) > 2 else ""
            # Enrich with model/agent info from the task record
            task_info = await get_task(task_id)
            model_used = ""
            agent_type = ""
            if task_info:
                model_used = task_info.get("model", "")
                agent_type = task_info.get("agent_type", "")
            await record_feedback(
                task_id=task_id, feedback_type=rating, reason=reason,
                model_used=model_used, agent_type=agent_type,
            )
            emoji = {"good": "👍", "bad": "👎", "partial": "🤷"}[rating]
            msg = f"{emoji} Feedback recorded for task #{task_id}: *{rating}*"
            if reason:
                msg += f"\nReason: _{reason}_"
            await update.message.reply_text(msg, parse_mode="Markdown")
        except (ValueError, IndexError):
            await update.message.reply_text("Usage: /feedback <task_id> <good|bad|partial> [reason]")
        except Exception as e:
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def cmd_autonomy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set or view risk autonomy threshold. /autonomy [low|medium|high|paranoid]"""
        args = context.args
        try:
            from src.security.risk_assessor import set_autonomy_threshold, get_autonomy_threshold
            levels = {
                "paranoid": 2,
                "low": 4,
                "medium": 6,
                "high": 8,
            }
            if not args:
                current = get_autonomy_threshold()
                level_name = next((k for k, v in levels.items() if v == current), f"custom ({current})")
                await update.message.reply_text(
                    f"🛡️ *Autonomy Level*\n\nCurrent threshold: *{level_name}* (score ≥{current} requires approval)",
                    parse_mode="Markdown",
                )
                return
            level = args[0].lower()
            if level not in levels:
                await update.message.reply_text(f"Unknown level. Choose: {', '.join(levels.keys())}")
                return
            threshold = levels[level]
            set_autonomy_threshold(threshold)
            desc = {
                "paranoid": "require approval for almost everything",
                "low": "require approval for medium+ risk tasks",
                "medium": "require approval for high-risk tasks only",
                "high": "require approval only for very dangerous tasks",
            }[level]
            await update.message.reply_text(
                f"🛡️ Autonomy set to *{level}* — will {desc}.",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def cmd_ingest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Ingest a URL or file into the knowledge base."""
        if not context.args:
            await update.message.reply_text(
                "Usage: /ingest <url\\_or\\_filepath>\n\n"
                "Examples:\n"
                "/ingest https://docs.example.com/api\n"
                "/ingest /path/to/document.pdf",
                parse_mode="Markdown",
            )
            return

        source = " ".join(context.args)
        await update.message.reply_text(f"📥 Ingesting: {source}...")

        try:
            result = await ingest_document(source)

            if result["status"] == "ok":
                await update.message.reply_text(
                    f"✅ Ingested *{result['chunks']}* chunks from "
                    f"`{result['source']}`\n\n"
                    f"Knowledge is now available to all agents.",
                    parse_mode="Markdown",
                )
            else:
                await update.message.reply_text(
                    f"❌ Ingestion failed: {result.get('error', 'unknown error')}"
                )
        except ImportError:
            await update.message.reply_text(
                "❌ Memory system not available. "
                "Install chromadb: pip install chromadb"
            )
        except Exception as e:
            await update.message.reply_text(
                f"❌ {_friendly_error(str(e))}"
            )

    # ─── Credential Management ────────────────────────────────────────

    async def cmd_credential(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manage stored credentials for external services."""
        if not context.args:
            await update.message.reply_text(
                "Usage:\n"
                "/credential list — Show stored services\n"
                "/credential add <service> <json\\_data> — Store credential\n"
                "/credential remove <service> — Delete credential\n\n"
                "Example:\n"
                '/credential add github \\{"token": "ghp\\_xxx"\\}',
                parse_mode="Markdown",
            )
            return

        sub = context.args[0].lower()

        if sub == "list":
            try:
                from ..security.credential_store import list_credentials

                services = await list_credentials()
                if not services:
                    await update.message.reply_text(
                        "No credentials stored. Use /credential add to add one."
                    )
                else:
                    lines = ["*Stored Credentials:*\n"]
                    for svc in services:
                        lines.append(f"  `{svc}`")
                    await update.message.reply_text(
                        "\n".join(lines), parse_mode="Markdown"
                    )
            except Exception as e:
                await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

        elif sub == "add":
            if len(context.args) < 3:
                await update.message.reply_text(
                    "Usage: /credential add <service> <json\\_data>\n"
                    'Example: /credential add github \\{"token": "ghp\\_xxx"\\}',
                    parse_mode="Markdown",
                )
                return

            service_name = context.args[1]
            json_str = " ".join(context.args[2:])

            try:
                import json as _json

                data = _json.loads(json_str)
            except (ValueError, TypeError):
                await update.message.reply_text(
                    "Invalid JSON data. Make sure to use proper JSON format."
                )
                return

            try:
                from ..security.credential_store import store_credential

                await store_credential(service_name, data)
                await update.message.reply_text(
                    f"Stored credential for `{service_name}`.",
                    parse_mode="Markdown",
                )
            except Exception as e:
                await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

        elif sub == "remove":
            if len(context.args) < 2:
                await update.message.reply_text(
                    "Usage: /credential remove <service>"
                )
                return

            service_name = context.args[1]
            try:
                from ..security.credential_store import delete_credential

                deleted = await delete_credential(service_name)
                if deleted:
                    await update.message.reply_text(
                        f"Removed credential for `{service_name}`.",
                        parse_mode="Markdown",
                    )
                else:
                    await update.message.reply_text(
                        f"No credential found for '{service_name}'."
                    )
            except Exception as e:
                await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

        else:
            await update.message.reply_text(
                "Unknown subcommand. Use: list, add, or remove."
            )

    # ─── Workflow Commands ──────────────────────────────────────────────

    async def cmd_wfstatus(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show workflow progress for a mission."""
        if not context.args:
            await update.message.reply_text("Usage: /wfstatus <mission\\_id>",
                                            parse_mode="Markdown")
            return

        try:
            mission_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Mission ID must be a number.")
            return

        try:
            from ..workflows.engine.status import (
                compute_phase_progress, format_status_message,
            )

            mission = await get_mission(mission_id)
            if not mission:
                await update.message.reply_text(f"Mission #{mission_id} not found.")
                return

            tasks = await get_tasks_for_mission(mission_id)
            if not tasks:
                await update.message.reply_text(
                    f"No tasks found for mission #{mission_id}."
                )
                return

            # Determine workflow name from mission context
            mission_ctx = mission.get("context", "{}")
            if isinstance(mission_ctx, str):
                import json as _json
                try:
                    mission_ctx = _json.loads(mission_ctx)
                except (ValueError, TypeError):
                    mission_ctx = {}
            workflow_name = mission_ctx.get("workflow_name", "idea_to_product_v2")

            progress = compute_phase_progress(tasks)
            msg = format_status_message(workflow_name, mission_id, progress)
            await update.message.reply_text(msg)
        except Exception as e:
            await update.message.reply_text(
                f"❌ {_friendly_error(str(e))}"
            )

    async def cmd_dlq(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Dead-letter queue management: /dlq [retry <task_id> | discard <task_id>]."""
        from ..infra.dead_letter import (
            get_dlq_summary, get_dlq_tasks, retry_dlq_task, resolve_dlq_task,
        )

        args = context.args or []

        try:
            if len(args) >= 2 and args[0] == "retry":
                task_id = int(args[1])
                await retry_dlq_task(task_id)
                await update.message.reply_text(
                    f"\u2705 Task #{task_id} re-queued from dead-letter queue."
                )
                return

            if len(args) >= 2 and args[0] == "discard":
                task_id = int(args[1])
                await resolve_dlq_task(task_id, resolution="discarded")
                await update.message.reply_text(
                    f"\U0001f5d1 Task #{task_id} discarded from dead-letter queue."
                )
                return

            # Default: show summary + recent entries
            summary = await get_dlq_summary()
            tasks = await get_dlq_tasks()

            lines = [
                "\u2620\ufe0f *Dead-Letter Queue*\n",
                f"Unresolved: *{summary['unresolved']}*  |  "
                f"Resolved: *{summary['resolved']}*  |  "
                f"Total: *{summary['total']}*",
            ]

            if summary["categories"]:
                cats = ", ".join(
                    f"{cat}: {cnt}" for cat, cnt in summary["categories"].items()
                )
                lines.append(f"Categories: {cats}\n")

            if tasks:
                lines.append("*Recent quarantined tasks:*")
                for t in tasks[:8]:
                    error_preview = (t.get("error") or "")[:60]
                    lines.append(
                        f"  \u2022 #{t['task_id']} (mission {t.get('mission_id', '?')}) "
                        f"[{t.get('error_category', '?')}] {error_preview}"
                    )
                lines.append(
                    "\nUse `/dlq retry <id>` or `/dlq discard <id>`"
                )
            else:
                lines.append("\n\u2705 No quarantined tasks!")

            await update.message.reply_text(
                "\n".join(lines), parse_mode="Markdown"
            )
        except Exception as e:
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume a failed or paused workflow."""
        if not context.args:
            await update.message.reply_text("Usage: /resume <mission\\_id>",
                                            parse_mode="Markdown")
            return

        try:
            mission_id_str = context.args[0]
            mission_id = int(mission_id_str)
        except ValueError:
            await update.message.reply_text("Mission ID must be a number.")
            return

        try:
            from ..workflows.engine.runner import WorkflowRunner

            runner = WorkflowRunner()
            resumed_id = await runner.resume(mission_id)

            await update.message.reply_text(
                f"\u25b6\ufe0f Workflow resumed for mission #{resumed_id}\n"
                f"Use /wfstatus {resumed_id} to track progress."
            )
        except ValueError as e:
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")
        except Exception as e:
            await update.message.reply_text(
                f"❌ {_friendly_error(str(e))}"
            )

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Pause a task or mission: /pause <mission_id|task_id>"""
        args = context.args or []
        if not args:
            await update.message.reply_text("Usage: /pause <mission\\_id>\nPauses all pending/processing tasks for the mission.",
                                           parse_mode="Markdown")
            return
        try:
            mission_id = int(args[0])
            from ..infra.db import get_db
            async with get_db() as db:
                result = await db.execute(
                    """UPDATE tasks SET status = 'paused'
                       WHERE mission_id = ? AND status IN ('pending', 'processing')""",
                    (mission_id,)
                )
                await db.commit()
                count = result.rowcount
            await update.message.reply_text(f"\u23f8 Mission #{mission_id}: paused {count} task(s).")
            logger.info("mission paused via command", mission_id=mission_id, tasks_paused=count)
        except ValueError:
            await update.message.reply_text("Please provide a valid integer mission ID.")
        except Exception as e:
            logger.exception("pause command failed", error=str(e))
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def cmd_load(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/load full|heavy|shared|minimal|auto — set GPU load mode"""
        args = context.args or []
        if not args:
            from src.infra.load_manager import get_load_mode, is_auto_managed
            current = await get_load_mode()
            auto_str = " (auto-managed)" if is_auto_managed() else " (manual)"
            await update.message.reply_text(
                f"Current load mode: *{current}*{auto_str}\n\n"
                "Usage: `/load full|heavy|shared|minimal|auto`\n"
                "• *full* — all GPU available\n"
                "• *heavy* — 90% VRAM cap\n"
                "• *shared* — 50% VRAM cap\n"
                "• *minimal* — cloud only\n"
                "• *auto* — enable auto-detection based on external GPU usage",
                parse_mode="Markdown",
            )
            return
        choice = args[0].lower()
        if choice == "auto":
            from src.infra.load_manager import enable_auto_management
            await enable_auto_management()
            await update.message.reply_text(
                "GPU load mode set to *auto-managed*. "
                "Will adjust based on external GPU usage.",
                parse_mode="Markdown",
            )
            logger.info("load mode set to auto via command")
            return
        from src.infra.load_manager import set_load_mode
        msg = await set_load_mode(choice, source="user")
        await update.message.reply_text(msg, parse_mode="Markdown")
        logger.info("load mode changed via command", mode=choice)

    async def cmd_tune(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/tune — force an auto-tuning cycle and report results."""
        await update.message.reply_text("Running tuning cycle...")
        try:
            from src.models.auto_tuner import maybe_run_tuning
            report = await maybe_run_tuning(force=True)

            tuned = report.get("tuned_models", {}) if report else {}
            if not tuned:
                await update.message.reply_text("No models needed tuning adjustment.")
                return

            lines = ["*Auto-Tuning Report*\n"]
            for model, info in tuned.items():
                changes = info["changes"]
                gw = info["grading_weight"]
                lines.append(f"*{model}* (grading weight: {gw:.0%})")
                for cap, vals in changes.items():
                    arrow = "\u2191" if vals["new"] > vals["old"] else "\u2193"
                    lines.append(f"  {cap}: {vals['old']:.1f} {arrow} {vals['new']:.1f}")
            lines.append(f"\n_{len(report.get('skipped', []))} models skipped (no data)_")

            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            logger.error("tune command failed", error=str(e))
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def cmd_improve(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/improve — run self-improvement analysis and show proposals."""
        await update.message.reply_text("Analyzing system performance...")
        try:
            from ..memory.self_improvement import (
                analyze_and_propose, format_proposals_for_telegram
            )
            proposals = await analyze_and_propose()
            msg = format_proposals_for_telegram(proposals)
            # Split if too long
            if len(msg) > 4000:
                await update.message.reply_text(msg[:4000], parse_mode="Markdown")
            else:
                await update.message.reply_text(msg, parse_mode="Markdown")

            # Save full report
            if proposals:
                try:
                    from ..memory.self_improvement import format_proposals_for_file
                    import os
                    report = await format_proposals_for_file(proposals)
                    os.makedirs("workspace/results", exist_ok=True)
                    path = f"workspace/results/improvement_report_{datetime.now().strftime('%Y%m%d')}.md"
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(report)
                    await update.message.reply_text(f"Full report saved: {path}")
                except Exception:
                    pass
        except Exception as e:
            logger.error("improve command failed", error=str(e))
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    # ── Phase 14.4: /remember and /recall ─────────────────────────────────────

    async def cmd_remember(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Store a user fact in the knowledge base for later recall."""
        text = " ".join(context.args) if context.args else ""
        if not text:
            await update.message.reply_text(
                "Usage: `/remember <fact or note>`\n"
                "Example: `/remember The staging server is at 10.0.1.50`",
                parse_mode="Markdown",
            )
            return
        try:
            from ..memory.rag import store_fact
            doc_id = await store_fact(
                text,
                category="user_knowledge",
                source="telegram",
                importance=8,
            )
            await update.message.reply_text(
                f"✅ Remembered! (id: `{doc_id or 'stored'}`)\n"
                f"Use `/recall` to search your notes later.",
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error("remember command failed", error=str(e))
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def cmd_recall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Search the knowledge base for previously stored facts."""
        query_text = " ".join(context.args) if context.args else ""
        if not query_text:
            await update.message.reply_text(
                "Usage: `/recall <search query>`\n"
                "Example: `/recall staging server address`",
                parse_mode="Markdown",
            )
            return
        try:
            from ..memory.vector_store import query as vs_query
            results = await vs_query(query_text, collection="semantic", top_k=5)
            if not results:
                await update.message.reply_text("No matching memories found.")
                return
            lines = ["🔍 *Recall results:*\n"]
            for i, r in enumerate(results, 1):
                text = r.get("text", r.get("document", ""))[:200]
                score = r.get("score", r.get("distance", 0))
                cat = r.get("metadata", {}).get("category", "")
                tag = f" [{cat}]" if cat else ""
                lines.append(f"{i}. {text}{tag}")
                if score:
                    lines.append(f"   _relevance: {score:.2f}_")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            logger.error("recall command failed", error=str(e))
            await update.message.reply_text(f"❌ {_friendly_error(str(e))}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Smart handler: LLM-classified message routing with clarification priority."""
        chat_id = update.message.chat_id

        # Handle file uploads
        if update.message.document:
            try:
                uploads_dir = Path("workspace/uploads")
                uploads_dir.mkdir(parents=True, exist_ok=True)

                file_obj = await context.bot.get_file(update.message.document.file_id)
                filename = update.message.document.file_name or f"file_{update.message.document.file_id}"
                filepath = uploads_dir / filename

                await file_obj.download_to_drive(str(filepath))

                await update.message.reply_text(
                    f"\U0001f4ce File received: `{filename}`\n"
                    f"Use `/ingest {filepath}` to add to knowledge base.",
                    parse_mode="Markdown"
                )
                logger.info("File uploaded", filename=filename, chat_id=chat_id)
                return
            except Exception as e:
                logger.error(f"Failed to handle file upload: {e}")
                await update.message.reply_text(f"❌ {_friendly_error(str(e))}")
                return

        text = update.message.text
        if not text:
            return

        logger.info("Message received", user_id=chat_id, text_preview=text[:50])

        # ═══════════════════════════════════════════════════════
        # PRIORITY 0: Button-initiated conversation flow
        # ═══════════════════════════════════════════════════════
        pending_action = self._pending_action.pop(chat_id, None)
        if pending_action:
            cmd = pending_action["command"]
            handler = self._resolve_cmd_handler(cmd)
            if handler:
                # Simulate command with text as args
                context.args = text.split() if text.strip() else []
                try:
                    await handler(update, context)
                except Exception as e:
                    await update.message.reply_text(f"❌ {_friendly_error(str(e))}")
                return

        # ═══════════════════════════════════════════════════════
        # PRIORITY 1: Check for pending clarification (state-based)
        # ═══════════════════════════════════════════════════════
        pending_task_id = self._pending_clarifications.get(chat_id)
        if pending_task_id:
            # Also check DB to confirm task is still in needs_clarification
            try:
                task_info = await get_task(pending_task_id)
                if task_info and task_info.get("status") == "needs_clarification":
                    await self._resume_with_clarification(
                        chat_id, pending_task_id, text, task_info, update
                    )
                    return
                else:
                    # Task no longer waiting — stale entry
                    self._pending_clarifications.pop(chat_id, None)
            except Exception:
                self._pending_clarifications.pop(chat_id, None)

        # Also check DB for ANY task in needs_clarification (handles bot restart)
        try:
            db = await get_db()
            cursor = await db.execute(
                """SELECT id, title, description, status, context FROM tasks
                   WHERE status = 'needs_clarification'
                   ORDER BY created_at DESC LIMIT 1"""
            )
            row = await cursor.fetchone()
            if row:
                waiting_task = dict(row)
                # LLM check: is this message likely a response to the pending question?
                is_response = await self._is_likely_clarification_response(
                    text, waiting_task
                )
                if is_response:
                    await self._resume_with_clarification(
                        chat_id, waiting_task["id"], text, waiting_task, update
                    )
                    return
        except Exception as e:
            logger.debug("clarification DB check failed", error=str(e))

        # ═══════════════════════════════════════════════════════
        # PRIORITY 2: LLM-based message classification
        # ═══════════════════════════════════════════════════════
        classification = await self._classify_user_message(text)
        msg_type = classification["type"]
        msg_workflow = classification.get("workflow")
        logger.info("message classified", msg_type=msg_type,
                    workflow=msg_workflow, text_preview=text[:50])

        # ── Route by classification ──

        if msg_type in ("bug_report", "feature_request", "ui_note", "feedback"):
            await self._handle_user_input(msg_type, text, chat_id, update)
            return

        if msg_type == "progress_inquiry":
            await update.message.reply_text("📊 Understood as a status question.")
            await self.cmd_progress(update, context)
            return

        if msg_type == "question":
            await update.message.reply_text("❓ Got your question — looking into it...")
            task_id = await add_task(
                title=f"Q: {text[:50]}",
                description=text,
                tier="auto",
                priority=TASK_PRIORITY.get("high", 8),
                agent_type="assistant",
            )
            self.user_last_task_id[chat_id] = task_id
            return

        if msg_type == "casual":
            await self._handle_casual(text, update)
            return

        if msg_type == "load_control":
            await self._handle_load_control(text, update)
            return

        if msg_type == "todo":
            await self._handle_todo_from_message(text, chat_id, update)
            return

        if msg_type in ("followup", "clarification_response"):
            parent_id = await self._find_followup_parent(chat_id, text)
            if parent_id:
                task_context = {"followup_to": parent_id}
                task_id = await add_task(
                    title=f"Follow-up to #{parent_id}: {text[:40]}",
                    description=text,
                    tier="auto",
                    parent_task_id=parent_id,
                    priority=TASK_PRIORITY.get("high", 8),
                    context=task_context,
                )
                self.user_last_task_id[chat_id] = task_id
                await update.message.reply_text(
                    f"🔗 Continuing task #{parent_id}. Queued as #{task_id}."
                )
                return

        # ═══════════════════════════════════════════════════════
        # PRIORITY 3: Mission vs task
        # ═══════════════════════════════════════════════════════
        # Only link to parent task if the message was classified as a followup.
        # New tasks ("task" type) should NOT inherit context from previous tasks —
        # "Can you do a web search" after a shoes task should not search for shoes.
        parent_id = None
        recent_context = None
        if msg_type == "followup":
            parent_id = self.user_last_task_id.get(chat_id)
            try:
                followup = await find_followup_context(chat_id, text)
                if followup.get("is_followup") and followup.get("parent_task_id"):
                    parent_id = int(followup["parent_task_id"])
                if followup.get("context"):
                    recent_context = format_recent_context(followup["context"])
            except Exception:
                pass

        if msg_type == "mission":
            # Classifier decides if this needs workflow engine
            if msg_workflow == "idea_to_product":
                try:
                    from ..workflows.engine.runner import WorkflowRunner
                    runner = WorkflowRunner(self.orchestrator)
                    mission_id = await runner.start(
                        workflow_name="idea_to_product_v2",
                        initial_input={"idea": text, "product_name": text[:50]},
                        title=text[:80],
                    )
                    await update.message.reply_text(
                        f"🔄 Workflow mission #{mission_id} created!\n"
                        f"_{text[:60]}_\n\n"
                        f"Use /wfstatus {mission_id} to track progress.",
                        parse_mode="Markdown",
                    )
                except Exception as e:
                    logger.error("workflow mission failed", error=str(e))
                    await update.message.reply_text(f"❌ {_friendly_error(str(e))}")
            else:
                mission_id = await add_mission(title=text[:80], description=text, priority=5)
                if self.orchestrator:
                    await self.orchestrator.plan_mission(mission_id, text[:80], text)
                await update.message.reply_text(
                    f"🎯 Mission #{mission_id} created. Planning now..."
                )
            self.user_last_task_id.pop(chat_id, None)
        else:
            task_context = {}
            if recent_context:
                task_context["recent_conversation"] = recent_context

            task_id = await add_task(
                title=text[:50],
                description=text,
                tier="auto",
                parent_task_id=parent_id,
                priority=TASK_PRIORITY["critical"],
                context=task_context if task_context else None,
            )
            self.user_last_task_id[chat_id] = task_id
            await update.message.reply_text(f"\u2705 Task #{task_id} queued.")

    # ─── Message Classification Helpers ───────────────────────────────────────

    MESSAGE_CLASSIFIER_PROMPT = """Classify this user message to an AI orchestrator. Respond with ONLY valid JSON.

Categories:
- "mission": complex multi-step project request (building something, research project, etc.)
- "task": specific actionable request
- "question": asking for information
- "bug_report": reporting a bug or error
- "feature_request": suggesting a new feature
- "ui_note": UI/UX feedback
- "feedback": general feedback on system behavior
- "progress_inquiry": asking about status of work
- "followup": continuing a previous conversation or task
- "clarification_response": answering a question the system asked
- "load_control": wanting to control GPU/resources (e.g. "I'm going to game", "free up GPU")
- "todo": adding a reminder, to-do item, or note to self (e.g., "remind me to buy milk", "don't forget the meeting", "add eggs to my list")
- "casual": greeting, thanks, small talk

If type is "mission", also decide if this needs a full product workflow:
- "workflow": "idea_to_product" if the user wants to BUILD a product, app, website, tool, platform, SaaS, bot, or any software from scratch
- "workflow": null if it's a simpler mission (research, analysis, fix something, write a report)

Context: {context}

Message: {message}

Respond as: {{"type": "mission", "confidence": 0.9, "workflow": "idea_to_product"}}
Or: {{"type": "task", "confidence": 0.8}}"""

    async def _classify_user_message(self, text: str) -> dict:
        """Classify user message using LLM with keyword fallback.

        Returns dict with at least {"type": str}. May also include
        "workflow" for mission-type messages that need the workflow engine.
        """
        try:
            from ..core.router import ModelRequirements, call_model

            # Build context hint
            context_parts = []
            if self._pending_clarifications:
                context_parts.append("System has pending clarification requests")

            reqs = ModelRequirements(
                task="router",
                agent_type="classifier",
                difficulty=2,
                prefer_speed=True,
                needs_json_mode=True,
                priority=2,
                estimated_input_tokens=300,
                estimated_output_tokens=50,
            )
            messages = [{
                "role": "user",
                "content": self.MESSAGE_CLASSIFIER_PROMPT.format(
                    message=text[:300],
                    context="; ".join(context_parts) if context_parts else "none",
                ),
            }]
            response = await call_model(reqs, messages)
            from ..core.task_classifier import _extract_json
            raw = response.get("content", "").strip()
            result = _extract_json(raw)
            msg_type = result.get("type", "task")
            confidence = result.get("confidence", 0.5)
            workflow = result.get("workflow")
            logger.debug("llm message classification",
                         type=msg_type, confidence=confidence, workflow=workflow)
            if confidence < 0.4:
                return {"type": "task"}
            classification = {"type": msg_type}
            if workflow:
                classification["workflow"] = workflow
            return classification
        except Exception as e:
            logger.debug("message classification failed, using keyword fallback",
                         error=str(e))
            return self._classify_message_by_keywords(text)

    @staticmethod
    def _classify_message_by_keywords(text: str) -> dict:
        """Fast keyword fallback for message classification."""
        lower = text.lower()
        # Todo items
        if any(w in lower for w in [
            "remind me", "don't forget", "dont forget", "todo",
            "add to list", "add to my list", "need to buy", "need to get",
            "remember to", "note to self", "hatirla", "unutma", "listeye ekle",
        ]):
            return {"type": "todo"}
        # Bug reports
        if any(w in lower for w in [
            "bug", "error", "broken", "crash", "doesn't work", "not working",
            "failed", "exception", "traceback", "issue with",
        ]):
            return {"type": "bug_report"}
        # Feature requests
        if any(w in lower for w in [
            "feature", "could you add", "would be nice", "suggestion",
            "it would help if", "can we have", "please add", "wish list",
        ]):
            return {"type": "feature_request"}
        # Feedback
        if any(w in lower for w in [
            "good job", "well done", "not great", "could be better",
            "i think", "in my opinion", "the output was",
        ]):
            return {"type": "feedback"}
        # Status / progress questions
        if any(w in lower for w in [
            "how's", "status", "progress", "how far", "eta", "update on",
            "what's happening", "is it done", "still running", "where are we",
        ]):
            return {"type": "progress_inquiry"}
        # Questions
        if any(w in lower for w in [
            "what is", "how do", "why does", "can you explain", "what does",
            "how does", "tell me about", "?",
        ]):
            return {"type": "question"}
        # Casual
        if any(w in lower for w in [
            "hi ", "hello", "thanks", "thank you", "hey", "good morning",
            "good night", "bye", "see you", "sup", "yo ",
        ]):
            return {"type": "casual"}
        # GPU/load control
        if any(w in lower for w in [
            "game", "gaming", "free up gpu", "gpu", "i'm going to play",
        ]):
            return {"type": "load_control"}
        # Mission (long or project-like)
        if len(text) > 200 or any(w in lower for w in [
            "research", "create a", "build", "analyze", "develop", "plan",
            "design a", "implement a", "set up", "write a report", "strategy",
        ]):
            result = {"type": "mission"}
            # Keyword fallback for workflow detection
            if _looks_like_product_idea(text):
                result["workflow"] = "idea_to_product"
            return result
        return {"type": "task"}

    async def _is_likely_clarification_response(
        self, text: str, waiting_task: dict
    ) -> bool:
        """Check if message is likely a response to a pending clarification.
        Uses heuristics first (fast), then LLM if uncertain."""
        # Heuristic: short responses are very likely clarification answers
        if len(text) < 100:
            return True
        # Heuristic: if it looks like a command or new task, probably not
        lower = text.lower()
        if any(kw in lower for kw in [
            "research", "create a", "build", "new mission"
        ]):
            return False
        # Default: if there's a pending clarification, assume it's an answer
        return True

    async def _resume_with_clarification(
        self, chat_id: int, task_id: int, answer: str,
        task_info: dict, update: Update
    ):
        """Resume a task that was waiting for clarification."""
        # Update the task: inject answer into context, reset to pending
        try:
            import json as _json
            existing_ctx = task_info.get("context", "{}")
            if isinstance(existing_ctx, str):
                try:
                    ctx = _json.loads(existing_ctx)
                except (ValueError, TypeError):
                    ctx = {}
            else:
                ctx = existing_ctx or {}

            # Append clarification to context
            ctx["user_clarification"] = answer
            clarifications = ctx.get("clarification_history", [])
            clarifications.append(answer)
            ctx["clarification_history"] = clarifications

            await update_task(
                task_id,
                status="pending",
                context=_json.dumps(ctx),
            )

            # Clean up tracking
            self._pending_clarifications.pop(chat_id, None)

            await update.message.reply_text(
                f"\u21a9\ufe0f Got it. Resuming task #{task_id} with your input."
            )
            logger.info("clarification received, task resumed",
                        task_id=task_id, answer_preview=answer[:50])

            # Record feedback
            try:
                await record_feedback(task_info, "modified", details=answer)
            except Exception:
                pass

        except Exception as e:
            logger.error("failed to resume clarification",
                         task_id=task_id, error=str(e))
            await update.message.reply_text(
                f"❌ {_friendly_error(str(e))}"
            )

    async def _handle_user_input(
        self, msg_type: str, text: str, chat_id: int, update: Update
    ):
        """Route bug reports, feature requests, feedback to user_inputs."""
        type_map = {
            "bug_report": "bug",
            "feature_request": "feature",
            "ui_note": "ui_note",
            "feedback": "feedback",
        }
        input_type = type_map.get(msg_type, "feedback")

        try:
            from ..infra.user_inputs import log_input
            # Try to find related mission
            related_mission = None
            try:
                missions = await get_active_missions()
                if missions:
                    # Simple fuzzy: check if any mission title words appear in the message
                    lower = text.lower()
                    for g in missions:
                        title_words = g["title"].lower().split()
                        if any(w in lower for w in title_words if len(w) > 3):
                            related_mission = g["id"]
                            break
            except Exception:
                pass

            input_id = await log_input(
                input_type=input_type,
                content=text,
                related_mission_id=related_mission,
            )

            type_emoji = {
                "bug": "\U0001f41b", "feature": "\U0001f4a1",
                "ui_note": "\U0001f3a8", "feedback": "\U0001f4ac",
            }
            emoji = type_emoji.get(input_type, "\U0001f4ac")
            mission_str = f" Linked to Mission #{related_mission}." if related_mission else ""
            await update.message.reply_text(
                f"{emoji} Logged as {input_type} #{input_id}.{mission_str}"
            )
        except Exception as e:
            logger.error("failed to log user input", error=str(e))
            await update.message.reply_text(f"Logged your {input_type}. (Note: save had an issue, check logs)")

    async def _handle_casual(self, text: str, update: Update):
        """Handle casual messages with a quick LLM response (no task creation)."""
        try:
            from ..core.router import ModelRequirements, call_model
            reqs = ModelRequirements(
                task="assistant",
                agent_type="assistant",
                difficulty=2,
                prefer_speed=True,
                priority=1,
                estimated_input_tokens=100,
                estimated_output_tokens=100,
            )
            response = await call_model(reqs, [{"role": "user", "content": text}])
            reply = response.get("content", "Hey! How can I help?")
            await update.message.reply_text(reply[:1000])
        except Exception:
            await update.message.reply_text("Hey! Send me a task or mission to work on.")

    async def _handle_load_control(self, text: str, update: Update):
        """Handle natural language GPU load control."""
        lower = text.lower()
        if any(w in lower for w in ["game", "gaming", "play"]):
            from ..infra.load_manager import set_load_mode
            msg = await set_load_mode("minimal", source="user")
            await update.message.reply_text(
                f"\U0001f3ae Switching to minimal mode for gaming.\n{msg}\n"
                "Use `/load auto` when you're done.",
                parse_mode="Markdown",
            )
        elif any(w in lower for w in ["free", "done", "finished", "back"]):
            from ..infra.load_manager import enable_auto_management
            await enable_auto_management()
            await update.message.reply_text(
                "GPU auto-management re-enabled. I'll use what's available."
            )
        else:
            await update.message.reply_text(
                "Use `/load full|heavy|shared|minimal|auto` to control GPU usage.",
                parse_mode="Markdown",
            )

    # ─── Todo Commands ──────────────────────────────────────────────────────

    async def cmd_todo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add a todo item. /todo <title>"""
        if not context.args:
            await update.message.reply_text(
                "Usage: /todo <what to remember>\n\n"
                "Example: /todo Buy groceries"
            )
            return

        title = " ".join(context.args)
        todo_id = await add_todo(title)
        buttons = [[InlineKeyboardButton(
            "✅ Done", callback_data=f"todo_toggle:{todo_id}"
        )]]
        await update.message.reply_text(
            f"📝 Added: *{title}*\n(#{todo_id})",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def cmd_todos(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all todos. /todos"""
        todos = await get_todos()
        if not todos:
            await update.message.reply_text("📋 No todo items yet. Use /todo to add one.")
            return

        lines = ["📋 *Your Todos*\n"]
        buttons = []
        for todo in todos:
            tid = todo["id"]
            title = todo["title"]
            status = todo.get("status", "pending")
            icon = "✅" if status == "done" else "⬜"
            priority = todo.get("priority", "normal")
            p_icon = {"high": "🔴", "normal": "🟡", "low": "⚪"}.get(priority, "🟡")

            lines.append(f"  {icon} {p_icon} *#{tid}* — {title}")
            if status != "done":
                buttons.append([InlineKeyboardButton(
                    f"✅ Done: {title[:25]}",
                    callback_data=f"todo_toggle:{tid}",
                )])

        text = "\n".join(lines)
        markup = InlineKeyboardMarkup(buttons) if buttons else None
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=markup)

    async def cmd_cleartodos(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Delete all completed todos. /cleartodos"""
        done_todos = await get_todos(status="done")
        if not done_todos:
            await update.message.reply_text("No completed todos to clear.")
            return

        for todo in done_todos:
            await delete_todo(todo["id"])

        await update.message.reply_text(
            f"🗑️ Cleared {len(done_todos)} completed todo(s)."
        )

    async def _handle_todo_from_message(self, text: str, chat_id: int, update):
        """Extract todo title from natural language and create a todo item."""
        import re
        # Strip common prefixes to get the actual todo title
        title = text
        for prefix in [
            r"remind me to\s+",
            r"don'?t forget to\s+",
            r"dont forget to\s+",
            r"remember to\s+",
            r"note to self[:\s]+",
            r"add to (?:my )?list[:\s]+",
            r"need to (?:buy|get)\s+",
            r"todo[:\s]+",
            r"hatirla[:\s]+",
            r"unutma[:\s]+",
            r"listeye ekle[:\s]+",
        ]:
            title = re.sub(f"^{prefix}", "", title, flags=re.IGNORECASE).strip()

        if not title:
            title = text  # Fallback to original if stripping removed everything

        todo_id = await add_todo(title, source="implicit")
        buttons = [[InlineKeyboardButton(
            "✅ Done", callback_data=f"todo_toggle:{todo_id}"
        )]]
        await update.message.reply_text(
            f"📝 Added: *{title}*",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _find_followup_parent(self, chat_id: int, text: str) -> int | None:
        """Find parent task for follow-up messages."""
        try:
            followup = await find_followup_context(chat_id, text)
            if followup.get("is_followup") and followup.get("parent_task_id"):
                return int(followup["parent_task_id"])
        except Exception:
            pass
        return self.user_last_task_id.get(chat_id)

    async def handle_reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle Telegram reply-to-message — catches explicit replies to clarification messages."""
        replied_to = update.message.reply_to_message
        if not replied_to:
            return

        text = replied_to.text or ""
        answer = update.message.text
        chat_id = update.message.chat_id

        # Check if this is a reply to a clarification request
        if "Clarification needed" in text:
            import re
            match = re.search(r"Task #(\d+)", text)
            if match:
                task_id = int(match.group(1))
                task_info = await get_task(task_id)
                if task_info and task_info.get("status") == "needs_clarification":
                    await self._resume_with_clarification(
                        chat_id, task_id, answer, task_info, update
                    )
                    return

        # Check if replying to an approval request
        if "Approval Required" in text:
            # Let handle_callback deal with button presses; text replies here
            # just fall through to normal message handling
            pass

        # Not a known reply type — treat as normal message
        await self.handle_message(update, context)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data

        # ── Interactive Menu Callbacks ──────────────────────────
        if data == "menu_back":
            await query.edit_message_text(
                "🤖 *Autonomous AI Orchestrator*\n\n"
                "Tap a category below, or just send a message.",
                parse_mode="Markdown",
                reply_markup=_build_category_keyboard()
            )
            return

        if data.startswith("menu_cat:"):
            cat_key = data.split(":", 1)[1]
            # Find category label
            cat_label = cat_key
            for label, key, _ in MENU_CATEGORIES:
                if key == cat_key:
                    cat_label = label
                    break
            await query.edit_message_text(
                f"{cat_label}\n\nTap a command:",
                reply_markup=_build_command_keyboard(cat_key)
            )
            return

        if data.startswith("menu_cmd:"):
            cmd = data.split(":", 1)[1]
            handler = self._resolve_cmd_handler(cmd)
            if handler:
                context.args = []
                # Send a message so the handler has something to reply_text on
                msg = await query.message.reply_text(f"⏳ /{cmd}...")
                # Wrap in a lightweight object matching what handlers expect
                class _CallbackUpdate:
                    """Minimal update shim so cmd_* handlers work from callbacks."""
                    def __init__(self, message):
                        self.message = message
                        self.effective_chat = message.chat
                try:
                    await handler(_CallbackUpdate(msg), context)
                except Exception as e:
                    await msg.reply_text(f"❌ {_friendly_error(str(e))}")
            else:
                await query.message.reply_text(f"Unknown command: /{cmd}")
            return

        if data.startswith("menu_ask:"):
            cmd = data.split(":", 1)[1]
            prompt = _CMD_ARG_PROMPTS.get(cmd, f"Enter argument for /{cmd}:")
            chat_id = query.message.chat_id
            self._pending_action[chat_id] = {"command": cmd}
            await query.message.reply_text(f"💬 {prompt}")
            return

        # ── Todo Callbacks ─────────────────────────────────────
        if data.startswith("todo_toggle:"):
            todo_id = int(data.split(":")[1])
            new_status = await toggle_todo(todo_id)
            icon = "✅" if new_status == "done" else "⬜"
            await query.answer(f"{icon} {'Done!' if new_status == 'done' else 'Reopened'}")
            # Refresh the message text to reflect the change
            try:
                todo = await get_todo(todo_id)
                if todo:
                    status_icon = "✅" if new_status == "done" else "📝"
                    await query.edit_message_text(
                        f"{status_icon} *#{todo_id}* — {todo['title']}\nStatus: {new_status}",
                        parse_mode="Markdown",
                    )
            except Exception:
                pass
            return

        if data.startswith("todo_ai:"):
            todo_id = int(data.split(":")[1])
            todo = await get_todo(todo_id)
            if todo:
                task_id = await add_task(
                    title=f"Help with: {todo['title'][:40]}",
                    description=f"Help the user with this todo item: {todo['title']}\n{todo.get('description', '')}",
                    tier="auto",
                    priority=8,
                )
                await query.answer(f"Task #{task_id} created!")
            return

        # ── Legacy Callbacks (approval, resetall confirm) ──────
        if data == "resetall_confirm":
            db = await get_db()
            await db.execute("DELETE FROM conversations")
            await db.execute("DELETE FROM tasks")
            await db.execute("DELETE FROM missions")
            await db.execute("DELETE FROM memory")
            await db.commit()
            await query.edit_message_text("☢️ Everything wiped. Fresh start.")
            return
        elif data == "resetall_cancel":
            await query.edit_message_text("Cancelled.")
            return

        # Approval/rejection buttons (approve_<id>, reject_<id>)
        if "_" in data:
            action, task_id_str = data.split("_", 1)
            try:
                task_id = int(task_id_str)
            except ValueError:
                return

            if action == "approve":
                await update_approval_status(task_id, "approved")
                if task_id in self._approval_events:
                    self._approval_events[task_id]["result"] = "approved"
                    self._approval_events[task_id]["event"].set()
                await query.edit_message_text(f"✅ Task #{task_id} approved.")
            elif action == "reject":
                await update_approval_status(task_id, "rejected")
                if task_id in self._approval_events:
                    self._approval_events[task_id]["result"] = "rejected"
                    self._approval_events[task_id]["event"].set()
                await update_task(task_id, status="rejected")
                await query.edit_message_text(f"❌ Task #{task_id} rejected.")

    # --- Outbound notifications ---

    async def send_notification(self, text: str, retries: int = 2):
        import asyncio as _asyncio

        # Phase 8.3: Redact secrets from outgoing messages
        try:
            from ..security.sensitivity import redact_secrets
            text = redact_secrets(text)
        except Exception:
            pass

        for attempt in range(retries + 1):
            try:
                await self.app.bot.send_message(
                    chat_id=TELEGRAM_ADMIN_CHAT_ID,
                    text=text,
                    parse_mode="Markdown"
                )
                return
            except Exception as e:
                # First fallback: retry without markdown
                try:
                    await self.app.bot.send_message(
                        chat_id=TELEGRAM_ADMIN_CHAT_ID, text=text
                    )
                    return
                except Exception:
                    if attempt < retries:
                        await _asyncio.sleep(1 * (attempt + 1))
                        continue
                    logger.error("Failed to send Telegram notification", error=str(e))

    async def send_result(self, task_id, title, result, model, cost):
        # Handle long results (>3000 chars) by sending as file attachment
        if len(result) > 3000:
            # Create results directory if needed
            results_dir = Path("workspace/results")
            results_dir.mkdir(parents=True, exist_ok=True)

            # Save full result to file
            result_file = results_dir / f"task_{task_id}.md"
            try:
                result_file.write_text(result, encoding='utf-8')
            except Exception as e:
                logger.error(f"Failed to save result file: {e}")

            # Send summary message first
            summary = result[:500] + "..." if len(result) > 500 else result
            summary_text = (
                f"✅ *Task #{task_id} Complete*\n"
                f"**{title}**\n"
                f"Model: `{model}` | Cost: ${cost:.4f}\n\n"
                f"{summary}"
            )
            await self.send_notification(summary_text)

            # Send full result as file attachment
            try:
                with open(result_file, 'rb') as doc:
                    await self.app.bot.send_document(
                        chat_id=TELEGRAM_ADMIN_CHAT_ID,
                        document=doc,
                        filename=f"task_{task_id}.md",
                        caption=f"📎 Full result for task #{task_id}"
                    )
            except Exception as e:
                logger.error(f"Failed to send result document: {e}")

            # Store exchange with summary
            try:
                chat_id = TELEGRAM_ADMIN_CHAT_ID or "system"
                await store_exchange(
                    chat_id=chat_id,
                    user_message=title,
                    ai_response=summary,
                    task_id=task_id,
                    task_title=title,
                )
            except Exception:
                pass
        else:
            # Short results (<= 3000 chars) send as plain text
            truncated = result
            await self.send_notification(
                f"✅ *Task #{task_id} Complete*\n"
                f"**{title}**\n"
                f"Model: `{model}` | Cost: ${cost:.4f}\n\n"
                f"{truncated}"
            )

            # Phase 11.4: Store exchange in conversation memory
            try:
                # Use admin chat ID as the chat_id for results
                chat_id = TELEGRAM_ADMIN_CHAT_ID or "system"
                await store_exchange(
                    chat_id=chat_id,
                    user_message=title,
                    ai_response=truncated[:500],
                    task_id=task_id,
                    task_title=title,
                )
            except Exception:
                pass

    async def send_error(self, task_id, title, error):
        # Show user-friendly message, not raw tracebacks
        friendly = _friendly_error(error)
        await self.send_notification(
            f"❌ *Task #{task_id} Failed*\n"
            f"**{title}**\n\n{friendly}"
        )

    async def request_clarification(self, task_id, title, question):
        # Track this as a pending clarification so handle_message can catch it
        chat_id = TELEGRAM_ADMIN_CHAT_ID
        if chat_id:
            try:
                self._pending_clarifications[int(chat_id)] = task_id
            except (ValueError, TypeError):
                pass

        await self.send_notification(
            f"\u2753 *Clarification needed \u2014 Task #{task_id}*\n"
            f"**{title}**\n\n"
            f"{question}\n\n"
            f"_Reply to this message or just type your answer._"
        )

    async def request_approval(self, task_id, title, plan, tier,
                               mission_id=None):
        # Persist approval request to DB
        details = f"Tier: {tier}\n\n{plan[:500]}"
        await insert_approval_request(task_id, mission_id, title, details)

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
            await update_approval_status(task_id, "timeout")
            await self.send_notification(f"⏰ Approval for #{task_id} timed out. Skipped.")
            return False
        finally:
            self._approval_events.pop(task_id, None)
