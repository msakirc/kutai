# telegram_bot.py
import asyncio
import os
import signal
from datetime import datetime
from pathlib import Path
from src.infra.logging_config import get_logger
from ..infra.times import utc_now, to_turkey, TZ_TR, tr_hour_to_utc
from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID, TASK_PRIORITY

logger = get_logger("app.telegram_bot")
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

from ..infra.db import (add_task, add_mission, get_active_missions,
                get_ready_tasks, get_daily_stats, update_task, get_recent_completed_tasks,
                get_db, cancel_task, reprioritize_task, get_task_tree,
                get_task, get_mission, get_budget, set_budget, get_model_stats,
                get_mission_locks, get_tasks_for_mission, update_mission,
                insert_approval_request, update_approval_status,
                add_todo, get_todos, get_todo, toggle_todo, delete_todo,
                update_todo, get_blocked_task_summary)
from ..memory.conversations import format_recent_context, find_followup_context, \
    store_exchange
from ..memory.ingest import ingest_document
from ..memory.preferences import record_feedback
from ..tools.workspace import list_mission_workspaces


pending_clarifications = {}  # task_id -> asyncio.Event + response


async def _reverse_geocode_photon(lat: float, lon: float) -> tuple[str, str]:
    """Reverse geocode coordinates via Photon (privacy-first, no logging, no API key).

    Returns (district, city). Logs warnings on failure, never silences errors.
    """
    import aiohttp
    district, city = "", ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://photon.komoot.io/reverse",
                params={"lat": str(lat), "lon": str(lon), "limit": "1", "lang": "default"},
                headers={"User-Agent": "KutAI/1.0"},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Photon reverse geocode HTTP {resp.status}")
                    return district, city
                data = await resp.json()
                features = data.get("features", [])
                if features:
                    props = features[0].get("properties", {})
                    district = (props.get("district") or props.get("locality")
                                or props.get("suburb") or props.get("county") or "")
                    city = (props.get("city") or props.get("state") or "")
    except Exception as e:
        logger.warning(f"Photon reverse geocode failed: {e}")
    return district, city

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
        return "Internal error. Will retry on restart."
    if "json" in e or "parsing" in e or "decode" in e:
        return "Failed to process the response. Will retry."
    if "cancelled" in e or "cancellederror" in e:
        return "Interrupted by shutdown. Will retry on restart."
    # Generic — show first 100 chars, strip tracebacks and Python internals
    first_line = error.split("\n")[0][:100]
    if any(kw in first_line.lower() for kw in ["object has no", "expected", "got an"]):
        return "Internal error. Will retry on restart."
    return "Interrupted. Will retry on restart."

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

# ─── Reply Keyboard Navigation System ────────────────────────────────────
# All navigation uses reply keyboard swaps. Inline keyboards only for
# contextual actions (todo toggles, mission management, confirmations).

import time as _time

def _make_keyboard(rows: list[list[str]], resize_keyboard: bool = True, **kwargs) -> ReplyKeyboardMarkup:
    """Build a ReplyKeyboardMarkup from a list of string rows."""
    return ReplyKeyboardMarkup(
        [[KeyboardButton(btn) for btn in row] for row in rows],
        resize_keyboard=resize_keyboard,
        one_time_keyboard=False,
        is_persistent=True,
        **kwargs,
    )

# ── Keyboard Definitions ─────────────────────────────────────────────────

REPLY_KEYBOARD = _make_keyboard([
    ["⚡ Hizmet", "🛒 Alışveriş", "📋 Listem"],
    ["🎯 Görevler", "⚙️ Sistem"],
])

KB_HIZMET = _make_keyboard([
    ["🏥 Eczane", "💰 Döviz", "🌤 Hava", "⛽ Yakıt"],
    ["🕌 Namaz", "📰 Haber", "🪙 Altın", "🌍 Deprem"],
    ["📍 Konum", "🔙 Geri"],
])

KB_ALISVERIS = _make_keyboard([
    ["⚡ Hızlı Ara", "🔬 Detaylı Araştır"],
    ["🔙 Geri"],
])

KB_LISTEM = _make_keyboard([
    ["📝 Yeni Ekle", "⏰ Hatırlat"],
    ["🔙 Geri"],
])

KB_GOREVLER = _make_keyboard([
    ["🎯 Yeni Görev", "📬 İş Kuyruğu", "⏰ Zamanla"],
    ["🔙 Geri"],
])

KB_WORKFLOW_SELECT = _make_keyboard([
    ["⚡ Hızlı Cevap", "📊 Araştır & Raporla"],
    ["🏗 Yeni Proje", "🤖 Otomatik"],
    ["💻 Kod / Diğer"],
    ["🔙 Geri"],
])

KB_SISTEM = _make_keyboard([
    ["🖥 Yük Modu", "🐛 Debug", "📭 DLQ", "📋 Loglar"],
    ["🖥️ Claude Code", "🔧 Yaşar Usta", "🗑 Reset Tasks", "☢️ Reset All"],
    ["🔄 Yeniden Başlat", "⏹ Durdur"],
    ["🔙 Geri"],
])

KB_YUK_MODU = _make_keyboard([
    ["⚡ Full", "🔋 Heavy", "⚖️ Shared"],
    ["🔻 Minimal", "🤖 Otomatik"],
    ["🔙 Geri"],
])

KB_BASLAT = _make_keyboard([
    ["▶️ Başlat"],
])

# ── Keyboard State Management ────────────────────────────────────────────

# Which parent keyboard each sub-keyboard returns to on "Geri"
_KB_PARENT: dict[str, ReplyKeyboardMarkup] = {
    "hizmet": REPLY_KEYBOARD,
    "alisveris": REPLY_KEYBOARD,
    "listem": REPLY_KEYBOARD,
    "gorevler": REPLY_KEYBOARD,
    "sistem": REPLY_KEYBOARD,
    "workflow_select": KB_GOREVLER,
    "yuk_modu": KB_SISTEM,
    "debug": KB_SISTEM,
    "dlq": KB_SISTEM,
}

# Map ALL known button labels to handler info.
# Format: button_text -> (action_type, value)
#   "cmd"      -> call cmd_<value> with no args
#   "cmd_args" -> set pending_action for <value>, prompt user
#   "category" -> switch keyboard to <value> category
#   "special"  -> handled by dedicated logic in _handle_special_button
_BUTTON_ACTIONS: dict[str, tuple[str, str]] = {
    # ── Top-level categories ──
    "⚡ Hizmet": ("category", "hizmet"),
    "🛒 Alışveriş": ("category", "alisveris"),
    "📋 Listem": ("category", "listem"),
    "🎯 Görevler": ("category", "gorevler"),
    "⚙️ Sistem": ("category", "sistem"),
    # ── Hizmet sub-buttons ──
    "🏥 Eczane": ("special", "pharmacy"),
    "💰 Döviz": ("special", "exchange"),
    "🌤 Hava": ("special", "weather"),
    "⛽ Yakıt": ("special", "fuel"),
    "🕌 Namaz": ("special", "prayer"),
    "📰 Haber": ("special", "news"),
    "🪙 Altın": ("special", "gold"),
    "🌍 Deprem": ("special", "earthquake"),
    # ── Alışveriş sub-buttons ──
    "⚡ Hızlı Ara": ("cmd_args", "shop"),
    "🔬 Detaylı Araştır": ("cmd_args", "research_product"),
    # ── Listem sub-buttons ──
    "📝 Yeni Ekle": ("cmd_args", "todo"),
    "⏰ Hatırlat": ("special", "reminder"),
    # ── Görevler sub-buttons ──
    "🎯 Yeni Görev": ("special", "new_mission"),
    "📬 İş Kuyruğu": ("cmd", "view_queue"),
    "⏰ Zamanla": ("special", "schedule_task"),
    # ── Workflow selection ──
    "⚡ Hızlı Cevap": ("special", "wf_quick"),
    "📊 Araştır & Raporla": ("special", "wf_research"),
    "🏗 Yeni Proje": ("special", "wf_project"),
    "🤖 Otomatik": ("special", "wf_auto"),
    "💻 Kod / Diğer": ("special", "wf_other"),
    # ── Sistem sub-buttons ──
    "📍 Konum": ("special", "location"),
    "🖥 Yük Modu": ("category", "yuk_modu"),
    "🐛 Debug": ("special", "debug"),
    "📭 DLQ": ("special", "dlq"),
    "📋 Loglar": ("cmd", "logs"),
    "🖥️ Claude Code": ("special", "claude_code"),
    "🔧 Yaşar Usta": ("special", "processes"),
    "🗑 Reset Tasks": ("special", "reset_tasks"),
    "☢️ Reset All": ("cmd", "reset_all"),
    "🔄 Yeniden Başlat": ("special", "restart"),
    "⏹ Durdur": ("special", "stop"),
    # ── Yük Modu sub-buttons ──
    "⚡ Full": ("special", "load_full"),
    "🔋 Heavy": ("special", "load_heavy"),
    "⚖️ Shared": ("special", "load_shared"),
    "🔻 Minimal": ("special", "load_minimal"),
    # ── Wrapper start ──
    "▶️ Başlat": ("special", "start_kutai"),
    # ── Back ──
    "🔙 Geri": ("special", "back"),
}

# Prompt texts for cmd_args actions
_CMD_ARG_PROMPTS: dict[str, str] = {
    "shop": "🔍 Ne arıyorsun?",
    "research_product": "🔬 Hangi ürünü araştıralım?",
    "todo": "📝 Ne ekleyelim?",
    "mission": "🎯 Ne yapılsın?",
}

# Pending action timeout (seconds)
_PENDING_ACTION_TIMEOUT = 300

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


def _format_log_entries(lines: list[str], n: int = 20) -> str:
    """Format the last N log lines for Telegram display."""
    if not lines:
        return "📋 No log entries found."

    import json as _json
    last_n = lines[-n:]
    formatted = []
    for line in last_n:
        line = line.strip()
        if not line:
            continue
        try:
            entry = _json.loads(line)
            ts = entry.get("timestamp", "?")
            if "T" in ts:
                ts = ts.split("T")[1][:8]
            elif " " in ts:
                ts = ts.split(" ")[1][:8]
            level = entry.get("level", "?")[:4]
            comp = entry.get("component", "?").split(".")[-1]
            msg = entry.get("message", "")[:120]
            icon = {"ERRO": "🔴", "CRIT": "🔴", "WARN": "🟡", "INFO": "⚪", "DEBU": "⚫"}.get(level, "⚪")
            # Escape Markdown special chars in log message to prevent parse errors
            msg = msg.replace("*", "\\*").replace("_", "\\_").replace("`", "\\`").replace("[", "\\[")
            formatted.append(f"{icon} `{ts}` *{comp}*: {msg}")
        except (ValueError, KeyError):
            formatted.append(f"⚫ {line[:120]}")

    if not formatted:
        return "📋 No log entries found."

    return "\n".join(formatted)


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
        # Reverse lookup: message_id → task_id for reply-to detection
        self._clarification_msg_ids: dict[int, int] = {}
        # Conversation flow: chat_id → {"command": str, "ts": float} for button-initiated arg prompts
        self._pending_action: dict[int, dict] = {}
        # Track current keyboard state per chat: chat_id → state name
        self._kb_state: dict[int, str] = {}
        # Store mission description during workflow selection
        self._pending_mission: dict[int, str] = {}

    async def restore_clarification_state(self):
        """Rebuild _pending_clarifications and Q&A queues from DB after restart.

        Called once after Telegram polling starts. For any task in
        waiting_human, restore the in-memory tracking and re-send
        the pending question so the user sees it again.
        """
        try:
            db = await get_db()
            cursor = await db.execute(
                """SELECT id, title, context FROM tasks
                   WHERE status = 'waiting_human'
                   ORDER BY created_at DESC"""
            )
            rows = await cursor.fetchall()
            if not rows:
                return

            import json as _json
            chat_id = int(TELEGRAM_ADMIN_CHAT_ID) if TELEGRAM_ADMIN_CHAT_ID else None
            if not chat_id:
                return

            for row in rows:
                task = dict(row)
                task_id = task["id"]
                title = task.get("title", "")
                self._pending_clarifications[chat_id] = task_id

                # Check for persisted Q&A queue
                ctx = task.get("context", "{}")
                if isinstance(ctx, str):
                    try:
                        ctx = _json.loads(ctx)
                    except (ValueError, TypeError):
                        ctx = {}

                qa_state = ctx.get("_clarification_queue")
                if qa_state and isinstance(qa_state, dict):
                    questions = qa_state.get("questions", [])
                    current = qa_state.get("current", 0)
                    answers = qa_state.get("answers", [])
                    if current < len(questions):
                        # Restore sequential Q&A — re-send current question
                        self._pending_clarification_queue = {
                            "task_id": task_id,
                            "title": title,
                            "questions": questions,
                            "current": current,
                            "answers": answers,
                        }
                        progress = f" ({current}/{len(questions)} answered)" if current else ""
                        await self.send_notification(
                            f"\u2753 *Clarification pending — Task #{task_id}*\n"
                            f"**{title}**{progress}\n\n"
                            f"_(Restored after restart)_"
                        )
                        await self._send_next_clarification_question()
                        logger.info("Restored sequential Q&A state",
                                    task_id=task_id, current=current,
                                    total=len(questions))
                        break  # Only one Q&A queue at a time
                else:
                    # Single question — re-send the original question
                    clarification_q = ctx.get("_clarification_question", "")
                    if clarification_q:
                        msg = await self.send_notification(
                            f"\u2753 *Clarification pending — Task #{task_id}*\n"
                            f"**{title}**\n\n"
                            f"{clarification_q}\n\n"
                            f"_Reply to this message or just type your answer._"
                        )
                        if msg:
                            self._clarification_msg_ids[msg.message_id] = task_id
                        logger.info("Restored clarification state",
                                    task_id=task_id)
                    else:
                        # No saved question text — just note it's pending
                        logger.info("Found waiting_human task without "
                                    "saved question", task_id=task_id)
                break  # One clarification at a time

        except Exception as e:
            logger.error("Failed to restore clarification state", error=str(e))

    async def _reply(self, update_or_msg, text: str, **kwargs):
        """Send a reply with the current keyboard state.

        If the caller supplies its own ``reply_markup`` (e.g. an
        InlineKeyboardMarkup) we pass it through instead so inline buttons
        still work.  Every other reply gets the current keyboard for the chat.
        """
        if "reply_markup" not in kwargs:
            # Use current keyboard state, default to REPLY_KEYBOARD
            chat_id = None
            msg = getattr(update_or_msg, "message", update_or_msg)
            if hasattr(msg, "chat"):
                chat_id = msg.chat.id
            kwargs["reply_markup"] = self._get_current_keyboard(chat_id)
        # Accept either an Update or a Message object
        msg = getattr(update_or_msg, "message", update_or_msg)
        try:
            return await msg.reply_text(text, **kwargs)
        except Exception as e:
            # Markdown parse error — retry without parse_mode
            if "parse_mode" in kwargs and "parse entities" in str(e).lower():
                kwargs.pop("parse_mode")
                return await msg.reply_text(text, **kwargs)
            raise

    def _get_current_keyboard(self, chat_id: int | None) -> ReplyKeyboardMarkup:
        """Return the keyboard for the current navigation state."""
        if chat_id is None:
            return REPLY_KEYBOARD
        state = self._kb_state.get(chat_id, "main")
        return {
            "main": REPLY_KEYBOARD,
            "hizmet": KB_HIZMET,
            "alisveris": KB_ALISVERIS,
            "listem": KB_LISTEM,
            "gorevler": KB_GOREVLER,
            "sistem": KB_SISTEM,
            "workflow_select": KB_WORKFLOW_SELECT,
            "yuk_modu": KB_YUK_MODU,
            "debug": KB_SISTEM,
            "dlq": KB_SISTEM,
        }.get(state, REPLY_KEYBOARD)

    async def _swap_keyboard(self, update, state: str, text: str = None, **kwargs):
        """Switch the reply keyboard to a new state and optionally send a message."""
        chat_id = update.effective_chat.id
        self._kb_state[chat_id] = state
        kb = self._get_current_keyboard(chat_id)
        if text:
            msg = getattr(update, "message", update)
            return await msg.reply_text(text, reply_markup=kb, **kwargs)
        # Send a minimal message to trigger keyboard swap
        msg = getattr(update, "message", update)
        return await msg.reply_text("⌨️", reply_markup=kb)

    def _resolve_cmd_handler(self, cmd: str):
        """Resolve a command string to its handler method."""
        method_name = _CMD_METHOD_MAP.get(cmd, f"cmd_{cmd}")
        return getattr(self, method_name, None)

    # ── Category & Special Button Handlers ────────────────────────────────

    async def _handle_category_button(self, update, context, category: str):
        """Handle a category button tap — swap keyboard and show auto-content."""
        chat_id = update.effective_chat.id

        if category == "listem":
            # Auto-list todos
            self._kb_state[chat_id] = "listem"
            try:
                from src.app.reminders import build_todo_list_message
                text, markup = await build_todo_list_message()
                if text:
                    await update.message.reply_text(
                        text, parse_mode="Markdown", reply_markup=markup,
                    )
                else:
                    await update.message.reply_text(
                        "📋 *Listem*\n\nHenüz bir şey yok.",
                        parse_mode="Markdown",
                    )
            except Exception as e:
                logger.error("Failed to load todos", error=str(e))
                await update.message.reply_text("📋 *Listem*\n\nTodo listesi yüklenemedi.",
                                                parse_mode="Markdown")
            # Send keyboard swap message
            await update.message.reply_text("⌨️", reply_markup=KB_LISTEM)
            return

        if category == "gorevler":
            # Auto-list active missions
            self._kb_state[chat_id] = "gorevler"
            try:
                missions = await get_active_missions()
                if missions:
                    lines = ["🎯 *Görevler*\n"]
                    buttons = []
                    for i, m in enumerate(missions[:10], 1):
                        status_icon = {"running": "🔄", "pending": "⏳",
                                       "waiting_human": "❓", "ungraded": "⏳"
                                       }.get(m.get("status", ""), "📋")
                        title = m.get("title", m.get("description", "?"))[:50]
                        lines.append(f"{i}. {status_icon} {title}")
                        buttons.append(InlineKeyboardButton(
                            f"{i}", callback_data=f"m:task:detail:{m['id']}"))
                    # Arrange buttons in rows of 5
                    btn_rows = [buttons[j:j+5] for j in range(0, len(buttons), 5)]
                    await update.message.reply_text(
                        "\n".join(lines), parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(btn_rows) if btn_rows else None,
                    )
                else:
                    await update.message.reply_text(
                        "🎯 *Görevler*\n\nAktif görev yok.",
                        parse_mode="Markdown",
                    )
            except Exception as e:
                logger.error("Failed to load missions", error=str(e))
                await update.message.reply_text("🎯 *Görevler*\n\nGörev listesi yüklenemedi.",
                                                parse_mode="Markdown")
            await update.message.reply_text("⌨️", reply_markup=KB_GOREVLER)
            return

        if category == "sistem":
            # Auto-print dashboard
            self._kb_state[chat_id] = "sistem"
            dashboard = await self._build_system_dashboard()
            await update.message.reply_text(dashboard, parse_mode="Markdown")
            await update.message.reply_text("⌨️", reply_markup=KB_SISTEM)
            return

        # Simple category swap (hizmet, alisveris, yuk_modu)
        kb_map = {
            "hizmet": KB_HIZMET,
            "alisveris": KB_ALISVERIS,
            "yuk_modu": KB_YUK_MODU,
        }
        kb = kb_map.get(category)
        if kb:
            self._kb_state[chat_id] = category
            await update.message.reply_text("⌨️", reply_markup=kb)

    async def _handle_special_button(self, update, context, action: str):
        """Handle special button actions (quick services, workflow, lifecycle, etc.)."""
        chat_id = update.effective_chat.id

        # ── Back button ──
        if action == "back":
            current_state = self._kb_state.get(chat_id, "main")
            parent_kb = _KB_PARENT.get(current_state, REPLY_KEYBOARD)
            # Find parent state name
            parent_state = "main"
            for state_name, kb in [("hizmet", KB_HIZMET), ("alisveris", KB_ALISVERIS),
                                    ("listem", KB_LISTEM), ("gorevler", KB_GOREVLER),
                                    ("sistem", KB_SISTEM), ("yuk_modu", KB_YUK_MODU)]:
                if parent_kb is kb or (state_name in _KB_PARENT and _KB_PARENT.get(current_state) is kb):
                    parent_state = state_name
                    break
            if parent_kb is REPLY_KEYBOARD:
                parent_state = "main"
            self._kb_state[chat_id] = parent_state
            await update.message.reply_text("⌨️", reply_markup=parent_kb)
            return

        # ── Quick Services ──
        if action in ("pharmacy", "exchange", "weather", "fuel", "prayer",
                      "news", "gold", "earthquake"):
            await self._handle_quick_service(update, context, action)
            return

        # ── New Mission (workflow selection flow) ──
        if action == "new_mission":
            self._pending_action[chat_id] = {
                "command": "_workflow_select",
                "ts": _time.time(),
            }
            await self._reply(update, "🎯 Ne yapılsın?")
            return

        # ── Workflow selection buttons ──
        if action.startswith("wf_"):
            mission_desc = self._pending_mission.pop(chat_id, None)
            if not mission_desc:
                await self._reply(update, "❌ Görev açıklaması bulunamadı. Tekrar dene.")
                self._kb_state[chat_id] = "gorevler"
                await update.message.reply_text("⌨️", reply_markup=KB_GOREVLER)
                return
            await self._create_mission_with_workflow(update, context, mission_desc, action)
            return

        # ── Debug ──
        if action == "debug":
            self._kb_state[chat_id] = "debug"
            await self._show_debug_tasks(update, context)
            return

        # ── DLQ ──
        if action == "dlq":
            self._kb_state[chat_id] = "dlq"
            await self._show_dlq_tasks(update, context)
            return

        # ── Location ──
        if action == "location":
            from src.infra.db import get_user_pref
            lat = await get_user_pref("location_lat")
            district = await get_user_pref("location_district") or ""
            city = await get_user_pref("location_city") or ""
            if lat:
                loc_desc = f"{district}, {city}" if district else f"{lat}"
                await update.message.reply_text(
                    f"📍 Kayıtlı konum: *{loc_desc}*\n\n"
                    f"Değiştirmek için aşağıdan seç veya\n"
                    f"Google Maps linki / koordinat gönder:",
                    parse_mode="Markdown",
                    reply_markup=ReplyKeyboardMarkup(
                        [[KeyboardButton("📍 Konumumu Paylaş", request_location=True)],
                         [KeyboardButton("✏️ İlçe Adı Yaz")],
                         [KeyboardButton("❌ İptal")]],
                        resize_keyboard=True,
                        one_time_keyboard=True,
                    ),
                )
                self._pending_action[chat_id] = {
                    "command": "_location_setup",
                    "original_service": None,
                    "ts": _time.time(),
                }
            else:
                await self._start_location_setup(update, None)
            return

        # ── Processes ──
        if action == "processes":
            await self._show_processes(update, context)
            return

        # ── Reset Tasks ──
        if action == "reset_tasks":
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("🗑 Evet, sil", callback_data="reset_tasks_confirm"),
                InlineKeyboardButton("İptal", callback_data="reset_tasks_cancel"),
            ]])
            await update.message.reply_text(
                "🗑 Tüm görevler ve misyonlar silinecek. Emin misin?",
                reply_markup=keyboard,
            )
            return

        # ── GPU Load Modes ──
        if action.startswith("load_"):
            mode = action.replace("load_", "")
            if mode == "auto":
                context.args = ["auto"]
            else:
                context.args = [mode]
            await self.cmd_load(update, context)
            return

        # ── Restart / Stop ──
        if action == "restart":
            await update.message.reply_text(
                "⚠️ Kutay yeniden başlatılsın mı?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ Evet", callback_data="m:confirm:restart"),
                     InlineKeyboardButton("❌ Hayır", callback_data="m:confirm:cancel")],
                ]),
            )
            return

        if action == "stop":
            await update.message.reply_text(
                "⚠️ Kutay durdurulsun mu?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ Evet", callback_data="m:confirm:stop"),
                     InlineKeyboardButton("❌ Hayır", callback_data="m:confirm:cancel")],
                ]),
            )
            return

        if action == "claude_code":
            context.args = []
            await self.cmd_claude(update, context)
            return

        if action == "start_kutai":
            # Handled by wrapper — just send the text so wrapper picks it up
            await update.message.reply_text("▶️ Başlatılıyor...")
            return

        # ── Reminder (one-shot) ──
        if action == "reminder":
            self._pending_action[chat_id] = {
                "command": "_reminder_time",
                "ts": _time.time(),
            }
            await self._reply(update,
                "⏰ Ne zaman hatırlatılsın?\n\n"
                "Örnekler:\n"
                "• 10dk\n"
                "• 1 saat\n"
                "• yarın 09:00\n"
                "• 14:30"
            )
            return

        # ── Schedule Task (recurring) ──
        if action == "schedule_task":
            try:
                from src.infra.db import get_scheduled_tasks
                tasks = await get_scheduled_tasks()
            except Exception:
                tasks = []
            lines = ["⏰ *Zamanlanmış Görevler*\n"]
            if tasks:
                day_map = {
                    "1": "Pzt", "2": "Sal", "3": "Çar",
                    "4": "Per", "5": "Cum", "6": "Cmt", "0": "Paz",
                }
                for i, t in enumerate(tasks, 1):
                    cron = t.get("cron_expression", "")
                    enabled = t.get("enabled", 1)
                    status = "✅" if enabled else "⏸"
                    # Human-readable cron summary
                    parts = cron.strip().split() if cron else []
                    if len(parts) == 5:
                        m, h, dom, mo, dow = parts
                        if h != "*" and m != "*" and "/" not in h and "," not in h:
                            if dow != "*":
                                day_label = day_map.get(dow, dow)
                                schedule_str = f"her {day_label} {h}:{m.zfill(2)}"
                            else:
                                schedule_str = f"her gün {h}:{m.zfill(2)}"
                        elif "/" in h:
                            interval = h.split("/")[1]
                            schedule_str = f"her {interval} saatte"
                        elif h == "*" and m != "*":
                            schedule_str = f"her saat {m}. dakikada"
                        else:
                            schedule_str = cron
                    else:
                        schedule_str = cron or "?"
                    lines.append(f"{i}. {status} {t['title']} — {schedule_str}")
            else:
                lines.append("_(henüz görev yok)_")
            lines.append("\n[Yeni eklemek için açıklama yaz]")
            self._pending_action[chat_id] = {
                "command": "_schedule_desc",
                "ts": _time.time(),
            }
            await self._reply(update, "\n".join(lines))
            return

        logger.warning("Unhandled special button", action=action)

    # ── Reminder / Schedule helpers ────────────────────────────────────────

    @staticmethod
    def _parse_maps_url(text: str) -> tuple[float, float] | None:
        """Extract lat/lon from a Google Maps URL or coordinate pair (sync).

        Supports:
        - https://maps.google.com/?q=39.95,32.84
        - https://www.google.com/maps/@39.95,32.84,15z
        - https://www.google.com/maps/place/.../@39.95,32.84,15z
        - Plain "39.95, 32.84" coordinate pair
        """
        import re
        text = text.strip()

        # Direct coordinate pair: "39.95, 32.84" or "39.95,32.84"
        m = re.match(r"^(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)$", text)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon

        # Google Maps @lat,lon pattern
        m = re.search(r"@(-?\d+\.?\d+),(-?\d+\.?\d+)", text)
        if m:
            return float(m.group(1)), float(m.group(2))

        # Google Maps ?q=lat,lon or query=lat,lon
        m = re.search(r"[?&]q(?:uery)?=(-?\d+\.?\d+),(-?\d+\.?\d+)", text)
        if m:
            return float(m.group(1)), float(m.group(2))

        # Google Maps /dir/ or place with coords in path
        m = re.search(r"/(-?\d+\.?\d+),(-?\d+\.?\d+)", text)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon

        return None

    @staticmethod
    async def _resolve_maps_url(text: str) -> tuple[float, float] | None:
        """Extract lat/lon, resolving short URLs (goo.gl, maps.app.goo.gl) if needed."""
        import re

        # Extract URL from text (user might send "address\nhttps://...")
        url_match = re.search(r"https?://\S+", text)
        url = url_match.group(0) if url_match else text.strip()

        # Try direct parsing first
        result = TelegramInterface._parse_maps_url(url)
        if result:
            return result

        # Short URL or maps link without coords? Resolve and extract from page
        if re.match(r"https?://(goo\.gl|maps\.app\.goo\.gl|bit\.ly|maps\.google|www\.google\.com/maps)/", url):
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, allow_redirects=True,
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        # Try the resolved URL first
                        resolved = str(resp.url)
                        result = TelegramInterface._parse_maps_url(resolved)
                        if result:
                            return result
                        # Extract coords from page content (center=lat%2Clon)
                        body = await resp.text()
                        m = re.search(r"center=(-?\d+\.\d+)%2C(-?\d+\.\d+)", body)
                        if m:
                            return float(m.group(1)), float(m.group(2))
                        # Fallback: [null,null,lat,lon] pattern
                        m = re.search(r"\[null,null,(-?\d+\.\d{4,}),(-?\d+\.\d{4,})\]", body)
                        if m:
                            return float(m.group(1)), float(m.group(2))
            except Exception:
                pass

        # Try the full text (might be "addr\nhttps://full-url")
        if url != text.strip():
            result = TelegramInterface._parse_maps_url(text.strip())
            if result:
                return result

        return None

    @staticmethod
    def _parse_reminder_time(text: str):
        """Parse a Turkish time expression into a datetime.

        Supported formats:
        - "10dk" / "10 dakika" / "10d" → now + N minutes
        - "1 saat" / "1s" / "2s" → now + N hours
        - "yarın 09:00" → tomorrow at HH:MM
        - "14:30" → today at 14:30 (or tomorrow if already past)
        - bare number without unit → treat as minutes

        Returns a datetime or None if parsing fails.
        """
        from datetime import datetime, timedelta, timezone
        import re

        text = text.strip().lower()
        # Work in Turkey local time for user-facing absolute times (HH:MM, yarın).
        # For relative offsets (N dk, N saat) we use UTC directly.
        now_utc = utc_now()
        now_local = now_utc.astimezone(TZ_TR)

        def _to_utc_naive(dt):
            """Convert a local (Turkey) datetime to a UTC naive datetime for DB storage."""
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            # dt is naive local Turkey time — attach tz then convert
            return dt.replace(tzinfo=TZ_TR).astimezone(timezone.utc).replace(tzinfo=None)

        # Turkish number words → digits
        _TR_NUMS = {
            "bir": 1, "iki": 2, "üç": 3, "uc": 3, "dört": 4, "dort": 4,
            "beş": 5, "bes": 5, "altı": 6, "alti": 6, "yedi": 7,
            "sekiz": 8, "dokuz": 9, "on": 10, "yarım": 0.5, "yarim": 0.5,
            "çeyrek": 0.25, "ceyrek": 0.25,
        }

        # "yarım saat" / "bir saat" / "iki dakika" etc.
        # Relative offsets: add to UTC now, return UTC naive
        m = re.match(r"^(\w+)\s+(?:saat|s)$", text)
        if m and m.group(1) in _TR_NUMS:
            val = _TR_NUMS[m.group(1)]
            return (now_utc + timedelta(hours=val)).replace(tzinfo=None)

        m = re.match(r"^(\w+)\s+(?:dk|dakika)$", text)
        if m and m.group(1) in _TR_NUMS:
            val = _TR_NUMS[m.group(1)]
            return (now_utc + timedelta(minutes=val)).replace(tzinfo=None)

        # N dakika / Ndk / Nd
        m = re.match(r"^(\d+)\s*(?:dk|dakika|d(?:akika)?)$", text)
        if m:
            return (now_utc + timedelta(minutes=int(m.group(1)))).replace(tzinfo=None)

        # N saat / Ns
        m = re.match(r"^(\d+)\s*(?:saat|s(?:aat)?)$", text)
        if m:
            return (now_utc + timedelta(hours=int(m.group(1)))).replace(tzinfo=None)

        # yarın HH:MM — user gives Turkey local time; convert to UTC for storage
        m = re.match(r"^yarın\s+(\d{1,2}):(\d{2})$", text)
        if m:
            tomorrow_local = now_local.date() + timedelta(days=1)
            local_dt = datetime(tomorrow_local.year, tomorrow_local.month, tomorrow_local.day,
                                int(m.group(1)), int(m.group(2)))
            return _to_utc_naive(local_dt)

        # HH:MM (today local, or tomorrow if past) — convert to UTC for storage
        m = re.match(r"^(\d{1,2}):(\d{2})$", text)
        if m:
            candidate_local = now_local.replace(
                hour=int(m.group(1)), minute=int(m.group(2)),
                second=0, microsecond=0,
            )
            if hasattr(candidate_local, 'tzinfo') and candidate_local.tzinfo is None:
                candidate_local = candidate_local.replace(tzinfo=TZ_TR)
            if candidate_local <= now_local:
                candidate_local += timedelta(days=1)
            return _to_utc_naive(candidate_local)

        # bare integer → minutes (relative, UTC)
        m = re.match(r"^(\d+)$", text)
        if m:
            return (now_utc + timedelta(minutes=int(m.group(1)))).replace(tzinfo=None)

        return None

    @staticmethod
    def _parse_cron_input(text: str):
        """Parse a Turkish schedule description into a cron expression.

        All hour values the user provides are in Turkey local time (UTC+3).
        The cron is stored and evaluated in UTC by the scheduler, so we convert
        Turkey hours to UTC by subtracting 3.  Turkey has no DST (always UTC+3).

        Supported patterns:
        - "her gün 09:00"        → "0 6 * * *"   (09:00 TR = 06:00 UTC)
        - "her gün 14:30"        → "30 11 * * *"  (14:30 TR = 11:30 UTC)
        - "her 2 saatte"         → "0 */2 * * *"  (relative — no offset needed)
        - "her saat"             → "0 * * * *"
        - "her pazartesi"        → "0 6 * * 1"   (09:00 TR = 06:00 UTC)
        - "her pazartesi 14:00"  → "0 11 * * 1"
        - "her salı 10:30"       → "30 7 * * 2"

        Returns a cron string or None if parsing fails.
        """
        import re

        text = text.strip().lower()

        day_map = {
            "pazartesi": "1", "salı": "2", "çarşamba": "3",
            "perşembe": "4", "cuma": "5", "cumartesi": "6", "pazar": "0",
        }

        # her N saatte — relative interval, no timezone conversion needed
        m = re.match(r"^her\s+(\d+)\s*saatte?$", text)
        if m:
            return f"0 */{m.group(1)} * * *"

        # her saat — every hour, no timezone conversion needed
        if re.match(r"^her\s*saat$", text):
            return "0 * * * *"

        # her gün HH:MM — daily at Turkey-local time, convert to UTC
        m = re.match(r"^her\s+gün\s+(\d{1,2}):(\d{2})$", text)
        if m:
            utc_h = tr_hour_to_utc(int(m.group(1)))
            return f"{int(m.group(2))} {utc_h} * * *"

        # her gün (default 09:00 TR = 06:00 UTC)
        if re.match(r"^her\s+gün$", text):
            return f"0 {tr_hour_to_utc(9)} * * *"

        # her <weekday> HH:MM — convert Turkey hour to UTC
        for day_name, day_num in day_map.items():
            m = re.match(rf"^her\s+{re.escape(day_name)}\s+(\d{{1,2}}):(\d{{2}})$", text)
            if m:
                utc_h = tr_hour_to_utc(int(m.group(1)))
                return f"{int(m.group(2))} {utc_h} * * {day_num}"
            if re.match(rf"^her\s+{re.escape(day_name)}$", text):
                return f"0 {tr_hour_to_utc(9)} * * {day_num}"

        return None

    async def _build_system_dashboard(self) -> str:
        """Build the system status dashboard text."""
        lines = ["📊 *Kutay Durum*\n━━━━━━━━━━━━━━━━━━━━"]
        try:
            # Model info
            try:
                from src.models.local_model_manager import get_local_manager
                lmm = get_local_manager()
            except Exception:
                lmm = None
            if lmm and lmm.current_model:
                    model_name = lmm.current_model
                    lines.append(f"🤖 Model: {model_name}")
                    if hasattr(lmm, 'runtime_state') and lmm.runtime_state:
                        rs = lmm.runtime_state
                        tps = getattr(rs, 'measured_tps', None)
                        gpu = getattr(rs, 'gpu_layers', None)
                        thinking = getattr(rs, 'thinking_enabled', None)
                        parts = []
                        if tps:
                            parts.append(f"{tps:.1f} t/s")
                        if gpu:
                            parts.append(f"GPU: {gpu} katman")
                        if thinking is not None:
                            parts.append(f"Thinking: {'ON' if thinking else 'OFF'}")
                        if parts:
                            lines.append(f"   {' | '.join(parts)}")
            else:
                lines.append("🤖 Model: yüklü değil")

            # Load mode
            try:
                from src.infra.load_manager import get_load_mode
                load_mode = await get_load_mode()
                lines.append(f"🖥 Yük Modu: {load_mode}")
            except Exception:
                pass

            # Queue stats
            try:
                ready = await get_ready_tasks()
                stats = await get_daily_stats()
                pending = len(ready) if ready else 0
                completed = stats.get("completed", 0) if stats else 0
                failed = stats.get("failed", 0) if stats else 0
                lines.append(f"\n📋 Kuyruk: {pending} bekleyen")
                lines.append(f"   Bugün: {completed} tamamlandı | {failed} başarısız")
            except Exception:
                lines.append("\n📋 Kuyruk: bilgi alınamadı")

            # Cost
            try:
                budget_info = await get_budget()
                if budget_info:
                    spent = budget_info.get("spent_today", 0)
                    limit = budget_info.get("daily_limit", 0)
                    lines.append(f"💰 Bugün: ${spent:.2f} / ${limit:.2f}")
            except Exception:
                pass

            # Uptime
            if self.orchestrator and hasattr(self.orchestrator, 'start_time'):
                import time
                uptime_s = time.time() - self.orchestrator.start_time
                hours = int(uptime_s // 3600)
                mins = int((uptime_s % 3600) // 60)
                lines.append(f"⏱ Çalışma: {hours}s {mins}dk")

        except Exception as e:
            lines.append(f"\n❌ Dashboard hatası: {_friendly_error(str(e))}")

        return "\n".join(lines)

    async def _show_debug_tasks(self, update, context):
        """Show recent tasks for debugging (all statuses)."""
        try:
            db = await get_db()
            cursor = await db.execute(
                """SELECT id, title, description, agent_type, status,
                          COALESCE(completed_at, started_at, created_at) AS last_ts
                   FROM tasks
                   ORDER BY COALESCE(completed_at, started_at, created_at) DESC
                   LIMIT 10"""
            )
            tasks = [dict(r) for r in await cursor.fetchall()]
            if not tasks:
                await self._reply(update, "🐛 Son görev bulunamadı.")
                return
            lines = ["🐛 Son Görevler\n"]
            buttons = []
            for i, t in enumerate(tasks, 1):
                title = (t.get("title") or t.get("description", "?"))[:40]
                agent = t.get("agent_type", "?")
                status_icon = {"completed": "✅", "failed": "❌", "running": "🔄",
                               "pending": "⏳", "ungraded": "⏳",
                               "waiting_human": "❓"
                               }.get(t.get("status", ""), "📋")
                # Time ago
                time_str = ""
                try:
                    updated = t.get("last_ts", "")
                    if updated:
                        updated_dt = datetime.fromisoformat(str(updated).replace(" ", "T"))
                        ago = (utc_now().replace(tzinfo=None) - updated_dt).total_seconds()
                        if ago < 60:
                            time_str = f"{int(ago)}sn önce"
                        elif ago < 3600:
                            time_str = f"{int(ago/60)}dk önce"
                        else:
                            time_str = f"{int(ago/3600)}s önce"
                except Exception:
                    pass
                lines.append(f"{i}. {status_icon} {title} — {agent} — {time_str}")
                buttons.append(InlineKeyboardButton(
                    f"{i}", callback_data=f"m:debug:detail:{t['id']}"))
            btn_rows = [buttons[j:j+5] for j in range(0, len(buttons), 5)]
            btn_rows.append([InlineKeyboardButton("📊 Skill Metrikleri", callback_data="m:debug:skillstats")])
            await update.message.reply_text(
                "\n".join(lines),
                reply_markup=InlineKeyboardMarkup(btn_rows),
            )
        except Exception as e:
            logger.error("Debug tasks failed", error=str(e))
            await self._reply(update, f"❌ {_friendly_error(str(e))}")

    async def _show_dlq_tasks(self, update, context):
        """Show quarantined tasks from the dead-letter queue."""
        try:
            from ..infra.dead_letter import get_dlq_tasks
            dlq_entries = await get_dlq_tasks(unresolved_only=True)
            if not dlq_entries:
                await self._reply(update, "📭 DLQ\n\nBaşarısız görev yok.")
                return
            lines = ["📭 Dead-Letter Queue\n"]
            buttons = []
            for i, entry in enumerate(dlq_entries[:10], 1):
                task_id = entry["task_id"]
                task = await get_task(task_id)
                title = (task.get("title") or "?")[:40] if task else f"Task #{task_id}"
                cat = entry.get("error_category", "unknown")
                error_short = (entry.get("error") or "")[:30]
                lines.append(f"{i}. ❌ {title} [{cat}]")
                if error_short:
                    lines.append(f"   {error_short}")
                buttons.append(InlineKeyboardButton(
                    f"{i}", callback_data=f"m:dlq:detail:{task_id}"))
            btn_rows = [buttons[j:j+5] for j in range(0, len(buttons), 5)]
            await update.message.reply_text(
                "\n".join(lines),
                reply_markup=InlineKeyboardMarkup(btn_rows) if btn_rows else None,
            )
        except Exception as e:
            logger.error("DLQ listing failed", error=str(e))
            await self._reply(update, f"❌ {_friendly_error(str(e))}")

    @staticmethod
    async def _check_yazbunu_health() -> str:
        """Check yazbunu health via HTTP and PID file."""
        import aiohttp
        yz_responding = False
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    "http://127.0.0.1:9880/",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as r:
                    yz_responding = r.status == 200
        except Exception:
            pass
        yz_pid = None
        try:
            pid_file = Path("logs/yazbunu.pid")
            pid = int(pid_file.read_text().strip())
            # Check alive
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if h:
                k32.CloseHandle(h)
                yz_pid = pid
        except Exception:
            pass
        if yz_responding:
            pid_str = f", PID {yz_pid}" if yz_pid else ""
            return f"📊 yazbunu: çalışıyor (port 9880{pid_str})"
        elif yz_pid:
            return f"🟠 yazbunu: süreç var ama yanıt yok (PID {yz_pid})"
        else:
            return "⚫ yazbunu: çalışmıyor"

    async def _build_proc_panel(self) -> tuple[str, list[list[InlineKeyboardButton]]]:
        """Build the Yaşar Usta status text and inline buttons."""
        import subprocess as _sp
        import time as _time

        lines = []
        wrappers = []
        orchestrators = []
        llama_pids = []

        # Python processes
        try:
            raw = _sp.check_output(
                ['wmic', 'process', 'where', "name='python.exe'",
                 'get', 'ProcessId,CommandLine'],
                text=True, timeout=5,
            )
            for line in raw.strip().splitlines():
                line = line.strip()
                if not line or line.startswith("CommandLine"):
                    continue
                pid = line.split()[-1] if line.split() else ""
                if "wrapper" in line.lower():
                    wrappers.append(pid)
                elif "run.py" in line:
                    orchestrators.append(pid)
        except Exception as e:
            lines.append(f"⚠️ Süreç listesi alınamadı: {e}")

        # llama-server
        try:
            llama_raw = _sp.check_output(
                ['tasklist', '/FI', 'IMAGENAME eq llama-server.exe'],
                text=True, timeout=5,
            )
            for ll in llama_raw.splitlines():
                if 'llama-server' in ll.lower():
                    parts = ll.split()
                    if len(parts) >= 2:
                        llama_pids.append(parts[1])
        except Exception:
            pass

        # Dedup: Windows venv stub + real Python show as 2 PIDs per process
        n_wrappers = (len(wrappers) + 1) // 2
        n_orchestrators = (len(orchestrators) + 1) // 2

        # ── Health: Yaşar Usta ──
        usta_line = f"🔵 Yaşar Usta: {n_wrappers} süreç"
        if n_wrappers > 1:
            usta_line += " ⚠️ duplicate!"

        # ── Health: Kutay ──
        kutay_healthy = False
        heartbeat_age = None
        for hb_path in ["logs/orchestrator.heartbeat", "logs/heartbeat"]:
            try:
                with open(hb_path, "r") as f:
                    last_beat = float(f.read().strip())
                heartbeat_age = _time.time() - last_beat
                kutay_healthy = heartbeat_age < 60
                break
            except (FileNotFoundError, ValueError):
                continue

        if n_orchestrators == 0:
            kutay_line = "💀 Kutay: çalışmıyor"
        elif kutay_healthy:
            age_str = f"{int(heartbeat_age)}sn önce" if heartbeat_age else ""
            kutay_line = f"💚 Kutay: sağlıklı (heartbeat {age_str})"
        elif heartbeat_age is not None:
            kutay_line = f"🔴 Kutay: YANIT VERMİYOR ({int(heartbeat_age)}sn sessiz)"
        else:
            kutay_line = "⚪ Kutay: heartbeat dosyası yok"

        # ── Health: llama-server ──
        if llama_pids:
            llama_line = f"🟡 llama-server: çalışıyor (PID {', '.join(llama_pids)})"
        else:
            llama_line = "⚫ llama-server: çalışmıyor"

        # ── Health: yazbunu ──
        yz_line = await self._check_yazbunu_health()

        ts = _time.strftime("%H:%M:%S")
        text = (
            f"🔧 *Yaşar Usta*\n\n"
            f"{usta_line}\n"
            f"{kutay_line}\n"
            f"{llama_line}\n"
            f"{yz_line}\n"
            f"\n_Son güncelleme: {ts}_"
        )

        # ── Buttons ──
        btn_rows = []
        if n_orchestrators > 0 and not kutay_healthy and heartbeat_age and heartbeat_age > 30:
            btn_rows.append([InlineKeyboardButton(
                f"☠️ Kutay'ı öldür ({int(heartbeat_age)}sn yanıtsız)",
                callback_data="m:proc:kill_kutai_only")])
        btn_rows.append([
            InlineKeyboardButton("♻️ Usta'yı Yeniden Başlat", callback_data="m:proc:kill_wrapper"),
            InlineKeyboardButton("📊 Yazbunu Yeniden Başlat", callback_data="m:proc:restart_yazbunu"),
        ])
        btn_rows.append([
            InlineKeyboardButton("🔃 Yenile", callback_data="m:proc:refresh"),
            InlineKeyboardButton("🔙 Geri", callback_data="m:proc:back"),
        ])

        return text, btn_rows

    async def _show_processes(self, update, context):
        """Show running processes, Kutay health, and management buttons."""
        try:
            text, btn_rows = await self._build_proc_panel()
            await update.message.reply_text(
                text, parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(btn_rows))
        except Exception as e:
            logger.error("Process check failed", error=str(e))
            await self._reply(update, f"❌ {_friendly_error(str(e))}")

    async def _handle_quick_service(self, update, context, service: str):
        """Handle quick service API calls (no LLM needed)."""
        # Check location for location-dependent services
        if service in ("pharmacy", "weather", "prayer"):
            try:
                from src.infra.db import get_user_pref
                lat = await get_user_pref("location_lat")
                if not lat:
                    await self._start_location_setup(update, service)
                    return
            except Exception:
                await self._start_location_setup(update, service)
                return

        service_handlers = {
            "pharmacy": self._quick_pharmacy,
            "exchange": self._quick_exchange,
            "weather": self._quick_weather,
            "fuel": self._quick_fuel,
            "prayer": self._quick_prayer,
            "news": self._quick_news,
            "gold": self._quick_gold,
            "earthquake": self._quick_earthquake,
        }
        handler = service_handlers.get(service)
        if handler:
            try:
                await handler(update, context)
            except Exception as e:
                logger.error(f"Quick service {service} failed", error=str(e))
                await self._reply(update, f"❌ {service} servisi şu an kullanılamıyor.")
        else:
            await self._reply(update, f"❌ Bilinmeyen servis: {service}")

    async def _start_location_setup(self, update, original_service: str):
        """Start the location setup flow for location-dependent services."""
        chat_id = update.effective_chat.id
        self._pending_action[chat_id] = {
            "command": "_location_setup",
            "original_service": original_service,
            "ts": _time.time(),
        }
        location_kb = ReplyKeyboardMarkup(
            [[KeyboardButton("📍 Konumumu Paylaş", request_location=True)],
             [KeyboardButton("✏️ İlçe Adı Yaz")],
             [KeyboardButton("❌ İptal")]],
            resize_keyboard=True,
            one_time_keyboard=True,
        )
        await update.message.reply_text(
            "📍 Konum bilgin henüz kayıtlı değil.\n"
            "Konumunu paylaş, ilçe adı yaz veya\n"
            "Google Maps linki gönder:",
            reply_markup=location_kb,
        )

    async def handle_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle shared GPS location from Telegram."""
        chat_id = update.effective_chat.id
        loc = update.message.location
        if not loc:
            return
        lat, lon = loc.latitude, loc.longitude
        # Reverse geocode to get district/city via Photon (privacy-first, no logging)
        district, city = await _reverse_geocode_photon(lat, lon)

        # Save location and restore keyboard immediately
        pending = self._pending_action.pop(chat_id, None)
        original_service = None
        if pending and pending.get("command") == "_location_setup":
            original_service = pending.get("original_service")

        # Restore keyboard state BEFORE sending any replies
        self._kb_state[chat_id] = "hizmet" if original_service else "main"
        restore_kb = KB_HIZMET if original_service else REPLY_KEYBOARD

        try:
            from src.infra.db import set_user_pref
            await set_user_pref("location_lat", str(lat))
            await set_user_pref("location_lon", str(lon))
            await set_user_pref("location_district", district)
            await set_user_pref("location_city", city)
            loc_desc = f"{district}, {city}" if district else f"{lat:.4f}, {lon:.4f}"
            await update.message.reply_text(
                f"📍 Konum kaydedildi: {loc_desc}",
                reply_markup=restore_kb,
            )
        except Exception as e:
            logger.error(f"Failed to save location: {e}")
            await update.message.reply_text("❌ Konum kaydedilemedi.", reply_markup=restore_kb)
            return

        # Resume original service if there was one
        if original_service:
            await self._handle_quick_service(update, context, original_service)

    async def _save_location_from_coords(self, update, context,
                                           lat: float, lon: float,
                                           original_service: str | None):
        """Reverse geocode coordinates, save to DB, and resume service if needed."""
        chat_id = update.effective_chat.id
        district, city = await _reverse_geocode_photon(lat, lon)

        from src.infra.db import set_user_pref
        await set_user_pref("location_lat", str(lat))
        await set_user_pref("location_lon", str(lon))
        await set_user_pref("location_district", district)
        await set_user_pref("location_city", city)
        loc_desc = f"{district}, {city}" if district else f"{lat:.4f}, {lon:.4f}"

        self._kb_state[chat_id] = "hizmet" if original_service else "main"
        restore_kb = KB_HIZMET if original_service else REPLY_KEYBOARD
        await update.message.reply_text(
            f"📍 Konum kaydedildi: {loc_desc}",
            reply_markup=restore_kb,
        )
        if original_service:
            await self._handle_quick_service(update, context, original_service)

    async def _geocode_district(self, update, context, district_text: str, original_service: str):
        """Geocode a district name via Photon (privacy-first) and save location."""
        chat_id = update.effective_chat.id
        try:
            import aiohttp
            query = f"{district_text}, Turkey"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://photon.komoot.io/api",
                    params={"q": query, "limit": "1", "lang": "default",
                            "osm_tag": "place"},
                    headers={"User-Agent": "KutAI/1.0"},
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Photon geocode HTTP {resp.status}")
                        raise RuntimeError(f"Photon geocode HTTP {resp.status}")
                    data = await resp.json()
                    features = data.get("features", [])
                    if features:
                        feat = features[0]
                        coords = feat["geometry"]["coordinates"]  # [lon, lat]
                        lat, lon = coords[1], coords[0]
                        props = feat.get("properties", {})
                        district = props.get("name", district_text)
                        city = props.get("city") or props.get("state") or ""
                        from src.infra.db import set_user_pref
                        await set_user_pref("location_lat", str(lat))
                        await set_user_pref("location_lon", str(lon))
                        await set_user_pref("location_district", district)
                        await set_user_pref("location_city", city)
                        await self._reply(update,
                            f"📍 Konum kaydedildi: {district}, {city}\n"
                            f"({lat:.4f}, {lon:.4f})")
                        if original_service:
                            self._kb_state[chat_id] = "hizmet"
                            await update.message.reply_text("⌨️", reply_markup=KB_HIZMET)
                            await self._handle_quick_service(update, context, original_service)
                        return
                    else:
                        await self._reply(update,
                            f"❌ '{district_text}' bulunamadı. Tekrar dene:")
                        self._pending_action[chat_id] = {
                            "command": "_location_district",
                            "original_service": original_service,
                            "ts": _time.time(),
                        }
                        return
        except Exception as e:
            logger.error(f"Geocode failed: {e}")
            await self._reply(update, f"❌ Konum araması başarısız: {_friendly_error(str(e))}")
        # Restore keyboard on failure
        self._kb_state[chat_id] = "hizmet"
        await update.message.reply_text("⌨️", reply_markup=KB_HIZMET)

    async def _create_mission_with_workflow(self, update, context, description: str, wf_action: str):
        """Create a mission with the selected workflow type."""
        chat_id = update.effective_chat.id
        workflow = None

        if wf_action == "wf_quick":
            # Single agent, no workflow
            workflow = None
        elif wf_action == "wf_research":
            workflow = "research"
        elif wf_action == "wf_project":
            workflow = "i2p_v3"
        elif wf_action == "wf_auto":
            # Let LLM decide — pass through to normal mission creation
            workflow = None  # Will be classified by _classify_user_message
        elif wf_action == "wf_other":
            await self._reply(update,
                "🚧 Bu iş akışları henüz menüden desteklenmiyor.\n"
                "Görevini yazarak gönder, LLM doğru akışa yönlendirecek.\n\n"
                "Örnekler:\n"
                "• \"router.py'deki hatayı düzelt\"\n"
                "• \"db modülünü refaktör et\"\n"
                "• \"agent base sınıfını dokümante et\""
            )
            self._kb_state[chat_id] = "gorevler"
            await update.message.reply_text("⌨️", reply_markup=KB_GOREVLER)
            return

        # Create the mission
        try:
            if wf_action == "wf_auto":
                # Use existing mission creation with LLM classification
                context.args = description.split()
                await self.cmd_mission(update, context)
            elif workflow == "i2p_v3":
                # i2p workflow — must go through WorkflowRunner, not plain task
                from src.workflows.engine.runner import WorkflowRunner
                runner = WorkflowRunner()
                mission_id = await runner.start(
                    workflow_name="i2p_v3",
                    initial_input={"raw_idea": description, "product_name": description[:50]},
                    title=description[:80],
                )
                await self._reply(update,
                    f"🔄 Workflow mission #{mission_id} oluşturuldu!\n"
                    f"_{description[:60]}_\n\n"
                    f"`/wfstatus {mission_id}` ile takip edebilirsin.",
                    parse_mode="Markdown")
            elif workflow:
                # Other workflows (research, etc.) — plain task with workflow context
                mission_id = await add_mission(title=description[:80], description=description)
                await add_task(description[:80], description, mission_id=mission_id,
                              priority=5,
                              context={"workflow": workflow})
                await self._reply(update,
                    f"✅ Görev oluşturuldu (#{mission_id})\n"
                    f"İş akışı: {workflow}\n"
                    f"📋 {description[:100]}")
            else:
                # Quick single task
                mission_id = await add_mission(title=description[:80], description=description)
                await add_task(description[:80], description, mission_id=mission_id, priority=5)
                await self._reply(update,
                    f"✅ Görev oluşturuldu (#{mission_id})\n"
                    f"📋 {description[:100]}")
        except Exception as e:
            logger.error("Mission creation failed", error=str(e))
            await self._reply(update, f"❌ Görev oluşturulamadı: {_friendly_error(str(e))}")

        self._kb_state[chat_id] = "gorevler"
        await update.message.reply_text("⌨️", reply_markup=KB_GOREVLER)

    # ── Quick Service Implementations ─────────────────────────────────────

    async def _quick_pharmacy(self, update, context):
        """Find pharmacies on duty with Google Maps buttons."""
        from src.infra.db import get_user_pref, set_user_pref
        district = await get_user_pref("location_district")
        city = await get_user_pref("location_city")
        lat = float(await get_user_pref("location_lat") or 0)
        lon = float(await get_user_pref("location_lon") or 0)
        # If coords exist but city is missing, reverse geocode now
        if lat and lon and not city:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    url = (f"https://api.bigdatacloud.net/data/reverse-geocode-client"
                           f"?latitude={lat}&longitude={lon}&localityLanguage=tr")
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            city = data.get("city", "") or data.get("principalSubdivision", "")
                            district = data.get("locality", "") or district
                            if city:
                                await set_user_pref("location_city", city)
                            if district:
                                await set_user_pref("location_district", district)
            except Exception:
                pass
        loc_str = f"{district}, {city}" if district else city
        if not city:
            await self._reply(update, "❌ Konum bilgisi eksik. 📍 Konum butonundan güncelle.")
            return
        await self._reply(update, f"🏥 Nöbetçi eczaneler aranıyor... (📍 {loc_str})")
        try:
            from src.tools.pharmacy import (
                find_pharmacies_structured, format_pharmacy_message,
                build_pharmacy_buttons,
            )
            pharmacies = await find_pharmacies_structured(
                city=city, include_route=True,
                user_lat=lat, user_lon=lon,
            )
            if pharmacies:
                header = f"🏥 Nöbetçi Eczaneler — {city.title()}"
                if district:
                    header += f" / {district.title()}"
                text = header + "\n\n" + format_pharmacy_message(pharmacies, show_all=False)
                btn_rows = build_pharmacy_buttons(pharmacies[:3], len(pharmacies))
                markup = InlineKeyboardMarkup(btn_rows) if btn_rows else None
                await update.message.reply_text(text, reply_markup=markup)
            else:
                # Fallback to plain text
                from src.tools.pharmacy import find_nearest_pharmacy
                result = await find_nearest_pharmacy(city=city, district=district)
                await self._reply(update, result or "Nöbetçi eczane bulunamadı.")
        except Exception as e:
            await self._reply(update, f"❌ Eczane araması başarısız: {_friendly_error(str(e))}")

    async def _call_api_by_name(self, api_name: str, endpoint: str = None, params: dict = None) -> str:
        """Look up a free API by name and call it."""
        from src.tools.free_apis import get_api, call_api
        api = get_api(api_name)
        if not api:
            return f"❌ API '{api_name}' bulunamadı."
        return await call_api(api, endpoint=endpoint, params=params)

    async def _fetch_truncgil(self) -> dict | None:
        """Fetch live financial data from finans.truncgil.com."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://finans.truncgil.com/today.json",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json(content_type=None)
        except Exception:
            pass
        return None

    @staticmethod
    def _change_arrow(change_str: str) -> str:
        """Convert '%0,03' or '%-0,64' to a colored arrow."""
        if not change_str:
            return ""
        try:
            val = float(change_str.replace("%", "").replace(",", "."))
            if val > 0:
                return f" ↑{change_str}"
            elif val < 0:
                return f" ↓{change_str}"
        except ValueError:
            pass
        return f" {change_str}"

    async def _quick_exchange(self, update, context):
        """Get live exchange rates from truncgil."""
        await self._reply(update, "💰 Döviz kurları alınıyor...")
        try:
            data = await self._fetch_truncgil()
            if not data:
                await self._reply(update, "❌ Döviz kuru kaynağına ulaşılamadı.")
                return

            ts = data.get("Update_Date", "?")
            _currencies = [
                ("USD", "🇺🇸", "Amerikan Doları"),
                ("EUR", "🇪🇺", "Euro"),
                ("GBP", "🇬🇧", "İngiliz Sterlini"),
                ("CHF", "🇨🇭", "İsviçre Frangı"),
            ]

            lines = [f"💰 *Döviz Kurları*", f"_{ts}_\n"]
            lines.append("```")
            lines.append(f"{'':2}{'Döviz':<6} {'Alış':>10} {'Satış':>10} {'Δ':>8}")
            lines.append(f"{'':2}{'-'*38}")
            for code, flag, _name in _currencies:
                item = data.get(code, {})
                buy = item.get("Alış", "—")
                sell = item.get("Satış", "—")
                chg = item.get("Değişim", "")
                lines.append(f"  {code:<6} {buy:>10} {sell:>10} {chg:>8}")
            lines.append("```")
            lines.append("\n_Kaynak: Serbest piyasa anlık kurları_")

            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await self._reply(update, f"❌ Döviz kuru alınamadı: {_friendly_error(str(e))}")

    async def _quick_weather(self, update, context):
        """Get weather using Open-Meteo (lat/lon based, no key)."""
        from src.infra.db import get_user_pref
        lat = await get_user_pref("location_lat")
        lon = await get_user_pref("location_lon")
        district = await get_user_pref("location_district")
        city = await get_user_pref("location_city")
        await self._reply(update, "🌤 Hava durumu alınıyor...")
        try:
            import aiohttp, json as _json
            url = (f"https://api.open-meteo.com/v1/forecast"
                   f"?latitude={lat}&longitude={lon}"
                   f"&current_weather=true"
                   f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"
                   f"&timezone=Europe/Istanbul&forecast_days=3")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()

            # Format current weather
            cw = data.get("current_weather", {})
            _WMO = {0: "☀️ Açık", 1: "🌤 Az bulutlu", 2: "⛅ Parçalı bulutlu",
                    3: "☁️ Bulutlu", 45: "🌫 Sisli", 48: "🌫 Kırağılı sis",
                    51: "🌧 Hafif çiseleme", 53: "🌧 Çiseleme", 55: "🌧 Yoğun çiseleme",
                    61: "🌧 Hafif yağmur", 63: "🌧 Yağmur", 65: "🌧 Şiddetli yağmur",
                    71: "🌨 Hafif kar", 73: "🌨 Kar", 75: "🌨 Yoğun kar",
                    80: "🌦 Sağanak", 81: "🌦 Kuvvetli sağanak", 82: "⛈ Şiddetli sağanak",
                    95: "⛈ Gök gürültülü fırtına", 96: "⛈ Dolu", 99: "⛈ Şiddetli dolu"}
            wcode = cw.get("weathercode", 0)
            condition = _WMO.get(wcode, f"Kod {wcode}")
            loc_name = f"{district}, {city}" if district else city or f"{lat}, {lon}"

            lines = [
                f"🌤 *Hava Durumu — {loc_name}*\n",
                f"🌡 Şu an: *{cw.get('temperature', '?')}°C* {condition}",
                f"💨 Rüzgar: {cw.get('windspeed', '?')} km/h",
            ]

            # Format daily forecast
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            maxs = daily.get("temperature_2m_max", [])
            mins = daily.get("temperature_2m_min", [])
            precip = daily.get("precipitation_sum", [])
            codes = daily.get("weathercode", [])
            _DAYS_TR = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]

            if dates:
                lines.append("\n📅 *3 Günlük Tahmin*")
                for i, d in enumerate(dates[:3]):
                    from datetime import datetime as _dt
                    try:
                        day_name = _DAYS_TR[_dt.strptime(d, "%Y-%m-%d").weekday()]
                    except Exception:
                        day_name = d
                    hi = maxs[i] if i < len(maxs) else "?"
                    lo = mins[i] if i < len(mins) else "?"
                    rain = precip[i] if i < len(precip) else 0
                    dcode = codes[i] if i < len(codes) else 0
                    dcond = _WMO.get(dcode, "")
                    rain_str = f" 🌧{rain}mm" if rain and rain > 0 else ""
                    lines.append(f"  {day_name}: {lo}°/{hi}° {dcond}{rain_str}")

            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await self._reply(update, f"❌ Hava durumu alınamadı: {_friendly_error(str(e))}")

    async def _quick_fuel(self, update, context):
        """Get fuel prices via web scraping."""
        await self._reply(update, "⛽ Yakıt fiyatları alınıyor...")
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.collectapi.com/gasPrice/turkeyGasPrice",
                    headers={"User-Agent": "KutAI/1.0",
                             "content-type": "application/json",
                             "authorization": f"apikey {__import__('os').getenv('COLLECTAPI_KEY', '')}"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("result", [])
                        if results:
                            lines = ["⛽ *Güncel Yakıt Fiyatları*\n"]
                            for item in results[:6]:
                                name = item.get("name", "?")
                                price = item.get("gasoline", item.get("price", "?"))
                                diesel = item.get("diesel", "?")
                                lpg = item.get("lpg", "?")
                                lines.append(f"  *{name}*: Benzin {price}₺ | Dizel {diesel}₺ | LPG {lpg}₺")
                            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
                            return
            # Fallback: no data available
            await self._reply(update,
                "⛽ Yakıt fiyatları şu an alınamıyor.\n"
                "Güncel fiyatlar için: https://www.opet.com.tr/akaryakit-fiyatlari")
        except Exception as e:
            await self._reply(update,
                "⛽ Yakıt fiyatları şu an alınamıyor.\n"
                "Güncel fiyatlar için: https://www.opet.com.tr/akaryakit-fiyatlari")

    async def _quick_prayer(self, update, context):
        """Get prayer times using Aladhan API with Diyanet method."""
        from src.infra.db import get_user_pref
        city = await get_user_pref("location_city") or "Istanbul"
        await self._reply(update, "🕌 Namaz vakitleri alınıyor...")
        try:
            import aiohttp, json as _json
            url = (f"https://api.aladhan.com/v1/timingsByCity"
                   f"?city={city}&country=Turkey&method=13")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
            timings = data.get("data", {}).get("timings", {})
            date_info = data.get("data", {}).get("date", {})
            readable = date_info.get("readable", "")
            hijri = date_info.get("hijri", {})
            hijri_str = f"{hijri.get('day', '')} {hijri.get('month', {}).get('en', '')} {hijri.get('year', '')}"
            _labels = {
                "Fajr": "🌅 İmsak",
                "Sunrise": "☀️ Güneş",
                "Dhuhr": "🕐 Öğle",
                "Asr": "🕑 İkindi",
                "Maghrib": "🌇 Akşam",
                "Isha": "🌙 Yatsı",
            }
            lines = [f"🕌 *Namaz Vakitleri — {city}*",
                     f"📅 {readable} / {hijri_str}\n"]
            for key, label in _labels.items():
                t = timings.get(key, "?")
                # Strip timezone suffix like " (EET)"
                if " (" in t:
                    t = t[:t.index(" (")]
                lines.append(f"  {label}: *{t}*")
            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await self._reply(update, f"❌ Namaz vakitleri alınamadı: {_friendly_error(str(e))}")

    async def _quick_news(self, update, context):
        """Get news headlines from RSS feeds (no key needed)."""
        await self._reply(update, "📰 Haberler alınıyor...")
        try:
            import aiohttp
            from xml.etree import ElementTree as ET
            feeds = [
                ("https://www.ntv.com.tr/son-dakika.rss", "NTV"),
                ("https://www.sozcu.com.tr/feeds-rss-category-gundem", "Sözcü"),
            ]
            articles = []
            async with aiohttp.ClientSession() as session:
                for feed_url, source in feeds:
                    try:
                        async with session.get(
                            feed_url,
                            headers={"User-Agent": "KutAI/1.0"},
                            timeout=aiohttp.ClientTimeout(total=8),
                        ) as resp:
                            if resp.status == 200:
                                text = await resp.text()
                                root = ET.fromstring(text)
                                for item in root.findall(".//item")[:5]:
                                    title_el = item.find("title")
                                    if title_el is not None and title_el.text:
                                        articles.append((source, title_el.text.strip()))
                    except Exception:
                        continue
                    if len(articles) >= 8:
                        break
            if articles:
                lines = ["📰 *Son Dakika Haberler*\n"]
                for source, title in articles[:10]:
                    lines.append(f"  • {title} _({source})_")
                await self._reply(update, "\n".join(lines), parse_mode="Markdown")
            else:
                await self._reply(update, "📰 Haber kaynakları şu an erişilemedi.")
        except Exception as e:
            await self._reply(update, f"❌ Haberler alınamadı: {_friendly_error(str(e))}")

    async def _quick_gold(self, update, context):
        """Get live gold prices from truncgil."""
        await self._reply(update, "🪙 Altın fiyatları alınıyor...")
        try:
            data = await self._fetch_truncgil()
            if not data:
                await self._reply(update, "❌ Altın fiyat kaynağına ulaşılamadı.")
                return

            ts = data.get("Update_Date", "?")
            _gold_items = [
                ("gram-altin", "Gram Altın"),
                ("ceyrek-altin", "Çeyrek"),
                ("yarim-altin", "Yarım"),
                ("tam-altin", "Tam"),
                ("cumhuriyet-altini", "Cumhuriyet"),
                ("ata-altin", "Ata"),
                ("22-ayar-bilezik", "22 Ayar Bilezik"),
            ]

            lines = [f"🪙 *Altın Fiyatları*", f"_{ts}_\n"]
            lines.append("```")
            lines.append(f"{'':2}{'Tür':<15} {'Alış':>11} {'Satış':>11}")
            lines.append(f"{'':2}{'-'*40}")
            for key, label in _gold_items:
                item = data.get(key, {})
                buy = item.get("Alış", "—")
                sell = item.get("Satış", "—")
                chg = item.get("Değişim", "")
                arrow = self._change_arrow(chg)
                lines.append(f"  {label:<15} {buy:>11} {sell:>11}")
            lines.append("```")

            # Gram altın change summary
            gram = data.get("gram-altin", {})
            gram_chg = gram.get("Değişim", "")
            if gram_chg:
                arrow = self._change_arrow(gram_chg)
                lines.append(f"\n_Gram altın{arrow} | Kaynak: Serbest piyasa_")

            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await self._reply(update, f"❌ Altın fiyatları alınamadı: {_friendly_error(str(e))}")

    async def _quick_earthquake(self, update, context):
        """Get recent earthquake info from Kandilli (no key needed)."""
        await self._reply(update, "🌍 Deprem bilgileri alınıyor...")
        try:
            import aiohttp, json as _json
            url = "https://api.orhanaydogdu.com.tr/deprem/kandilli/live"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
            quakes = data.get("result", [])
            if not quakes:
                await self._reply(update, "Son 24 saatte kayda değer deprem yok.")
                return
            # Show significant ones (mag >= 3.0) or last 5
            significant = [q for q in quakes if float(q.get("mag", 0)) >= 3.0]
            show = significant[:8] if significant else quakes[:5]
            lines = ["🌍 *Son Depremler (Kandilli)*\n"]
            for q in show:
                mag = q.get("mag", "?")
                loc = q.get("title", q.get("location_properties", {}).get("epiCenter", {}).get("name", "?"))
                date = q.get("date", "?")
                depth = q.get("depth", "?")
                lines.append(f"  *{mag}* — {loc}\n  📍 Derinlik: {depth} km | {date}\n")
            if significant:
                lines.append(f"_Son 24 saatte {len(significant)} adet M3.0+ deprem_")
            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await self._reply(update, f"❌ Deprem bilgileri alınamadı: {_friendly_error(str(e))}")

    async def set_bot_commands(self):
        """Register the / command list with Telegram so autocomplete is up to date."""
        commands = [
            BotCommand("start", "Ana menü"),
            BotCommand("usta", "Yaşar Usta süreç yönetimi"),
            BotCommand("restart", "Kutay yeniden başlat"),
            BotCommand("retry", "Retry a task"),
        ]
        try:
            await self.app.bot.set_my_commands(commands)
            logger.info("Telegram command list updated")
        except Exception as e:
            logger.warning(f"Failed to set bot commands: {e}")


    def _setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
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
        self.app.add_handler(CommandHandler("retry", self.cmd_retry))
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
        # Shopping commands
        self.app.add_handler(CommandHandler("shop", self.cmd_shop))
        self.app.add_handler(CommandHandler("research_product", self.cmd_research_product))
        self.app.add_handler(CommandHandler("price", self.cmd_price))
        self.app.add_handler(CommandHandler("watch", self.cmd_watch))
        self.app.add_handler(CommandHandler("deals", self.cmd_deals))
        self.app.add_handler(CommandHandler("mystuff", self.cmd_mystuff))
        self.app.add_handler(CommandHandler("compare", self.cmd_compare))
        self.app.add_handler(CommandHandler("result", self.cmd_result))
        self.app.add_handler(CommandHandler("skillstats", self.cmd_skillstats))
        self.app.add_handler(CommandHandler("smartsearch", self.cmd_smartsearch))
        self.app.add_handler(CommandHandler("trace", self.cmd_trace))
        self.app.add_handler(CommandHandler("logs", self.cmd_logs))
        # Wrapper control commands
        self.app.add_handler(CommandHandler("kutai_restart", self.cmd_kutai_restart))
        self.app.add_handler(CommandHandler("usta", self.cmd_usta))
        self.app.add_handler(CommandHandler("restart", self.cmd_kutai_restart))
        self.app.add_handler(CommandHandler("kutai_stop", self.cmd_kutai_stop))
        self.app.add_handler(CommandHandler("stop", self.cmd_kutai_stop))
        self.app.add_handler(CommandHandler("claude", self.cmd_claude))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_handler(MessageHandler(filters.LOCATION, self.handle_location))
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.REPLY,
            self.handle_reply
        ))
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_message
        ))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show main menu and reset keyboard to default state."""
        chat_id = update.effective_chat.id
        self._pending_action.pop(chat_id, None)
        self._kb_state[chat_id] = "main"
        await self._reply(update,
            "🤖 *Kutay Online*\n\n"
            "Aşağıdaki butonları kullan veya mesaj yaz.",
            parse_mode="Markdown",
            reply_markup=REPLY_KEYBOARD,
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show command reference."""
        help_text = (
            "*Kutay Commands*\n\n"
            "*Shopping:* /shop, /price, /compare, /watch, /deals, /mystuff\n"
            "*Todo:* /todo, /todos, /cleartodos\n"
            "*Missions:* /mission, /wfstatus, /queue, /cancel, /resume\n"
            "*Memory:* /remember, /recall, /ingest\n"
            "*Monitor:* /status, /trace, /skillstats, /metrics, /cost\n"
            "*System:* /restart, /stop, /debug, /load, /autonomy\n"
            "\nOr just type naturally — I understand Turkish and English."
        )
        await self._reply(update, help_text, parse_mode="Markdown")

    # ─── Mission Commands ──────────────────────────────────────────────────────

    async def cmd_mission(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Create a new mission. /mission <description> or /mish <description>"""
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "mission"}
            await self._reply(update, "🎯 Describe your mission:")
            return

        text_args = list(context.args)
        workflow = None

        if "--workflow" in text_args:
            text_args.remove("--workflow")
            workflow = "i2p_v3"

        description = " ".join(text_args)
        if not description:
            await self._reply(update,"Please provide a mission description.")
            return

        # Let the classifier decide if this needs a workflow
        if not workflow:
            classification = await self._classify_user_message(description)
            if classification.get("workflow") == "i2p":
                workflow = "i2p_v3"

        chat_id = update.message.chat_id

        if workflow:
            # Workflow mission — delegate to workflow runner
            try:
                from ..workflows.engine.runner import WorkflowRunner
                runner = WorkflowRunner()
                mission_id = await runner.start(
                    workflow_name=workflow,
                    initial_input={"raw_idea": description, "product_name": description[:50]},
                    title=description[:80],
                )
                await self._reply(update,
                    f"🔄 Workflow mission #{mission_id} created!\n"
                    f"_{description[:60]}_\n\n"
                    f"Use /wfstatus {mission_id} to track progress.",
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.error("workflow mission failed", error=str(e))
                await self._reply(update,f"❌ {_friendly_error(str(e))}")
            return

        # Regular mission — create and plan
        mission_id = await add_mission(
            title=description[:80],
            description=description,
            priority=7,
        )

        if self.orchestrator:
            await self.orchestrator.plan_mission(mission_id, description[:80], description)

        await self._reply(update,
            f"🎯 Mission #{mission_id} created. Planning now...\n"
            f"_{description[:60]}_",
            parse_mode="Markdown",
        )
        self.user_last_task_id[chat_id] = None

    async def cmd_mission_workflow(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Create a workflow mission from menu button."""
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "mission_wf"}
            await self._reply(update, "🔄 Describe your product/workflow idea:")
            return
        # Inject --workflow and delegate
        context.args = ["--workflow"] + list(context.args)
        await self.cmd_mission(update, context)

    async def cmd_missions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List active missions."""
        missions = await get_active_missions()
        if not missions:
            await self._reply(update,"No active missions.")
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

        await self._reply(update,"\n".join(lines), parse_mode="Markdown")

    async def cmd_add_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "task"}
            await self._reply(update, "📋 Describe the task:")
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
        await self._reply(update,f"✅ Task #{task_id} queued.{pin_msg}")


    async def cmd_view_queue(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from src.infra.db import get_db
        db = await get_db()
        # Fetch in-progress tasks
        cursor = await db.execute(
            """SELECT * FROM tasks WHERE status = 'processing'
               ORDER BY priority DESC, started_at ASC LIMIT 10"""
        )
        processing = [dict(row) for row in await cursor.fetchall()]
        # Fetch ready (pending with deps met)
        ready = await get_ready_tasks(limit=15)
        # Fetch blocked task summary
        blocked_summary = await get_blocked_task_summary()
        blocked_count = blocked_summary["blocked_count"]

        if not processing and not ready and blocked_count == 0:
            await self._reply(update, "No pending tasks. System is idle.")
            return

        msg = "📬 Task Queue:\n\n"
        if processing:
            msg += "⚙️ In Progress:\n"
            for t in processing:
                agent = t.get('agent_type', '?')
                msg += f"  #{t['id']} [{agent}] {t['title'][:50]}\n"
            msg += "\n"
        if ready:
            msg += "⏳ Ready:\n"
            for t in ready:
                agent = t.get('agent_type', '?')
                msg += f"  #{t['id']} [{agent}|{t['tier']}] {t['title'][:50]}\n"
            msg += "\n"
        if blocked_count > 0:
            msg += f"🚫 {blocked_count} tasks blocked (waiting on dependencies)\n"
            top_blockers = blocked_summary["top_blockers"]
            if top_blockers:
                blocker_parts = [f"#{tid} ({cnt} tasks)" for tid, cnt in top_blockers]
                msg += f"  Top blockers: {', '.join(blocker_parts)}\n"
        await self._reply(update, msg)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = await get_daily_stats()
        await self._reply(update,
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
            await self._reply(update,"Orchestrator not connected.")

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
            await self._reply(update,"Database is empty.")
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
            "waiting_human": "❓",
            "cancelled": "🚫",
            "ungraded": "⏳",
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
                await self._reply(update,part, parse_mode="Markdown")
        else:
            await self._reply(update,msg, parse_mode="Markdown")


    async def cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Reset a specific stuck task back to pending."""
        if not context.args:
            await self._reply(update,
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
            await self._reply(update,f"♻️ Reset {count} failed task(s) to pending.")

        elif arg == "stuck":
            db = await get_db()
            cursor = await db.execute(
                """UPDATE tasks SET status = 'pending'
                   WHERE status = 'processing'"""
            )
            count = cursor.rowcount
            await db.commit()
            await self._reply(update,f"♻️ Reset {count} stuck task(s) to pending.")

        elif arg == "blocked":
            # Clear all dependency references so blocked tasks can run
            db = await get_db()
            cursor = await db.execute(
                """UPDATE tasks SET depends_on = '[]'
                   WHERE status = 'pending' AND depends_on != '[]'"""
            )
            count = cursor.rowcount
            await db.commit()
            await self._reply(update,
                f"♻️ Cleared dependencies on {count} blocked task(s). They'll run now."
            )

        else:
            try:
                task_id = int(arg)
                await update_task(task_id, status="pending", retry_count=0, error=None)
                await self._reply(update,f"♻️ Task #{task_id} reset to pending.")
            except ValueError:
                await self._reply(update,"Invalid argument. Use a task ID, 'failed', 'stuck', or 'blocked'.")


    async def cmd_reset_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Nuclear option: wipe everything and start fresh."""
        keyboard = [[
            InlineKeyboardButton("☢️ Yes, wipe everything", callback_data="resetall_confirm"),
            InlineKeyboardButton("Cancel", callback_data="resetall_cancel"),
        ]]
        await self._reply(update,
            "⚠️ This will delete ALL missions, tasks, memory, and conversations.\n"
            "Are you sure?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # ─── Wrapper Control Commands ─────────────────────────────────────

    async def cmd_usta(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show Yaşar Usta process management panel via /usta command."""
        await self._show_processes(update, context)

    async def cmd_kutai_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Restart KutAI via the wrapper (exit code 42).

        Uses a hard exit after a short delay as a fallback in case the
        graceful shutdown path is blocked (e.g. stuck LLM call).
        """
        await self._reply(update,"🔄 Kutay yeniden başlatılıyor...")
        if self.orchestrator:
            self.orchestrator.requested_exit_code = 42
            self.orchestrator.shutdown_event.set()
        # Hard exit fallback — fires only if graceful shutdown is truly stuck.
        # 45s allows: task drain (30s) + DB close + Telegram stop + llama stop.
        import threading
        threading.Timer(45.0, lambda: os._exit(42)).start()

    async def cmd_kutai_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop KutAI via the wrapper (exit code 0). Requires confirmation."""
        keyboard = [[
            InlineKeyboardButton("⏹ Evet, durdur", callback_data="stop_confirm"),
            InlineKeyboardButton("Cancel", callback_data="stop_cancel"),
        ]]
        await self._reply(update,
            "⚠️ *Kutay durdurulsun mu?*\nManuel olarak yeniden başlatmanız gerekecek.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def _do_kutai_stop(self):
        """Actually perform the KutAI stop after confirmation."""
        if self.orchestrator:
            self.orchestrator.requested_exit_code = 0
            self.orchestrator.shutdown_event.set()
        import threading
        threading.Timer(45.0, lambda: os._exit(0)).start()

    # ─── Claude Code Remote Control ─────────────────────────────────────

    async def cmd_claude(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Request the wrapper to start a Claude Code remote-control session."""
        signal_file = Path("logs/claude_remote.signal")
        signal_file.write_text(utc_now().isoformat(), encoding="utf-8")
        await self._reply(update, "🖥️ Claude Code remote-control session requested.\nYaşar Usta will start it shortly.")

    # ─── Result Command ─────────────────────────────────────────────────

    async def cmd_result(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View the result of a completed task. /result <task_id>"""
        if not context.args:
            recent = await get_recent_completed_tasks(limit=5)
            if not recent:
                await self._reply(update,"No completed tasks found.")
                return
            lines = ["📄 *Recent completed tasks:*\n"]
            for t in recent:
                tid = t["id"]
                title = t.get("title", "untitled")[:60]
                lines.append(f"• #{tid} — {title}")
            lines.append("\nUse /result <id> to view full result.")
            await self._reply(update,"\n".join(lines), parse_mode="Markdown")
            return
        try:
            task_id = int(context.args[0])
        except ValueError:
            await self._reply(update,"Task ID must be a number.")
            return
        task = await get_task(task_id)
        if not task:
            await self._reply(update,f"Task #{task_id} not found.")
            return
        result = task.get("result", "")
        status = task.get("status", "unknown")
        title = task.get("title", "untitled")
        if not result:
            await self._reply(update,
                f"Task #{task_id} ({status}): no result yet."
            )
            return
        header = f"📄 *Result — Task #{task_id}*\n_{title}_\n\n"
        text = header + str(result)
        if len(text) > 4000:
            text = text[:3950] + "\n\n_(truncated — result too long)_"
        await self._reply(update,text, parse_mode="Markdown")

    # ─── Phase 3 Commands ──────────────────────────────────────────────

    async def cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cancel a task or mission and its children."""
        if not context.args:
            await self._reply(update,
                "Usage: /cancel <task_id or mission_id>\n"
                "Cancels a task (and children) or an entire mission."
            )
            return
        try:
            item_id = int(context.args[0])
        except ValueError:
            await self._reply(update,"ID must be a number.")
            return

        # Try cancelling as a task first
        success = await cancel_task(item_id)
        if success:
            await self._reply(update,
                f"🚫 Task #{item_id} and its children cancelled."
            )
            return

        # Not a task — try as a mission
        mission = await get_mission(item_id)
        if mission and mission.get("status") not in ("completed", "cancelled"):
            await update_mission(item_id, status="cancelled")
            # Also cancel all pending tasks for this mission
            tasks = await get_tasks_for_mission(item_id)
            cancelled_count = 0
            for t in tasks:
                if t.get("status") in ("pending", "processing", "blocked"):
                    await cancel_task(t["id"])
                    cancelled_count += 1
            await self._reply(update,
                f"🚫 Mission #{item_id} cancelled ({cancelled_count} tasks also cancelled)."
            )
        else:
            await self._reply(update,
                f"#{item_id} not found or already finished."
            )

    async def cmd_priority(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Change task priority."""
        if len(context.args) < 2:
            await self._reply(update,
                "Usage: /priority <task_id> <1-10>"
            )
            return
        try:
            task_id = int(context.args[0])
            level = int(context.args[1])
            if not 1 <= level <= 10:
                raise ValueError
        except ValueError:
            await self._reply(update,
                "Task ID and priority (1-10) must be numbers."
            )
            return

        success = await reprioritize_task(task_id, level)
        if success:
            await self._reply(update,
                f"✅ Task #{task_id} priority set to {level}."
            )
        else:
            await self._reply(update,
                f"Task #{task_id} not found or not pending/processing."
            )

    async def cmd_graph(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show text DAG of task dependencies for a mission."""
        if not context.args:
            await self._reply(update,"Usage: /graph <mission_id>")
            return
        try:
            mission_id = int(context.args[0])
        except ValueError:
            await self._reply(update,"Mission ID must be a number.")
            return

        tasks = await get_task_tree(mission_id)
        if not tasks:
            await self._reply(update,
                f"No tasks found for mission #{mission_id}."
            )
            return

        # Build text DAG
        lines = [f"📊 *Task Graph — Mission #{mission_id}*\n"]
        status_icons = {
            "pending": "⏳", "processing": "⚙️",
            "completed": "✅", "failed": "❌",
            "cancelled": "🚫", "waiting_subtasks": "🔄",
            "waiting_human": "❓", "needs_review": "👀", "ungraded": "⏳",
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

        await self._reply(update,
            "\n".join(lines), parse_mode="Markdown"
        )

    async def cmd_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View or set daily cost budget."""
        if context.args:
            try:
                new_limit = float(context.args[0])
                await set_budget("daily", daily_limit=new_limit)
                await self._reply(update,
                    f"💰 Daily budget set to ${new_limit:.2f}"
                )
            except ValueError:
                await self._reply(update,
                    "Usage: /budget [daily_limit]\n"
                    "Example: /budget 1.50"
                )
            return

        budget = await get_budget("daily")
        if not budget:
            await self._reply(update,
                "💰 No daily budget set.\n"
                "Use /budget <amount> to set one.\n"
                "Example: /budget 1.00"
            )
            return

        today = budget.get("last_reset_date", "N/A")
        await self._reply(update,
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
            await self._reply(update,"Usage: /cost <mission\\_id>",
                                            parse_mode="Markdown")
            return
        try:
            mission_id = int(context.args[0])
        except ValueError:
            await self._reply(update,"Mission ID must be a number.")
            return

        try:
            from ..collaboration.blackboard import read_blackboard
            cost_data = await read_blackboard(mission_id, "cost_tracking")
            if not isinstance(cost_data, dict):
                cost_data = {}
        except Exception:
            cost_data = {}

        if not cost_data or cost_data.get("total_cost", 0) == 0:
            await self._reply(update,f"No cost data for mission #{mission_id}")
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

        await self._reply(update,"\n".join(lines), parse_mode="Markdown")

    async def cmd_model_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show model performance statistics."""
        stats = await get_model_stats()
        if not stats:
            await self._reply(update,"📊 No model stats yet.")
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

        await self._reply(update,
            "\n".join(lines), parse_mode="Markdown"
        )

    async def cmd_trace(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show end-to-end trace for a task."""
        args = context.args
        if not args:
            await self._reply(update, "Usage: /trace <task_id>")
            return

        try:
            task_id = int(args[0])
        except ValueError:
            await self._reply(update, "Invalid task ID")
            return

        from src.infra.db import get_db
        db = await get_db()

        # Get task info
        cursor = await db.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        )
        task = await cursor.fetchone()
        if not task:
            await self._reply(update, f"Task #{task_id} not found")
            return

        task = dict(task)
        lines = [f"📋 *Task #{task_id} Trace*\n"]
        lines.append(f"Title: {task.get('title', '?')[:60]}")
        lines.append(f"Agent: {task.get('agent_type', '?')}")
        lines.append(f"Status: {task.get('status', '?')}")
        lines.append(f"Priority: {task.get('priority', '?')}")

        # Timeline
        created = task.get('created_at', '?')
        started = task.get('started_at', '')
        completed = task.get('completed_at', '')

        lines.append(f"\n⏱ *Timeline*")
        lines.append(f"Created: {created}")
        if started:
            lines.append(f"Started: {started}")
        if completed:
            lines.append(f"Completed: {completed}")
        if started and completed:
            try:
                t1 = datetime.strptime(started.replace("T", " ")[:19], "%Y-%m-%d %H:%M:%S")
                t2 = datetime.strptime(completed.replace("T", " ")[:19], "%Y-%m-%d %H:%M:%S")
                duration = (t2 - t1).total_seconds()
                lines.append(f"Duration: {duration:.0f}s")
            except Exception:
                pass

        # Cost
        cost = task.get('cost', 0) or 0
        if cost > 0:
            lines.append(f"\n💰 Cost: ${cost:.4f}")

        # Error
        error = task.get('error', '')
        if error:
            lines.append(f"\n❌ Error: {error[:200]}")

        # Check for injected skills
        import json as _json_trace
        ctx = task.get('context', '{}')
        try:
            ctx_dict = _json_trace.loads(ctx) if isinstance(ctx, str) else ctx
            skills = ctx_dict.get('injected_skills', [])
            if skills:
                lines.append(f"\n🎯 Skills: {', '.join(skills)}")
        except Exception:
            pass

        # Result preview
        result = task.get('result', '')
        if result:
            lines.append(f"\n📄 Result: {result[:300]}...")

        await self._reply(update, "\n".join(lines), parse_mode="Markdown")

    async def cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show recent orchestrator log entries."""
        args = context.args
        n = 20
        if args:
            try:
                n = min(int(args[0]), 50)
            except ValueError:
                pass

        log_path = os.path.join("logs", "orchestrator.jsonl")
        if not os.path.exists(log_path):
            await self._reply(update, "📋 No log file found.")
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 100_000))
                chunk = f.read()
                lines = chunk.strip().split("\n")
        except Exception as e:
            await self._reply(update, f"❌ Error reading logs: {e}")
            return

        text = _format_log_entries(lines, n=n)
        try:
            await self._reply(update, text, parse_mode="Markdown")
        except Exception:
            # Markdown parse error from special chars in log messages — send as plain text
            await self._reply(update, text)

    async def cmd_skillstats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show skill injection A/B metrics."""
        from src.infra.db import get_skill_metrics_summary

        summary = await get_skill_metrics_summary()
        overall = summary.get("overall", {})
        per_skill = summary.get("per_skill", [])

        lines = ["*Skill Injection Metrics*\n"]

        # Overall comparison
        with_skills = overall.get("with_skills", {})
        baseline = overall.get("baseline", {})

        if with_skills or baseline:
            lines.append("*A/B Comparison:*")
            if with_skills:
                lines.append(
                    f"  With skills: {with_skills['success_rate']}% success "
                    f"({with_skills['successes']}/{with_skills['total']}), "
                    f"avg {with_skills['avg_iterations']} iter, "
                    f"{with_skills['avg_duration']:.0f}s"
                )
            if baseline:
                lines.append(
                    f"  No skills: {baseline['success_rate']}% success "
                    f"({baseline['successes']}/{baseline['total']}), "
                    f"avg {baseline['avg_iterations']} iter, "
                    f"{baseline['avg_duration']:.0f}s"
                )

            # Calculate lift
            if with_skills and baseline and baseline.get("success_rate", 0) > 0:
                lift = with_skills["success_rate"] - baseline["success_rate"]
                indicator = "+" if lift > 0 else ""
                lines.append(f"  Lift: {indicator}{lift:.1f}%")
            lines.append("")
        else:
            lines.append("No data yet. Skills will be tracked as tasks complete.\n")

        # Per-skill breakdown
        if per_skill:
            lines.append("*Top Skills:*")
            for s in per_skill[:10]:
                lines.append(f"  {s['name']}: {s['success_rate']}% ({s['successes']}/{s['total']})")

        await self._reply(update, "\n".join(lines), parse_mode="Markdown")

    async def cmd_smartsearch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show smart search stats and API/MCP observability."""
        try:
            from src.infra.db import get_smart_search_stats, get_api_reliability_all

            stats = await get_smart_search_stats(days=7)
            reliability = await get_api_reliability_all()

            today = stats.get("today", 0)
            layers = stats.get("layers", {})
            total_7d = sum(l["count"] for l in layers.values())

            lines = ["Smart Search Stats", "-" * 25]
            lines.append(f"Queries today: {today}")

            layer_names = {0: "Layer 0 (fast-path)", 1: "Layer 1 (enriched)", 2: "Layer 2 (smart_search)", 3: "Fell through to web"}
            for layer_num in sorted(layers.keys()):
                info = layers[layer_num]
                pct = int(info["count"] / max(total_7d, 1) * 100)
                name = layer_names.get(layer_num, f"Layer {layer_num}")
                lines.append(f"  {name}: {info['count']}  ({pct}%)")

            top = [r for r in reliability if r["status"] == "active" and (r["success_count"] + r["failure_count"]) > 0]
            top.sort(key=lambda r: r["success_count"], reverse=True)
            if top[:5]:
                lines.append("")
                lines.append("Top APIs (7d)")
                for r in top[:5]:
                    total = r["success_count"] + r["failure_count"]
                    rate = int(r["success_count"] / max(total, 1) * 100)
                    lines.append(f"  {r['api_name']:<20} {total} calls, {rate}% success")

            worst = [r for r in reliability if r["status"] in ("warning", "demoted", "suspended")]
            if worst:
                lines.append("")
                lines.append("Worst Performers (7d)")
                for r in worst[:5]:
                    total = r["success_count"] + r["failure_count"]
                    rate = int(r["success_count"] / max(total, 1) * 100)
                    status_label = {"warning": "!", "demoted": "demoted", "suspended": "suspended"}
                    lines.append(f"  {r['api_name']:<20} {r['success_count']}/{total} ({rate}%) {status_label.get(r['status'], '')}")

            top_sources = stats.get("top_sources", [])
            if top_sources:
                lines.append("")
                lines.append("Top Sources (7d)")
                for s in top_sources[:5]:
                    lines.append(f"  {s['source']:<20} {s['count']} calls")

            # Registry summary
            from src.tools.free_apis import API_REGISTRY, _db_api_cache
            static_count = len(API_REGISTRY)
            discovered_count = len(_db_api_cache)
            lines.append("")
            lines.append(f"Registry: {static_count} static + {discovered_count} discovered APIs")

            # Show loaded API categories
            categories = {}
            for api in API_REGISTRY:
                categories[api.category] = categories.get(api.category, 0) + 1
            for api in _db_api_cache:
                categories[api.category] = categories.get(api.category, 0) + 1
            if categories:
                sorted_cats = sorted(categories.items(), key=lambda x: -x[1])[:10]
                lines.append("Categories: " + ", ".join(f"{c}({n})" for c, n in sorted_cats))

            # MCP tools
            from src.tools import TOOL_REGISTRY
            mcp_tools = [k for k in TOOL_REGISTRY if k.startswith("mcp_")]
            if mcp_tools:
                lines.append("")
                lines.append(f"MCP Tools: {len(mcp_tools)}")
                for t in mcp_tools:
                    entry = TOOL_REGISTRY[t]
                    stub = " (stub)" if entry.get("_mcp_stub") else " (connected)"
                    lines.append(f"  {t}{stub}")

            text = "\n".join(lines)

            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Refresh Now", callback_data="ss:refresh"),
                    InlineKeyboardButton("Top Failures", callback_data="ss:failures"),
                ],
                [
                    InlineKeyboardButton("Unsuspend All", callback_data="ss:unsuspend"),
                    InlineKeyboardButton("List APIs", callback_data="ss:list_apis"),
                ],
            ])

            await self._reply(update, text, reply_markup=keyboard)

        except Exception as e:
            await self._reply(update, f"Error loading stats: {e}")

    async def cmd_workspace(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show active mission workspaces."""
        workspaces = list_mission_workspaces()
        if not workspaces:
            await self._reply(update,"📁 No mission workspaces active.")
            return

        lines = ["📁 *Mission Workspaces*\n"]
        for ws in workspaces:
            locks = await get_mission_locks(ws["mission_id"])
            lock_str = f" ({len(locks)} locks)" if locks else ""
            lines.append(
                f"  Mission #{ws['mission_id']}: "
                f"{ws['file_count']} files{lock_str}"
            )

        await self._reply(update,
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
            await self._reply(update,msg, parse_mode="Markdown")
        except (ValueError, IndexError):
            await self._reply(update,"Usage: /progress [mission_id]")
        except Exception as e:
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

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
            await self._reply(update,msg, parse_mode="Markdown")
        except (ValueError, IndexError):
            await self._reply(update,"Usage: /audit [task_id]")
        except Exception as e:
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

    async def cmd_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show system metrics summary. /metrics"""
        try:
            from src.infra.metrics import format_metrics_summary
            msg = format_metrics_summary()
            await self._reply(update,msg, parse_mode="Markdown")
        except Exception as e:
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

    async def cmd_replay(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Replay task execution trace. /replay <task_id>"""
        args = context.args
        if not args:
            await self._reply(update,"Usage: /replay <task_id>")
            return
        try:
            from src.infra.tracing import get_trace, format_trace
            task_id = int(args[0])
            trace = await get_trace(task_id)
            trace_text = format_trace(trace)
            msg = f"🔄 *Trace for Task #{task_id}*\n\n{trace_text}"
            if len(msg) > 4000:
                msg = msg[:4000] + "\n... (truncated)"
            await self._reply(update,msg, parse_mode="Markdown")
        except (ValueError, IndexError):
            await self._reply(update,"Usage: /replay <task_id>")
        except Exception as e:
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

    async def cmd_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Rate a completed task. /feedback <task_id> <good|bad|partial> [reason]"""
        args = context.args
        if len(args) < 2:
            await self._reply(update,"Usage: /feedback <task_id> <good|bad|partial> [reason]")
            return
        try:
            from src.memory.feedback import record_feedback
            task_id = int(args[0])
            rating = args[1].lower()
            if rating not in ("good", "bad", "partial"):
                await self._reply(update,"Rating must be: good, bad, or partial")
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
            await self._reply(update,msg, parse_mode="Markdown")
        except (ValueError, IndexError):
            await self._reply(update,"Usage: /feedback <task_id> <good|bad|partial> [reason]")
        except Exception as e:
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

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
                await self._reply(update,
                    f"🛡️ *Autonomy Level*\n\nCurrent threshold: *{level_name}* (score ≥{current} requires approval)",
                    parse_mode="Markdown",
                )
                return
            level = args[0].lower()
            if level not in levels:
                await self._reply(update,f"Unknown level. Choose: {', '.join(levels.keys())}")
                return
            threshold = levels[level]
            set_autonomy_threshold(threshold)
            desc = {
                "paranoid": "require approval for almost everything",
                "low": "require approval for medium+ risk tasks",
                "medium": "require approval for high-risk tasks only",
                "high": "require approval only for very dangerous tasks",
            }[level]
            await self._reply(update,
                f"🛡️ Autonomy set to *{level}* — will {desc}.",
                parse_mode="Markdown",
            )
        except Exception as e:
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

    async def cmd_ingest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Ingest a URL or file into the knowledge base."""
        if not context.args:
            await self._reply(update,
                "Usage: /ingest <url\\_or\\_filepath>\n\n"
                "Examples:\n"
                "/ingest https://docs.example.com/api\n"
                "/ingest /path/to/document.pdf",
                parse_mode="Markdown",
            )
            return

        source = " ".join(context.args)
        await self._reply(update,f"📥 Ingesting: {source}...")

        try:
            result = await ingest_document(source)

            if result["status"] == "ok":
                await self._reply(update,
                    f"✅ Ingested *{result['chunks']}* chunks from "
                    f"`{result['source']}`\n\n"
                    f"Knowledge is now available to all agents.",
                    parse_mode="Markdown",
                )
            else:
                await self._reply(update,
                    f"❌ Ingestion failed: {result.get('error', 'unknown error')}"
                )
        except ImportError:
            await self._reply(update,
                "❌ Memory system not available. "
                "Install chromadb: pip install chromadb"
            )
        except Exception as e:
            await self._reply(update,
                f"❌ {_friendly_error(str(e))}"
            )

    # ─── Credential Management ────────────────────────────────────────

    async def cmd_credential(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manage stored credentials for external services."""
        if not context.args:
            await self._reply(update,
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
                    await self._reply(update,
                        "No credentials stored. Use /credential add to add one."
                    )
                else:
                    lines = ["*Stored Credentials:*\n"]
                    for svc in services:
                        lines.append(f"  `{svc}`")
                    await self._reply(update,
                        "\n".join(lines), parse_mode="Markdown"
                    )
            except Exception as e:
                await self._reply(update,f"❌ {_friendly_error(str(e))}")

        elif sub == "add":
            if len(context.args) < 3:
                await self._reply(update,
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
                await self._reply(update,
                    "Invalid JSON data. Make sure to use proper JSON format."
                )
                return

            try:
                from ..security.credential_store import store_credential

                await store_credential(service_name, data)
                await self._reply(update,
                    f"Stored credential for `{service_name}`.",
                    parse_mode="Markdown",
                )
            except Exception as e:
                await self._reply(update,f"❌ {_friendly_error(str(e))}")

        elif sub == "remove":
            if len(context.args) < 2:
                await self._reply(update,
                    "Usage: /credential remove <service>"
                )
                return

            service_name = context.args[1]
            try:
                from ..security.credential_store import delete_credential

                deleted = await delete_credential(service_name)
                if deleted:
                    await self._reply(update,
                        f"Removed credential for `{service_name}`.",
                        parse_mode="Markdown",
                    )
                else:
                    await self._reply(update,
                        f"No credential found for '{service_name}'."
                    )
            except Exception as e:
                await self._reply(update,f"❌ {_friendly_error(str(e))}")

        else:
            await self._reply(update,
                "Unknown subcommand. Use: list, add, or remove."
            )

    # ─── Workflow Commands ──────────────────────────────────────────────

    async def cmd_wfstatus(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show workflow progress for a mission. Without args, lists active workflow missions."""
        if not context.args:
            # List all active workflow missions with inline buttons
            try:
                db = await get_db()
                cursor = await db.execute(
                    """SELECT id, title, status FROM missions
                       WHERE status NOT IN ('completed', 'cancelled', 'failed')
                       ORDER BY id DESC LIMIT 10"""
                )
                rows = await cursor.fetchall()
                if not rows:
                    await self._reply(update,"No active missions.")
                    return
                buttons = []
                for r in rows:
                    label = f"#{r['id']} — {r['title'][:40]} ({r['status']})"
                    buttons.append([InlineKeyboardButton(label, callback_data=f"wfstatus:{r['id']}")])
                buttons.append([InlineKeyboardButton("❌ Cancel", callback_data="wfstatus_dismiss")])
                await self._reply(update,
                    "*Active Missions* — tap to view details:",
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(buttons),
                )
            except Exception as e:
                await self._reply(update,f"Usage: /wfstatus <mission\\_id>",
                                                parse_mode="Markdown")
            return

        try:
            mission_id = int(context.args[0])
        except ValueError:
            await self._reply(update,"Mission ID must be a number.")
            return

        try:
            from ..workflows.engine.status import (
                compute_phase_progress, format_status_message,
            )

            mission = await get_mission(mission_id)
            if not mission:
                await self._reply(update,f"Mission #{mission_id} not found.")
                return

            tasks = await get_tasks_for_mission(mission_id)
            if not tasks:
                await self._reply(update,
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
            workflow_name = mission_ctx.get("workflow_name", "i2p_v3")

            progress = compute_phase_progress(tasks)
            msg = format_status_message(workflow_name, mission_id, progress)
            cancel_button = InlineKeyboardMarkup([[
                InlineKeyboardButton("🗑 Cancel Mission", callback_data=f"wfcancel:{mission_id}")
            ]])
            await self._reply(update,msg, reply_markup=cancel_button)
        except Exception as e:
            await self._reply(update,
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
                await self._reply(update,
                    f"\u2705 Task #{task_id} re-queued from dead-letter queue."
                )
                return

            if len(args) >= 2 and args[0] == "discard":
                task_id = int(args[1])
                await resolve_dlq_task(task_id, resolution="discarded")
                await self._reply(update,
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

            await self._reply(update,
                "\n".join(lines), parse_mode="Markdown"
            )
        except Exception as e:
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

    async def cmd_retry(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Retry a failed task: /retry <task_id>."""
        if not context.args:
            await self._reply(update, "Usage: /retry <task\\_id>", parse_mode="Markdown")
            return

        try:
            task_id = int(context.args[0])
        except ValueError:
            await self._reply(update, "❌ Invalid task ID — must be a number.")
            return

        try:
            from ..infra.db import get_task
            task = await get_task(task_id)
            if task is None:
                await self._reply(update, f"❌ Task #{task_id} not found.")
                return

            status = task.get("status", "")

            if status == "failed":
                from ..infra.dead_letter import retry_dlq_task
                await retry_dlq_task(task_id)
                # Phase-aware: re-read task to show actual new status
                refreshed = await get_task(task_id)
                new_status = refreshed.get("status", "pending") if refreshed else "pending"
                await self._reply(update, f"🔄 Task #{task_id} re-queued from DLQ → {new_status}.")
            else:
                await self._reply(
                    update,
                    f"⚠️ Task #{task_id} has status '{status}' — only failed tasks can be retried."
                )
        except Exception as e:
            await self._reply(update, f"❌ {_friendly_error(str(e))}")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume a failed workflow."""
        if not context.args:
            await self._reply(update,"Usage: /resume <mission\\_id>",
                                            parse_mode="Markdown")
            return

        try:
            mission_id_str = context.args[0]
            mission_id = int(mission_id_str)
        except ValueError:
            await self._reply(update,"Mission ID must be a number.")
            return

        try:
            from ..workflows.engine.runner import WorkflowRunner

            runner = WorkflowRunner()
            resumed_id = await runner.resume(mission_id)

            await self._reply(update,
                f"\u25b6\ufe0f Workflow resumed for mission #{resumed_id}\n"
                f"Use /wfstatus {resumed_id} to track progress."
            )
        except ValueError as e:
            await self._reply(update,f"❌ {_friendly_error(str(e))}")
        except Exception as e:
            await self._reply(update,
                f"❌ {_friendly_error(str(e))}"
            )

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Cancel pending tasks in a mission: /pause <mission_id>"""
        args = context.args or []
        if not args:
            await self._reply(update,"Usage: /pause <mission\\_id>\nCancels all pending tasks for the mission.",
                                           parse_mode="Markdown")
            return
        try:
            mission_id = int(args[0])
            from ..infra.db import get_db
            async with get_db() as db:
                result = await db.execute(
                    """UPDATE tasks SET status = 'cancelled'
                       WHERE mission_id = ? AND status IN ('pending')""",
                    (mission_id,)
                )
                await db.commit()
                count = result.rowcount
            await self._reply(update,f"🚫 Mission #{mission_id}: cancelled {count} task(s).")
            logger.info("mission cancelled via command", mission_id=mission_id, tasks_cancelled=count)
        except ValueError:
            await self._reply(update,"Please provide a valid integer mission ID.")
        except Exception as e:
            logger.exception("pause command failed", error=str(e))
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

    async def cmd_load(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/load full|heavy|shared|minimal|auto — set GPU load mode"""
        args = context.args or []
        if not args:
            from src.infra.load_manager import get_load_mode, is_auto_managed
            current = await get_load_mode()
            auto_str = " (auto-managed)" if is_auto_managed() else " (manual)"
            await self._reply(update,
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
            await self._reply(update,
                "GPU load mode set to *auto-managed*. "
                "Will adjust based on external GPU usage.",
                parse_mode="Markdown",
            )
            logger.info("load mode set to auto via command")
            return
        from src.infra.load_manager import set_load_mode
        msg = await set_load_mode(choice, source="user")
        await self._reply(update,msg, parse_mode="Markdown")
        logger.info("load mode changed via command", mode=choice)

    async def cmd_tune(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/tune — force an auto-tuning cycle and report results."""
        await self._reply(update,"Running tuning cycle...")
        try:
            from src.models.auto_tuner import maybe_run_tuning
            report = await maybe_run_tuning(force=True)

            tuned = report.get("tuned_models", {}) if report else {}
            if not tuned:
                await self._reply(update,"No models needed tuning adjustment.")
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

            await self._reply(update,"\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            logger.error("tune command failed", error=str(e))
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

    async def cmd_improve(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/improve — run self-improvement analysis and show proposals."""
        await self._reply(update,"Analyzing system performance...")
        try:
            from ..memory.self_improvement import (
                analyze_and_propose, format_proposals_for_telegram
            )
            proposals = await analyze_and_propose()
            msg = format_proposals_for_telegram(proposals)
            # Split if too long
            if len(msg) > 4000:
                await self._reply(update,msg[:4000], parse_mode="Markdown")
            else:
                await self._reply(update,msg, parse_mode="Markdown")

            # Save full report
            if proposals:
                try:
                    from ..memory.self_improvement import format_proposals_for_file
                    import os
                    report = await format_proposals_for_file(proposals)
                    os.makedirs("workspace/results", exist_ok=True)
                    path = f"workspace/results/improvement_report_{utc_now().strftime('%Y%m%d')}.md"
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(report)
                    await self._reply(update,f"Full report saved: {path}")
                except Exception:
                    pass
        except Exception as e:
            logger.error("improve command failed", error=str(e))
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

    # ── Phase 14.4: /remember and /recall ─────────────────────────────────────

    async def cmd_remember(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Store a user fact in the knowledge base for later recall."""
        text = " ".join(context.args) if context.args else ""
        if not text:
            await self._reply(update,
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
            await self._reply(update,
                f"✅ Remembered! (id: `{doc_id or 'stored'}`)\n"
                f"Use `/recall` to search your notes later.",
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error("remember command failed", error=str(e))
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

    async def cmd_recall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Search the knowledge base for previously stored facts."""
        query_text = " ".join(context.args) if context.args else ""
        if not query_text:
            await self._reply(update,
                "Usage: `/recall <search query>`\n"
                "Example: `/recall staging server address`",
                parse_mode="Markdown",
            )
            return
        try:
            from ..memory.vector_store import query as vs_query
            results = await vs_query(query_text, collection="semantic", top_k=5)
            if not results:
                await self._reply(update,"No matching memories found.")
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
            await self._reply(update,"\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            logger.error("recall command failed", error=str(e))
            await self._reply(update,f"❌ {_friendly_error(str(e))}")

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

                await self._reply(update,
                    f"\U0001f4ce File received: `{filename}`\n"
                    f"Use `/ingest {filepath}` to add to knowledge base.",
                    parse_mode="Markdown"
                )
                logger.info("File uploaded", filename=filename, chat_id=chat_id)
                return
            except Exception as e:
                logger.error(f"Failed to handle file upload: {e}")
                await self._reply(update,f"❌ {_friendly_error(str(e))}")
                return

        text = update.message.text
        if not text:
            return

        # ═══════════════════════════════════════════════════════
        # PRIORITY -1: Reply-keyboard button taps → route via _BUTTON_ACTIONS
        # ═══════════════════════════════════════════════════════
        btn_action = _BUTTON_ACTIONS.get(text.strip())
        if btn_action:
            # Clear any stale pending action — user tapped a new button
            self._pending_action.pop(chat_id, None)
            action_type, action_value = btn_action
            try:
                if action_type == "category":
                    await self._handle_category_button(update, context, action_value)
                elif action_type == "cmd":
                    handler = self._resolve_cmd_handler(action_value)
                    if handler:
                        context.args = []
                        await handler(update, context)
                elif action_type == "cmd_args":
                    prompt = _CMD_ARG_PROMPTS.get(action_value, f"Enter input:")
                    self._pending_action[chat_id] = {
                        "command": action_value,
                        "ts": _time.time(),
                    }
                    await self._reply(update, prompt)
                elif action_type == "special":
                    await self._handle_special_button(update, context, action_value)
            except Exception as e:
                logger.error("Button action failed", action=action_value, error=str(e))
                await self._reply(update, f"❌ {_friendly_error(str(e))}")
            return

        logger.info("Message received", user_id=chat_id, text_preview=text[:50])

        # ═══════════════════════════════════════════════════════
        # PRIORITY 0: Button-initiated conversation flow (with timeout)
        # ═══════════════════════════════════════════════════════
        pending_action = self._pending_action.pop(chat_id, None)
        if pending_action:
            # Check timeout
            if _time.time() - pending_action.get("ts", 0) > _PENDING_ACTION_TIMEOUT:
                logger.info("Pending action expired", command=pending_action.get("command"))
                # Fall through to normal routing
            else:
                cmd = pending_action["command"]
                if cmd == "_todo_help":
                    self._last_todo_help = pending_action
                    await self.cmd__todo_help(update, context)
                    return

                if cmd == "_todo_edit":
                    await self._handle_todo_edit(update, context, pending_action)
                    return

                # ── Workflow selection flow ──
                if cmd == "_workflow_select":
                    # User typed mission description, now show workflow picker
                    self._pending_mission[chat_id] = text
                    self._kb_state[chat_id] = "workflow_select"
                    await update.message.reply_text(
                        "🎯 Bu görev nasıl çalışsın?",
                        reply_markup=KB_WORKFLOW_SELECT,
                    )
                    return

                # ── Location setup flow ──
                if cmd in ("_location_setup", "_location_district"):
                    original_service = pending_action.get("original_service", "")
                    stripped = text.strip()
                    if stripped == "✏️ İlçe Adı Yaz":
                        self._pending_action[chat_id] = {
                            "command": "_location_district",
                            "original_service": original_service,
                            "ts": _time.time(),
                        }
                        await self._reply(update,
                            "📍 İlçe adı, koordinat veya Google Maps linki gönder:\n"
                            "• Kadıköy\n"
                            "• 39.95, 32.84\n"
                            "• https://maps.google.com/...")
                        return
                    if stripped == "❌ İptal":
                        self._kb_state[chat_id] = "hizmet" if original_service else "main"
                        restore_kb = KB_HIZMET if original_service else REPLY_KEYBOARD
                        await update.message.reply_text("❌ İptal edildi.", reply_markup=restore_kb)
                        return
                    # Try parsing as maps URL or coordinate pair (resolve short URLs)
                    coords = await self._resolve_maps_url(stripped)
                    if coords:
                        lat, lon = coords
                        await self._save_location_from_coords(
                            update, context, lat, lon, original_service)
                        return
                    # Fall back to district name geocoding
                    await self._geocode_district(update, context, stripped, original_service)
                    return

                # ── Reminder flow ──
                if cmd == "_reminder_time":
                    parsed_dt = self._parse_reminder_time(text)
                    if parsed_dt is None:
                        await self._reply(update,
                            "❌ Zaman anlaşılamadı. Tekrar dene:\n"
                            "• 10dk\n• 1 saat\n• 14:30\n• yarın 09:00"
                        )
                        # Re-set so user can retry
                        self._pending_action[chat_id] = {
                            "command": "_reminder_time",
                            "ts": _time.time(),
                        }
                        return
                    self._pending_action[chat_id] = {
                        "command": "_reminder_text",
                        "reminder_time": parsed_dt,
                        "ts": _time.time(),
                    }
                    # parsed_dt is UTC naive — show Turkey local time to user
                    _display_dt = to_turkey(parsed_dt)
                    time_str = _display_dt.strftime("%d.%m.%Y %H:%M")
                    await self._reply(update, f"📝 Ne hatırlatılsın? _(Saat: {time_str})_")
                    return

                if cmd == "_reminder_text":
                    reminder_time = pending_action.get("reminder_time")
                    if reminder_time is None:
                        await self._reply(update, "❌ Hatırlatma zamanı kayboldu. Tekrar dene.")
                        return
                    try:
                        from src.infra.db import add_scheduled_task
                        next_run_str = reminder_time.strftime("%Y-%m-%d %H:%M:%S")
                        task_id = await add_scheduled_task(
                            title=f"Hatırlatma: {text[:60]}",
                            description=text,
                            cron_expression="",  # one-shot — no cron
                            agent_type="notifier",
                            tier="cheap",
                            context={
                                "one_shot": True,
                                "chat_id": chat_id,
                                "reminder_text": text,
                                "next_run_override": next_run_str,
                            },
                        )
                        # Override next_run directly so scheduler picks it up
                        from src.infra.db import get_db
                        db = await get_db()
                        await db.execute(
                            "UPDATE scheduled_tasks SET next_run=?, enabled=1 WHERE id=?",
                            (next_run_str, task_id),
                        )
                        await db.commit()
                        time_str = reminder_time.strftime("%d.%m.%Y %H:%M")
                        await self._reply(update,
                            f"✅ Hatırlatma kaydedildi!\n"
                            f"⏰ {time_str}\n"
                            f"📝 {text[:100]}"
                        )
                    except Exception as e:
                        logger.error("Reminder creation failed", error=str(e))
                        await self._reply(update, f"❌ Hatırlatma oluşturulamadı: {_friendly_error(str(e))}")
                    return

                # ── Schedule task flow ──
                if cmd == "_schedule_desc":
                    self._pending_action[chat_id] = {
                        "command": "_schedule_cron",
                        "schedule_desc": text,
                        "ts": _time.time(),
                    }
                    await self._reply(update,
                        "⏰ Ne sıklıkla?\n\n"
                        "Örnekler:\n"
                        "• her gün 09:00\n"
                        "• her 2 saatte\n"
                        "• her pazartesi\n"
                        "• her saat"
                    )
                    return

                if cmd == "_schedule_cron":
                    schedule_desc = pending_action.get("schedule_desc", "")
                    cron_expr = self._parse_cron_input(text)
                    if cron_expr is None:
                        await self._reply(update,
                            "❌ Program anlaşılamadı. Tekrar dene:\n"
                            "• her gün 09:00\n• her 2 saatte\n• her pazartesi\n• her saat"
                        )
                        self._pending_action[chat_id] = {
                            "command": "_schedule_cron",
                            "schedule_desc": schedule_desc,
                            "ts": _time.time(),
                        }
                        return
                    try:
                        from src.infra.db import add_scheduled_task
                        task_id = await add_scheduled_task(
                            title=schedule_desc[:100],
                            description=schedule_desc,
                            cron_expression=cron_expr,
                            agent_type="executor",
                            tier="cheap",
                            context={"chat_id": chat_id},
                        )
                        await self._reply(update,
                            f"✅ Görev zamanlandı!\n"
                            f"📋 {schedule_desc[:100]}\n"
                            f"⏰ Cron: `{cron_expr}`"
                        )
                    except Exception as e:
                        logger.error("Schedule task creation failed", error=str(e))
                        await self._reply(update, f"❌ Görev zamanlanamadı: {_friendly_error(str(e))}")
                    return

                handler = self._resolve_cmd_handler(cmd)
                if handler:
                    # Simulate command with text as args
                    context.args = text.split() if text.strip() else []
                    try:
                        await handler(update, context)
                    except Exception as e:
                        await self._reply(update, f"❌ {_friendly_error(str(e))}")
                else:
                    logger.warning("pending_action handler not found", command=cmd)
                    await self._reply(update, f"❌ Unknown command: {cmd}")
                return

        # ═══════════════════════════════════════════════════════
        # PRIORITY 1: Check for pending clarification (state-based)
        # ═══════════════════════════════════════════════════════
        pending_task_id = self._pending_clarifications.get(chat_id)
        if pending_task_id:
            lower = text.lower().strip()
            # Allow user to escape the clarification loop
            escape_phrases = [
                "skip", "cancel", "dismiss", "ignore", "forget it",
                "never mind", "nevermind", "no this is", "new task",
                "stop", "different",
            ]
            if lower.startswith("/") or any(p in lower for p in escape_phrases):
                # User wants out — cancel the clarification task
                old_id = self._pending_clarifications.pop(chat_id, None)
                if old_id:
                    try:
                        await update_task(old_id, status="failed",
                                          error="Clarification dismissed by user")
                    except Exception:
                        pass
                    await self._reply(update,
                        f"🗑 Dismissed clarification for task #{old_id}."
                    )
                # Fall through to normal routing (don't return)
            else:
                # Also check DB to confirm task is still in waiting_human
                try:
                    task_info = await get_task(pending_task_id)
                    if task_info and task_info.get("status") == "waiting_human":
                        await self._resume_with_clarification(
                            chat_id, pending_task_id, text, task_info, update
                        )
                        return
                    else:
                        # Task no longer waiting — stale entry
                        self._pending_clarifications.pop(chat_id, None)
                except Exception:
                    self._pending_clarifications.pop(chat_id, None)

        # Also check DB for ANY task in waiting_human (handles bot restart)
        try:
            db = await get_db()
            cursor = await db.execute(
                """SELECT id, title, description, status, context FROM tasks
                   WHERE status = 'waiting_human'
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
        # PRIORITY 2: Keyword pre-check then LLM classification
        # ═══════════════════════════════════════════════════════
        # Run keyword classifier first — high-confidence pattern types
        # (status_query, todo, etc.) skip the LLM entirely. This prevents
        # small LLMs from misclassifying "how is the coffee machine search
        # going" as shopping just because it contains a product noun.
        keyword_result = self._classify_message_by_keywords(text)
        _KEYWORD_AUTHORITATIVE_TYPES = {
            "status_query", "todo", "load_control", "bug_report",
            "feature_request", "casual",
        }
        if keyword_result["type"] in _KEYWORD_AUTHORITATIVE_TYPES:
            classification = keyword_result
        else:
            classification = await self._classify_user_message(text)

        msg_type = classification["type"]
        msg_workflow = classification.get("workflow")
        logger.info("message classified", msg_type=msg_type,
                    workflow=msg_workflow, text_preview=text[:50])

        # ── Route by classification ──

        if msg_type in ("bug_report", "feature_request", "ui_note", "feedback"):
            await self._handle_user_input(msg_type, text, chat_id, update)
            return

        if msg_type in ("progress_inquiry", "status_query"):
            await self._handle_status_query(text, chat_id, update, context)
            return

        if msg_type == "question":
            task_id = await add_task(
                title=f"Q: {text[:50]}",
                description=text,
                tier="auto",
                priority=TASK_PRIORITY.get("high", 8),
                agent_type="assistant",
            )
            self.user_last_task_id[chat_id] = task_id
            await self._reply(update,f"❓ Task #{task_id} queued.")
            return

        if msg_type == "shopping":
            task_id = await add_task(
                title=text[:80],
                description=text,
                tier="auto",
                priority=TASK_PRIORITY.get("high", 8),
                agent_type="shopping_advisor",
                context={"chat_id": chat_id},
            )
            if task_id is None:
                await self._reply(update,
                    "🛒 A shopping search for this is already in progress.",
                )
                return
            self.user_last_task_id[chat_id] = task_id
            await self._reply(update,
                f"🛒 Shopping task #{task_id} queued.\n"
                f"I'll search prices and compare options for you.",
            )
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
                await self._reply(update,
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
            if msg_workflow == "i2p":
                try:
                    from ..workflows.engine.runner import WorkflowRunner
                    runner = WorkflowRunner()
                    mission_id = await runner.start(
                        workflow_name="i2p_v3",
                        initial_input={"raw_idea": text, "product_name": text[:50]},
                        title=text[:80],
                    )
                    await self._reply(update,
                        f"🔄 Workflow mission #{mission_id} created!\n"
                        f"_{text[:60]}_\n\n"
                        f"Use /wfstatus {mission_id} to track progress.",
                        parse_mode="Markdown",
                    )
                except Exception as e:
                    logger.error("workflow mission failed", error=str(e))
                    await self._reply(update,f"❌ {_friendly_error(str(e))}")
            else:
                mission_id = await add_mission(title=text[:80], description=text, priority=5)
                if self.orchestrator:
                    await self.orchestrator.plan_mission(mission_id, text[:80], text)
                await self._reply(update,
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
            if task_id is None:
                await self._reply(update, "⏳ A similar task is already in progress.")
                return
            self.user_last_task_id[chat_id] = task_id
            await self._reply(update,f"\u2705 Task #{task_id} queued.")

    # ─── Message Classification Helpers ───────────────────────────────────────

    MESSAGE_CLASSIFIER_PROMPT = """Classify this user message to an AI orchestrator. Respond with ONLY valid JSON.

Categories:
- "status_query": asking about the status/progress of an EXISTING task, mission, or search. Examples: "how is the coffee machine search going", "any update on the motherboard", "did you find anything for X", "what's the status of mission 5", "how far along is the CPU task". The user is NOT requesting something new — they want to know about something already in progress.
- "shopping": buying, comparing prices, finding deals, product recommendations, "I want to buy X", "how much is X", budget shopping, upgrading hardware/equipment — anything about purchasing products or services
- "mission": complex multi-step project request (building something, research project, etc.)
- "task": specific actionable request
- "question": asking for information
- "bug_report": reporting a bug or error
- "feature_request": suggesting a new feature
- "ui_note": UI/UX feedback
- "feedback": general feedback on system behavior
- "progress_inquiry": asking about GENERAL system status (no specific task/mission mentioned)
- "followup": continuing a previous conversation or task
- "clarification_response": answering a question the system asked
- "load_control": wanting to control GPU/resources (e.g. "I'm going to game", "free up GPU")
- "todo": adding a reminder, to-do item, or note to self (e.g., "remind me to buy milk", "don't forget the meeting", "add eggs to my list")
- "casual": greeting, thanks, small talk

If type is "mission", also decide if this needs a full product workflow:
- "workflow": "i2p" ONLY if the user explicitly wants to BUILD/CREATE/DEVELOP a new product from scratch
- "workflow": null if it's a simpler mission (research, analysis, fix something, write a report)

IMPORTANT: If the user asks about progress/status/updates on a SPECIFIC existing task or search, classify as "status_query", NOT "shopping" or "mission".
IMPORTANT: Shopping/buying/comparing products is ALWAYS "shopping", never "mission". "Mission" is only for building/creating new projects from scratch.
IMPORTANT: Asking about existing apps/tools is a research TASK or QUESTION, NOT a mission.

Context: {context}

Message: {message}

Respond as: {{"type": "mission", "confidence": 0.9, "workflow": "i2p"}}
Or: {{"type": "task", "confidence": 0.8}}"""

    async def _classify_user_message(self, text: str) -> dict:
        """Classify user message using LLM with keyword fallback.

        Returns dict with at least {"type": str}. May also include
        "workflow" for mission-type messages that need the workflow engine.
        """
        try:
            from ..core.router import ModelRequirements

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
            from ..core.llm_dispatcher import get_dispatcher, CallCategory
            response = await get_dispatcher().request(
                CallCategory.OVERHEAD, reqs, messages,
            )
            from ..core.task_classifier import _extract_json
            raw = response.get("content", "").strip()
            result = _extract_json(raw)
            msg_type = result.get("type", "task")
            confidence = result.get("confidence", 0.5)
            workflow = result.get("workflow")
            logger.debug("llm message classification",
                         type=msg_type, confidence=confidence, workflow=workflow)
            if confidence < 0.4:
                return self._classify_message_by_keywords(text)
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
        lower = text.lower().strip()
        # Very short or purely numeric messages are ambiguous — don't let
        # the LLM classifier turn them into missions/tasks.
        # Exclude meaningful short words that may be clarification responses.
        _SHORT_PASSTHROUGH = {
            "yes", "no", "ok", "evet", "hayır", "yep", "nah", "nope",
            "yea", "yeah", "hay", "aha", "ehm",
        }
        if (len(lower) <= 3 or lower.isdigit()) and lower not in _SHORT_PASSTHROUGH:
            return {"type": "casual"}
        # Todo items
        if any(w in lower for w in [
            "remind me", "don't forget", "dont forget", "todo",
            "add to list", "add to my list", "need to buy", "need to get",
            "remember to", "note to self", "hatirla", "unutma", "listeye ekle",
        ]):
            return {"type": "todo"}
        # Status queries about existing tasks — MUST be checked BEFORE shopping
        _status_phrases = [
            "how is the", "how's the", "how is my", "how's my",
            "status of", "update on", "any update", "any progress",
            "how far along", "how far is", "did you find",
            "have you found", "is it done", "still running",
            "what's the status", "whats the status",
            "where are we on", "where are we with",
            "search going", "task going", "mission going",
            "nasıl gidiyor", "durum ne", "ne durumda",
        ]
        import re
        _mission_id_pattern = re.compile(
            r"(?:mission|task|görev)\s*#?\d+", re.IGNORECASE
        )
        if any(p in lower for p in _status_phrases) or _mission_id_pattern.search(lower):
            return {"type": "status_query"}
        # Shopping
        if any(w in lower for w in [
            "shopping", "buy", "purchase", "fiyat", "price", "compare price",
            "en ucuz", "cheapest", "deal", "indirim", "kampanya",
            "almak istiyorum", "want to buy", "should i buy", "almalı",
            "karşılaştır", "hediye", "gift", "tavsiye", "öneri",
            "upgrade", "yükseltme", "how much", "ne kadar",
        ]):
            return {"type": "shopping"}
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
        # General status / progress questions (no specific task detected above)
        if any(w in lower for w in [
            "how's", "status", "progress", "how far", "eta",
            "what's happening", "still running", "where are we",
        ]):
            return {"type": "status_query"}
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
                result["workflow"] = "i2p"
            return result
        return {"type": "task"}

    async def _is_likely_clarification_response(
        self, text: str, waiting_task: dict
    ) -> bool:
        """Check if message is likely a response to a pending clarification.

        Conservative: only return True when the message clearly answers
        the pending question. New tasks / commands must not be swallowed.
        """
        lower = text.lower().strip()

        # Clearly NOT a clarification response
        not_response_phrases = [
            "new task", "new mission", "let's", "lets ", "i want to",
            "can you", "please ", "do some", "shopping", "research",
            "create", "build", "help me with", "start a", "no this is",
            "forget it", "skip", "cancel", "never mind", "nevermind",
            "ignore", "dismiss", "stop", "different",
        ]
        if any(phrase in lower for phrase in not_response_phrases):
            return False

        # Starts with a slash → command, not a clarification answer
        if lower.startswith("/"):
            return False

        # Very short single-word/phrase answers that directly address
        # the question are likely responses (e.g. "air purifier", "yes", "the blue one")
        task_title = (waiting_task.get("title") or "").lower()
        if len(text) < 60 and not any(c in lower for c in [".", "!", "?"]):
            # Short, declarative — likely an answer
            return True

        # Default: treat as a new message, not a clarification response
        return False

    async def _resume_with_clarification(
        self, chat_id: int, task_id: int, answer: str,
        task_info: dict, update: Update
    ):
        """Resume a task that was waiting for clarification.

        If a sequential question queue is active, collect the answer and
        send the next question.  Only resume the task when all questions
        are answered.
        """
        q = getattr(self, "_pending_clarification_queue", None)

        # Sequential Q&A mode — collect answer and advance
        if q and q.get("task_id") == task_id and q["current"] < len(q["questions"]):
            q["answers"].append(answer)
            q["current"] += 1

            if q["current"] < len(q["questions"]):
                # More questions — persist progress and send next one
                await self._persist_clarification_state(
                    task_id, numbered=q["questions"],
                    current=q["current"], answers=q["answers"],
                )
                await self._reply(update,
                    f"\u2705 Got it ({q['current']}/{len(q['questions'])})"
                )
                await self._send_next_clarification_question()
                return  # Don't resume task yet
            else:
                # All answered — merge answers and resume
                merged = "\n\n".join(
                    f"Q{i+1}: {q['questions'][i]}\nA: {a}"
                    for i, a in enumerate(q["answers"])
                )
                answer = merged  # Use merged as the full clarification
                self._pending_clarification_queue = None
                await self._reply(update,
                    f"\u2705 All {len(q['answers'])} questions answered. Resuming..."
                )

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
            # Clean up persisted clarification state
            ctx.pop("_clarification_queue", None)
            ctx.pop("_clarification_question", None)

            await update_task(
                task_id,
                status="pending",
                context=_json.dumps(ctx),
            )

            # Clean up tracking
            self._pending_clarifications.pop(chat_id, None)
            # Remove all message-ID entries for this task
            self._clarification_msg_ids = {
                mid: tid for mid, tid in self._clarification_msg_ids.items()
                if tid != task_id
            }

            if not q or q.get("task_id") != task_id:
                await self._reply(update,
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
            await self._reply(update,
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
            await self._reply(update,
                f"{emoji} Logged as {input_type} #{input_id}.{mission_str}"
            )
        except Exception as e:
            logger.error("failed to log user input", error=str(e))
            await self._reply(update,f"Logged your {input_type}. (Note: save had an issue, check logs)")

    async def _handle_status_query(self, text: str, chat_id: int, update: Update, context):
        """Look up existing tasks/missions matching the user's status question."""
        import re

        lower = text.lower()

        # 1) Check for explicit mission/task ID reference
        id_match = re.search(r"(?:mission|task|görev)\s*#?(\d+)", lower)
        if id_match:
            ref_id = int(id_match.group(1))
            # Try as mission first, then task
            mission = await get_mission(ref_id)
            if mission:
                tasks = await get_tasks_for_mission(ref_id)
                total = len(tasks)
                done = sum(1 for t in tasks if t["status"] == "completed")
                failed = sum(1 for t in tasks if t["status"] == "failed")
                running = sum(1 for t in tasks if t["status"] in ("processing", "running"))
                pending = total - done - failed - running
                msg = (
                    f"📊 *Mission #{ref_id}*: {mission['title']}\n"
                    f"Status: *{mission['status']}*\n"
                    f"Tasks: {done}✅ {running}🔄 {pending}⏳ {failed}❌ / {total} total"
                )
                # Show latest completed task result preview
                completed = [t for t in tasks if t["status"] == "completed" and t.get("result")]
                if completed:
                    latest = completed[-1]
                    preview = (latest["result"] or "")[:200]
                    msg += f"\n\nLatest result from _{latest['title']}_:\n{preview}"
                await self._reply(update,msg, parse_mode="Markdown")
                return

            task = await get_task(ref_id)
            if task:
                msg = f"📊 *Task #{ref_id}*: {task['title']}\nStatus: *{task['status']}*"
                if task["status"] == "completed" and task.get("result"):
                    preview = (task["result"] or "")[:300]
                    msg += f"\n\nResult:\n{preview}"
                elif task["status"] == "failed" and task.get("error"):
                    msg += f"\n\nError: {task['error'][:200]}"
                await self._reply(update,msg, parse_mode="Markdown")
                return

            await self._reply(update,f"Could not find mission or task #{ref_id}.")
            return

        # 2) Search recent tasks/missions by keyword matching
        # Extract likely subject from the status question
        subject = lower
        for strip in [
            "how is the", "how's the", "how is my", "how's my",
            "what's the status of", "whats the status of",
            "status of", "any update on", "any progress on",
            "did you find anything for", "did you find",
            "have you found anything for", "have you found",
            "how far along is", "how far is",
            "where are we on", "where are we with",
            "the", "my", "search", "task", "going", "?",
            "nasıl gidiyor", "durum ne", "ne durumda",
        ]:
            subject = subject.replace(strip, "")
        subject = subject.strip(" ?.,!")

        matches = []
        if subject:
            # Search active missions
            missions = await get_active_missions()
            for m in missions:
                title_lower = (m.get("title") or "").lower()
                desc_lower = (m.get("description") or "").lower()
                if subject in title_lower or subject in desc_lower:
                    tasks = await get_tasks_for_mission(m["id"])
                    total = len(tasks)
                    done = sum(1 for t in tasks if t["status"] == "completed")
                    running = sum(1 for t in tasks if t["status"] in ("processing", "running"))
                    matches.append(
                        f"• *Mission #{m['id']}*: {m['title']} "
                        f"— {m['status']} ({done}/{total} done, {running} running)"
                    )

            # Search recent tasks (last 20)
            db = await get_db()
            cursor = await db.execute(
                """SELECT id, title, status, result, error, agent_type
                   FROM tasks
                   WHERE parent_task_id IS NULL
                   ORDER BY created_at DESC LIMIT 20"""
            )
            recent_tasks = [dict(r) for r in await cursor.fetchall()]
            for t in recent_tasks:
                title_lower = (t.get("title") or "").lower()
                if subject in title_lower:
                    line = f"• *Task #{t['id']}*: {t['title']} — {t['status']}"
                    if t["status"] == "completed" and t.get("result"):
                        preview = (t["result"] or "")[:150]
                        line += f"\n  _{preview}_"
                    elif t["status"] == "failed" and t.get("error"):
                        line += f"\n  Error: {t['error'][:100]}"
                    matches.append(line)

        if matches:
            header = f"📊 Found {len(matches)} matching item(s):\n\n"
            msg = header + "\n\n".join(matches[:5])
            if len(msg) > 4000:
                msg = msg[:4000] + "\n... (truncated)"
            await self._reply(update,msg, parse_mode="Markdown")
        else:
            # Fallback: show general progress
            await self._reply(update,"📊 No matching tasks found. Showing general progress:")
            await self.cmd_progress(update, context)

    async def _handle_casual(self, text: str, update: Update):
        """Handle casual messages with a quick LLM response (no task creation)."""
        try:
            from ..core.router import ModelRequirements
            from ..core.llm_dispatcher import get_dispatcher, CallCategory
            reqs = ModelRequirements(
                task="assistant",
                agent_type="assistant",
                difficulty=2,
                prefer_speed=True,
                priority=1,
                estimated_input_tokens=100,
                estimated_output_tokens=100,
            )
            response = await get_dispatcher().request(
                CallCategory.OVERHEAD, reqs,
                [{"role": "user", "content": text}],
            )
            reply = response.get("content", "Hey! How can I help?")
            await self._reply(update,reply[:1000])
        except Exception:
            await self._reply(update,"Hey! Send me a task or mission to work on.")

    async def _handle_load_control(self, text: str, update: Update):
        """Handle natural language GPU load control."""
        lower = text.lower()
        if any(w in lower for w in ["game", "gaming", "play"]):
            from ..infra.load_manager import set_load_mode
            msg = await set_load_mode("minimal", source="user")
            await self._reply(update,
                f"\U0001f3ae Switching to minimal mode for gaming.\n{msg}\n"
                "Use `/load auto` when you're done.",
                parse_mode="Markdown",
            )
        elif any(w in lower for w in ["free", "done", "finished", "back"]):
            from ..infra.load_manager import enable_auto_management
            await enable_auto_management()
            await self._reply(update,
                "GPU auto-management re-enabled. I'll use what's available."
            )
        else:
            await self._reply(update,
                "Use `/load full|heavy|shared|minimal|auto` to control GPU usage.",
                parse_mode="Markdown",
            )

    # ─── Todo Commands ──────────────────────────────────────────────────────

    async def cmd_todo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add a todo item. /todo <title>"""
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "todo"}
            await self._reply(update, "📝 What do you need to remember?")
            return

        title = " ".join(context.args)
        todo_id = await add_todo(title)
        buttons = [[InlineKeyboardButton(
            "✅ Done", callback_data=f"todo_toggle:{todo_id}"
        )]]
        await self._reply(update,
            f"📝 Added: *{title}*\n(#{todo_id})",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def cmd_todos(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all todos. /todos"""
        from src.app.reminders import build_todo_list_message
        text, markup = await build_todo_list_message()
        if not text:
            await self._reply(update, "📋 No todo items yet. Use /todo to add one.")
            return
        await self._reply(update, text, parse_mode="Markdown", reply_markup=markup)

    async def cmd_cleartodos(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Delete all completed todos. /cleartodos"""
        done_todos = await get_todos(status="done")
        if not done_todos:
            await self._reply(update,"No completed todos to clear.")
            return

        for todo in done_todos:
            await delete_todo(todo["id"])

        await self._reply(update,
            f"🗑️ Cleared {len(done_todos)} completed todo(s)."
        )

    # ─── Shopping Commands ──────────────────────────────────────────────────

    # ── Two-tier shopping: simple queries vs research missions ──

    _SHOPPING_MISSION_KEYWORDS_TR = {"araştır", "karşılaştır", "analiz et", "detaylı"}
    _SHOPPING_MISSION_KEYWORDS_EN = {"compare", "research", "analyze", "vs"}
    _SHOPPING_MISSION_SUB_INTENTS = {"research", "exploration", "compare"}

    @staticmethod
    def _is_complex_shopping_query(query: str, sub_intent: str | None = None) -> bool:
        """Determine if a shopping query needs a full research mission.

        Returns True for queries that benefit from the 3-task pipeline
        (researcher -> analyst -> advisor).  Simple price checks and
        single-product lookups return False.
        """
        if sub_intent and sub_intent in TelegramInterface._SHOPPING_MISSION_SUB_INTENTS:
            return True

        q_lower = query.lower()

        # Turkish research keywords
        for kw in TelegramInterface._SHOPPING_MISSION_KEYWORDS_TR:
            if kw in q_lower:
                return True

        # English keywords — "vs" requires 2+ products (avoid false positives)
        for kw in TelegramInterface._SHOPPING_MISSION_KEYWORDS_EN:
            if kw == "vs":
                # Must appear between two product-like tokens
                import re
                if re.search(r'\S+\s+vs\.?\s+\S+', q_lower):
                    return True
            elif kw in q_lower:
                return True

        return False

    async def _create_shopping_mission(
        self, query: str, chat_id: int, sub_intent: str | None = None,
    ) -> int:
        """Create a shopping mission with researcher -> analyst -> advisor pipeline."""
        mission_id = await add_mission(
            title=f"Shopping: {query[:60]}",
            description=query,
            priority=8,
            context={"chat_id": chat_id, "shopping_query": query, "sub_intent": sub_intent or "research"},
        )

        # Task 1: Product research (no dependencies)
        task1_id = await add_task(
            title=f"Research: {query[:50]}",
            description=(
                f"Research products for: {query}\n\n"
                "Use shopping_search to find products across sources. "
                f"Write findings to blackboard key 'shopping_top_products' (mission_id={mission_id}). "
                f"Write price comparisons to blackboard key 'shopping_price_comparisons' (mission_id={mission_id})."
            ),
            agent_type="product_researcher",
            priority=8,
            mission_id=mission_id,
            context={
                "chat_id": chat_id,
                "silent": True,
                "mission_id": mission_id,
                "shopping_query": query,
            },
        )

        # Task 2: Deal analysis (depends on research)
        task2_id = await add_task(
            title=f"Analyze deals: {query[:50]}",
            description=(
                f"Analyze deals and timing for: {query}\n\n"
                f"Read blackboard keys 'shopping_top_products' and 'shopping_price_comparisons' (mission_id={mission_id}). "
                "Evaluate value, detect fake discounts, check timing. "
                f"Write findings to blackboard key 'shopping_deal_verdicts' (mission_id={mission_id})."
            ),
            agent_type="deal_analyst",
            priority=8,
            depends_on=[task1_id],
            mission_id=mission_id,
            context={
                "chat_id": chat_id,
                "silent": True,
                "mission_id": mission_id,
                "shopping_query": query,
            },
        )

        # Task 3: Synthesis & recommendation (depends on both)
        task3_id = await add_task(
            title=f"Recommend: {query[:50]}",
            description=(
                f"Final shopping recommendation for: {query}\n\n"
                f"Read ALL blackboard keys: 'shopping_top_products', "
                f"'shopping_price_comparisons', 'shopping_deal_verdicts' (mission_id={mission_id}). "
                "Synthesize into a clear recommendation with top pick, budget option, "
                "alternatives, warnings, and timing advice."
            ),
            agent_type="shopping_advisor",
            priority=7,
            depends_on=[task1_id, task2_id],
            mission_id=mission_id,
            context={
                "chat_id": chat_id,
                "mission_id": mission_id,
                "shopping_query": query,
                "is_synthesis": True,
            },
        )

        logger.info(
            "shopping mission created",
            mission_id=mission_id,
            tasks=[task1_id, task2_id, task3_id],
            query=query[:80],
        )
        return mission_id

    async def cmd_shop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """General shopping assistant. /shop <query>"""
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "shop"}
            await self._reply(update, "🛒 What are you looking for?")
            return
        query = " ".join(context.args)
        chat_id = update.effective_chat.id

        # Two-tier routing: complex queries get a full research mission
        if self._is_complex_shopping_query(query):
            mission_id = await self._create_shopping_mission(query, chat_id)
            await self._reply(
                update,
                f"🔬 Shopping research mission #{mission_id} started.\n"
                "I'll research products, analyze deals, and send you a "
                "recommendation. This may take a few minutes.",
            )
            return

        # Simple query: single shopping_advisor task (existing flow)
        task_id = await add_task(
            title=query[:80],
            description=query,
            tier="auto",
            priority=TASK_PRIORITY.get("high", 8),
            agent_type="shopping_advisor",
            context={"chat_id": chat_id},
        )
        if task_id is None:
            await self._reply(update,
                "🛒 A shopping search for this is already in progress.",
            )
            return
        self.user_last_task_id[chat_id] = task_id
        await self._reply(update,
            f"🛒 Shopping task #{task_id} queued.\n"
            f"I'll search prices and compare options for you.",
        )

    async def cmd_research_product(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Deep product research. /research_product <product>"""
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "research_product"}
            await self._reply(update, "🔍 Which product to research in depth?")
            return
        product = " ".join(context.args)
        chat_id = update.effective_chat.id
        task_id = await add_task(
            title=f"Research: {product[:60]}",
            description=f"Deep research on: {product}. "
                        f"Compare options, check reviews, analyze value, "
                        f"check for fakes/grey market, suggest alternatives.",
            tier="auto",
            priority=TASK_PRIORITY.get("high", 8),
            agent_type="product_researcher",
            context={"shopping_sub_intent": "deep_research", "chat_id": chat_id},
        )
        if task_id is None:
            await self._reply(update, "🔬 A research task for this product is already in progress.")
            return
        await self._reply(update,
            f"🔬 Deep research queued for *{product}* (task #{task_id})",
            parse_mode="Markdown",
        )

    async def cmd_price(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Quick price check. /price <product>"""
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "price"}
            await self._reply(update, "💰 Which product to check prices for?")
            return
        product = " ".join(context.args)
        chat_id = update.effective_chat.id
        task_id = await add_task(
            title=f"Price check: {product[:40]}",
            description=f"Quick price check for: {product}. "
                        f"Find current prices in Turkey, report top 3 options with prices and links.",
            tier="auto",
            priority=TASK_PRIORITY.get("high", 8),
            agent_type="shopping_advisor",
            context={"shopping_sub_intent": "price_check", "chat_id": chat_id},
        )
        if task_id is None:
            await self._reply(update, "🔍 A price check for this is already in progress.")
            return
        await self._reply(update,
            f"🔍 Price check queued for *{product}* (task #{task_id})",
            parse_mode="Markdown",
        )

    async def cmd_watch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set up price watch. /watch <product> [target_price]"""
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "watch"}
            await self._reply(update, "⏰ Which product to watch? (e.g. 'RTX 4070 under 25k')")
            return
        args = list(context.args)
        target_price = None
        # Check if last arg is a number (target price)
        try:
            target_price = float(args[-1].replace(",", "."))
            args = args[:-1]
        except (ValueError, IndexError):
            pass
        product = " ".join(args)
        if not product:
            await self._reply(update,"Usage: /watch <product> [target\\_price]")
            return
        chat_id = update.effective_chat.id
        price_info = f" Target: {target_price} TL." if target_price else ""
        task_id = await add_task(
            title=f"Price watch: {product[:40]}",
            description=f"Set up price watch for: {product}.{price_info} "
                        f"Monitor prices and alert when a good deal appears.",
            tier="auto",
            priority=TASK_PRIORITY.get("normal", 5),
            agent_type="shopping_advisor",
            context={"shopping_sub_intent": "deal_hunt", "chat_id": chat_id,
                     "target_price": target_price},
        )
        if task_id is None:
            await self._reply(update, "👁️ A price watch for this product is already active.")
            return
        msg = f"👁️ Price watch set for *{product}*"
        if target_price:
            msg += f" (target: {target_price} TL)"
        msg += f" — task #{task_id}"
        await self._reply(update, msg, parse_mode="Markdown")

    async def cmd_deals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show currently tracked deals. /deals"""
        try:
            from src.shopping.memory.price_watch import get_all_active_watches
            watches = await get_all_active_watches()
            lines = ["🏷️ *Active Deals & Watches*\n"]
            if watches:
                # Separate watches with price drops (deals) from regular watches
                deals = []
                watching = []
                for w in watches:
                    if (
                        w.get("historical_low") is not None
                        and w.get("current_price") is not None
                        and w["current_price"] < w.get("historical_low", float("inf"))
                    ):
                        deals.append(w)
                    else:
                        watching.append(w)

                if watching:
                    lines.append("*Watching:*")
                    for w in watching[:10]:
                        name = w.get("product_name", "?")
                        price_str = f" — {w['current_price']:.0f} TL" if w.get("current_price") else ""
                        target = f" (target: {w['target_price']:.0f} TL)" if w.get("target_price") else ""
                        lines.append(f"  • {name}{price_str}{target}")

                if deals:
                    lines.append("\n*Recent deals:*")
                    for d in deals[:5]:
                        name = d.get("product_name", "?")
                        current = d.get("current_price", 0)
                        historical = d.get("historical_low", current)
                        lines.append(f"  🔥 {name} — {current:.0f} TL (was {historical:.0f} TL)")

                if not watching and not deals:
                    lines.append("No active price watches.")
            else:
                lines.append("No active price watches.")
            await self._reply(update,"\n".join(lines), parse_mode="Markdown")
        except ImportError:
            await self._reply(update,"Shopping module not yet available.")
        except Exception as e:
            logger.warning("deals command failed", error=str(e))
            await self._reply(update,"Could not fetch deals right now.")

    async def cmd_mystuff(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user profile — owned items, preferences. /mystuff"""
        try:
            from src.shopping.memory import get_user_profile
            chat_id = update.effective_chat.id
            profile = await get_user_profile(chat_id)
            lines = ["📦 *My Stuff*\n"]
            owned = profile.get("owned_items", [])
            if owned:
                lines.append("*Owned items:*")
                for item in owned[:15]:
                    lines.append(f"  • {item.get('name', '?')}")
            prefs = profile.get("preferences", {})
            if prefs:
                lines.append("\n*Preferences:*")
                for k, v in list(prefs.items())[:10]:
                    lines.append(f"  • {k}: {v}")
            if len(lines) == 1:
                lines.append("No items or preferences recorded yet.")
            await self._reply(update,"\n".join(lines), parse_mode="Markdown")
        except ImportError:
            await self._reply(update,"Shopping module not yet available.")
        except Exception as e:
            logger.warning("mystuff command failed", error=str(e))
            await self._reply(update,"Could not fetch your profile right now.")

    async def cmd_compare(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Direct comparison. /compare <product1> vs <product2>"""
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "compare"}
            await self._reply(update, "⚖️ Compare what? (e.g. 'iPhone 15 vs Samsung S24')")
            return
        text = " ".join(context.args)
        # Split on " vs " (case-insensitive)
        import re
        parts = re.split(r"\s+vs\.?\s+", text, flags=re.IGNORECASE)
        if len(parts) < 2:
            await self._reply(update,
                "Please use 'vs' to separate products.\nExample: /compare iPhone 15 vs Samsung S24"
            )
            return
        product1, product2 = parts[0].strip(), parts[1].strip()
        chat_id = update.effective_chat.id
        query = f"{product1} vs {product2}"

        # Comparisons always use the research mission pipeline
        mission_id = await self._create_shopping_mission(
            query, chat_id, sub_intent="compare",
        )
        await self._reply(
            update,
            f"⚖️ Comparison mission #{mission_id} started: *{product1}* vs *{product2}*\n"
            "I'll research both products, analyze deals, and send a recommendation.",
            parse_mode="Markdown",
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
        await self._reply(update,
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

        # ── Primary: message-ID lookup (works for ALL clarification messages) ──
        task_id = self._clarification_msg_ids.get(replied_to.message_id)
        if task_id:
            task_info = await get_task(task_id)
            if task_info and task_info.get("status") == "waiting_human":
                await self._resume_with_clarification(
                    chat_id, task_id, answer, task_info, update
                )
                return

        # ── Fallback: text-based detection (survives restart — IDs are lost) ──
        if "Clarification needed" in text or "Question " in text:
            import re
            match = re.search(r"Task #(\d+)", text)
            if match:
                task_id = int(match.group(1))
                task_info = await get_task(task_id)
                if task_info and task_info.get("status") == "waiting_human":
                    await self._resume_with_clarification(
                        chat_id, task_id, answer, task_info, update
                    )
                    return
            # Sequential questions don't have Task # — check pending_clarifications
            pending_task_id = self._pending_clarifications.get(chat_id)
            if not pending_task_id:
                # DB fallback: any task in waiting_human
                try:
                    db = await get_db()
                    cursor = await db.execute(
                        """SELECT id FROM tasks
                           WHERE status = 'waiting_human'
                           ORDER BY created_at DESC LIMIT 1"""
                    )
                    row = await cursor.fetchone()
                    if row:
                        pending_task_id = row[0]
                except Exception:
                    pass
            if pending_task_id:
                task_info = await get_task(pending_task_id)
                if task_info and task_info.get("status") == "waiting_human":
                    await self._resume_with_clarification(
                        chat_id, pending_task_id, answer, task_info, update
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

        # ── Mission Detail Callbacks ──────────────────────────────
        if data.startswith("m:task:detail:"):
            mission_id = int(data.split(":")[-1])
            try:
                mission = await get_mission(mission_id)
                if not mission:
                    await query.message.reply_text(f"Görev #{mission_id} bulunamadı.")
                    return
                title = mission.get("title", mission.get("description", "?"))[:50]
                status = mission.get("status", "?")
                priority = mission.get("priority", "?")
                status_tr = {"running": "çalışıyor", "pending": "bekliyor",
                            "completed": "tamamlandı",
                            "failed": "başarısız", "waiting_human": "yanıt bekleniyor",
                            "ungraded": "derecelendirme bekleniyor", "cancelled": "iptal"
                            }.get(status, status)
                text = (f"🎯 *{title}*\n"
                        f"Durum: {status_tr} | Öncelik: {priority}")
                buttons = []
                if status in ("running", "pending"):
                    buttons.append([
                        InlineKeyboardButton("🚫 İptal", callback_data=f"m:task:cancel:{mission_id}"),
                    ])
                buttons.append([
                    InlineKeyboardButton("📄 Sonuç", callback_data=f"m:task:result:{mission_id}"),
                ])
                await query.message.reply_text(
                    text, parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(buttons) if buttons else None,
                )
            except Exception as e:
                await query.message.reply_text(f"❌ Hata: {e}")
            return

        if data.startswith("m:task:pause:"):
            mid = int(data.split(":")[-1])
            try:
                await update_mission(mid, status="cancelled")
                await query.message.reply_text(f"🚫 Görev #{mid} iptal edildi.")
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        if data.startswith("m:task:cancel:"):
            mid = int(data.split(":")[-1])
            try:
                await update_mission(mid, status="cancelled")
                await query.message.reply_text(f"🚫 Görev #{mid} iptal edildi.")
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        if data.startswith("m:task:result:"):
            mid = int(data.split(":")[-1])
            try:
                tasks = await get_tasks_for_mission(mid)
                completed = [t for t in tasks if t.get("status") == "completed"]
                if completed:
                    last = completed[-1]
                    result_text = last.get("result", "Sonuç yok.")
                    if len(result_text) > 3000:
                        result_text = result_text[:3000] + "..."
                    await query.message.reply_text(f"📄 Sonuç:\n\n{result_text}")
                else:
                    await query.message.reply_text("📄 Henüz tamamlanmış görev yok.")
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        # ── Debug/DLQ Detail Callbacks ────────────────────────────
        if data.startswith("m:debug:detail:"):
            task_id = int(data.split(":")[-1])
            try:
                task = await get_task(task_id)
                if not task:
                    await query.message.reply_text(f"Görev #{task_id} bulunamadı.")
                    return
                title = (task.get("title") or task.get("description", "?"))[:50]
                agent = task.get("agent_type", "?")
                status = task.get("status", "?")
                error = task.get("error", "")

                # Get model from conversations table
                model = "?"
                try:
                    db = await get_db()
                    cur = await db.execute(
                        """SELECT model_used FROM conversations
                           WHERE task_id = ? AND model_used IS NOT NULL
                           ORDER BY id DESC LIMIT 1""",
                        (task_id,),
                    )
                    row = await cur.fetchone()
                    if row:
                        model = row[0]
                except Exception:
                    pass

                # Calculate duration from started_at / completed_at
                duration = "?"
                try:
                    started = task.get("started_at")
                    completed = task.get("completed_at")
                    if started and completed:
                        s = datetime.fromisoformat(str(started).replace(" ", "T"))
                        c = datetime.fromisoformat(str(completed).replace(" ", "T"))
                        dur = (c - s).total_seconds()
                        duration = f"{dur:.0f}"
                except Exception:
                    pass

                text = (f"🐛 Görev #{task_id}: {title}\n\n"
                        f"Agent: {agent}\n"
                        f"Model: {model}\n"
                        f"Durum: {status}\n"
                        f"Süre: {duration}s")
                if error:
                    text += f"\n\nHata:\n{error[:500]}"
                trace_btn = InlineKeyboardMarkup([[
                    InlineKeyboardButton("📍 Trace", callback_data=f"m:debug:trace:{task_id}"),
                ]])
                await query.message.reply_text(text, reply_markup=trace_btn)
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        if data.startswith("m:debug:trace:"):
            task_id = int(data.split(":")[-1])
            try:
                # Reuse cmd_trace logic
                context.args = [str(task_id)]
                class _CallbackUpdate:
                    def __init__(self, message):
                        self.message = message
                        self.effective_chat = message.chat
                await self.cmd_trace(_CallbackUpdate(query.message), context)
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        if data == "m:debug:skillstats":
            try:
                class _CallbackUpdate:
                    def __init__(self, message):
                        self.message = message
                        self.effective_chat = message.chat
                context.args = []
                await self.cmd_skillstats(_CallbackUpdate(query.message), context)
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        if data.startswith("m:dlq:detail:"):
            task_id = int(data.split(":")[-1])
            try:
                task = await get_task(task_id)
                if not task:
                    await query.message.reply_text(f"Görev #{task_id} bulunamadı.")
                    return
                title = (task.get("title") or task.get("description", "?"))[:50]
                error = task.get("error", "Hata bilgisi yok")
                text = (f"📭 DLQ #{task_id}: {title}\n\n"
                        f"Hata:\n{error[:1000]}")
                retry_btn = InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔄 Tekrar Dene", callback_data=f"m:dlq:retry:{task_id}"),
                ]])
                await query.message.reply_text(text, reply_markup=retry_btn)
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        if data.startswith("m:dlq:retry:"):
            task_id = int(data.split(":")[-1])
            try:
                from ..infra.dead_letter import retry_dlq_task
                await retry_dlq_task(task_id)
                await query.message.reply_text(f"🔄 Görev #{task_id} kuyruğa geri eklendi.")
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        # ── Process Management Callbacks ──────────────────────────
        if data.startswith("m:proc:"):
            action = data.split(":")[-1]
            if action == "back":
                try:
                    await query.delete_message()
                except Exception:
                    pass
                return

            # Confirmation prompts (same pattern as restart/stop)
            if action == "kill_wrapper":
                await query.edit_message_text(
                    "💀 Tüm süreçleri kapatıp Yaşar Usta'yı yeniden başlatmak istediğinden emin misin?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("✅ Evet", callback_data="m:confirm:proc_kill_wrapper"),
                         InlineKeyboardButton("❌ Hayır", callback_data="m:confirm:cancel")],
                    ]))
                return
            if action == "kill_kutai":
                await query.edit_message_text(
                    "🔄 Tüm süreçleri kapatmak istediğinden emin misin?\n"
                    "Yaşar Usta otomatik yeniden başlatacak.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("✅ Evet", callback_data="m:confirm:proc_kill_kutai"),
                         InlineKeyboardButton("❌ Hayır", callback_data="m:confirm:cancel")],
                    ]))
                return
            if action == "kill_kutai_only":
                await query.edit_message_text(
                    "☠️ Kutay'ı öldürmek istediğinden emin misin?\n"
                    "Yaşar Usta otomatik yeniden başlatacak.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("✅ Evet", callback_data="m:confirm:proc_kill_kutai_only"),
                         InlineKeyboardButton("❌ Hayır", callback_data="m:confirm:cancel")],
                    ]))
                return

            if action == "refresh":
                try:
                    text, btn_rows = await self._build_proc_panel()
                    await query.edit_message_text(
                        text, parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(btn_rows))
                except Exception as e:
                    await query.answer(f"Refresh failed: {e}")
                return

            if action == "restart_yazbunu":
                try:
                    # Kill existing yazbunu by PID file
                    pid_file = Path("logs/yazbunu.pid")
                    try:
                        pid = int(pid_file.read_text().strip())
                        os.kill(pid, signal.SIGTERM)
                        pid_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                    # Signal wrapper to start a new one via ensure_yazbunu
                    # (wrapper checks on each loop iteration)
                    await query.answer("📊 Yazbunu yeniden başlatılıyor...")
                    # Wait briefly for old process to die, then refresh
                    import asyncio as _aio
                    await _aio.sleep(2)
                    text, btn_rows = await self._build_proc_panel()
                    await query.edit_message_text(
                        text, parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(btn_rows))
                except Exception as e:
                    await query.answer(f"Restart failed: {e}")
                return

            return

        # ── Lifecycle Confirmation Callbacks ──────────────────────
        if data == "m:confirm:restart":
            try:
                handler = getattr(self, 'cmd_kutai_restart', None)
                if handler:
                    context.args = []
                    class _ConfirmUpdate:
                        def __init__(self, message):
                            self.message = message
                            self.effective_chat = message.chat
                    await handler(_ConfirmUpdate(query.message), context)
                else:
                    await query.message.reply_text("❌ Restart handler bulunamadı.")
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        if data == "m:confirm:stop":
            try:
                handler = getattr(self, 'cmd_kutai_stop', None)
                if handler:
                    context.args = ["--force"]
                    class _ConfirmUpdate:
                        def __init__(self, message):
                            self.message = message
                            self.effective_chat = message.chat
                    await handler(_ConfirmUpdate(query.message), context)
                else:
                    await query.message.reply_text("❌ Stop handler bulunamadı.")
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        if data == "m:confirm:proc_kill_kutai_only":
            import subprocess as _sp
            try:
                raw = _sp.check_output(
                    ['wmic', 'process', 'where', "name='python.exe'",
                     'get', 'ProcessId,CommandLine'],
                    text=True, timeout=5,
                )
                killed = []
                for line in raw.strip().splitlines():
                    line = line.strip()
                    if not line or line.startswith("CommandLine"):
                        continue
                    pid = line.split()[-1]
                    if "run.py" in line:
                        try:
                            _sp.run(['taskkill', '/F', '/PID', pid],
                                    capture_output=True, timeout=5)
                            killed.append(pid)
                        except Exception:
                            pass
                msg = (f"☠️ Kutay killed: PID {', '.join(killed) if killed else 'none'}\n"
                       f"⏳ Yaşar Usta otomatik yeniden başlatacak...")
            except Exception as e:
                msg = f"❌ Kill failed: {e}"
            try:
                await query.edit_message_text(msg)
            except Exception:
                await query.message.reply_text(msg)
            return

        if data in ("m:confirm:proc_kill_wrapper", "m:confirm:proc_kill_kutai"):
            import subprocess as _sp
            start_wrapper = data.endswith("kill_wrapper")
            try:
                raw = _sp.check_output(
                    ['wmic', 'process', 'where', "name='python.exe'",
                     'get', 'ProcessId,CommandLine'],
                    text=True, timeout=5,
                )
                # Collect PIDs to kill — wrappers first, then orchestrators
                wrapper_pids = []
                orch_pids = []
                my_pid = os.getpid()
                for line in raw.strip().splitlines():
                    line = line.strip()
                    if not line or line.startswith("CommandLine"):
                        continue
                    pid = line.split()[-1]
                    try:
                        pid_int = int(pid)
                    except ValueError:
                        continue
                    if pid_int == my_pid:
                        continue
                    if "wrapper" in line.lower():
                        wrapper_pids.append(pid)
                    elif "run.py" in line:
                        orch_pids.append(pid)

                # Spawn new wrapper BEFORE killing (it will wait on lock until old one dies)
                if start_wrapper:
                    import sys
                    _sp.Popen(
                        [sys.executable, "kutai_wrapper.py"],
                        cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__)))),
                        creationflags=0x00000008,  # DETACHED_PROCESS
                    )

                # Kill wrappers first (releases lock for the new one)
                killed = []
                for pid in wrapper_pids:
                    try:
                        _sp.run(['taskkill', '/F', '/PID', pid],
                                capture_output=True, timeout=5)
                        killed.append(pid)
                    except Exception:
                        pass
                # Clean lock files so new wrapper can start
                for lf in ["logs/wrapper.lock", "logs/wrapper.lk"]:
                    try:
                        os.remove(lf)
                    except Exception:
                        pass
                # Kill orchestrators (including self — do this LAST)
                for pid in orch_pids:
                    try:
                        _sp.run(['taskkill', '/F', '/PID', pid],
                                capture_output=True, timeout=5)
                        killed.append(pid)
                    except Exception:
                        pass

                if start_wrapper:
                    msg = (f"💀 Killed: PID {', '.join(killed) if killed else 'none'}\n"
                           f"✅ Yaşar Usta başlatıldı.")
                else:
                    msg = (f"💀 Killed: PID {', '.join(killed) if killed else 'none'}\n"
                           f"⏳ /kutai\\_start gönderin.")
            except Exception as e:
                msg = f"❌ Kill failed: {e}"
            # This message may not arrive if we kill ourselves first,
            # but the new wrapper's "Bennn Yaşar Usta" will confirm success
            try:
                await query.edit_message_text(msg, parse_mode="Markdown")
            except Exception:
                pass
            return

        if data == "m:confirm:cancel":
            try:
                await query.edit_message_text("❌ İptal edildi.")
            except Exception:
                pass
            return

        # ── Pharmacy Callbacks ─────────────────────────────────
        if data == "pharm:all":
            await query.answer("Yükleniyor...")
            try:
                from src.infra.db import get_user_pref
                city = await get_user_pref("location_city")
                lat = float(await get_user_pref("location_lat") or 0)
                lon = float(await get_user_pref("location_lon") or 0)
                from src.tools.pharmacy import (
                    find_pharmacies_structured, format_pharmacy_message,
                )
                # Skip OSRM for full list — haversine sorting is enough
                pharmacies = await find_pharmacies_structured(
                    city=city, include_route=False,
                    user_lat=lat, user_lon=lon,
                )
                if pharmacies:
                    # Paginate: send in chunks of 10 to avoid message too long
                    header = f"🏥 Tüm Nöbetçi Eczaneler — {city} ({len(pharmacies)})\n\n"
                    chunk_size = 10
                    for i in range(0, len(pharmacies), chunk_size):
                        chunk = pharmacies[i:i + chunk_size]
                        text = format_pharmacy_message(chunk, show_all=True, start_index=i + 1)
                        msg = header + text if i == 0 else text
                        # Truncate if still too long
                        if len(msg) > 4000:
                            msg = msg[:3990] + "\n..."
                        await query.message.reply_text(msg)
                else:
                    await query.message.reply_text("Eczane bulunamadı.")
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        # ── Todo Callbacks ─────────────────────────────────────
        if data.startswith("todo_detail:"):
            try:
                todo_id = int(data.split(":")[1])
            except (ValueError, IndexError):
                await query.answer("Invalid todo ID")
                return
            todo = await get_todo(todo_id)
            if not todo:
                await query.answer("Todo not found")
                return
            from src.app.reminders import build_todo_detail_message
            text, markup = build_todo_detail_message(todo)
            try:
                await query.edit_message_text(
                    text, parse_mode="Markdown", reply_markup=markup,
                )
            except Exception:
                await query.message.reply_text(
                    text, parse_mode="Markdown", reply_markup=markup,
                )
            return

        if data.startswith("todo_toggle:"):
            try:
                todo_id = int(data.split(":")[1])
            except (ValueError, IndexError):
                await query.answer("Invalid todo ID")
                return
            new_status = await toggle_todo(todo_id)
            icon = "✅" if new_status == "done" else "⬜"
            await query.answer(f"{icon} {'Done!' if new_status == 'done' else 'Reopened'}")
            # Return to list view
            from src.app.reminders import build_todo_list_message
            text, markup = await build_todo_list_message()
            if text:
                try:
                    await query.edit_message_text(
                        text, parse_mode="Markdown", reply_markup=markup,
                    )
                except Exception:
                    pass
            else:
                await query.edit_message_text("🎉 All todos done!")
            return

        if data == "todo_list":
            from src.app.reminders import build_todo_list_message
            text, markup = await build_todo_list_message()
            if text:
                try:
                    await query.edit_message_text(
                        text, parse_mode="Markdown", reply_markup=markup,
                    )
                except Exception:
                    pass
            else:
                await query.edit_message_text("🎉 All todos done!")
            return

        if data == "todo_close":
            try:
                await query.delete_message()
            except Exception:
                await query.edit_message_text("(closed)")
            return

        if data.startswith("todo_edit:"):
            try:
                todo_id = int(data.split(":")[1])
            except (ValueError, IndexError):
                await query.answer("Invalid todo ID")
                return
            todo = await get_todo(todo_id)
            if not todo:
                await query.answer("Todo not found")
                return
            chat_id = query.message.chat_id
            self._pending_action[chat_id] = {
                "command": "_todo_edit",
                "todo_id": todo_id,
                "ts": _time.time(),
            }
            # Prefill keyboard with current title for easy editing
            prefill_kb = ReplyKeyboardMarkup(
                [[KeyboardButton(todo["title"])]],
                resize_keyboard=True,
                one_time_keyboard=True,
            )
            await query.message.reply_text(
                f"✏️ Edit todo *#{todo_id}*\nType the new title (keyboard prefilled):",
                parse_mode="Markdown",
                reply_markup=prefill_kb,
            )
            return

        if data.startswith("todo_delete:"):
            try:
                todo_id = int(data.split(":")[1])
            except (ValueError, IndexError):
                await query.answer("Invalid todo ID")
                return
            await delete_todo(todo_id)
            await query.answer("❌ Deleted")
            # Return to list
            from src.app.reminders import build_todo_list_message
            text, markup = await build_todo_list_message()
            if text:
                try:
                    await query.edit_message_text(
                        text, parse_mode="Markdown", reply_markup=markup,
                    )
                except Exception:
                    pass
            else:
                await query.edit_message_text("🎉 All todos done!")
            return

        if data.startswith("todo_help:"):
            parts = data.split(":")
            try:
                todo_id = int(parts[1])
            except (ValueError, IndexError):
                await query.answer("Invalid todo ID")
                return
            agent_type = parts[2] if len(parts) > 2 else "researcher"
            todo = await get_todo(todo_id)
            if not todo:
                await query.answer("Todo not found")
                return
            suggestion = todo.get("suggestion") or f"Help me with: {todo['title']}"
            chat_id = query.message.chat_id
            self._pending_action[chat_id] = {
                "command": "_todo_help",
                "todo_id": todo_id,
                "todo_title": todo["title"],
                "suggestion": suggestion,
                "agent_type": agent_type,
                "ts": _time.time(),
            }
            # Prefill keyboard with suggestion
            prefill_kb = ReplyKeyboardMarkup(
                [[KeyboardButton(suggestion[:60] if len(suggestion) > 60 else suggestion)]],
                resize_keyboard=True,
                one_time_keyboard=True,
            )
            await query.message.reply_text(
                f"🤖 *Help with: {todo['title']}*\n\n"
                f"💡 _{suggestion}_\n\n"
                f"Tap the suggestion, type a custom request, or /cancel.",
                parse_mode="Markdown",
                reply_markup=prefill_kb,
            )
            return

        # ── Workflow Status Callbacks ────────────────────────────
        if data == "wfstatus_dismiss":
            try:
                await query.delete_message()
            except Exception:
                await query.edit_message_text("(dismissed)")
            return

        if data.startswith("wfstatus:"):
            mission_id = int(data.split(":", 1)[1])
            try:
                from ..workflows.engine.status import (
                    compute_phase_progress, format_status_message,
                )
                mission = await get_mission(mission_id)
                if not mission:
                    await query.edit_message_text(f"Mission #{mission_id} not found.")
                    return
                tasks = await get_tasks_for_mission(mission_id)
                if not tasks:
                    await query.edit_message_text(f"No tasks found for mission #{mission_id}.")
                    return
                mission_ctx = mission.get("context", "{}")
                if isinstance(mission_ctx, str):
                    import json as _json
                    try:
                        mission_ctx = _json.loads(mission_ctx)
                    except (ValueError, TypeError):
                        mission_ctx = {}
                workflow_name = mission_ctx.get("workflow_name", "i2p_v3")
                progress = compute_phase_progress(tasks)
                msg = format_status_message(workflow_name, mission_id, progress)
                cancel_button = InlineKeyboardMarkup([[
                    InlineKeyboardButton("🗑 Cancel Mission", callback_data=f"wfcancel:{mission_id}")
                ]])
                await query.edit_message_text(msg, reply_markup=cancel_button)
            except Exception as e:
                await query.edit_message_text(f"❌ {_friendly_error(str(e))}")
            return

        if data.startswith("wfcancel:"):
            mission_id = int(data.split(":", 1)[1])
            try:
                await update_mission(mission_id, status="cancelled")
                await query.edit_message_text(f"🗑 Mission #{mission_id} has been cancelled.")
            except Exception as e:
                await query.edit_message_text(f"❌ {_friendly_error(str(e))}")
            return

        if data.startswith("ss:"):
            await self._handle_smartsearch_callback(query, data)
            return

        # ── Reset Tasks confirm ──────────────────────────────────
        if data == "reset_tasks_confirm":
            # Cancel any in-progress task futures first
            cancelled = 0
            if self.orchestrator:
                for fut in list(getattr(self.orchestrator, "_running_futures", [])):
                    if not fut.done():
                        fut.cancel()
                        cancelled += 1
                self.orchestrator._running_futures = []
                self.orchestrator._current_task_future = None

            db = await get_db()
            await db.execute("DELETE FROM tasks")
            await db.execute("DELETE FROM missions")
            await db.execute("DELETE FROM dead_letter_tasks")
            await db.execute("DELETE FROM workflow_checkpoints")
            await db.execute("DELETE FROM blackboards")
            await db.execute("DELETE FROM approval_requests")
            await db.execute("DELETE FROM scheduled_tasks")
            await db.commit()
            # Also wipe shopping cache (separate DB)
            try:
                import aiosqlite as _aiosqlite
                from ..shopping.cache import CACHE_DB_PATH as _CACHE_DB_PATH
                async with _aiosqlite.connect(_CACHE_DB_PATH) as sdb:
                    await sdb.execute("DELETE FROM products")
                    await sdb.execute("DELETE FROM reviews")
                    await sdb.execute("DELETE FROM price_history")
                    await sdb.execute("DELETE FROM search_cache")
                    await sdb.commit()
                shopping_note = " + alışveriş cache"
            except Exception as e:
                logger.warning(f"Shopping cache reset failed: {e}")
                shopping_note = " (alışveriş cache silinemedi)"
            cancel_note = f", {cancelled} aktif görev iptal edildi" if cancelled else ""
            await query.edit_message_text(f"🗑 Görevler, misyonlar ve alışveriş cache silindi{shopping_note}{cancel_note}.")
            return
        elif data == "reset_tasks_cancel":
            await query.edit_message_text("İptal edildi.")
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

        if data == "stop_confirm":
            await query.edit_message_text("⏹ Kutay durduruluyor...")
            await self._do_kutai_stop()
            return
        elif data == "stop_cancel":
            await query.edit_message_text("İptal edildi. Kutay çalışmaya devam ediyor.")
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
                await update_task(task_id, status="cancelled")
                await query.edit_message_text(f"❌ Task #{task_id} cancelled.")

    # --- Outbound notifications ---

    async def send_notification(self, text: str, retries: int = 2):
        """Send a notification message. Returns the sent Message or None."""
        import asyncio as _asyncio

        # Phase 8.3: Redact secrets from outgoing messages
        try:
            from ..security.sensitivity import redact_secrets
            text = redact_secrets(text)
        except Exception:
            pass

        for attempt in range(retries + 1):
            try:
                msg = await self.app.bot.send_message(
                    chat_id=TELEGRAM_ADMIN_CHAT_ID,
                    text=text,
                    parse_mode="Markdown",
                    reply_markup=REPLY_KEYBOARD,
                )
                return msg
            except Exception as e:
                # First fallback: retry without markdown
                try:
                    msg = await self.app.bot.send_message(
                        chat_id=TELEGRAM_ADMIN_CHAT_ID, text=text,
                        reply_markup=REPLY_KEYBOARD,
                    )
                    return msg
                except Exception:
                    if attempt < retries:
                        await _asyncio.sleep(1 * (attempt + 1))
                        continue
                    logger.error("Failed to send Telegram notification", error=str(e))
        return None

    async def send_result(self, task_id, title, result, model, cost,
                          mission_id=None):
        # Handle long results (>3000 chars) by sending as file attachment
        if len(result) > 3000:
            # Use mission-specific directory if available
            # Import WORKSPACE_DIR to match where write_file puts files
            try:
                from src.tools.workspace import WORKSPACE_DIR as _ws_dir
            except ImportError:
                _ws_dir = "workspace"
            if mission_id:
                results_dir = Path(_ws_dir) / f"mission_{mission_id}"
            else:
                results_dir = Path(_ws_dir) / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Save full result to file — use step ID from title if available
            # Title format: "[0.3] assumption_identification" → "0.3_assumption_identification"
            safe_name = f"task_{task_id}"
            if title and title.startswith("["):
                try:
                    step_part = title.split("]", 1)[0].strip("[")
                    name_part = title.split("]", 1)[1].strip().replace(" ", "_")[:40]
                    safe_name = f"{step_part}_{name_part}"
                except (IndexError, ValueError):
                    pass
            result_file = results_dir / f"{safe_name}.md"
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
                        filename=f"{safe_name}.md",
                        caption=f"📎 Full result for task #{task_id} ({safe_name}.md)"
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
        """Send clarification request.

        If the question contains numbered items (1. ... 2. ... etc.),
        split into individual messages sent one at a time.  Each answer
        is collected sequentially; the merged set is stored when done.
        """
        chat_id = TELEGRAM_ADMIN_CHAT_ID
        if not chat_id:
            return

        int_chat = int(chat_id)

        # Try to split numbered questions
        # Matches: "1. ", "1) ", "### 1. ", "### 1) ", "**1.**", "**1)**"
        import re as _re
        parts = _re.split(r'\n(?=(?:#{1,4}\s*)?\*{0,2}\d+[\.\)]\s)', question.strip())
        # Filter to actual numbered questions
        numbered = [p.strip() for p in parts
                    if _re.match(r'^(?:#{1,4}\s*)?\*{0,2}\d+[\.\)]\s', p.strip())]

        if len(numbered) >= 2:
            # Sequential Q&A mode
            self._pending_clarification_queue = {
                "task_id": task_id,
                "title": title,
                "questions": numbered,
                "current": 0,
                "answers": [],
            }
            self._pending_clarifications[int_chat] = task_id

            # Persist Q&A state to task context (survives restart)
            await self._persist_clarification_state(task_id, numbered=numbered)

            header = (
                f"\u2753 *Clarification needed \u2014 Task #{task_id}*\n"
                f"**{title}**\n\n"
                f"I have {len(numbered)} questions. Answering one at a time:\n"
            )
            msg = await self.send_notification(header)
            if msg:
                self._clarification_msg_ids[msg.message_id] = task_id
            # Send first question
            await self._send_next_clarification_question()
        else:
            # Single question or free-form — original behavior
            self._pending_clarifications[int_chat] = task_id

            # Persist question text to task context (survives restart)
            await self._persist_clarification_state(task_id, question=question)

            msg = await self.send_notification(
                f"\u2753 *Clarification needed \u2014 Task #{task_id}*\n"
                f"**{title}**\n\n"
                f"{question}\n\n"
                f"_Reply to this message or just type your answer._"
            )
            if msg:
                self._clarification_msg_ids[msg.message_id] = task_id

    async def _send_next_clarification_question(self):
        """Send the next question from the sequential clarification queue."""
        q = getattr(self, "_pending_clarification_queue", None)
        if not q:
            return
        idx = q["current"]
        questions = q["questions"]
        if idx >= len(questions):
            return
        total = len(questions)
        question_text = questions[idx]
        msg = await self.send_notification(
            f"*Question {idx + 1}/{total}:*\n\n"
            f"{question_text}\n\n"
            f"_Type your answer:_"
        )
        if msg:
            self._clarification_msg_ids[msg.message_id] = q["task_id"]

    async def _persist_clarification_state(
        self, task_id: int, *,
        question: str = "",
        numbered: list[str] | None = None,
        current: int = 0,
        answers: list[str] | None = None,
    ):
        """Save clarification state into the task's context so it survives restart.

        For single questions: stores _clarification_question.
        For sequential Q&A: stores _clarification_queue with questions/current/answers.
        """
        import json as _json
        try:
            task_info = await get_task(task_id)
            if not task_info:
                return
            existing_ctx = task_info.get("context", "{}")
            if isinstance(existing_ctx, str):
                try:
                    ctx = _json.loads(existing_ctx)
                except (ValueError, TypeError):
                    ctx = {}
            else:
                ctx = existing_ctx or {}

            if numbered:
                ctx["_clarification_queue"] = {
                    "questions": numbered,
                    "current": current,
                    "answers": answers or [],
                }
                ctx.pop("_clarification_question", None)
            elif question:
                ctx["_clarification_question"] = question
                ctx.pop("_clarification_queue", None)

            await update_task(task_id, context=_json.dumps(ctx))
        except Exception as e:
            logger.debug("Failed to persist clarification state",
                         task_id=task_id, error=str(e))

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

    async def _handle_todo_edit(self, update, context, pending):
        """Handle the user's reply with a new todo title."""
        todo_id = pending["todo_id"]
        new_title = (update.message.text or "").strip()
        if not new_title:
            await self._reply(update, "Title can't be empty.")
            return
        # Reset suggestion so it gets regenerated next cycle
        await update_todo(
            todo_id, title=new_title,
            suggestion=None, suggestion_agent=None, suggestion_at=None,
        )
        await self._reply(update, f"✏️ Updated *#{todo_id}*: {new_title}", parse_mode="Markdown")

    async def cmd__todo_help(self, update, context):
        """Handle the user's reply to a todo help suggestion."""
        text = update.message.text or ""
        todo_info = getattr(self, "_last_todo_help", None)
        if not todo_info:
            await self._reply(update, "❌ Help session expired. Try again from the reminder.")
            return
        self._last_todo_help = None
        todo_id = todo_info["todo_id"]
        todo_title = todo_info["todo_title"]
        suggestion = todo_info.get("suggestion", "")
        agent_type = todo_info.get("agent_type", "researcher")
        # Handle prefilled keyboard taps and text responses
        stripped = text.strip()
        if stripped == "❌ Cancel":
            await self._reply(update, "(cancelled)")
            return
        # Accept via prefilled keyboard button (starts with ✅)
        if stripped.startswith("✅") and suggestion:
            description = suggestion
        elif suggestion and stripped.lower() in ("yes", "ok", "do it", "evet", "tamam"):
            description = suggestion
        elif suggestion:
            description = f"User request: {text}\nOriginal suggestion: {suggestion}"
        else:
            description = text or f"Help with: {todo_title}"
        task_id = await add_task(
            title=f"Help with: {todo_title[:40]}",
            description=description,
            agent_type=agent_type,
            tier="auto",
            priority=6,
            context={"todo_id": todo_id},
        )
        await self._reply(update, f"✅ Task #{task_id} created!")

    async def _handle_smartsearch_callback(self, query, data: str):
        """Handle /smartsearch inline button callbacks."""
        action = data.split(":", 1)[1] if ":" in data else ""

        if action == "refresh":
            try:
                from src.tools.free_apis import discover_new_apis, build_keyword_index
                new_count = await discover_new_apis("all")
                await build_keyword_index()
                await query.edit_message_text(f"Discovery complete: {new_count} new APIs found.")
            except Exception as e:
                await query.edit_message_text(f"Discovery failed: {e}")

        elif action == "failures":
            try:
                from src.infra.db import get_api_reliability_all
                all_rel = await get_api_reliability_all()
                failures = [r for r in all_rel if r["failure_count"] > 0]
                failures.sort(key=lambda r: r["failure_count"], reverse=True)
                if not failures:
                    await query.edit_message_text("No failures recorded.")
                    return
                lines = ["API Failures (all time)", "-" * 25]
                for r in failures[:15]:
                    total = r["success_count"] + r["failure_count"]
                    rate = int(r["success_count"] / max(total, 1) * 100)
                    lines.append(f"{r['api_name']}: {r['failure_count']} failures ({rate}% success) [{r['status']}]")
                await query.edit_message_text("\n".join(lines))
            except Exception as e:
                await query.edit_message_text(f"Error: {e}")

        elif action == "unsuspend":
            try:
                from src.infra.db import get_api_reliability_all, unsuspend_api
                all_rel = await get_api_reliability_all()
                suspended = [r for r in all_rel if r["status"] in ("suspended", "demoted")]
                for r in suspended:
                    await unsuspend_api(r["api_name"])
                await query.edit_message_text(f"Unsuspended {len(suspended)} APIs. Counters reset.")
            except Exception as e:
                await query.edit_message_text(f"Error: {e}")

        elif action == "list_apis":
            try:
                from src.tools.free_apis import API_REGISTRY, _db_api_cache
                lines = ["📦 API Registry", "=" * 25]

                # Static APIs — show all (small list)
                lines.append(f"\nStatic ({len(API_REGISTRY)}):")
                for api in API_REGISTRY:
                    lines.append(f"  [{api.category}] {api.name}")

                # Discovered APIs — summary by category only
                if _db_api_cache:
                    cats = {}
                    for api in _db_api_cache:
                        cats.setdefault(api.category, []).append(api.name)
                    lines.append(f"\nDiscovered ({len(_db_api_cache)}) by category:")
                    for cat, names in sorted(cats.items(), key=lambda x: -len(x[1])):
                        lines.append(f"  {cat}: {len(names)} APIs")
                else:
                    lines.append("\nNo discovered APIs (run Refresh)")

                text = "\n".join(lines)
                if len(text) > 4000:
                    text = text[:4000] + "\n... (truncated)"
                await query.edit_message_text(text)
            except Exception as e:
                await query.edit_message_text(f"Error: {e}")
