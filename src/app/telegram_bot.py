# telegram_bot.py
import asyncio
import os
import signal
from datetime import datetime
from pathlib import Path
from src.infra.logging_config import get_logger
from ..infra.times import utc_now, to_turkey, to_db, TZ_TR, tr_hour_to_utc
from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID, TASK_PRIORITY, DB_PATH

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
    if "no models available" in e or "no models matched" in e or "no model candidates" in e:
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


async def _enqueue_inline_chat(
    *,
    title: str,
    description: str,
    agent_type: str,
    kind: str,
    llm_call_kwargs: dict,
    parent_id: int | None = None,
) -> dict:
    """Enqueue an LLM call as a Beckman task with await_inline=True.

    Returns the dispatcher response dict (same shape today's
    dispatcher.request returns), so callers can do
    `response.get("content", ...)` unchanged.
    """
    import general_beckman
    spec = {
        "title": title,
        "description": description,
        "agent_type": agent_type,
        "kind": kind,
        "context": {
            "llm_call": {
                "raw_dispatch": True,
                **llm_call_kwargs,
            },
        },
    }
    tr = await general_beckman.enqueue(spec, parent_id=parent_id, await_inline=True)
    if tr.status == "failed":
        from src.core.router import ModelCallFailed
        raise ModelCallFailed(tr.error or "call failed", error_category="availability")
    res = tr.result
    if isinstance(res, str):
        import json
        try:
            res = json.loads(res)
        except Exception:
            res = {"content": res}
    return res or {}


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
    ["🎯 Görevler", "📬 İş Kuyruğu", "⚙️ Sistem"],
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
    ["🎯 Yeni Görev", "⏰ Zamanla"],
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
    ["🗑 Reset Tasks", "☢️ Reset All"],
    ["🔙 Geri"],
])

KB_YUK_MODU = _make_keyboard([
    ["⚡ Full", "🔋 Heavy", "⚖️ Shared"],
    ["🔻 Minimal", "🤖 Otomatik"],
    ["🔙 Geri"],
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
    "🔬 Detaylı Araştır": ("cmd", "research_product"),
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
    "🗑 Reset Tasks": ("special", "reset_tasks"),
    "☢️ Reset All": ("cmd", "reset_all"),
    # ── Yük Modu sub-buttons ──
    "⚡ Full": ("special", "load_full"),
    "🔋 Heavy": ("special", "load_heavy"),
    "⚖️ Shared": ("special", "load_shared"),
    "🔻 Minimal": ("special", "load_minimal"),
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
_PENDING_ACTION_TIMEOUT = 1800

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


_TG_INSTANCE: "TelegramInterface | None" = None


def get_telegram() -> "TelegramInterface":
    """Module-level accessor for the singleton TelegramInterface.

    Raises RuntimeError if called before the interface is constructed. Used by
    mechanical executors (e.g. mr_roboto clarify/notify_user) to reach Telegram
    without tight coupling.
    """
    if _TG_INSTANCE is None:
        raise RuntimeError("telegram not initialized")
    return _TG_INSTANCE


def set_telegram(instance: "TelegramInterface") -> None:
    global _TG_INSTANCE
    _TG_INSTANCE = instance


async def enqueue_launch_mission(spec: dict, **kwargs) -> int:
    """Z7 T3A (A2) — Thin wrapper to enqueue a launch mission via Beckman.

    Separated from cmd_launch so tests can monkeypatch this function
    without importing the full TelegramInterface.
    """
    from general_beckman import enqueue as beckman_enqueue
    return await beckman_enqueue(spec, **kwargs)


class TelegramInterface:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self._setup_handlers()
        set_telegram(self)
        self._approval_events = {}
        self._clarification_events = {}
        self.user_last_task_id = {}
        # Explicit clarification tracking: chat_id → task_id
        # TODO(Z10 T2B / D6): the mission-scoped subset of this mapping
        # belongs to ``mission_events`` (kind='asking'). When a task that
        # owns a clarification is tied to a mission_id, the future
        # ``post_event(kind='asking', ...)`` path replaces this dict for
        # that task. Standalone /ask + sequential Q&A queues keep using
        # this dict — they aren't mission-events. Audit:
        #   * `restore_clarification_state` (line ~388) → keeps using dict
        #   * `handle_reply` (line ~5947) → keeps text fallback
        #   * `_resume_with_clarification` (search) → standalone path
        # All ad-hoc clarifications continue to live here; mission-scoped
        # clarifications should be created via ``post_event(kind='asking')``
        # going forward.
        self._pending_clarifications: dict[int, int] = {}
        # Reverse lookup: message_id → task_id for reply-to detection
        self._clarification_msg_ids: dict[int, int] = {}
        # Conversation flow: chat_id → {"command": str, "ts": float} for button-initiated arg prompts
        self._pending_action: dict[int, dict] = {}
        # Shopping intent fork: chat_id → "specific" | "category"
        self._pending_shop_subintent: dict[int, str] = {}
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
                    # Single-saved question OR recovered from mechanical
                    # child. Route through request_clarification so its
                    # numbered-question splitter (same one used on the
                    # original send) turns a multi-Q payload back into a
                    # sequential Q&A queue instead of dumping all five
                    # onto one screen.
                    clarification_q = ctx.get("_clarification_question", "")
                    if not clarification_q:
                        clarification_q = await self._recover_question_from_child(
                            db, task_id,
                        )
                    if clarification_q:
                        await self.request_clarification(
                            task_id, title, clarification_q,
                        )
                        logger.info("Restored clarification state",
                                    task_id=task_id)
                    else:
                        logger.info("Found waiting_human task without "
                                    "saved question", task_id=task_id)
                break  # One clarification at a time

        except Exception as e:
            logger.exception(
                "Failed to restore clarification state: %s", e,
            )

    async def provision_mission_thread(
        self, chat_id: int, mission_id: int, title: str,
    ) -> "int | None":
        """Create a forum topic for the mission. Returns thread_id or None on failure.

        Caller must persist returned thread_id to missions.message_thread_id.
        """
        topic_name = f"#{mission_id} {title[:40]}"
        try:
            topic = await self.bot.create_forum_topic(chat_id=chat_id, name=topic_name)
            thread_id = topic.message_thread_id
        except Exception as e:
            logger.warning(
                "forum topic creation failed for mission %d: %s — falling back to tag-prefix",
                mission_id, e,
            )
            return None

        try:
            msg = await self.bot.send_message(
                chat_id=chat_id,
                message_thread_id=thread_id,
                text=self._format_pinned_status(mission_id, title, spent=0, ceiling=None),
            )
            await self.bot.pin_chat_message(
                chat_id=chat_id, message_id=msg.message_id, disable_notification=True,
            )
        except Exception as e:
            logger.warning("pin_chat_message failed: %s", e)

        return thread_id

    def _format_pinned_status(
        self, mission_id: int, title: str,
        spent: float, ceiling: "float | None",
        state: str = "active",
        tasks_done: int = 0, tasks_running: int = 0, tasks_queued: int = 0,
    ) -> str:
        if ceiling:
            pct = (spent / ceiling) * 100 if ceiling > 0 else 0
            budget_line = f"Spent: ${spent:.2f} / ${ceiling:.2f} ({pct:.1f}%)"
        else:
            budget_line = f"Spent: ${spent:.2f} (no ceiling)"
        return (
            f"Mission #{mission_id} — \"{title}\"\n"
            f"Status: {state}\n"
            f"{budget_line}\n"
            f"Tasks: {tasks_done} done, {tasks_running} in flight, {tasks_queued} queued"
        )

    def _is_admin_chat(self, chat_id) -> bool:
        """Return True if ``chat_id`` is the configured admin/owner chat.

        When ``TELEGRAM_ADMIN_CHAT_ID`` is unset the bot is single-tenant
        and every chat is treated as the owner (gate is a no-op). When it
        is set, only that chat passes — used to gate privileged commands
        (e.g. /lifecycle, which triggers real email sends).
        """
        if not TELEGRAM_ADMIN_CHAT_ID:
            return True
        try:
            return int(chat_id) == int(TELEGRAM_ADMIN_CHAT_ID)
        except (TypeError, ValueError):
            return False

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
            try:
                await update.message.reply_text(dashboard, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(dashboard)
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
            total = len(dlq_entries)
            lines = [f"📭 Dead-Letter Queue ({total} total)\n"]
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
            if total > 10:
                lines.append(f"\n… +{total - 10} more")
            btn_rows = [buttons[j:j+5] for j in range(0, len(buttons), 5)]
            btn_rows.append([InlineKeyboardButton(
                f"🔄 Tümünü Yeniden Dene ({total})",
                callback_data="m:dlq:retry_all")])
            await update.message.reply_text(
                "\n".join(lines),
                reply_markup=InlineKeyboardMarkup(btn_rows),
            )
        except Exception as e:
            logger.error("DLQ listing failed", error=str(e))
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
                    chat_id=chat_id,
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
        # Z8 T1D — /stop_ops <mission_id> revokes an ongoing mission.
        self.app.add_handler(CommandHandler("stop_ops", self.cmd_stop_ops))
        self.app.add_handler(CommandHandler("digest", self.cmd_digest))
        # Z9 T2C — /digest_now force-fires the weekly growth digest cron.
        self.app.add_handler(CommandHandler("digest_now", self.cmd_digest_now))
        self.app.add_handler(CommandHandler("debug", self.cmd_debug))
        self.app.add_handler(CommandHandler("reset", self.cmd_reset))
        self.app.add_handler(CommandHandler("resetall", self.cmd_reset_all))
        self.app.add_handler(CommandHandler("cancel", self.cmd_cancel))
        self.app.add_handler(CommandHandler("priority", self.cmd_priority))
        self.app.add_handler(CommandHandler("graph", self.cmd_graph))
        self.app.add_handler(CommandHandler("budget", self.cmd_budget))
        self.app.add_handler(CommandHandler("signoff", self.cmd_signoff))
        self.app.add_handler(CommandHandler("preflight", self.cmd_preflight))
        self.app.add_handler(CommandHandler("audit", self.cmd_audit))
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
        # Z10 T2A: per-mission cost gauge + quality_mode dial
        self.app.add_handler(CommandHandler("mission_cost", self.cmd_mission_cost))
        self.app.add_handler(CommandHandler("quality_mode", self.cmd_quality_mode))
        self.app.add_handler(CommandHandler("dlq", self.cmd_dlq))
        self.app.add_handler(CommandHandler("rework", self.cmd_rework))
        # Z3 T1C: review density founder dials
        self.app.add_handler(CommandHandler("density", self.cmd_density))
        self.app.add_handler(CommandHandler("regen", self.cmd_regen))
        # Z1 Tier 4 (T4B): asset->spec propagation + two-way HTML edit reflection
        self.app.add_handler(CommandHandler("edit_html", self.cmd_edit_html))
        self.app.add_handler(CommandHandler("propagate", self.cmd_propagate))
        # Z1 Tier 4 (T4C): tunneled preview URL surface
        self.app.add_handler(CommandHandler("preview", self.cmd_preview))
        self.app.add_handler(CommandHandler("preview_off", self.cmd_preview_off))
        # Z1 Tier 5A (A5): founder attention budget
        self.app.add_handler(CommandHandler("budget", self.cmd_budget))
        # Z1 Tier 5A (P6): founder compliance signoffs
        self.app.add_handler(CommandHandler("signoff", self.cmd_signoff))
        # Z0 minimal slice: mission preflight contract
        self.app.add_handler(CommandHandler("preflight", self.cmd_preflight))
        # Z1 audit-log inspector (critic/regen/preview/paraflow/streaming)
        self.app.add_handler(CommandHandler("audit", self.cmd_audit))
        # Z1 Tier 6 (C18): per-mission GitHub repo (init / view / visibility)
        self.app.add_handler(CommandHandler("github", self.cmd_github))
        # Z1 Tier 7B (C21): bundle-quality regression vs Paraflow goldens
        self.app.add_handler(
            CommandHandler("paraflow_check", self.cmd_paraflow_check)
        )
        self.app.add_handler(CommandHandler("retry", self.cmd_retry))
        self.app.add_handler(CommandHandler("load", self.cmd_load))
        self.app.add_handler(CommandHandler("tune", self.cmd_tune))
        self.app.add_handler(CommandHandler("feedback", self.cmd_feedback))
        self.app.add_handler(CommandHandler("improve", self.cmd_improve))
        self.app.add_handler(CommandHandler("remember", self.cmd_remember))
        self.app.add_handler(CommandHandler("recall", self.cmd_recall))
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
        self.app.add_handler(CommandHandler("bench_picks", self.cmd_bench_picks))
        # Z10 T4B — confidence reliability matrix
        self.app.add_handler(CommandHandler("calibration", self.cmd_calibration))
        self.app.add_handler(CommandHandler("revive", self.cmd_revive))
        self.app.add_handler(CommandHandler("dead", self.cmd_dead))
        self.app.add_handler(CommandHandler("pause_mission", self.cmd_pause_mission))
        self.app.add_handler(CommandHandler("resume_mission", self.cmd_resume_mission))
        self.app.add_handler(CommandHandler("kill_mission", self.cmd_kill_mission))
        # Z10 T2B: per-mission thread surfacing + cost peek
        self.app.add_handler(CommandHandler("mission_thread", self.cmd_mission_thread))
        self.app.add_handler(CommandHandler("missions_active", self.cmd_missions_active))
        # (mission_cost handler registered above at T2A line)
        # Z10 T3C: rollback mission to last green checkpoint
        self.app.add_handler(
            CommandHandler("rollback_mission", self.cmd_rollback_mission)
        )
        # Z6 T1D: founder_actions surface (real-world bridge handoff queue)
        self.app.add_handler(CommandHandler("actions", self.cmd_actions))
        self.app.add_handler(CommandHandler("action_done", self.cmd_action_done))
        # Z8 T4D: ops log — show recent on-call agent actions per mission.
        self.app.add_handler(CommandHandler("ops_log", self.cmd_ops_log))
        # Z7 A0: founder hours saved ROI summary
        self.app.add_handler(
            CommandHandler("founder_hours_saved", self.cmd_founder_hours_saved)
        )
        # --- Z9 growth (T1D stubs) — reserve the founder-track command surface ---
        self.app.add_handler(CommandHandler("northstar", self.cmd_northstar))
        self.app.add_handler(CommandHandler("hypothesis", self.cmd_hypothesis))
        self.app.add_handler(CommandHandler("backlog", self.cmd_backlog))
        self.app.add_handler(CommandHandler("sunset", self.cmd_sunset))
        self.app.add_handler(CommandHandler("experiment", self.cmd_experiment))
        # Z9 T5D — A/B experiment founder-track commands.
        self.app.add_handler(
            CommandHandler("experiment_ship", self.cmd_experiment_ship)
        )
        self.app.add_handler(
            CommandHandler("experiment_rollback", self.cmd_experiment_rollback)
        )
        self.app.add_handler(
            CommandHandler("experiment_disable", self.cmd_experiment_disable)
        )
        # Z7 T1D (B5) — founder attention budget queue
        self.app.add_handler(CommandHandler("attention", self.cmd_attention))
        # Z7 T1D (B9) — external comms audit log search
        self.app.add_handler(CommandHandler("audit_comms", self.cmd_audit_comms))
        # Z7 T2A — email-send shared service (config/upgrade/test)
        self.app.add_handler(CommandHandler("email", self.cmd_email))
        # Z7 T3E (B6) — crisis comms tiered playbook
        self.app.add_handler(CommandHandler("crisis", self.cmd_crisis))
        # Z7 T3A (A2) — launch playbook
        self.app.add_handler(CommandHandler("launch", self.cmd_launch))
        # Z7 T4 A10 — CRM-as-interaction-log + A10.r1 consent ledger
        self.app.add_handler(CommandHandler("contact", self.cmd_contact))
        self.app.add_handler(CommandHandler("log", self.cmd_crm_log))
        self.app.add_handler(CommandHandler("contacts", self.cmd_contacts))
        self.app.add_handler(CommandHandler("follow_ups", self.cmd_follow_ups))
        self.app.add_handler(CommandHandler("consent", self.cmd_consent))
        # Z7 T4 B4 — meeting brief auto-generation
        self.app.add_handler(CommandHandler("meeting", self.cmd_meeting))
        # Z7 T4 B7 — customer interview pipeline
        self.app.add_handler(CommandHandler("interview", self.cmd_interview))
        # Z7 T5 B1 — lifecycle email engine
        self.app.add_handler(CommandHandler("lifecycle", self.cmd_lifecycle))
        # Z7 T6 A7 — cold outreach + deliverability spine
        self.app.add_handler(CommandHandler("outreach", self.cmd_outreach))
        # Z7 B2 — changelog publish (founder-gated announcement trigger)
        self.app.add_handler(CommandHandler("changelog", self.cmd_changelog))
        # Z7 A11 — mention monitor registration
        self.app.add_handler(CommandHandler("mention_monitor", self.cmd_mention_monitor))
        # Z8 — manual on-call action trigger
        self.app.add_handler(CommandHandler("force_action", self.cmd_force_action))
        # Z9 T5E — full-params typed confirmation for irreversible pricing A/B.
        self.app.add_handler(CommandHandler("confirm", self.cmd_confirm))
        self.app.add_handler(CommandHandler("approve", self.cmd_approve))
        self.app.add_handler(
            CommandHandler("approve_sunset", self.cmd_approve_sunset)
        )
        self.app.add_handler(CommandHandler("ask", self.cmd_ask))
        # Yalayut Phase 3 — catalog auth + MCP control commands
        self.app.add_handler(CommandHandler("yalayut", self.cmd_yalayut))
        self.app.add_handler(CallbackQueryHandler(
            self._handle_founder_action_callback,
            pattern=r"^fa_(done|inprogress|block)_\d+$",
        ))
        # Register the notifier with the founder_actions module so create()
        # can surface new cards to the mission thread automatically.
        try:
            import src.founder_actions as _fa_mod
            _fa_mod.register_notifier(self._notify_founder_action)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"founder_actions notifier register failed: {e}")
        self.app.add_handler(CallbackQueryHandler(
            self._handle_variant_choice, pattern=r"^(vc|variant_choice):"
        ))
        # Z10 T2B: typed event + confirmation reactions
        self.app.add_handler(CallbackQueryHandler(
            self._handle_mission_event_callback,
            pattern=r"^(confirm|event):",
        ))
        self.app.add_handler(CallbackQueryHandler(
            self._handle_artifact_confirm_callback, pattern=r"^rpc:",
        ))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_handler(MessageHandler(filters.LOCATION, self.handle_location))
        # B7+C16: founder photo upload → mission .intake/visuals/ + clarify-shape
        self.app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        # Z1 T4B: founder document upload — paired with /edit_html via
        # _pending_action[chat_id]={"command":"_edit_html_upload",...}
        self.app.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
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
        """Create or view a mission.

        Usage:
          /mission                — prompt for description
          /mission <id>           — show mission details + pacing (Z10 T3A)
          /mission <description>  — create new mission
        """
        if not context.args:
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {"command": "mission"}
            await self._reply(update, "🎯 Describe your mission:")
            return

        # Z10 T3A: if single int arg, render the per-mission view with
        # the Pacing block appended.
        if len(context.args) == 1 and context.args[0].isdigit():
            try:
                from src.app.mission_view import format_mission_view
                mid = int(context.args[0])
                body = await format_mission_view(mid)
                await self._reply(update, body, parse_mode="Markdown")
            except Exception as e:  # noqa: BLE001
                logger.warning("cmd_mission view failed", error=str(e))
                await self._reply(
                    update, f"❌ Could not show mission: {_friendly_error(str(e))}",
                )
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

        # Z10 T2B: provision per-mission forum topic (falls back silently
        # to flat-prefix mode if chat is not a forum supergroup). Persists
        # the topic to missions.telegram_thread_id — the canonical column
        # read by beckman threshold-notify and the lifecycle commands.
        try:
            from .telegram_topics import ensure_mission_topic
            await ensure_mission_topic(
                self.app.bot, mission_id, description[:80], chat_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("ensure_mission_topic skipped: %s", e)

        if self.orchestrator:
            await self.orchestrator.plan_mission(mission_id, description[:80], description)

        # Z0: prompt for cost ceiling
        await self._reply(
            update,
            "Cost ceiling for this mission ($)? Reply with a number, or `none` for unlimited.",
        )
        self._pending_action[chat_id] = {
            "kind": "z0_ceiling",
            "mission_id": mission_id,
        }
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
        # Z6 T7B — also surface missions parked in blocked_on_founder_action
        # so the founder can see them and act. Without this, /missions
        # showed an empty list every time T1E blocked a mission.
        try:
            from src.infra.db import get_db
            db = await get_db()
            cur = await db.execute(
                "SELECT * FROM missions "
                "WHERE status = 'blocked_on_founder_action' "
                "ORDER BY priority DESC"
            )
            blocked = [dict(r) for r in await cur.fetchall()]
            if blocked:
                missions = list(missions) + blocked
        except Exception:
            pass
        if not missions:
            await self._reply(update,"No active missions.")
            return

        # Z6 T7B — fetch pending founder_action counts per mission so we
        # can surface a "⚠ N action(s) pending" badge inline. Best-effort:
        # any failure just drops the badge silently.
        action_counts: dict[int, int] = {}
        try:
            import src.founder_actions as fa
            for m in missions:
                try:
                    rows = await fa.list_by_mission(
                        int(m["id"]),
                        status_filter=["pending", "in_progress"],
                    )
                    if rows:
                        action_counts[int(m["id"])] = len(rows)
                except Exception:
                    pass
        except Exception:
            pass

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
            action_n = action_counts.get(int(mid), 0)
            action_suffix = (
                f" · ⚠ {action_n} action(s) pending" if action_n else ""
            )
            lines.append(
                f"  {badge} #{mid} {title}{wf_tag}{task_info}{action_suffix}"
            )

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

        description = " ".join(raw_args)

        task_id = await add_task(
            title=description[:50],
            description=description,
            tier="auto",
            parent_task_id=parent_id,
            priority=TASK_PRIORITY["critical"],
        )
        self.user_last_task_id[chat_id] = task_id
        await self._reply(update, f"✅ Task #{task_id} queued.")


    async def cmd_view_queue(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from src.infra.db import get_db
        db = await get_db()
        # Fetch in-progress tasks
        cursor = await db.execute(
            """SELECT * FROM tasks WHERE status = 'processing'
               ORDER BY priority DESC, started_at ASC LIMIT 10"""
        )
        processing = [dict(row) for row in await cursor.fetchall()]
        # Fetch ready (pending with deps met). Pass a high cap so we get an
        # accurate total count for the queue header — render only the first
        # 5 to keep the Telegram message short.
        ready = await get_ready_tasks(limit=500)
        # Fetch retry-pending: pending tasks with future next_retry_at.
        # These were getting lost in "blocked" before — user couldn't tell
        # whether a phase was deadlocked vs just waiting on backoff timer.
        cursor_retry = await db.execute(
            """SELECT id, title, agent_type, tier, worker_attempts,
                      max_worker_attempts, next_retry_at, error_category, error
                 FROM tasks
                WHERE status = 'pending'
                  AND next_retry_at IS NOT NULL
                  AND next_retry_at > datetime('now')
                ORDER BY next_retry_at ASC
                LIMIT 500"""
        )
        retry_pending = [dict(row) for row in await cursor_retry.fetchall()]
        # Fetch blocked task summary
        blocked_summary = await get_blocked_task_summary()
        blocked_count = blocked_summary["blocked_count"]
        # Don't double-count retry-pending against blocked count — they're
        # already in their own section.
        retry_ids = {t["id"] for t in retry_pending}
        blocked_count = max(0, blocked_count - len(retry_ids))

        if (not processing and not ready and not retry_pending
                and blocked_count == 0):
            await self._reply(update, "No pending tasks. System is idle.")
            return

        msg = "📬 Task Queue:\n\n"
        if processing:
            from src.core.in_flight import get_task_entry, get_recent_cloud
            msg += "⚙️ In Progress:\n"
            for t in processing:
                agent = t.get('agent_type', '?')
                entry = get_task_entry(t['id'])
                if entry is not None:
                    where = "local" if entry.is_local else (entry.provider or "?")
                    model = entry.model or "?"
                    tag = f"{agent}→{where}:{model}"
                else:
                    tag = f"{agent}→?"
                # If current slot is local but cloud was tried recently, show
                # the bounce — fast-failing cloud retries are otherwise
                # invisible because the slot reverts to local within ms.
                bounce = ""
                if entry is None or entry.is_local:
                    rc = get_recent_cloud(t['id'])
                    if rc is not None:
                        cprov, cmodel, age = rc
                        bounce = f" ↔{cprov}:{cmodel} ({int(age)}s ago)"
                msg += f"  #{t['id']} [{tag}]{bounce} {t['title'][:50]}\n"
            msg += "\n"
        if ready:
            ready_total = len(ready)
            msg += f"⏳ Ready ({ready_total}):\n"
            for t in ready[:5]:
                agent = t.get('agent_type', '?')
                msg += f"  #{t['id']} [{agent}|{t['tier']}] {t['title'][:50]}\n"
            if ready_total > 5:
                msg += f"  … +{ready_total - 5} more\n"
            msg += "\n"
        if retry_pending:
            retry_total = len(retry_pending)
            msg += f"🔁 Retry pending ({retry_total}):\n"
            from datetime import datetime
            from src.infra.times import from_db, utc_now
            now = utc_now()
            for t in retry_pending[:5]:
                agent = t.get('agent_type', '?')
                att = t.get('worker_attempts') or 0
                mx = t.get('max_worker_attempts') or 0
                cat = t.get('error_category') or "?"
                # ETA in seconds. from_db tolerates either ISO or
                # "YYYY-MM-DD HH:MM:SS" (sqlite default).
                try:
                    eta_dt = from_db(str(t['next_retry_at']))
                    eta_s = max(0, int((eta_dt - now).total_seconds()))
                    if eta_s < 60:
                        eta_str = f"{eta_s}s"
                    elif eta_s < 3600:
                        eta_str = f"{eta_s // 60}m{eta_s % 60:02d}s"
                    else:
                        eta_str = f"{eta_s // 3600}h{(eta_s % 3600) // 60:02d}m"
                except Exception:
                    eta_str = "?"
                msg += (
                    f"  #{t['id']} [{agent}] {t['title'][:42]} "
                    f"att={att}/{mx} cat={cat} eta={eta_str}\n"
                )
            if retry_total > 5:
                msg += f"  … +{retry_total - 5} more\n"
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
        status_msg = (
            f"⚙️ *System Status*\n\n"
            f"✅ Completed: {stats['completed']}\n"
            f"⏳ Pending: {stats['pending']}\n"
            f" Processing: {stats['processing']}\n"
            f"❌ Failed: {stats['failed']}\n"
            f" Cost today: ${stats['today_cost']:.4f}"
        )
        try:
            from src.core.router import get_kdv
            _kdv_warnings = get_kdv().no_data_warnings(min_age_hours=24)
            if _kdv_warnings:
                _lines = [
                    f"  • {w['provider']} — {w['age_hours']:.1f}h since enable, no calls"
                    for w in _kdv_warnings
                ]
                status_msg += "\n\n⚠️ Cloud providers using cold-start defaults:\n" + "\n".join(_lines)
        except Exception:
            pass
        await self._reply(update, status_msg, parse_mode="Markdown")

    async def cmd_stop_ops(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z8 T1D — revoke an ongoing mission.

        Usage: ``/stop_ops <mission_id>``. Transitions
        ``missions.lifecycle_state`` → ``'revoked'`` and stamps
        ``revoked_at`` so the orchestrator stops resuming it on boot
        and webhook/cron handlers drop their subscriptions. No-op for
        oneshot missions or unknown ids.
        """
        if not context.args:
            await self._reply(update, "Usage: /stop_ops <mission_id>")
            return
        try:
            mid = int(context.args[0])
        except (TypeError, ValueError):
            await self._reply(update, "mission_id must be an integer")
            return
        try:
            from general_beckman.resumption import revoke
            ok = await revoke(mid)
        except Exception as e:
            logger.warning(f"/stop_ops failed for mission {mid}: {e}")
            await self._reply(update, f"failed to revoke mission {mid}: {e}")
            return
        if ok:
            await self._reply(update, f"mission {mid} revoked")
        else:
            await self._reply(
                update,
                f"mission {mid} not ongoing or not found"
            )

    async def cmd_bench_picks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show 7-day model pick distribution from model_pick_log."""
        import aiosqlite
        query = """
            SELECT task_name, picked_model, COUNT(*) AS n,
                   ROUND(AVG(picked_score), 2) AS avg_score
            FROM model_pick_log
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY task_name, picked_model
            ORDER BY task_name, n DESC
        """
        try:
            from src.infra.db import connect_aux
            async with connect_aux(DB_PATH, _label="bench_picks_query") as db:
                cursor = await db.execute(query)
                rows = await cursor.fetchall()
        except Exception as exc:
            await self._reply(update, f"❌ bench_picks query failed: {exc}")
            return

        if not rows:
            await self._reply(update, "📊 No pick log entries in last 7 days.")
            return

        MAX_ROWS = 40
        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]

        lines = [
            f"{'task':<20} {'model':<28} {'n':>4} {'avg':>5}",
            "─" * 60,
        ]
        for task, model, n, avg in rows:
            lines.append(
                f"{(task or '?')[:20]:<20} {(model or '?')[:28]:<28} "
                f"{n:>4} {(avg or 0.0):>5.2f}"
            )
        body = "\n".join(lines)
        footer = "\n\n… (truncated)" if truncated else ""
        await self._reply(
            update,
            f"📊 *Model picks — last 7 days*\n```\n{body}\n```{footer}",
            parse_mode="Markdown",
        )

    async def cmd_calibration(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z10 T4B — per-(model, task_kind) confidence reliability matrix.

        Buckets with sample_n < 5 render as '—' to avoid noise. Color cues:
            🟢 reliability ≥ 0.85   🟡 0.65–0.85   🔴 < 0.65
        Empty matrix → friendly nudge that the recompute job hasn't run
        yet (or no resolved outcomes exist).
        """
        from src.infra.db import calibration_matrix
        try:
            rows = await calibration_matrix()
        except Exception as exc:
            await self._reply(update, f"❌ calibration query failed: {exc}")
            return

        if not rows:
            await self._reply(
                update,
                "🎯 No calibration data yet "
                "(need ≥30 outcomes per model+domain).",
            )
            return

        # Pivot: dict[(model, kind)] -> {bucket: row}
        pivot: dict[tuple[str, str], dict[str, dict]] = {}
        for r in rows:
            key = (r["model_id"], r["task_kind"])
            pivot.setdefault(key, {})[r["confidence_bucket"]] = r

        MIN_SAMPLE_VISIBLE = 5

        def _fmt(rel_row: dict | None) -> str:
            if not rel_row or int(rel_row.get("sample_n") or 0) < MIN_SAMPLE_VISIBLE:
                return "—"
            rel = float(rel_row.get("reliability") or 0.0)
            if rel >= 0.85:
                icon = "🟢"
            elif rel >= 0.65:
                icon = "🟡"
            else:
                icon = "🔴"
            return f"{icon}{rel:.2f}"

        def _samples(rel_row: dict | None) -> str:
            if not rel_row:
                return "0"
            return str(int(rel_row.get("sample_n") or 0))

        lines = [
            f"{'model/kind':<28} {'low':>7} {'med':>7} {'high':>7}  samples"
        ]
        lines.append("─" * 60)
        for (model_id, task_kind) in sorted(pivot.keys()):
            buckets = pivot[(model_id, task_kind)]
            low = buckets.get("low")
            med = buckets.get("med")
            high = buckets.get("high")
            label = f"{(model_id or '?')[:14]}/{(task_kind or '?')[:12]}"
            samples = f"{_samples(low)}/{_samples(med)}/{_samples(high)}"
            lines.append(
                f"{label[:28]:<28} {_fmt(low):>7} {_fmt(med):>7} "
                f"{_fmt(high):>7}  ({samples})"
            )

        body = "\n".join(lines)
        await self._reply(
            update,
            f"🎯 *Calibration matrix*\n```\n{body}\n```",
            parse_mode="Markdown",
        )

    async def cmd_dead(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List currently-dead providers + models in the SQLite registry.
        Diagnostic sibling of /revive — operator can see what /revive
        would target."""
        from src.infra import registry_store
        providers = registry_store.list_dead_providers()
        models = registry_store.list_dead()
        if not providers and not models:
            await self._reply(update, "No providers or models marked dead.")
            return
        lines: list[str] = []
        if providers:
            lines.append(f"🔥 {len(providers)} dead provider(s):")
            for p in providers:
                lines.append(
                    f"  • {p['name']} — {p['cause']} (since {p['marked_at']})"
                )
            lines.append("")
        if models:
            lines.append(f"💀 {len(models)} dead model(s):")
            for r in models[:30]:
                ttl = r["expires_at"] or "never (manual /revive)"
                lines.append(
                    f"  • {r['litellm_name']} — {r['cause']} → {ttl}"
                )
            if len(models) > 30:
                lines.append(f"  … (+{len(models) - 30} more)")
        await self._reply(update, "\n".join(lines))

    async def cmd_revive(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Revive a dead model or provider.

        Usage:
          /revive                    — list current dead models/providers
          /revive <litellm_name>     — revive single model
          /revive provider <name>    — revive provider (releases all its models)
        """
        from src.infra import registry_store
        args = context.args or []
        if not args:
            dead_models = registry_store.list_dead()
            lines = []
            if dead_models:
                lines.append(f"💀 {len(dead_models)} dead model(s):")
                for r in dead_models[:20]:
                    lines.append(f"  • {r['litellm_name']} ({r['cause']})")
                if len(dead_models) > 20:
                    lines.append(f"  … (+{len(dead_models) - 20} more)")
            else:
                lines.append("No models marked dead.")
            lines.append("")
            lines.append("Usage:")
            lines.append("  /revive <litellm_name>")
            lines.append("  /revive provider <name>")
            await self._reply(update, "\n".join(lines))
            return

        # Provider form: /revive provider <name>
        if args[0].lower() == "provider":
            if len(args) < 2:
                await self._reply(update, "Usage: /revive provider <name>")
                return
            provider = args[1]
            was_dead = registry_store.is_provider_dead(provider)
            registry_store.revive_provider(provider, actor="user")
            if was_dead:
                await self._reply(update, f"✅ Revived provider: {provider}")
            else:
                await self._reply(update, f"Provider {provider} was not dead.")
            return

        # Model form: /revive <litellm_name>
        identifier = args[0]
        was_dead = registry_store.is_dead(identifier)
        registry_store.revive(identifier, actor="user")
        if was_dead:
            await self._reply(update, f"✅ Revived model: {identifier}")
        else:
            await self._reply(
                update,
                f"Model {identifier} was not dead. "
                f"(Use exact litellm_name; for provider, use "
                f"`/revive provider {identifier}`)",
            )

    async def cmd_digest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.orchestrator:
            await self.orchestrator.daily_digest()
        else:
            await self._reply(update,"Orchestrator not connected.")
        # Z9 T2C — append the latest stored weekly growth digest.
        await self._send_growth_digest_section(update)

    async def _send_growth_digest_section(self, update: Update) -> None:
        """Z9 T2C — surface the most recent stored weekly growth digest.

        Reads the latest ``growth_events`` row with ``kind="weekly_digest"``
        (written by the digest synthesis agent's on_complete continuation).
        Sent through ``_reply`` so the persistent keyboard survives.
        """
        try:
            from src.infra.db import get_growth_events

            rows = await get_growth_events(kind="weekly_digest")
        except Exception as exc:  # noqa: BLE001
            logger.debug("growth digest fetch failed: %s", exc)
            return
        if not rows:
            await self._reply(
                update,
                "📈 *Growth*\n\nNo weekly digest yet — one is generated "
                "weekly after a product launches. Use /digest_now to "
                "force-fire it for the latest launched mission.",
                parse_mode="Markdown",
            )
            return
        latest = rows[0]
        props = latest.get("properties_json") or {}
        markdown = ""
        if isinstance(props, dict):
            markdown = str(props.get("markdown") or "").strip()
        iso_yw = props.get("iso_year_week") if isinstance(props, dict) else None
        header = f"📈 *Growth — weekly digest*"
        if iso_yw:
            header += f" ({iso_yw})"
        body = markdown or "_Digest content unavailable._"
        msg = f"{header}\n\n{body}"
        # Telegram 4096-char cap — chunk if needed.
        if len(msg) > 4000:
            for i in range(0, len(msg), 4000):
                await self._reply(update, msg[i:i + 4000], parse_mode="Markdown")
        else:
            await self._reply(update, msg, parse_mode="Markdown")

    async def cmd_digest_now(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T2C — force-fire the weekly analytics_digest cron action.

        Enqueues the ``analytics_digest`` mechanical task immediately for the
        most recently launched mission (or a mission id given as an arg).
        The mechanical does the data pull and enqueues the synthesis agent;
        the result lands as a ``weekly_digest`` growth event minutes later.
        """
        mission_id = None
        if context.args:
            try:
                mission_id = int(context.args[0])
            except (TypeError, ValueError):
                await self._reply(
                    update, "Usage: /digest_now [mission_id]"
                )
                return
        try:
            from src.infra.db import get_db

            db = await get_db()
            if mission_id is None:
                # Most recent mission with an analytics_digest cron armed.
                cur = await db.execute(
                    "SELECT id FROM missions "
                    "WHERE cursor LIKE '%analytics_digest%' "
                    "ORDER BY id DESC LIMIT 1"
                )
                row = await cur.fetchone()
                if not row:
                    # Fall back to the most recent mission overall.
                    cur = await db.execute(
                        "SELECT id FROM missions ORDER BY id DESC LIMIT 1"
                    )
                    row = await cur.fetchone()
                if not row:
                    await self._reply(
                        update, "No missions found to run a digest for."
                    )
                    return
                mission_id = int(row[0])

            from general_beckman import enqueue
            from general_beckman.apply import _mechanical_context

            task_id = await enqueue(
                {
                    "title": f"digest_now: analytics_digest (mid={mission_id})",
                    "description": "Founder-triggered weekly growth digest.",
                    "agent_type": "mechanical",
                    "context": _mechanical_context(
                        "analytics_digest", mission_id=mission_id
                    ),
                    "depends_on": [],
                    "mission_id": mission_id,
                }
            )
            await self._reply(
                update,
                f"📈 Weekly growth digest queued for mission #{mission_id} "
                f"(task #{task_id}). The synthesized digest will appear "
                f"under /digest shortly.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("cmd_digest_now failed: %s", exc)
            await self._reply(
                update, f"Could not queue digest: {exc}"
            )

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
                """UPDATE tasks SET status = 'pending', worker_attempts = 0, error = NULL
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
                await update_task(task_id, status="pending", worker_attempts=0, error=None)
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

    async def cmd_mission_cost(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z10 T2A D3 — per-mission cost gauge.

        Renders the breakdown from ``src.infra.cost_wiring.format_mission_cost``.
        T2B will reuse the same formatter when auto-posting on
        ``[milestone]`` events.
        """
        if not context.args:
            await self._reply(update, "Usage: /mission_cost <mission_id>")
            return
        try:
            mission_id = int(context.args[0])
        except ValueError:
            await self._reply(update, "Mission ID must be a number.")
            return
        try:
            from src.infra.cost_wiring import format_mission_cost
            body = await format_mission_cost(mission_id)
        except Exception as e:
            await self._reply(update, f"Failed to render mission cost: {e}")
            return
        await self._reply(update, body)

    async def cmd_quality_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z10 T2A D7 — set or read a mission's quality_mode dial.

        Usage:
          /quality_mode <mission_id>          → read
          /quality_mode <mission_id> <mode>   → set (quick/balanced/thorough)
        """
        if not context.args:
            await self._reply(
                update,
                "Usage: /quality_mode <mission_id> [quick|balanced|thorough]",
            )
            return
        try:
            mission_id = int(context.args[0])
        except ValueError:
            await self._reply(update, "Mission ID must be a number.")
            return
        if len(context.args) >= 2:
            mode = context.args[1].strip().lower()
            try:
                from src.infra.db import set_mission_quality_mode
                await set_mission_quality_mode(mission_id, mode)
            except ValueError as ve:
                await self._reply(update, str(ve))
                return
            except Exception as e:
                await self._reply(update, f"Failed to set quality_mode: {e}")
                return
            await self._reply(
                update,
                f"Mission {mission_id} quality_mode set to {mode}",
            )
            return
        try:
            from src.infra.db import get_mission_quality_mode
            cur = await get_mission_quality_mode(mission_id)
        except Exception as e:
            await self._reply(update, f"Failed to read quality_mode: {e}")
            return
        await self._reply(
            update,
            f"Mission {mission_id} quality_mode = {cur}",
        )

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

    async def cmd_changelog(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /changelog command — the founder-gated publish trigger.

        Subcommands:
          /changelog [list]        — list unpublished changelog drafts
          /changelog publish <id>  — publish a draft (marks it released,
                                     invalidates caches, queues the B1
                                     announcement email blast to subscribers)

        The changelog/publish mr_roboto verb had no production caller — i2p
        step 11.4 produces a changelog artifact but never publishes it, and
        the changelog_freshness posthook only emits advisory text. This
        command is that missing trigger; publish broadcasts to subscribers,
        so it stays an explicit founder action.
        """
        chat_id = update.effective_chat.id
        product_id = str(chat_id)
        args = context.args or []
        sub = (args[0].lower() if args else "list")

        if sub == "list":
            try:
                from src.infra.db import get_db
                db = await get_db()
                cur = await db.execute(
                    "SELECT entry_id, version, title FROM changelog_entries "
                    "WHERE product_id=? AND published=0 "
                    "ORDER BY entry_id DESC LIMIT 20",
                    (product_id,),
                )
                rows = await cur.fetchall()
            except Exception as e:
                await self._reply(update, f"changelog list failed: {e}")
                return
            if not rows:
                await self._reply(
                    update,
                    "No unpublished changelog drafts.\n"
                    "Drafts are created by the `changelog/draft` verb during "
                    "a release mission.",
                    parse_mode="Markdown",
                )
                return
            lines = ["*Unpublished changelog drafts:*"]
            for entry_id, version, title in rows:
                lines.append(f"  `{entry_id}` — {version or '(no version)'}: {title or '(untitled)'}")
            lines.append("\nPublish with `/changelog publish <id>`")
            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
            return

        if sub == "publish":
            if len(args) < 2 or not args[1].isdigit():
                await self._reply(
                    update,
                    "Usage: `/changelog publish <entry_id>`\n"
                    "Run `/changelog` to see draft entry ids.",
                    parse_mode="Markdown",
                )
                return
            entry_id = int(args[1])
            try:
                from src.infra.db import get_db
                db = await get_db()
                cur = await db.execute(
                    "SELECT published FROM changelog_entries "
                    "WHERE entry_id=? AND product_id=?",
                    (entry_id, product_id),
                )
                row = await cur.fetchone()
            except Exception as e:
                await self._reply(update, f"changelog lookup failed: {e}")
                return
            if row is None:
                await self._reply(update, f"No changelog draft #{entry_id} for this product.")
                return
            if row[0]:
                await self._reply(update, f"Changelog #{entry_id} is already published.")
                return
            import general_beckman
            await general_beckman.enqueue(
                {"agent_type": "mechanical",
                 "title": f"Publish changelog #{entry_id}",
                 "payload": {"action": "changelog/publish",
                             "entry_id": entry_id,
                             "product_id": product_id}},
                lane="oneshot",
            )
            await self._reply(
                update,
                f"Changelog #{entry_id} queued for publish — entry will be "
                "marked released and the announcement email blast queued to "
                "subscribers.",
            )
            return

        await self._reply(
            update,
            "Usage:\n"
            "  `/changelog` — list unpublished drafts\n"
            "  `/changelog publish <id>` — publish a draft + announce",
            parse_mode="Markdown",
        )

    async def cmd_mention_monitor(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /mention_monitor command — register A11 mention monitoring.

        Subcommands:
          /mention_monitor [list]                  — show this product's monitor
          /mention_monitor add <name> [channels..] — register; default channels
                                                     hn google (free, no key)
          /mention_monitor remove                  — disable monitoring

        Once registered, the hourly mention_monitor_sweep cron polls the
        enabled channels and surfaces a digest. Without a row here the
        sweep has nothing to poll — the A11 monitor never runs.
        """
        chat_id = update.effective_chat.id
        product_id = str(chat_id)
        args = context.args or []
        sub = (args[0].lower() if args else "list")
        valid = {"hn", "reddit", "google", "discord"}

        from src.infra.db import get_db
        import json as _json
        db = await get_db()

        if sub == "list":
            cur = await db.execute(
                "SELECT product_name, channels_json, enabled, last_run_at "
                "FROM mention_monitors WHERE product_id=?",
                (product_id,),
            )
            row = await cur.fetchone()
            if row is None:
                await self._reply(
                    update,
                    "No mention monitor for this product.\n"
                    "Register with `/mention_monitor add <product_name> [channels]`",
                    parse_mode="Markdown",
                )
                return
            name, channels_json, enabled, last_run = row
            try:
                channels = _json.loads(channels_json or "[]")
            except Exception:
                channels = []
            await self._reply(
                update,
                f"*Mention monitor:* {name}\n"
                f"  channels: {', '.join(channels) or '(none)'}\n"
                f"  enabled: {'yes' if enabled else 'no'}\n"
                f"  last run: {last_run or 'never'}",
                parse_mode="Markdown",
            )
            return

        if sub == "add":
            if len(args) < 2:
                await self._reply(
                    update,
                    "Usage: `/mention_monitor add <product_name> [channels]`\n"
                    "Channels: hn reddit google discord (default: hn google)",
                    parse_mode="Markdown",
                )
                return
            product_name = args[1]
            channels = [c.lower() for c in args[2:] if c.lower() in valid]
            if not channels:
                channels = ["hn", "google"]
            await db.execute(
                "INSERT INTO mention_monitors "
                "(product_id, product_name, channels_json, enabled) "
                "VALUES (?, ?, ?, 1) "
                "ON CONFLICT(product_id) DO UPDATE SET "
                "  product_name=excluded.product_name, "
                "  channels_json=excluded.channels_json, enabled=1",
                (product_id, product_name, _json.dumps(channels)),
            )
            await db.commit()
            await self._reply(
                update,
                f"Mention monitor registered for *{product_name}*.\n"
                f"Channels: {', '.join(channels)}. The hourly sweep will "
                "poll these and surface a digest.",
                parse_mode="Markdown",
            )
            return

        if sub == "remove":
            cur = await db.execute(
                "UPDATE mention_monitors SET enabled=0 WHERE product_id=?",
                (product_id,),
            )
            await db.commit()
            if cur.rowcount:
                await self._reply(update, "Mention monitor disabled.")
            else:
                await self._reply(update, "No mention monitor to remove.")
            return

        await self._reply(
            update,
            "Usage:\n"
            "  `/mention_monitor` — show monitor\n"
            "  `/mention_monitor add <name> [channels]` — register\n"
            "  `/mention_monitor remove` — disable",
            parse_mode="Markdown",
        )

    async def _resolve_or_create_outreach_mission(self, product_id: str) -> int:
        """Find (or lazily create) the per-product mission that hosts cold-
        outreach founder_actions. founder_actions.mission_id is NOT NULL but
        ad-hoc outreach has no natural mission — one ongoing mission per
        product owns all its outreach cards."""
        from src.infra.db import get_db, add_mission
        db = await get_db()
        title = f"Cold outreach: {product_id}"
        cur = await db.execute(
            "SELECT id FROM missions WHERE title=? LIMIT 1", (title,))
        row = await cur.fetchone()
        if row:
            return int(row[0])
        return int(await add_mission(title, "Cold-outreach campaign container"))

    async def cmd_force_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /force_action command — manually trigger an on-call action.

        Usage: /force_action <mission_id> <verb> [params_json]

        The Z8 on-call agent normally chooses + dispatches ops verbs
        autonomously; /force_action lets the founder force one. The verb
        still goes through oncall_action's cooldown + handler + record path —
        only the *decision* is manual. Z8 shipped without this command.
        """
        args = context.args or []
        from mr_roboto.executors.oncall_action import WHITELISTED_VERBS
        if len(args) < 2:
            await self._reply(
                update,
                "Usage: `/force_action <mission_id> <verb> [params_json]`\n"
                f"Verbs: {', '.join(sorted(WHITELISTED_VERBS))}",
                parse_mode="Markdown",
            )
            return
        if not args[0].isdigit():
            await self._reply(update, "mission_id must be an integer.")
            return
        mission_id = int(args[0])
        verb = args[1]
        if verb not in WHITELISTED_VERBS:
            await self._reply(
                update,
                f"Unknown on-call verb `{verb}`.\n"
                f"Valid: {', '.join(sorted(WHITELISTED_VERBS))}",
                parse_mode="Markdown",
            )
            return
        params: dict = {}
        if len(args) > 2:
            import json as _json
            try:
                parsed = _json.loads(" ".join(args[2:]))
                if isinstance(parsed, dict):
                    params = parsed
            except Exception:
                await self._reply(update, "params must be valid JSON object.")
                return
        import general_beckman
        await general_beckman.enqueue(
            {"agent_type": "mechanical",
             "title": f"Force on-call action: {verb} (mission {mission_id})",
             "mission_id": mission_id,
             "payload": {"action": "oncall_action", "verb": verb,
                         "params": params}},
            lane="oneshot",
        )
        await self._reply(
            update,
            f"Forced on-call action `{verb}` queued for mission {mission_id} "
            "— routes through the cooldown + handler + audit path.",
            parse_mode="Markdown",
        )

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

    # ── B7+C16: visual ingest (sketch / screenshot upload) ───────────────────
    # Founder uploads a photo to the chat → save into the most-recent active
    # mission's ``.intake/visuals/`` directory, then prompt for "what's this
    # for?" and on the founder's next reply enqueue the ``ingest_visual``
    # mechanical action via Beckman. Image GENERATION is Z2 work
    # (``gorsel_ustasi``); this path is INGEST-only.

    _VISUAL_PURPOSE_LABELS: dict[str, str] = {
        "🖼 Rakip Ekran": "competitor_screenshot",
        "🎨 Moodboard": "moodboard",
        "✏️ Wireframe": "wireframe_sketch",
        "💡 İlham": "inspiration",
        "📱 Mevcut Ürün": "screenshot_of_existing_product",
    }

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """B7+C16: route founder-uploaded photos into a mission's intake dir.

        - Save the highest-resolution variant of the uploaded photo into
          ``workspace/mission_{id}/.intake/visuals/``.
        - Look up the most-recent active mission for context. If none, tell
          the founder to start a mission first (we don't synthesize one — the
          intent must be explicit).
        - Stash a ``_pending_action`` of kind ``_visual_purpose`` carrying the
          saved file path + mission_id, then prompt with a REPLY_KEYBOARD of
          purpose buttons. The founder's next tap is consumed in
          ``handle_message`` and turned into a ``ingest_visual`` Beckman task.
        """
        chat_id = update.message.chat_id
        if not update.message.photo:
            return

        # Resolve target mission. We use the most-recent active mission (highest
        # id). If the founder hasn't started one yet, surface that explicitly.
        try:
            missions = await get_active_missions()
        except Exception as exc:
            logger.error("handle_photo: get_active_missions failed", error=str(exc))
            await self._reply(update,
                "❌ Aktif görev listesi alınamadı. Tekrar dene.")
            return
        if not missions:
            await self._reply(update,
                "📷 Görsel aldım ama aktif bir görev yok.\n"
                "Önce `/mission <açıklama>` ile bir görev başlat, sonra "
                "görseli tekrar gönder.",
                parse_mode="Markdown")
            return
        mission = max(missions, key=lambda m: int(m.get("id") or 0))
        mission_id = int(mission["id"])

        # Pick the largest variant — Telegram sends multiple sizes.
        photo = update.message.photo[-1]
        try:
            from src.tools.workspace import get_mission_workspace
            ws = get_mission_workspace(mission_id)
            visuals_dir = Path(ws) / ".intake" / "visuals"
            visuals_dir.mkdir(parents=True, exist_ok=True)
            file_obj = await context.bot.get_file(photo.file_id)
            # Telegram photos are JPEG. file_unique_id is a stable short slug.
            filename = f"photo_{photo.file_unique_id}.jpg"
            filepath = visuals_dir / filename
            await file_obj.download_to_drive(str(filepath))
        except Exception as exc:
            logger.error("handle_photo: download failed",
                         chat_id=chat_id, error=str(exc))
            await self._reply(update, f"❌ Görsel kaydedilemedi: {_friendly_error(str(exc))}")
            return

        # Stash and prompt.
        self._pending_action[chat_id] = {
            "command": "_visual_purpose",
            "mission_id": mission_id,
            "file_path": str(filepath),
            "ts": _time.time(),
        }
        labels = list(self._VISUAL_PURPOSE_LABELS.keys()) + ["❌ İptal"]
        kb = _make_keyboard([labels[:3], labels[3:]])
        await self._reply(update,
            "📷 Görsel kaydedildi. Bu ne için?\n"
            "(a) Rakip ekran  (b) Moodboard  (c) Wireframe\n"
            "(d) İlham  (e) Mevcut ürün ekranı",
            reply_markup=kb)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z1 T4B: pair a founder document upload with a pending `/edit_html`.

        Consumes ``_pending_action[chat_id]["command"] == "_edit_html_upload"``
        stashed by ``cmd_edit_html``. Downloads the edited HTML next to the
        original under ``mission_<id>/.web/edited/`` and enqueues a
        ``propose_spec_patch_from_html_diff`` mechanical task on Beckman.

        Without a matching ``_pending_action`` we ignore the upload (silent
        — telegram users drop random documents into chats; surfacing every
        one as a usage error would be noise).
        """
        chat_id = update.message.chat_id
        doc = update.message.document
        if doc is None:
            return
        pending = self._pending_action.get(chat_id) or {}
        if pending.get("command") != "_edit_html_upload":
            return
        # 5-min staleness window — drop without acting so the founder
        # doesn't get a confusing pair-up with a stale `/edit_html`.
        if _time.time() - float(pending.get("ts") or 0) > 300:
            self._pending_action.pop(chat_id, None)
            await self._reply(update,
                "⏱ `/edit_html` istemi zamanaşımına uğradı. Tekrar başlat.",
                parse_mode="Markdown")
            return
        mission_id = int(pending.get("mission_id") or 0)
        screen_slug = str(pending.get("screen_slug") or "")
        original_path = str(pending.get("original_path") or "")
        if not (mission_id and screen_slug and original_path):
            self._pending_action.pop(chat_id, None)
            await self._reply(update,
                "❌ `/edit_html` bağlamı eksik. Tekrar başlat.",
                parse_mode="Markdown")
            return

        # Download the edited HTML next to the original.
        try:
            from src.tools.workspace import get_mission_workspace
            ws = Path(get_mission_workspace(mission_id))
            edited_dir = ws / ".web" / "edited"
            edited_dir.mkdir(parents=True, exist_ok=True)
            # Telegram document file_name may collide across uploads — namespace by ts.
            base = doc.file_name or f"{screen_slug}.html"
            base = base.replace("/", "_").replace("\\", "_")[:80]
            ts = int(_time.time())
            edited_path = edited_dir / f"{ts}_{base}"
            file_obj = await context.bot.get_file(doc.file_id)
            await file_obj.download_to_drive(str(edited_path))
        except Exception as exc:
            logger.error("handle_document: download failed",
                         chat_id=chat_id, error=str(exc))
            await self._reply(update,
                f"❌ Belge indirilemedi: {_friendly_error(str(exc))}")
            return
        finally:
            # Always clear the stash — consumed (success) or aborted (fail).
            self._pending_action.pop(chat_id, None)

        # Enqueue propose_spec_patch_from_html_diff via Beckman.
        proposal_dir = ws / ".propagation"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = proposal_dir / f"spec_patch_proposal_{ts}.md"
        try:
            import general_beckman
            await general_beckman.enqueue({
                "title": f"propose_spec_patch:{screen_slug}",
                "description": (
                    f"Diff founder-edited HTML vs original for {screen_slug} "
                    f"(mission #{mission_id})."
                ),
                "agent_type": "mechanical",
                "kind": "main_work",
                "priority": 5,
                "mission_id": mission_id,
                "context": {
                    "executor": "mechanical",
                    "payload": {
                        "action": "propose_spec_patch_from_html_diff",
                        "html_path": original_path,
                        "edited_html_path": str(edited_path),
                        "out_path": str(proposal_path),
                    },
                },
            })
        except Exception as exc:
            logger.error("propose_spec_patch enqueue failed", error=str(exc))
            await self._reply(update,
                f"❌ Spec-patch enqueue failed: {_friendly_error(str(exc))}")
            return
        await self._reply(update,
            f"📋 `{screen_slug}` için spec-patch proposal kuyruğa alındı.\n"
            f"Sonuç: `{proposal_path}`",
            parse_mode="Markdown")

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
                "/credential add <service> <json\\_data> \\[--unsafe\\] — Store credential\n"
                "/credential remove <service> — Delete credential\n"
                "/credential schema <service> — Show required/optional fields\n"
                "/credential log <service> \\[N\\] — Show last N access events\n\n"
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
            tail = list(context.args[2:])
            # Allow --unsafe anywhere after the service name; strip it before
            # parsing the JSON payload.
            unsafe = False
            if "--unsafe" in tail:
                unsafe = True
                tail = [t for t in tail if t != "--unsafe"]
            json_str = " ".join(tail)

            try:
                import json as _json

                data = _json.loads(json_str)
            except (ValueError, TypeError):
                await self._reply(update,
                    "Invalid JSON data. Make sure to use proper JSON format."
                )
                return

            try:
                from ..security.credential_store import (
                    CredentialSchemaError,
                    store_credential,
                )

                await store_credential(service_name, data, unsafe=unsafe)
                await self._reply(update,
                    f"Stored credential for `{service_name}`"
                    + (" \\(unsafe\\)" if unsafe else "")
                    + ".",
                    parse_mode="Markdown",
                )
            except CredentialSchemaError as e:
                await self._reply(update,
                    f"❌ Schema validation failed:\n{e}\n\n"
                    "Re-run with `--unsafe` to bypass.",
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

        elif sub == "log":
            # /credential log <service> [N]
            if len(context.args) < 2:
                await self._reply(update,
                    "Usage: /credential log <service> \\[N\\]",
                    parse_mode="Markdown",
                )
                return
            service_name = context.args[1]
            try:
                limit = int(context.args[2]) if len(context.args) >= 3 else 20
            except ValueError:
                limit = 20
            try:
                from ..security.credential_audit import recent_events

                events = await recent_events(service_name, limit=limit)
                if not events:
                    await self._reply(update,
                        f"No access events for `{service_name}`.",
                        parse_mode="Markdown",
                    )
                    return
                lines = [f"*Access log: {service_name}* (last {len(events)})"]
                for e in events:
                    ok = "✓" if e["success"] else "✗"
                    when = (e["accessed_at"] or "")[:19]
                    actor = e.get("agent") or "—"
                    err = f" err={e['error']}" if e.get("error") else ""
                    lines.append(
                        f"`{when}` {ok} {e['action']} "
                        f"agent={actor} m={e.get('mission_id') or '—'}"
                        f"{err}"
                    )
                await self._reply(update,
                    "\n".join(lines), parse_mode="Markdown"
                )
            except Exception as e:
                await self._reply(update, f"❌ {_friendly_error(str(e))}")

        elif sub == "schema":
            if len(context.args) < 2:
                await self._reply(update,
                    "Usage: /credential schema <service>"
                )
                return
            service_name = context.args[1]
            try:
                from ..security.credential_schemas import describe_schema

                await self._reply(update,
                    describe_schema(service_name),
                    parse_mode="Markdown",
                )
            except Exception as e:
                await self._reply(update,f"❌ {_friendly_error(str(e))}")

        else:
            await self._reply(update,
                "Unknown subcommand. Use: list, add, remove, schema, or log."
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

    async def cmd_rework(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show B10 rework metric for recent missions: count + reasons summary.

        Backs the spec-first bet (docs/i2p-evolution §B10): high rework
        means the spec leaked into build, low rework means the spec
        held. Reads ``missions.phase_7_rework_loops`` for counts and
        the most recent ``logs/kutai.jsonl`` events for reason
        breakdown.
        """
        from src.infra.db import get_mission_rework_summary
        import os
        import json as _json

        try:
            limit_arg = (context.args or ["10"])[0]
            limit = max(1, min(int(limit_arg), 50))
        except (ValueError, TypeError):
            limit = 10

        rows = await get_mission_rework_summary(limit=limit)

        # Aggregate reason counts from the JSONL log if present
        reason_totals: dict[str, int] = {}
        log_path = os.path.join("logs", "kutai.jsonl")
        if os.path.isfile(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "phase_rollback" not in line:
                            continue
                        try:
                            row = _json.loads(line)
                        except Exception:
                            continue
                        if row.get("event") != "phase_rollback":
                            continue
                        reason = row.get("reason") or "other"
                        reason_totals[reason] = reason_totals.get(reason, 0) + 1
            except Exception:
                pass

        lines = ["*Rework Metric (B10)*", ""]
        total = sum(int(r.get("phase_7_rework_loops") or 0) for r in rows)
        lines.append(
            f"Recent {len(rows)} missions: *{total}* phase-7+ rework loops total"
        )
        lines.append("")

        if rows:
            lines.append("*Per mission:*")
            for r in rows[:10]:
                count = int(r.get("phase_7_rework_loops") or 0)
                marker = "!" if count > 0 else " "
                title = (r.get("title") or "")[:40]
                lines.append(
                    f"  {marker} #{r['id']} [{r.get('status','?')}] "
                    f"{count} loops — {title}"
                )
            lines.append("")

        if reason_totals:
            lines.append("*Rollback reasons (all-time log):*")
            for reason, n in sorted(
                reason_totals.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"  {reason:<18} {n}")
        else:
            lines.append(
                "_No phase_rollback events logged yet — "
                "spec-first bet is holding._"
            )

        await self._reply(update, "\n".join(lines), parse_mode="Markdown")

    # ─────────────────────────────────────────────────────────────────
    # Z3 T1C — review density founder dials
    # ─────────────────────────────────────────────────────────────────
    async def cmd_density(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show or update review-density dials for the active mission.

        Usage::

          /density                  — show current dials + usage examples
          /density <key> <value>    — set one dial; confirm + print full state

        Valid keys and values::

          qa_dial               quick | standard | strict
          accessibility_dial    on | off
          multi_file_expansion  true | false
          integration_replay    quick | standard | strict

        Example::

          /density qa_dial strict
          /density accessibility_dial on
          /density multi_file_expansion true
        """
        from src.infra.db import get_active_missions
        from src.workflows.review_density import get_dials, set_dial

        # Resolve active mission
        missions = await get_active_missions()
        if not missions:
            await self._reply(
                update,
                "No active mission. Start one with /mission first.",
            )
            return
        mission = missions[0]
        mission_id: int = mission["id"]
        mission_title: str = mission.get("title") or f"mission #{mission_id}"

        args = context.args or []

        if not args:
            # Show current dials
            dials = await get_dials(mission_id)
            lines = [
                f"*Review density dials* — {mission_title} (#{mission_id})",
                "",
                f"  qa_dial: `{dials.qa_dial}`",
                f"  accessibility_dial: `{dials.accessibility_dial}`",
                f"  multi_file_expansion: `{dials.multi_file_expansion}`",
                f"  integration_replay: `{dials.integration_replay}`",
                "",
                "*Usage:*",
                "  `/density qa_dial quick|standard|strict`",
                "  `/density accessibility_dial on|off`",
                "  `/density multi_file_expansion true|false`",
                "  `/density integration_replay quick|standard|strict`",
            ]
            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
            return

        if len(args) < 2:
            await self._reply(
                update,
                "Usage: `/density <key> <value>` — e.g. `/density qa_dial strict`",
                parse_mode="Markdown",
            )
            return

        key = args[0]
        value_raw = args[1]

        try:
            updated = await set_dial(mission_id, key, value_raw)
        except ValueError as e:
            await self._reply(update, f"Invalid dial: {e}")
            return

        lines = [
            f"Updated *{key}* = `{value_raw}` on {mission_title} (#{mission_id})",
            "",
            "*Current dials:*",
            f"  qa_dial: `{updated.qa_dial}`",
            f"  accessibility_dial: `{updated.accessibility_dial}`",
            f"  multi_file_expansion: `{updated.multi_file_expansion}`",
            f"  integration_replay: `{updated.integration_replay}`",
        ]
        await self._reply(update, "\n".join(lines), parse_mode="Markdown")

    # ─────────────────────────────────────────────────────────────────
    # Z1 Tier 4A — regen surface (C11+A15 + C19)
    # ─────────────────────────────────────────────────────────────────
    async def cmd_regen(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Trigger a regen of one artifact or a directional bundle.

        Usage:
          /regen <mission_id> <artifact_path>       → prompts for change description
          /regen <mission_id> bundle <axis> <dir...> → bundle regen (axis: tone|density|scope)

        With no args, /regen stashes a `_pending_action` that asks the
        founder to describe what to regen. Inline `🔄 regen` callback
        buttons on artifact-emit notifications enter this flow with the
        artifact pre-filled.
        """
        chat_id = update.effective_chat.id
        args = context.args or []

        # No args → stash pending and prompt.
        if not args:
            self._pending_action[chat_id] = {
                "command": "regen",
                "stage": "ask_target",
                "ts": _time.time(),
            }
            await self._reply(
                update,
                "🔄 *Regen*\n\n"
                "Reply with one of:\n"
                "• `<mission_id> <artifact_path>` — re-emit one artifact\n"
                "• `<mission_id> bundle <axis> <direction>` — directional bundle "
                "(axis: tone | density | scope)",
                parse_mode="Markdown",
            )
            return

        # Bundle path: /regen <mid> bundle <axis> <direction...>
        if len(args) >= 4 and args[1].lower() == "bundle":
            try:
                mission_id = int(args[0])
            except ValueError:
                await self._reply(update, "❌ mission_id must be an integer.")
                return
            axis = args[2]
            direction = " ".join(args[3:])
            await self._enqueue_regen_bundle(update, mission_id, axis, direction)
            return

        # Artifact path: /regen <mid> <artifact_path> [change description...]
        try:
            mission_id = int(args[0])
        except ValueError:
            await self._reply(update, "❌ mission_id must be an integer.")
            return
        artifact_path = args[1]
        change_description = " ".join(args[2:]) if len(args) > 2 else ""

        if not change_description.strip():
            self._pending_action[chat_id] = {
                "command": "regen",
                "stage": "ask_change",
                "mission_id": mission_id,
                "artifact_path": artifact_path,
                "ts": _time.time(),
            }
            await self._reply(
                update,
                f"📝 Describe the change for `{artifact_path}` (mission #{mission_id}):",
                parse_mode="Markdown",
            )
            return

        await self._enqueue_regen_artifact(
            update, mission_id, artifact_path, change_description
        )

    async def _enqueue_regen_artifact(
        self, update, mission_id: int, artifact_path: str, change_description: str
    ):
        """Enqueue a mr_roboto `regen_artifact` mechanical task."""
        try:
            import general_beckman
            await general_beckman.enqueue({
                "title": f"regen_artifact:{os.path.basename(artifact_path)}",
                "description": (
                    f"Founder regen of {artifact_path}: {change_description}"
                ),
                "agent_type": "mechanical",
                "kind": "main_work",
                "priority": 5,
                "mission_id": int(mission_id),
                "context": {
                    "executor": "mechanical",
                    "payload": {
                        "action": "regen_artifact",
                        "artifact_path": artifact_path,
                        "change_description": change_description,
                    },
                },
            })
        except Exception as exc:
            logger.error("regen_artifact enqueue failed", error=str(exc))
            await self._reply(update, f"❌ Regen enqueue failed: {exc}")
            return
        await self._reply(
            update,
            f"🔄 Regen queued for `{artifact_path}` — change: _{change_description}_",
            parse_mode="Markdown",
        )

    async def _enqueue_regen_bundle(
        self, update, mission_id: int, axis: str, direction: str
    ):
        """Enqueue a mr_roboto `regen_bundle` mechanical task."""
        try:
            from mr_roboto import known_regen_axes
            valid = set(known_regen_axes())
        except Exception:
            valid = {"tone", "density", "scope"}
        if axis not in valid:
            await self._reply(
                update,
                f"❌ Unknown axis `{axis}`. Known: {sorted(valid)}",
                parse_mode="Markdown",
            )
            return
        try:
            import general_beckman
            await general_beckman.enqueue({
                "title": f"regen_bundle:{axis}:{direction}",
                "description": (
                    f"Founder bundle regen mission #{mission_id}: "
                    f"axis={axis} direction={direction}"
                ),
                "agent_type": "mechanical",
                "kind": "main_work",
                "priority": 5,
                "mission_id": int(mission_id),
                "context": {
                    "executor": "mechanical",
                    "payload": {
                        "action": "regen_bundle",
                        "axis": axis,
                        "direction": direction,
                    },
                },
            })
        except Exception as exc:
            logger.error("regen_bundle enqueue failed", error=str(exc))
            await self._reply(update, f"❌ Bundle regen enqueue failed: {exc}")
            return
        await self._reply(
            update,
            f"🔄 Bundle regen queued — axis: *{axis}*, direction: _{direction}_",
            parse_mode="Markdown",
        )

    async def cmd_edit_html(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z1 Tier 4 (T4B / C17+A20): two-way HTML edit reflection.

        Founder edits an annotated HTML offline and uploads the modified
        file. We store ``_pending_action`` so the next document upload
        is paired with the original screen for ``propose_spec_patch``.

        Usage: ``/edit_html <screen_slug>``
        """
        chat_id = update.effective_chat.id
        args = context.args or []
        if not args:
            await self._reply(update,
                "Usage: `/edit_html <screen_slug>` — then upload the "
                "edited HTML as a document. The slug must match a screen "
                "produced in your active mission (e.g. `screen_5_3`).",
                parse_mode="Markdown")
            return
        screen_slug = args[0].strip()
        # Find the matching original HTML in the most-recent active mission.
        try:
            missions = await get_active_missions()
        except Exception as exc:
            await self._reply(update,
                f"❌ Aktif görev listesi alınamadı: {_friendly_error(str(exc))}")
            return
        if not missions:
            await self._reply(update,
                "📝 Önce `/mission <açıklama>` ile bir görev başlat.",
                parse_mode="Markdown")
            return
        mission = max(missions, key=lambda m: int(m.get("id") or 0))
        mission_id = int(mission["id"])
        try:
            from src.tools.workspace import get_mission_workspace
            ws = Path(get_mission_workspace(mission_id))
            web_dir = ws / ".web"
            candidates = list(web_dir.glob(f"*{screen_slug}*.html")) if web_dir.exists() else []
        except Exception:
            candidates = []
        if not candidates:
            await self._reply(update,
                f"⚠️ `{screen_slug}` için orijinal HTML bulunamadı "
                f"(`mission_{mission_id}/.web/`).",
                parse_mode="Markdown")
            return
        original_path = str(candidates[0])
        self._pending_action[chat_id] = {
            "command": "_edit_html_upload",
            "mission_id": mission_id,
            "screen_slug": screen_slug,
            "original_path": original_path,
            "ts": _time.time(),
        }
        await self._reply(update,
            f"📤 `{screen_slug}` için düzenlenmiş HTML'i şimdi *belge olarak* "
            f"(document, photo değil) yükle.\n"
            f"Orijinal: `{original_path}`",
            parse_mode="Markdown")

    async def cmd_propagate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z1 Tier 4 (T4B / B2): asset->spec propagation primitive.

        Usage: ``/propagate <asset_path> <change description...>``
        Walks the produces/consumes graph in i2p_v3.json and emits a
        ``propagation_proposal.md`` listing affected dependents.
        """
        args = context.args or []
        if len(args) < 2:
            await self._reply(update,
                "Usage: `/propagate <asset_path> <change description>`\n"
                "Example: `/propagate mission_1/.style/design_tokens.json "
                "switch primary from blue to teal`",
                parse_mode="Markdown")
            return
        asset_path = args[0]
        change_description = " ".join(args[1:])
        # Pick most-recent active mission for context.
        try:
            missions = await get_active_missions()
        except Exception as exc:
            await self._reply(update,
                f"❌ Aktif görev listesi alınamadı: {_friendly_error(str(exc))}")
            return
        mission_id = (
            max(missions, key=lambda m: int(m.get("id") or 0))["id"]
            if missions else 0
        )
        try:
            from mr_roboto.propagate_asset_change import propagate_asset_change
            from src.tools.workspace import get_mission_workspace
            wf_path = str(Path("src/workflows/i2p/i2p_v3.json").resolve())
            out_dir = Path(get_mission_workspace(int(mission_id))) / ".propagation"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"propagation_proposal_{int(_time.time())}.md"
            res = propagate_asset_change(
                asset_path=asset_path,
                change_description=change_description,
                workflow_path=wf_path,
                mission_id=str(mission_id),
                out_path=str(out_path),
            )
        except Exception as exc:
            await self._reply(update, f"❌ propagate failed: {_friendly_error(str(exc))}")
            return
        if not res.get("ok"):
            await self._reply(update, f"⚠️ {res.get('error')}", parse_mode="Markdown")
            return
        deps = res.get("dependents") or []
        ups = res.get("upstream_candidates") or []
        lines = [
            f"🎯 *Propagation* — `{asset_path}`",
            f"Origin step: `{res.get('origin_step_id')}`",
            f"Downstream dependents: {len(deps)}",
            f"Upstream candidates: {len(ups)}",
            f"Proposal: `{out_path}`",
        ]
        for d in deps[:5]:
            lines.append(f"  • `{d['step_id']}` — {d.get('step_name','')}")
        await self._reply(update, "\n".join(lines), parse_mode="Markdown")

    async def cmd_preview(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z1 T4C — emit a tunneled preview URL for a mission's HTML prototypes.

        Usage: ``/preview <mission_id>``. Calls
        :func:`mr_roboto.emit_preview_url` and replies with the live URL or
        a ``pending: hosting deferred to Z2`` message when the operator has
        not opted into ``KUTAI_PREVIEW_PROVIDER=cloudflared``.
        """
        args = context.args or []
        if not args:
            await self._reply(update, "Usage: /preview <mission_id>")
            return
        try:
            mission_id = int(args[0])
        except (ValueError, TypeError):
            await self._reply(update, "Mission id must be an integer.")
            return
        try:
            from mr_roboto.emit_preview_url import emit_preview_url
            res = await emit_preview_url(mission_id=mission_id)
        except Exception as e:
            await self._reply(update, f"Preview emit failed: {e}")
            return
        if res.get("pending"):
            await self._reply(
                update,
                f"Preview pending for mission #{mission_id}\n"
                f"Hosting deferred to Z2 — surface written to "
                f"`{res.get('path')}`",
                parse_mode="Markdown",
            )
            return
        url = res.get("url") or "(no url)"
        await self._reply(
            update,
            f"📡 Preview ready for mission #{mission_id}\n{url}",
        )

    async def cmd_preview_off(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ):
        """Z1 T4C — terminate the preview tunnel for a mission.

        Usage: ``/preview_off <mission_id>``.
        """
        args = context.args or []
        if not args:
            await self._reply(update, "Usage: /preview_off <mission_id>")
            return
        try:
            mission_id = int(args[0])
        except (ValueError, TypeError):
            await self._reply(update, "Mission id must be an integer.")
            return
        try:
            from mr_roboto.kill_preview_url import kill_preview_url
            res = await kill_preview_url(mission_id=mission_id)
        except Exception as e:
            await self._reply(update, f"Preview kill failed: {e}")
            return
        pid = res.get("killed_pid")
        if pid:
            await self._reply(
                update,
                f"Preview tunnel for mission #{mission_id} terminated "
                f"(pid={pid}).",
            )
        else:
            await self._reply(
                update,
                f"No active preview tunnel for mission #{mission_id}; "
                "surface files cleaned.",
            )

    async def cmd_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z1 T5A (A5) — founder attention budget inspector / setter.

        Usage:
          /budget                  — show remaining minutes for the
                                     most-recent active mission.
          /budget set <minutes>    — set the budget for the most-recent
                                     active mission.
        """
        from src.infra.db import get_db

        args = context.args or []
        db = await get_db()

        # Find the most-recent active mission for this chat.
        cur = await db.execute(
            "SELECT id, status FROM missions "
            "WHERE status IN ('active', 'pending') "
            "ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
        if not row:
            # Fall back to any mission if nothing active.
            cur = await db.execute(
                "SELECT id, status FROM missions ORDER BY id DESC LIMIT 1"
            )
            row = await cur.fetchone()
        if not row:
            await self._reply(update, "No missions found.")
            return
        mission_id = int(row[0])

        if args and args[0] == "set":
            if len(args) < 2:
                await self._reply(update, "Usage: /budget set <minutes>")
                return
            try:
                minutes = int(args[1])
            except (ValueError, TypeError):
                await self._reply(update, "Minutes must be an integer.")
                return
            if minutes < 0:
                await self._reply(update, "Minutes must be >= 0.")
                return
            await db.execute(
                "UPDATE missions SET founder_attention_budget_minutes = ? "
                "WHERE id = ?",
                (minutes, mission_id),
            )
            await db.commit()
            await self._reply(
                update,
                f"Budget set: mission #{mission_id} = {minutes} minutes.",
            )
            return

        # Read-only view.
        from mr_roboto.attention_check import attention_check
        try:
            res = await attention_check(mission_id=mission_id, reserve_minutes=0)
        except Exception as e:
            await self._reply(update, f"Budget check failed: {e}")
            return
        if not res.get("budget_set"):
            await self._reply(
                update,
                f"Mission #{mission_id}: no attention budget set.\n"
                f"Use `/budget set <minutes>` to declare one.",
                parse_mode="Markdown",
            )
            return
        await self._reply(
            update,
            (
                f"Mission #{mission_id} attention budget\n"
                f"  budget:    {res.get('budget')} min\n"
                f"  spent:     {res.get('spent')} min\n"
                f"  remaining: {res.get('remaining')} min"
            ),
        )

    async def cmd_attention(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T1D (B5) — Founder attention budget queue.

        Subcommands:
          /attention status [product_id]        — show queue (today/this_week/deferred/idle)
          /attention defer <card_id> to <when>  — defer a card to next morning
          /attention budget <product_id> <min>  — set daily cap (minutes)
        """
        from src.app.attention_budget import (
            check_budget, get_queue, record_deferred, set_daily_budget,
            next_review_window,
        )
        args = context.args or []
        sub = args[0].lower() if args else "status"

        if sub == "status":
            try:
                product_id = int(args[1]) if len(args) > 1 else None
            except (ValueError, TypeError):
                product_id = None
            try:
                queue = await get_queue(product_id)
            except Exception as e:
                await self._reply(update, f"Attention queue error: {e}")
                return
            cap = queue["cap"]
            spent = queue["spent"]
            remaining = queue["remaining"]
            over = " OVER BUDGET" if queue["over_budget"] else ""
            lines = [
                f"*Attention Queue* — {spent}/{cap} min spent today{over}",
                "",
            ]
            for section, label in [
                ("today", "Today (p0+p1)"),
                ("this_week", "This Week (p2)"),
                ("deferred", "Deferred"),
                ("when_idle", "When Idle (p3)"),
            ]:
                cards = queue.get(section) or []
                if not cards:
                    continue
                lines.append(f"*{label}:*")
                for c in cards[:5]:
                    fold = " _(below fold)_" if c.get("below_fold") else ""
                    urgent = " ⚡" if c.get("urgent") else ""
                    lines.append(
                        f"  [{c['id']}] {c['title'][:50]}{urgent}{fold}"
                    )
                lines.append("")
            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
            return

        if sub == "defer":
            # /attention defer <card_id> to <when>
            # <when>: tomorrow | morning | YYYY-MM-DD
            if len(args) < 4 or args[2].lower() != "to":
                await self._reply(
                    update,
                    "Usage: `/attention defer <card_id> to <tomorrow|morning|YYYY-MM-DD>`",
                    parse_mode="Markdown",
                )
                return
            try:
                card_id = int(args[1])
            except (ValueError, TypeError):
                await self._reply(update, "card_id must be an integer.")
                return
            when_str = args[3].lower()
            import datetime as _dt
            nrw = next_review_window({})
            if when_str in ("tomorrow", "morning"):
                deferred_to = nrw.strftime("%Y-%m-%d %H:%M:%S")
            else:
                try:
                    _dt.datetime.strptime(when_str, "%Y-%m-%d")
                    deferred_to = when_str + " 09:00:00"
                except ValueError:
                    await self._reply(update, "when must be: tomorrow, morning, or YYYY-MM-DD")
                    return
            try:
                await record_deferred(card_id=card_id, product_id=0, deferred_to=deferred_to)
                await self._reply(
                    update,
                    f"Card #{card_id} deferred to {deferred_to}.",
                )
            except Exception as e:
                await self._reply(update, f"Defer failed: {e}")
            return

        if sub == "budget":
            # /attention budget <product_id> <minutes>
            if len(args) < 3:
                await self._reply(
                    update,
                    "Usage: `/attention budget <product_id> <minutes>`",
                    parse_mode="Markdown",
                )
                return
            try:
                product_id = int(args[1])
                minutes = int(args[2])
            except (ValueError, TypeError):
                await self._reply(update, "product_id and minutes must be integers.")
                return
            try:
                await set_daily_budget(product_id=product_id, minutes=minutes)
                await self._reply(
                    update,
                    f"Daily attention cap for product #{product_id} set to {minutes} min.",
                )
            except Exception as e:
                await self._reply(update, f"Budget set failed: {e}")
            return

        await self._reply(
            update,
            "Usage: `/attention <status|defer|budget> [args...]`",
            parse_mode="Markdown",
        )

    async def cmd_audit_comms(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T1D (B9) — External comms audit log search.

        Usage:
          /audit_comms search recipient:<x>        — filter by recipient
          /audit_comms search channel:<x>          — filter by channel
          /audit_comms search since:<YYYY-MM-DD>   — filter by date
          /audit_comms search mission:<id>         — filter by mission
          /audit_comms gaps                        — show pending audit gaps
        """
        from mr_roboto.audit_log import search_sends, pending_audit_gaps
        args = context.args or []
        sub = args[0].lower() if args else "search"

        if sub == "gaps":
            try:
                gaps = await pending_audit_gaps(window_minutes=5)
                if not gaps:
                    await self._reply(update, "No audit gaps found.")
                    return
                lines = [f"*Audit gaps ({len(gaps)} found):*"]
                for g in gaps[:10]:
                    lines.append(
                        f"  vc_id={g.get('vendor_call_id')} verb={g.get('verb')!r}"
                        f" mission={g.get('mission_id')} at {g.get('created_at')}"
                    )
                await self._reply(update, "\n".join(lines), parse_mode="Markdown")
            except Exception as e:
                await self._reply(update, f"Gap check failed: {e}")
            return

        # Parse search filters from args like "recipient:@foo channel:telegram"
        filters: dict = {}
        for arg in args[1:]:
            if ":" in arg:
                k, v = arg.split(":", 1)
                filters[k.lower()] = v
        try:
            results = await search_sends(
                recipient=filters.get("recipient"),
                channel=filters.get("channel"),
                since=filters.get("since"),
                mission_id=int(filters["mission"]) if "mission" in filters else None,
                limit=15,
            )
        except Exception as e:
            await self._reply(update, f"Audit search failed: {e}")
            return
        if not results:
            await self._reply(update, "No external comms log entries found.")
            return
        lines = [f"*External comms log ({len(results)} rows):*"]
        for r in results:
            revoked = " ~~REVOKED~~" if r.get("revoked_at") else ""
            lines.append(
                f"  [{r['log_id']}] {r['sent_at']} {r['channel']}"
                f" → {r.get('recipient') or '(broadcast)'}{revoked}\n"
                f"    hash={r['content_hash'][:12]}... rev={r['reversibility']}"
                f" m={r.get('source_mission_id')}"
            )
        await self._reply(update, "\n".join(lines), parse_mode="Markdown")

    async def cmd_email(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T2A — Email-send shared service management.

        Subcommands:
          /email config <product_id>            — show provider config for product
          /email upgrade <product_id>           — flip tier from 'free' to 'paid'
          /email test <product_id> <address>    — send a test email to <address>
        """
        args = context.args or []
        if not args:
            await self._reply(
                update,
                "Usage:\n"
                "`/email config <product_id>` — show email config\n"
                "`/email upgrade <product_id>` — flip tier free→paid\n"
                "`/email test <product_id> <addr>` — send test email",
                parse_mode="Markdown",
            )
            return

        sub = args[0].lower()

        if sub == "config":
            if len(args) < 2:
                await self._reply(update, "Usage: `/email config <product_id>`", parse_mode="Markdown")
                return
            product_id = args[1]
            try:
                from src.infra.db import get_db
                db = await get_db()
                cur = await db.execute(
                    "SELECT provider, from_domain, api_key_ref, monthly_quota, tier, created_at "
                    "FROM product_email_config WHERE product_id = ?",
                    (product_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    await self._reply(update, f"No email config found for product `{product_id}`.", parse_mode="Markdown")
                    return
                provider, from_domain, api_key_ref, monthly_quota, tier, created_at = row
                msg = (
                    f"*Email config for* `{product_id}`\n"
                    f"Provider: `{provider}`\n"
                    f"Tier: `{tier}`\n"
                    f"From domain: `{from_domain or '(not set)'}`\n"
                    f"API key ref: `{api_key_ref or '(not set)'}`\n"
                    f"Monthly quota: `{monthly_quota or 'unlimited'}`\n"
                    f"Created: `{created_at}`"
                )
                await self._reply(update, msg, parse_mode="Markdown")
            except Exception as e:
                await self._reply(update, f"Error: {e}")

        elif sub == "upgrade":
            if len(args) < 2:
                await self._reply(update, "Usage: `/email upgrade <product_id>`", parse_mode="Markdown")
                return
            product_id = args[1]
            try:
                from src.infra.db import get_db
                db = await get_db()
                cur = await db.execute(
                    "UPDATE product_email_config SET tier = 'paid', updated_at = datetime('now') "
                    "WHERE product_id = ?",
                    (product_id,),
                )
                await db.commit()
                if cur.rowcount == 0:
                    await self._reply(update, f"No email config found for product `{product_id}`.", parse_mode="Markdown")
                else:
                    await self._reply(
                        update,
                        f"Email tier for `{product_id}` flipped to *paid*.\n"
                        "Remember to set a paid provider (postmark/ses) and API key ref.",
                        parse_mode="Markdown",
                    )
            except Exception as e:
                await self._reply(update, f"Error: {e}")

        elif sub == "test":
            if len(args) < 3:
                await self._reply(
                    update,
                    "Usage: `/email test <product_id> <email_address>`",
                    parse_mode="Markdown",
                )
                return
            product_id = args[1]
            addr = args[2]
            try:
                from src.integrations.email.service import send_email
                result = await send_email(
                    product_id=product_id,
                    to=addr,
                    subject=f"[KutAI Test] Email from product {product_id}",
                    body_md=(
                        f"This is a test email sent from KutAI for product `{product_id}`.\n\n"
                        "If you received this, email delivery is working correctly."
                    ),
                    idempotency_key=f"test-{product_id}-{addr}",
                )
                status = result.get("status", "unknown")
                msg_id = result.get("message_id")
                if status == "sent":
                    await self._reply(
                        update,
                        f"Test email sent to `{addr}`.\n"
                        f"Provider: `{result.get('provider')}`\n"
                        f"Message ID: `{msg_id}`",
                        parse_mode="Markdown",
                    )
                elif status == "quota_blocked":
                    await self._reply(
                        update,
                        f"Send blocked — quota exhausted ({result.get('sent_count')}/{result.get('quota')} this month).",
                    )
                elif status == "suppressed":
                    await self._reply(
                        update,
                        f"Address `{addr}` is suppressed (reason: {result.get('reason')}).",
                        parse_mode="Markdown",
                    )
                else:
                    await self._reply(
                        update,
                        f"Send failed (status={status}): {result.get('error') or result}",
                    )
            except Exception as e:
                await self._reply(update, f"Error sending test email: {e}")

        else:
            await self._reply(
                update,
                "Unknown subcommand. Use: `config`, `upgrade`, or `test`.",
                parse_mode="Markdown",
            )

    async def cmd_crisis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T3E (B6) — Crisis comms tiered playbook.

        Subcommands:
          /crisis open [tier] [summary]    — open a crisis event (tier 1-4; default 1)
          /crisis resume <product_id>      — resume marketing (clear freeze) for product
          /crisis status                   — list active crisis events
        """
        args = context.args or []
        full_text = "/crisis " + " ".join(args) if args else "/crisis status"

        try:
            from mr_roboto.crisis_open import parse_crisis_cmd
        except Exception as exc:
            await self._reply(update, f"Crisis module unavailable: {exc}")
            return

        cmd = parse_crisis_cmd(full_text)
        sub = cmd.get("subcommand", "status")

        if sub == "status":
            try:
                from src.infra.db import get_db
                db = await get_db()
                async with db.execute(
                    "SELECT event_id, product_id, tier, source, summary, opened_at "
                    "FROM crisis_events WHERE status='active' "
                    "ORDER BY opened_at DESC LIMIT 10"
                ) as cur:
                    rows = await cur.fetchall()
                if not rows:
                    await self._reply(update, "No active crisis events.")
                    return
                tier_labels = {1: "Brand misstep", 2: "Outage", 3: "Security breach", 4: "Existential/Legal"}
                lines = ["*Active crisis events:*"]
                for row in rows:
                    eid, pid, tier, src, summ, opened = row
                    label = tier_labels.get(tier, f"Tier {tier}")
                    lines.append(
                        f"• Event #{eid} | {pid} | T{tier} {label} | "
                        f"src={src} | {(summ or '')[:60]} | opened={opened}"
                    )
                await self._reply(update, "\n".join(lines), parse_mode="Markdown")
            except Exception as exc:
                await self._reply(update, f"Error fetching crisis status: {exc}")

        elif sub == "open":
            tier = int(cmd.get("tier") or 1)
            summary = cmd.get("summary") or ""
            if not summary and len(args) > 1:
                # Try to infer product from args
                summary = " ".join(args[2:]) if len(args) > 2 else ""

            # Use product_id from summary context or prompt
            product_id = "default"
            if args and len(args) > 1:
                # First non-tier arg might be a product_id
                candidate = args[1] if not args[1].isdigit() else (args[2] if len(args) > 2 else "")
                if candidate and not candidate.isdigit():
                    product_id = candidate

            try:
                from mr_roboto.crisis_open import open_crisis_event
                from mr_roboto.crisis_freeze_marketing import run as freeze_run

                event = await open_crisis_event(
                    product_id=product_id,
                    tier=tier,
                    source="manual",
                    summary=summary or f"Manual crisis open (Tier {tier})",
                )
                event_id = event["event_id"]

                # Auto-freeze for Tier 2+
                if tier >= 2:
                    await freeze_run({"product_id": product_id, "event_id": event_id})
                    freeze_msg = f"\nMarketing freeze activated for `{product_id}`."
                else:
                    freeze_msg = ""

                tier_labels = {1: "Brand misstep / pile-on", 2: "Outage / data issue",
                               3: "Security incident / breach", 4: "Existential / legal"}
                label = tier_labels.get(tier, f"Tier {tier}")
                await self._reply(
                    update,
                    f"*Crisis event opened* (#{event_id})\n"
                    f"Product: `{product_id}`\n"
                    f"Tier: {tier} — {label}\n"
                    f"Summary: {summary or '(none)'}"
                    f"{freeze_msg}\n\n"
                    f"See `playbooks/crisis_comms_tier{tier}.md` for next steps.\n"
                    f"Use `/crisis status` to monitor.",
                    parse_mode="Markdown",
                )
            except Exception as exc:
                await self._reply(update, f"Error opening crisis event: {exc}")

        elif sub == "resume":
            product_id = cmd.get("product_id") or ""
            if not product_id:
                await self._reply(update, "Usage: `/crisis resume <product_id>`", parse_mode="Markdown")
                return
            try:
                from mr_roboto.crisis_freeze_marketing import resume_marketing_freeze
                from src.infra.db import get_db

                result = await resume_marketing_freeze(product_id)
                cleared = result.get("cleared", 0)

                # Mark active events resolved if product matches
                db = await get_db()
                await db.execute(
                    "UPDATE crisis_events SET status='resolved', "
                    "resolved_at=strftime('%Y-%m-%d %H:%M:%S','now') "
                    "WHERE product_id=? AND status='active'",
                    (product_id,),
                )
                await db.commit()

                if cleared > 0:
                    await self._reply(
                        update,
                        f"Marketing freeze lifted for `{product_id}` ({cleared} freeze row(s) cleared).\n"
                        "Active crisis events for this product marked resolved.\n"
                        "Remember to publish a postmortem if Tier 2+.",
                        parse_mode="Markdown",
                    )
                else:
                    await self._reply(
                        update,
                        f"No active freeze found for `{product_id}`. "
                        "Active crisis events (if any) marked resolved.",
                        parse_mode="Markdown",
                    )
            except Exception as exc:
                await self._reply(update, f"Error resuming crisis: {exc}")

        elif "error" in cmd or sub == "unknown":
            await self._reply(
                update,
                "Usage:\n"
                "`/crisis status` — list active crises\n"
                "`/crisis open [tier] [summary]` — open crisis (tier 1-4)\n"
                "`/crisis resume <product_id>` — resume marketing / resolve crisis",
                parse_mode="Markdown",
            )

        else:
            await self._reply(
                update,
                "Unknown /crisis subcommand. Try `/crisis status`.",
                parse_mode="Markdown",
            )

    async def cmd_launch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T3A (A2) — Launch playbook.

        Usage:
          /launch <YYYY-MM-DD> <channels>   — create a launch mission
          /launch status                    — list active launches
          /launch help                      — show usage

        Examples:
          /launch 2026-06-15 hn,twitter,linkedin
          /launch 2026-06-15 all
          /launch status
        """
        args = context.args or []

        if not args or args[0] in ("help", "--help"):
            await self._reply(
                update,
                (
                    "*Launch playbook (A2)*\n\n"
                    "Usage: `/launch <YYYY-MM-DD> <channels>`\n"
                    "Channels: `hn`, `ph`, `twitter`, `linkedin`, `reddit` (comma-separated) or `all`\n\n"
                    "Examples:\n"
                    "  `/launch 2026-06-15 hn,twitter,linkedin`\n"
                    "  `/launch 2026-06-15 all`\n\n"
                    "Use `/launch status` to list active launches."
                ),
                parse_mode="Markdown",
            )
            return

        if args[0] == "status":
            try:
                from src.infra.db import get_db
                db = await get_db()
                async with db.execute(
                    "SELECT launch_id, product_id, scheduled_publish_at, status, channels_json "
                    "FROM launches ORDER BY created_at DESC LIMIT 10"
                ) as cur:
                    rows = await cur.fetchall()
                if not rows:
                    await self._reply(update, "No launches found.")
                    return
                lines = ["*Active launches:*"]
                for row in rows:
                    lid, pid, pub_at, status, ch_json = row
                    lines.append(
                        f"• Launch #{lid} | {pid} | T-0: {pub_at} | "
                        f"status={status} | {ch_json}"
                    )
                await self._reply(update, "\n".join(lines), parse_mode="Markdown")
            except Exception as exc:
                await self._reply(update, f"Error fetching launches: {exc}")
            return

        # Parse date + channels
        if len(args) < 2:
            await self._reply(
                update,
                "Usage: `/launch <YYYY-MM-DD> <channels>`\n"
                "Example: `/launch 2026-06-15 hn,twitter`",
                parse_mode="Markdown",
            )
            return

        date_str = args[0]
        channels_raw = args[1]

        VALID_CHANNELS = {"hn", "ph", "twitter", "linkedin", "reddit"}
        if channels_raw.strip().lower() == "all":
            channels = sorted(VALID_CHANNELS)
        else:
            channels = [c.strip().lower() for c in channels_raw.split(",") if c.strip()]
            unknown = [c for c in channels if c not in VALID_CHANNELS]
            if unknown:
                await self._reply(
                    update,
                    f"Unknown channels: {', '.join(unknown)}. "
                    f"Supported: {', '.join(sorted(VALID_CHANNELS))}",
                )
                return

        try:
            from datetime import datetime
            pub_dt = datetime.strptime(date_str, "%Y-%m-%d")
            # Default to 09:00 UTC on the specified date
            scheduled_publish_at = pub_dt.strftime("%Y-%m-%d 09:00:00")
        except ValueError:
            await self._reply(
                update,
                f"Invalid date format: `{date_str}`. Use `YYYY-MM-DD`.",
                parse_mode="Markdown",
            )
            return

        # Determine product_id (default to "default")
        product_id = "default"
        if len(args) >= 3:
            product_id = args[2]

        try:
            import json
            task_id = await enqueue_launch_mission(
                spec={
                    "title": f"Launch playbook for '{product_id}' on {date_str}",
                    "description": (
                        f"A2 launch playbook: schedule={scheduled_publish_at}, "
                        f"channels={channels}, product={product_id}"
                    ),
                    "agent_type": "assistant",
                    "kind": "main_work",
                    "context": {
                        "product_id": product_id,
                        "scheduled_publish_at": scheduled_publish_at,
                        "channels": channels,
                        "workflow": "launch_playbook",
                    },
                },
            )
        except Exception as exc:
            await self._reply(update, f"Failed to create launch mission: {exc}")
            return

        import json
        ch_str = ", ".join(channels)
        await self._reply(
            update,
            (
                f"*Launch playbook created!*\n\n"
                f"Product: `{product_id}`\n"
                f"T-0: `{scheduled_publish_at}` UTC\n"
                f"Channels: `{ch_str}`\n"
                f"Task ID: `{task_id}`\n\n"
                f"Phase clock:\n"
                f"  T-72h — Asset prep (drafts)\n"
                f"  T-24h — Founder approval\n"
                f"  T-0   — Synchronized publish\n"
                f"  T+7d  — Lessons writeback"
            ),
            parse_mode="Markdown",
        )

    async def cmd_audit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z1 audit-log inspector — view recent rows of write-only Z1 logs.

        Subcommands:
          /audit critic [mission_id] [N]      — last N critic_log rows
          /audit regen [mission_id] [N]       — last N regen_log rows
          /audit preview [mission_id] [N]     — last N preview_log rows
          /audit paraflow [mission_id] [N]    — last N paraflow_diff_log rows
          /audit streaming [N]                — last N streaming_guard_log rows

        Default N=10. mission_id omitted = aggregate across missions.
        """
        from src.infra.db import get_db
        args = context.args or []
        if not args:
            await self._reply(
                update,
                "Usage: `/audit <critic|regen|preview|paraflow|streaming> "
                "[mission_id] [N]`",
                parse_mode="Markdown",
            )
            return

        kind = args[0].lower()
        # Optional mission_id (integer in pos 1) + N (integer in pos 2)
        mid: int | None = None
        limit = 10
        rest = args[1:]
        if rest:
            try:
                mid = int(rest[0])
                rest = rest[1:]
            except (ValueError, TypeError):
                mid = None
        if rest:
            try:
                limit = max(1, min(50, int(rest[0])))
            except (ValueError, TypeError):
                pass

        db = await get_db()
        if kind == "critic":
            where = "WHERE mission_id = ?" if mid is not None else ""
            sql = (
                f"SELECT mission_id, action_name, verdict, reasons_json, "
                f"created_at FROM critic_log {where} "
                f"ORDER BY id DESC LIMIT ?"
            )
            params = ((mid, limit) if mid is not None else (limit,))
            cur = await db.execute(sql, params)
            rows = await cur.fetchall() or []
            if not rows:
                await self._reply(update, "No critic_log rows.")
                return
            lines = ["🛡 *critic_log* (latest)"]
            for r in rows:
                emoji = "✅" if r[2] == "pass" else "🛑"
                lines.append(
                    f"{emoji} #{r[0]} {r[1]} — {r[2]} _{r[4]}_"
                )
            await self._reply(
                update, "\n".join(lines), parse_mode="Markdown",
            )
            return

        if kind == "regen":
            where = "WHERE mission_id = ?" if mid is not None else ""
            sql = (
                f"SELECT mission_id, artifact_path, change_description, "
                f"prev_version, new_version, scope, created_at "
                f"FROM regen_log {where} ORDER BY id DESC LIMIT ?"
            )
            params = ((mid, limit) if mid is not None else (limit,))
            cur = await db.execute(sql, params)
            rows = await cur.fetchall() or []
            if not rows:
                await self._reply(update, "No regen_log rows.")
                return
            lines = ["🔄 *regen_log* (latest)"]
            for r in rows:
                lines.append(
                    f"#{r[0]} `{r[1]}` ({r[5]}) — _{r[6]}_\n"
                    f"  ↳ {r[2][:80]}"
                )
            await self._reply(
                update, "\n".join(lines), parse_mode="Markdown",
            )
            return

        if kind == "preview":
            where = "WHERE mission_id = ?" if mid is not None else ""
            sql = (
                f"SELECT mission_id, action, url, exit_code, created_at "
                f"FROM preview_log {where} ORDER BY id DESC LIMIT ?"
            )
            params = ((mid, limit) if mid is not None else (limit,))
            cur = await db.execute(sql, params)
            rows = await cur.fetchall() or []
            if not rows:
                await self._reply(update, "No preview_log rows.")
                return
            lines = ["🌐 *preview_log* (latest)"]
            for r in rows:
                ok = "✅" if r[3] == 0 else "❌"
                lines.append(
                    f"{ok} #{r[0]} {r[1]} {r[2] or '(no url)'} _{r[4]}_"
                )
            await self._reply(
                update, "\n".join(lines), parse_mode="Markdown",
            )
            return

        if kind == "paraflow":
            where = "WHERE mission_id = ?" if mid is not None else ""
            sql = (
                f"SELECT mission_id, archetype, verdict, score, created_at "
                f"FROM paraflow_diff_log {where} ORDER BY id DESC LIMIT ?"
            )
            params = ((mid, limit) if mid is not None else (limit,))
            cur = await db.execute(sql, params)
            rows = await cur.fetchall() or []
            if not rows:
                await self._reply(update, "No paraflow_diff_log rows.")
                return
            lines = ["📐 *paraflow_diff_log* (latest)"]
            for r in rows:
                emoji = {"paraflow_par": "✅",
                         "paraflow_partial": "🟡",
                         "paraflow_gap": "🛑"}.get(r[2], "•")
                score = f"{r[3]:.2f}" if r[3] is not None else "—"
                lines.append(
                    f"{emoji} #{r[0]} {r[1]} ({score}) — {r[2]} _{r[4]}_"
                )
            await self._reply(
                update, "\n".join(lines), parse_mode="Markdown",
            )
            return

        if kind == "streaming":
            sql = (
                "SELECT mission_id, task_id, guard_name, action, note, "
                "created_at FROM streaming_guard_log "
                "ORDER BY id DESC LIMIT ?"
            )
            cur = await db.execute(sql, (limit,))
            rows = await cur.fetchall() or []
            if not rows:
                await self._reply(update, "No streaming_guard_log rows.")
                return
            lines = ["🚦 *streaming_guard_log* (latest)"]
            for r in rows:
                emoji = {"halt": "🛑", "warn": "⚠️", "fix": "🩹"}.get(
                    r[3], "•"
                )
                lines.append(
                    f"{emoji} m{r[0] or '?'} t{r[1] or '?'} "
                    f"{r[2]}: {r[3]} _{r[5]}_"
                )
            await self._reply(
                update, "\n".join(lines), parse_mode="Markdown",
            )
            return

        await self._reply(
            update,
            "Unknown audit kind. Use one of: critic, regen, preview, "
            "paraflow, streaming.",
        )

    async def cmd_preflight(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z0 minimal slice — view / set mission preflight contract.

        Usage:
          /preflight <mission_id>                      — show preflight
          /preflight <mission_id> tier <name>          — set ambition_tier
          /preflight <mission_id> cost <usd>           — set cost ceiling
          /preflight <mission_id> attention <minutes>  — override attention budget

        ``tier`` ∈ {prototype, private_beta, public_launch, revenue_product}.
        Setting `tier` without `attention` applies the spec'd default for
        that tier (120/240/480/None).
        """
        args = context.args or []
        if not args:
            await self._reply(
                update,
                "Usage: `/preflight <mission_id> [tier|cost|attention] [value]`",
                parse_mode="Markdown",
            )
            return
        try:
            mission_id = int(args[0])
        except (ValueError, TypeError):
            await self._reply(update, "❌ mission_id must be an integer.")
            return

        if len(args) == 1:
            # Show current preflight.
            import os
            import json as _json
            from src.tools.workspace import get_mission_workspace
            from src.infra.db import get_db
            ws = get_mission_workspace(mission_id)
            path = os.path.join(ws, ".preflight", "mission_preflight.json")
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as fh:
                        body = fh.read()
                    await self._reply(
                        update,
                        f"Mission #{mission_id} preflight:\n```json\n{body}\n```",
                        parse_mode="Markdown",
                    )
                    return
                except Exception:
                    pass
            # Fall back to DB columns.
            db = await get_db()
            cur = await db.execute(
                "SELECT ambition_tier, cost_ceiling_usd, "
                "founder_attention_budget_minutes FROM missions WHERE id=?",
                (mission_id,),
            )
            row = await cur.fetchone()
            if not row:
                await self._reply(update, f"Mission #{mission_id} not found.")
                return
            await self._reply(
                update,
                (
                    f"Mission #{mission_id} preflight (no JSON yet)\n"
                    f"  ambition_tier:           {row[0] or '(unset)'}\n"
                    f"  cost_ceiling_usd:        {row[1] if row[1] is not None else '(unset)'}\n"
                    f"  attention_budget_min:    {row[2] if row[2] is not None else '(unbounded)'}"
                ),
            )
            return

        if len(args) < 3:
            await self._reply(
                update,
                "Usage: `/preflight <mission_id> <tier|cost|attention> <value>`",
                parse_mode="Markdown",
            )
            return

        key = args[1].lower()
        value = args[2]

        from mr_roboto.z0_preflight import z0_preflight_write, VALID_TIERS
        kwargs: dict = {"mission_id": mission_id}
        if key == "tier":
            if value not in VALID_TIERS:
                await self._reply(
                    update,
                    f"❌ tier must be one of: {', '.join(VALID_TIERS)}",
                )
                return
            kwargs["ambition_tier"] = value
        elif key == "cost":
            try:
                kwargs["cost_ceiling_usd"] = float(value)
            except ValueError:
                await self._reply(update, "❌ cost must be a number (USD).")
                return
        elif key == "attention":
            try:
                kwargs["attention_budget_minutes"] = int(value)
            except ValueError:
                await self._reply(update, "❌ attention must be an integer.")
                return
        else:
            await self._reply(
                update,
                "❌ key must be one of: tier, cost, attention.",
            )
            return

        try:
            res = await z0_preflight_write(**kwargs)
        except Exception as e:
            await self._reply(update, f"❌ preflight write failed: {e}")
            return
        if not res.get("ok"):
            await self._reply(update, f"❌ {res.get('error')}")
            return
        await self._reply(
            update,
            f"✅ Preflight updated for mission #{mission_id}\n"
            f"  {key}: {value}\n"
            f"  written: `{res.get('preflight_path')}`",
            parse_mode="Markdown",
        )

    async def cmd_signoff(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z1 T5A (P6) — record a founder signoff for a compliance doc.

        Usage:
          /signoff <mission_id> <doc_type>           — record signoff
          /signoff <mission_id> list                 — list signed doc_types

        ``doc_type`` matches the value from ``compliance_overlay.required_documents[].doc_type``
        (e.g. ``privacy_policy``, ``tos``, ``dpa``).
        """
        args = context.args or []
        if len(args) < 2:
            await self._reply(
                update,
                "Usage: `/signoff <mission_id> <doc_type>` or "
                "`/signoff <mission_id> list`",
                parse_mode="Markdown",
            )
            return
        try:
            mission_id = int(args[0])
        except (ValueError, TypeError):
            await self._reply(update, "❌ mission_id must be an integer.")
            return

        if args[1].lower() == "list":
            from src.infra.db import get_founder_signoffs
            signed = await get_founder_signoffs(mission_id)
            if not signed:
                await self._reply(
                    update,
                    f"Mission #{mission_id}: no signoffs recorded.",
                )
                return
            await self._reply(
                update,
                f"Mission #{mission_id} signoffs:\n"
                + "\n".join(f"  • {dt}" for dt in sorted(signed)),
            )
            return

        doc_type = args[1].strip()
        # Compute signature_hash from the rendered template body when we
        # can find one — best-effort, leave NULL otherwise. Lets a later
        # drift check detect that the founder signed v1 but v2 is now on disk.
        import hashlib
        import os
        from src.infra.db import record_founder_signoff
        from src.tools.workspace import get_mission_workspace

        sig = None
        try:
            ws = get_mission_workspace(mission_id)
            for cand in (
                os.path.join(ws, f"compliance/{doc_type}.md"),
                os.path.join(ws, f"compliance/{doc_type}.html"),
                os.path.join(ws, f"{doc_type}.md"),
            ):
                if os.path.isfile(cand):
                    with open(cand, "rb") as fh:
                        sig = hashlib.sha256(fh.read()).hexdigest()[:16]
                    break
        except Exception:
            pass

        try:
            await record_founder_signoff(
                mission_id=mission_id, doc_type=doc_type, signature_hash=sig
            )
        except Exception as e:
            await self._reply(update, f"❌ Signoff failed: {e}")
            return
        await self._reply(
            update,
            f"✅ Signoff recorded: mission #{mission_id}, doc `{doc_type}`"
            + (f" (sig {sig})" if sig else ""),
            parse_mode="Markdown",
        )

    async def cmd_github(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z1 Tier 6 (C18) — per-mission GitHub repo surface.

        Usage:
          /github                                    — show repo URL for
                                                       most-recent active mission
          /github init <mission_id>                  — manually re-run init
                                                       (for missions that hit
                                                       fail-soft)
          /github visibility <mission_id> <public|private>
                                                     — flip repo visibility
        """
        from src.infra.db import get_db
        args = context.args or []
        db = await get_db()

        # Subcommand: init <mission_id>
        if args and args[0] == "init":
            if len(args) < 2:
                await self._reply(update, "Usage: /github init <mission_id>")
                return
            try:
                mid = int(args[1])
            except (ValueError, TypeError):
                await self._reply(update, "Mission id must be an integer.")
                return
            try:
                from mr_roboto.init_mission_github_repo import (
                    init_mission_github_repo,
                )
                res = await init_mission_github_repo(mission_id=mid)
            except Exception as e:
                await self._reply(update, f"GitHub init failed: {e}")
                return
            if res.get("pending"):
                await self._reply(
                    update,
                    f"GitHub init pending for mission #{mid}\n"
                    f"reason: {res.get('reason')}\n"
                    f"status: `{res.get('status_path')}`",
                    parse_mode="Markdown",
                )
                return
            await self._reply(
                update,
                f"✅ GitHub repo ready for mission #{mid}\n"
                f"{res.get('repo_url')}\n"
                f"commit: {res.get('commit_sha') or '(unknown)'}\n"
                f"files: {len(res.get('files') or [])}",
            )
            return

        # Subcommand: visibility <mission_id> <public|private> [--confirm]
        # Z1 T6C — visibility flips are GitHub-side irreversible in
        # practice (public→private leaves clones cached on github.com
        # for ~7 days, doesn't unwind forks; private→public can leak
        # mission notes). Default flow now requires confirmation via
        # inline buttons; pass --confirm at the end to skip the prompt.
        if args and args[0] == "visibility":
            if len(args) < 3:
                await self._reply(
                    update,
                    "Usage: /github visibility <mission_id> <public|private>",
                )
                return
            try:
                mid = int(args[1])
            except (ValueError, TypeError):
                await self._reply(update, "Mission id must be an integer.")
                return
            vis = args[2].lower()
            if vis not in ("public", "private"):
                await self._reply(update, "Visibility must be public or private.")
                return
            skip_confirm = "--confirm" in args

            if not skip_confirm:
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                markup = InlineKeyboardMarkup([[
                    InlineKeyboardButton(
                        f"✅ Confirm → {vis}",
                        callback_data=f"gh_vis:y:{mid}:{vis}",
                    ),
                    InlineKeyboardButton(
                        "❌ Cancel", callback_data="gh_vis:n",
                    ),
                ]])
                warning = (
                    "Public → Private does NOT unwind forks or "
                    "remove cached clones."
                    if vis == "private"
                    else "Private → Public reveals all committed "
                    "mission files (charter, prd, screen plans, "
                    "interview transcripts)."
                )
                await update.message.reply_text(
                    f"⚠️ Confirm visibility flip\n\n"
                    f"Mission #{mid} → *{vis}*\n\n_{warning}_",
                    parse_mode="Markdown",
                    reply_markup=markup,
                )
                return

            try:
                from mr_roboto.init_mission_github_repo import (
                    set_repo_visibility,
                )
                res = await set_repo_visibility(mission_id=mid, visibility=vis)
            except Exception as e:
                await self._reply(update, f"GitHub visibility flip failed: {e}")
                return
            if not res.get("ok"):
                await self._reply(
                    update,
                    f"GitHub visibility flip failed: {res.get('error')}",
                )
                return
            await self._reply(
                update,
                f"✅ mission #{mid} visibility set to {vis}\n"
                f"{res.get('repo_url')}",
            )
            return

        # Default: show URL for most-recent active mission.
        cur = await db.execute(
            "SELECT id, github_repo_url FROM missions "
            "WHERE status IN ('active', 'pending') "
            "ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
        if not row:
            cur = await db.execute(
                "SELECT id, github_repo_url FROM missions "
                "ORDER BY id DESC LIMIT 1"
            )
            row = await cur.fetchone()
        if not row:
            await self._reply(update, "No missions found.")
            return
        mid, repo_url = int(row[0]), row[1]
        if repo_url:
            await self._reply(
                update,
                f"Mission #{mid}\n{repo_url}",
            )
        else:
            await self._reply(
                update,
                f"Mission #{mid}: GitHub repo not initialised yet.\n"
                f"Use `/github init {mid}` to create one.",
                parse_mode="Markdown",
            )

    async def cmd_paraflow_check(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Z1 Tier 7B (C21) — bundle-quality regression vs Paraflow goldens.

        Usage:
          /paraflow_check <mission_id> [archetype]

        Default archetype is ``truthrate``. Replies with verdict, score,
        and a compact gap summary.
        """
        args = context.args or []
        if not args:
            await self._reply(
                update,
                "Usage: /paraflow_check <mission_id> [archetype]\n"
                "Default archetype: truthrate",
            )
            return
        try:
            mid = int(args[0])
        except (ValueError, TypeError):
            await self._reply(update, "Mission id must be an integer.")
            return
        archetype = args[1] if len(args) > 1 else "truthrate"
        try:
            from mr_roboto.verify_against_paraflow_goldens import (
                verify_against_paraflow_goldens,
            )
            res = await verify_against_paraflow_goldens(
                mission_id=mid, archetype=archetype
            )
        except Exception as e:
            await self._reply(update, f"paraflow_check failed: {e}")
            return
        if res.get("error"):
            await self._reply(
                update,
                f"paraflow_check error for mission #{mid}: "
                f"{res.get('error')}",
            )
            return
        verdict = res.get("verdict") or "unknown"
        score = res.get("score")
        gaps = res.get("gaps") or []
        emoji = {
            "paraflow_par": "✅",
            "paraflow_partial": "⚠️",
            "paraflow_gap": "❌",
        }.get(verdict, "❓")
        gaps_str = "\n".join(f"  - {g}" for g in gaps[:15]) or "  (none)"
        if len(gaps) > 15:
            gaps_str += f"\n  ... +{len(gaps) - 15} more"
        await self._reply(
            update,
            f"{emoji} Paraflow check — mission #{mid}\n"
            f"archetype: {archetype}\n"
            f"verdict: {verdict}  (score {score})\n"
            f"gaps:\n{gaps_str}",
        )

    async def cmd_dlq(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Dead-letter queue management: /dlq [retry <task_id> | discard <task_id>]."""
        from ..infra.dead_letter import (
            get_dlq_summary, get_dlq_tasks, retry_dlq_task, resolve_dlq_task,
        )

        args = context.args or []

        if args and args[0] == "unpause":
            try:
                from general_beckman import paused_patterns as _pp
                cleared = list(_pp.all_paused())
                for p in cleared:
                    _pp.unpause(p)
                if cleared:
                    await self._reply(update, f"Unpaused {len(cleared)} patterns:\n" +
                                      "\n".join(f"- {p}" for p in cleared))
                else:
                    await self._reply(update, "No patterns currently paused.")
            except Exception as e:
                await self._reply(update, f"Error: {e}")
            return

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

    async def cmd_pause_mission(self, update, context):
        if not context.args:
            await self._reply(update, "Usage: /pause_mission <id>")
            return
        try:
            mid = int(context.args[0])
        except ValueError:
            await self._reply(update, "Invalid mission id.")
            return
        from general_beckman.lifecycle_events import emit_pause
        changed = await emit_pause(mid, reason="founder_pause", triggered_by="founder")
        if changed:
            await self._reply(update, f"Mission {mid} paused. In-flight tasks will finish.")
        else:
            await self._reply(update, f"Mission {mid}: not in active state.")

    async def cmd_resume_mission(self, update, context):
        if not context.args:
            await self._reply(update, "Usage: /resume_mission <id>")
            return
        try:
            mid = int(context.args[0])
        except ValueError:
            await self._reply(update, "Invalid mission id.")
            return
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute("SELECT lifecycle_state FROM missions WHERE id = ?", (mid,))
        row = await cur.fetchone()
        if not row:
            await self._reply(update, f"Mission {mid}: not found.")
            return
        state = row[0]
        if state in ("killed", "completed"):
            await self._reply(update, f"Mission {mid} is {state}; cannot resume.")
            return
        from general_beckman.lifecycle_events import emit_resume
        changed = await emit_resume(mid, triggered_by="founder")
        if changed:
            await self._reply(update, f"Mission {mid} resumed.")
        else:
            await self._reply(update, f"Mission {mid} not paused.")

    async def cmd_kill_mission(self, update, context):
        if not context.args:
            await self._reply(update, "Usage: /kill_mission <id>")
            return
        try:
            mid = int(context.args[0])
        except ValueError:
            await self._reply(update, "Invalid mission id.")
            return
        from general_beckman.lifecycle_events import emit_kill
        changed = await emit_kill(mid, reason="founder_kill", triggered_by="founder")
        if not changed:
            await self._reply(update, f"Mission {mid}: cannot kill (already terminal or missing).")
            return
        await self._snapshot_mission(mid)
        await self._reply(update, f"Mission {mid} killed. Snapshot written.")

    async def _snapshot_mission(self, mission_id: int):
        """Write mission state to artifact store as `mission_kill_<id>`."""
        import json as _json
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute("SELECT * FROM missions WHERE id = ?", (mission_id,))
        cols = [c[0] for c in cur.description]
        mrow = await cur.fetchone()
        mission = dict(zip(cols, mrow)) if mrow else {}
        cur = await db.execute(
            "SELECT id, title, status, completed_at FROM tasks WHERE mission_id = ?",
            (mission_id,),
        )
        tcols = [c[0] for c in cur.description]
        tasks = [dict(zip(tcols, r)) for r in await cur.fetchall()]
        snapshot = {"mission": mission, "tasks": tasks}
        try:
            from src.workflows.engine.hooks import get_artifact_store
            store = get_artifact_store()
            await store.store(mission_id, f"mission_kill_{mission_id}", _json.dumps(snapshot))
        except Exception as e:
            logger.error("snapshot write failed for mission %d: %s", mission_id, e)

    async def cmd_load(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/load full|heavy|shared|minimal|auto — set GPU load mode"""
        args = context.args or []
        if not args:
            from src.infra.load_manager import get_load_mode, is_auto_managed_async
            current = await get_load_mode()
            auto_str = " (auto-managed)" if await is_auto_managed_async() else " (manual)"
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
                expired_cmd = pending_action.get("command", "")
                logger.info("Pending action expired", command=expired_cmd)
                # Surface the timeout to the user so their message doesn't
                # silently fall through to the generic classifier (which
                # has no way to recover the conversational context).
                if not expired_cmd.startswith("_"):
                    try:
                        await self._reply(update,
                            f"⏱ Önceki istek (`/{expired_cmd}`) zaman aşımına "
                            "uğradı. Mesajın genel görev olarak işleniyor — "
                            "akışı yenilemek için komutu tekrar başlat.",
                            parse_mode="Markdown",
                        )
                    except Exception:
                        pass
                # Fall through to normal routing
            else:
                # ── Z0: Cost ceiling Q ──
                if pending_action.get("kind") == "z0_ceiling":
                    raw = (update.message.text or "").strip().lower()
                    if raw in ("none", "skip", ""):
                        ceiling = None
                    else:
                        try:
                            ceiling = float(raw)
                        except ValueError:
                            await self._reply(update, "Invalid number. Skipping (no ceiling).")
                            ceiling = None
                    if ceiling is not None:
                        from src.infra.db import get_db
                        _db = await get_db()
                        await _db.execute(
                            "UPDATE missions SET cost_ceiling_usd = ? WHERE id = ?",
                            (ceiling, pending_action["mission_id"]),
                        )
                        await _db.commit()
                    self._pending_action.pop(chat_id, None)
                    await self._reply(update, "Mission starting…")
                    return

                # ── Artifact-inline-edit: next message is the rewritten
                # markdown for the attached file. Overwrite + regenerate.
                if pending_action.get("kind") == "artifact_edit_inline":
                    if (text or "").strip().lower() in ("/cancel", "cancel"):
                        await self._reply(update, "Edit cancelled.")
                        return
                    import os as _os, os.path as _osp
                    from src.tools.workspace import WORKSPACE_DIR
                    files = pending_action.get("files") or []
                    mission_id = int(pending_action.get("mission_id") or 0)
                    task_id = int(pending_action.get("task_id") or 0)
                    regen_step = str(pending_action.get("regenerate_step_id") or "")
                    if not files:
                        await self._reply(update, "❌ No attached file to overwrite.")
                        return
                    rel = files[0]
                    abs_p = rel if _osp.isabs(rel) else _osp.join(WORKSPACE_DIR, rel)
                    try:
                        _os.makedirs(_osp.dirname(abs_p), exist_ok=True)
                        with open(abs_p, "w", encoding="utf-8") as fh:
                            fh.write(text)
                    except OSError as e:
                        await self._reply(update, f"❌ Write failed: {e}")
                        return
                    # Edit = founder's text IS the final artifact. Mark
                    # confirm task completed; DO NOT reset the writer step
                    # (that would overwrite the founder's edits).
                    from src.infra.db import update_task as _update_task
                    try:
                        await _update_task(task_id, status="completed", result='{"confirmed": true, "edited_inline": true}')
                    except Exception as e:
                        logger.exception(f"artifact_edit_inline complete failed: {e}")
                        await self._reply(update, f"❌ Complete failed: {e}")
                        return
                    await self._reply(update,
                        f"✅ Overwrote `{rel}` and accepted as final. Mission advancing.",
                        parse_mode="Markdown",
                    )
                    return

                cmd = pending_action.get("command") or ""
                if cmd == "_todo_help":
                    self._last_todo_help = pending_action
                    await self.cmd__todo_help(update, context)
                    return

                # ── Z1 Tier 4A: regen flow (C11+A15 / C19) ──
                if cmd == "regen":
                    stage = pending_action.get("stage")
                    if stage == "ask_change":
                        # Founder is replying with the change description for
                        # a previously-selected artifact.
                        mid = pending_action.get("mission_id")
                        ap = pending_action.get("artifact_path")
                        change = (text or "").strip()
                        if not change:
                            await self._reply(update, "❌ Empty change description; aborted.")
                            return
                        await self._enqueue_regen_artifact(update, int(mid), ap, change)
                        return
                    if stage == "ask_target":
                        # Founder is providing the full target line; re-route
                        # via cmd_regen with the parsed args.
                        parts = (text or "").strip().split()
                        context.args = parts
                        await self.cmd_regen(update, context)
                        return
                    # Unknown stage — clear and fall through.
                    logger.warning("regen pending: unknown stage", stage=stage)

                # ── Z1 Tier 4B: propagate flow (B2 / inline button) ──
                if cmd == "propagate":
                    stage = pending_action.get("stage")
                    if stage == "ask_change":
                        mid = pending_action.get("mission_id")
                        asset = pending_action.get("asset_path")
                        change = (text or "").strip()
                        if not change:
                            await self._reply(update, "❌ Empty change description; aborted.")
                            return
                        # Reuse cmd_propagate by injecting args.
                        context.args = [asset, *change.split()]
                        await self.cmd_propagate(update, context)
                        return
                    logger.warning("propagate pending: unknown stage", stage=stage)

                # ── B7+C16: visual ingest purpose-tap → enqueue mechanical task ──
                if cmd == "_visual_purpose":
                    stripped = text.strip()
                    if stripped == "❌ İptal":
                        # Restore main keyboard, leave the saved file in place
                        # (founder can re-trigger by uploading again).
                        await self._reply(update,
                            "❌ İptal edildi. Görsel kaydedildi ama "
                            "işlenmedi.",
                            reply_markup=REPLY_KEYBOARD)
                        return
                    purpose = self._VISUAL_PURPOSE_LABELS.get(stripped)
                    if purpose is None:
                        # Unknown tap — re-prompt with the same keyboard.
                        self._pending_action[chat_id] = {
                            **pending_action,
                            "ts": _time.time(),
                        }
                        labels = list(self._VISUAL_PURPOSE_LABELS.keys()) + ["❌ İptal"]
                        kb = _make_keyboard([labels[:3], labels[3:]])
                        await self._reply(update,
                            "Lütfen butonlardan birini seç.",
                            reply_markup=kb)
                        return
                    mission_id = pending_action.get("mission_id")
                    file_path = pending_action.get("file_path")
                    if not mission_id or not file_path:
                        await self._reply(update,
                            "❌ Görsel bağlamı kayboldu. Tekrar yükle.",
                            reply_markup=REPLY_KEYBOARD)
                        return
                    try:
                        import general_beckman
                        await general_beckman.enqueue({
                            "title": f"ingest_visual:{purpose}",
                            "description": (
                                f"Founder uploaded a {purpose} for mission "
                                f"{mission_id}; extract structural elements "
                                f"into visual_brief.md."
                            ),
                            "agent_type": "mechanical",
                            "kind": "main_work",
                            "priority": 5,
                            "mission_id": int(mission_id),
                            "context": {
                                "executor": "mechanical",
                                "payload": {
                                    "action": "ingest_visual",
                                    "mission_id": int(mission_id),
                                    "file_paths": [file_path],
                                    "purpose": purpose,
                                },
                            },
                        })
                    except Exception as exc:
                        logger.error("ingest_visual enqueue failed",
                                     mission_id=mission_id, error=str(exc))
                        await self._reply(update,
                            f"❌ Görev kuyruğa eklenemedi: {_friendly_error(str(exc))}",
                            reply_markup=REPLY_KEYBOARD)
                        return
                    await self._reply(update,
                        f"✅ Görsel `{purpose}` olarak kuyruğa eklendi. "
                        f"`visual_brief.md` üretildiğinde göreceksin.",
                        parse_mode="Markdown",
                        reply_markup=REPLY_KEYBOARD)
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
                        next_run_str = to_db(reminder_time)
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
                        time_str = to_turkey(reminder_time).strftime("%d.%m.%Y %H:%M")
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
            task_context = {"chat_id": chat_id}
            if recent_context:
                task_context["recent_conversation"] = recent_context

            task_id = await add_task(
                title=text[:50],
                description=text,
                tier="auto",
                parent_task_id=parent_id,
                priority=TASK_PRIORITY["critical"],
                context=task_context,
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
            # Build context hint
            context_parts = []
            if self._pending_clarifications:
                context_parts.append("System has pending clarification requests")

            messages = [{
                "role": "user",
                "content": self.MESSAGE_CLASSIFIER_PROMPT.format(
                    message=text[:300],
                    context="; ".join(context_parts) if context_parts else "none",
                ),
            }]
            response = await _enqueue_inline_chat(
                title="telegram-classifier",
                description=f"Classify telegram message: {text[:80]!r}",
                agent_type="classifier",
                kind="classifier",
                llm_call_kwargs={
                    "task": "router",
                    "agent_type": "classifier",
                    "difficulty": 2,
                    "messages": messages,
                    "prefer_speed": True,
                    "needs_json_mode": True,
                    "priority": 2,
                    "estimated_input_tokens": 300,
                    "estimated_output_tokens": 50,
                    "call_category": "overhead",
                },
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

    async def _resume_needs_review_tasks(
        self, mission_id: int, posthook_kind: str,
        note: str = "founder ack", cancel: bool = False,
    ) -> int:
        """Resolve any waiting needs_review tasks for a given posthook kind.

        Z1 T6A — when the founder clicks Continue/Branch/Abort on a
        similar-missions notification, the find_similar_missions source
        task is sitting in needs_review. This helper either completes
        (Continue/Branch) or cancels (Abort) all such tasks for the
        mission.

        Returns the count of tasks transitioned.
        """
        import json as _json
        from src.infra.db import get_db, update_task

        try:
            db = await get_db()
            cur = await db.execute(
                "SELECT id, context FROM tasks "
                "WHERE mission_id = ? AND status = 'waiting_human'",
                (int(mission_id),),
            )
            rows = await cur.fetchall() or []
        except Exception as exc:
            logger.warning(
                "resume_needs_review query failed",
                mission_id=mission_id, error=str(exc),
            )
            return 0

        count = 0
        target_status = "cancelled" if cancel else "completed"
        for row in rows:
            tid = row[0]
            raw_ctx = row[1] or "{}"
            try:
                ctx = _json.loads(raw_ctx) if isinstance(raw_ctx, str) else {}
            except (ValueError, TypeError):
                ctx = {}
            payload = (ctx.get("payload") or {})
            if payload.get("action") != posthook_kind and ctx.get(
                "posthook_kind"
            ) != posthook_kind:
                continue
            ctx["founder_review_note"] = note
            try:
                await update_task(
                    tid, status=target_status,
                    context=_json.dumps(ctx),
                )
                count += 1
            except Exception as exc:
                logger.warning(
                    "resume_needs_review update failed",
                    task_id=tid, error=str(exc),
                )
        return count

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
            question = ctx.get("_clarification_question", "")
            clarifications = ctx.get("clarification_history", [])
            clarifications.append({"question": question, "answer": answer})
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

            # Z1 T5A — auto-fire attention_debit. Minutes elapsed = answer
            # timestamp minus the question's sent timestamp (replied_to). Floor
            # at 1 minute (any answer costs at least one). Best-effort:
            # failures must not block resume.
            try:
                mission_id = task_info.get("mission_id")
                if mission_id:
                    step_id = ctx.get("step_id") or ctx.get("_step_id") or "clarify"
                    minutes = 1
                    try:
                        replied_to = (
                            update.message.reply_to_message
                            if update and update.message else None
                        )
                        if replied_to and replied_to.date and update.message.date:
                            delta = update.message.date - replied_to.date
                            minutes = max(1, int(delta.total_seconds() // 60))
                    except Exception:
                        pass
                    import general_beckman as _beckman
                    import json as _json_dbg
                    await _beckman.enqueue({
                        "title": f"attention_debit:m{mission_id}:t{task_id}",
                        "agent_type": "mechanical",
                        "mission_id": mission_id,
                        "context": _json_dbg.dumps({
                            "executor": "mechanical",
                            "payload": {
                                "action": "attention_debit",
                                "mission_id": mission_id,
                                "step_id": str(step_id),
                                "debit_action": "clarify_reply",
                                "minutes_debited": minutes,
                            },
                        }),
                    })
            except Exception as _debit_exc:
                logger.warning("attention_debit auto-fire failed",
                               task_id=task_id, error=str(_debit_exc))

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
            response = await _enqueue_inline_chat(
                title="telegram-casual-chat",
                description=f"Casual chat reply: {text[:80]!r}",
                agent_type="assistant",
                kind="chat",
                llm_call_kwargs={
                    "task": "assistant",
                    "agent_type": "assistant",
                    "difficulty": 2,
                    "messages": [{"role": "user", "content": text}],
                    "prefer_speed": True,
                    "priority": 1,
                    "estimated_input_tokens": 100,
                    "estimated_output_tokens": 100,
                    "call_category": "overhead",
                },
            )
            reply = response.get("content", "Hey! How can I help?")
            await self._reply(update, reply[:1000])
        except Exception:
            await self._reply(update, "Hey! Send me a task or mission to work on.")

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
        """Create a shopping mission via the workflow runner."""
        from src.workflows.engine.runner import WorkflowRunner

        # Map sub_intents to workflow names
        wf_map = {
            "deep_research": "shopping_v2",
            "research": "shopping_v2",
            "compare": "combo_research",
            "gift": "gift_recommendation",
            "deals": "exploration",
            "quick_search": "quick_search_v2",
            "product_research": "product_research_v2",
        }
        workflow_name = wf_map.get(sub_intent or "shopping", "shopping_v2")

        runner = WorkflowRunner()
        mission_id = await runner.start(
            workflow_name=workflow_name,
            initial_input={
                "user_query": query,
                "chat_id": chat_id,
            },
            title=f"Shopping: {query[:60]}",
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
        else:
            # Simple query → quick_search workflow
            mission_id = await self._create_shopping_mission(
                query, chat_id, sub_intent="quick_search"
            )
            await self._reply(
                update,
                f"🛒 Searching for *{query}*... (mission #{mission_id})",
                parse_mode="Markdown",
            )

    async def cmd_research_product(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Deep research. /research_product <query>

        First message (no args) shows an intent fork: specific product vs
        category. The callback stores the sub_intent; the follow-up query
        message re-enters this handler with args and routes accordingly.
        """
        if not context.args:
            chat_id = update.effective_chat.id
            buttons = [[
                InlineKeyboardButton("🎯 Belirli ürün", callback_data="shop:specific"),
                InlineKeyboardButton("🏷 Kategori", callback_data="shop:category"),
            ]]
            await update.message.reply_text(
                "🔬 Ne araştıralım?",
                reply_markup=InlineKeyboardMarkup(buttons),
            )
            return
        product = " ".join(context.args)
        chat_id = update.effective_chat.id

        sub = self._pending_shop_subintent.pop(chat_id, None)
        if sub == "category":
            mission_id = await self._create_shopping_mission(
                product, chat_id, sub_intent="deep_research"
            )
            await self._reply(
                update,
                f"🏷 Kategori araştırması başladı: *{product}* (mission #{mission_id})",
                parse_mode="Markdown",
            )
            return

        # Default (including sub == "specific"): route to product_research
        mission_id = await self._create_shopping_mission(
            product, chat_id, sub_intent="product_research"
        )
        await self._reply(
            update,
            f"🔬 Ürün araştırması başladı: *{product}* (mission #{mission_id})",
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

        # ── Z10 T2B: reply-to a mission_event → resolution='comment' ──
        try:
            from .mission_events import get_event_by_message_id, resolve_event
            event = await get_event_by_message_id(replied_to.message_id)
            if event:
                await resolve_event(event["id"], "comment")
                logger.info(
                    "mission_event comment recorded",
                    event_id=event["id"],
                    mission_id=event.get("mission_id"),
                    text_preview=(answer or "")[:80],
                )
                # Skeleton: revision-task plumbing for artifact-linked events
                # is deferred; just persist + log + ack the user so they know
                # the comment landed.
                await self._reply(
                    update,
                    f"💬 Comment recorded on mission "
                    f"#{event.get('mission_id')} event.",
                )
                return
        except Exception as e:  # noqa: BLE001
            logger.debug("mission_event reply lookup skipped", error=str(e))

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

        # ── Mission Lifecycle Button Callbacks ───────────────────
        if data.startswith("mission_resume:"):
            mid = int(data.split(":", 1)[1])
            from general_beckman.lifecycle_events import emit_resume
            await emit_resume(mid, triggered_by="founder")
            await query.answer("Resumed.")
            await query.edit_message_text(f"Mission {mid} resumed.")
            return
        if data.startswith("mission_pause:"):
            mid = int(data.split(":", 1)[1])
            from general_beckman.lifecycle_events import emit_pause
            await emit_pause(mid, reason="founder_pause", triggered_by="founder")
            await query.answer("Paused.")
            await query.edit_message_text(f"Mission {mid} paused.")
            return
        if data.startswith("mission_kill:"):
            mid = int(data.split(":", 1)[1])
            from general_beckman.lifecycle_events import emit_kill
            await emit_kill(mid, triggered_by="founder")
            await self._snapshot_mission(mid)
            await query.answer("Killed.")
            await query.edit_message_text(f"Mission {mid} killed.")
            return

        if data.startswith("yal:"):
            await self.handle_yalayut_callback(update, context)
            return

        # ── Z1 T6C: github visibility flip confirmation ────────────
        if data.startswith("gh_vis:"):
            parts = data.split(":")
            if parts[1] == "n":
                await query.message.reply_text("❌ Visibility flip cancelled.")
                return
            if parts[1] == "y" and len(parts) >= 4:
                try:
                    mid = int(parts[2])
                except (ValueError, IndexError):
                    await query.message.reply_text("❌ Bad mission id.")
                    return
                vis = parts[3]
                if vis not in ("public", "private"):
                    await query.message.reply_text("❌ Bad visibility.")
                    return
                try:
                    from mr_roboto.init_mission_github_repo import (
                        set_repo_visibility,
                    )
                    res = await set_repo_visibility(
                        mission_id=mid, visibility=vis
                    )
                except Exception as e:
                    await query.message.reply_text(
                        f"❌ Flip failed: {e}"
                    )
                    return
                if not res.get("ok"):
                    await query.message.reply_text(
                        f"❌ Flip failed: {res.get('error')}"
                    )
                    return
                await query.message.reply_text(
                    f"✅ mission #{mid} → {vis}\n{res.get('repo_url')}"
                )
                return
            await query.message.reply_text("❌ Bozuk gh_vis butonu.")
            return

        # ── Z1 Tier 6A: similar-missions review inline buttons ─────
        # Format:
        #   simrev:c:<mid>           → Continue (index + clear needs_review)
        #   simrev:b:<from>:<mid>    → Branch from prior #from
        #   simrev:a:<mid>           → Abort current mission
        if data.startswith("simrev:"):
            try:
                parts = data.split(":")
                kind = parts[1]
                mid = int(parts[-1])
            except (ValueError, IndexError):
                await query.message.reply_text("❌ Bozuk similar-review butonu.")
                return
            if kind == "c":
                # Continue: enqueue index_idea_fingerprint + resolve any
                # waiting find_similar_missions tasks on this mission.
                try:
                    import general_beckman as _beckman
                    import json as _json_dbg
                    await _beckman.enqueue({
                        "title": f"index_idea_fingerprint:m{mid}",
                        "agent_type": "mechanical",
                        "mission_id": mid,
                        "context": _json_dbg.dumps({
                            "executor": "mechanical",
                            "payload": {
                                "action": "index_idea_fingerprint",
                                "mission_id": mid,
                            },
                        }),
                    })
                    await self._resume_needs_review_tasks(
                        mid, "find_similar_missions", note="founder: continue",
                    )
                except Exception as e:
                    await query.message.reply_text(
                        f"❌ Continue failed: {e}"
                    )
                    return
                await query.message.reply_text(
                    f"▶️ Continuing mission #{mid}. Idea indexed."
                )
                return
            if kind == "b":
                try:
                    from_mid = int(parts[2])
                except (ValueError, IndexError):
                    await query.message.reply_text("❌ Bozuk branch butonu.")
                    return
                try:
                    db = await get_db()
                    await db.execute(
                        "UPDATE missions SET branched_from_mission_id = ? "
                        "WHERE id = ?",
                        (from_mid, mid),
                    )
                    await db.commit()
                    await self._resume_needs_review_tasks(
                        mid, "find_similar_missions",
                        note=f"founder: branch_from_{from_mid}",
                    )
                except Exception as e:
                    await query.message.reply_text(
                        f"❌ Branch failed: {e}"
                    )
                    return
                await query.message.reply_text(
                    f"🌿 Mission #{mid} branched from prior mission "
                    f"#{from_mid}. You can now `/clone_artifacts {from_mid}` "
                    f"or proceed."
                )
                return
            if kind == "a":
                try:
                    from src.infra.db import update_mission
                    await update_mission(mid, status="cancelled")
                    await self._resume_needs_review_tasks(
                        mid, "find_similar_missions", note="founder: abort",
                        cancel=True,
                    )
                except Exception as e:
                    await query.message.reply_text(
                        f"❌ Abort failed: {e}"
                    )
                    return
                await query.message.reply_text(
                    f"🛑 Mission #{mid} aborted."
                )
                return
            await query.message.reply_text("❌ Unknown similar-review action.")
            return

        # ── Z4 T4B/T4C: visual-review founder-loop inline buttons ───
        # Format: visrev:<mid>:<token> — token resolves against the
        # per-mission cbmap sidecar (mission_<mid>/.visual/.cbmap.json):
        #   {kind: "approve", step_id, frame} → copy captured frame to the
        #       per-mission baseline AND promote it to the cross-mission
        #       baseline store keyed by the mission's design-token hash.
        #   {kind: "cal", verdict, lesson_pattern} → upsert a mission lesson.
        if data.startswith("visrev:"):
            import json as _json
            try:
                _, _mid_s, _token = data.split(":", 2)
                mid = int(_mid_s)
            except (ValueError, IndexError):
                await query.message.reply_text("❌ Bozuk visrev butonu.")
                return

            from src.tools.workspace import WORKSPACE_DIR as _WS
            _cbmap_path = os.path.join(
                _WS, f"mission_{mid}", ".visual", ".cbmap.json"
            )
            entry = None
            try:
                with open(_cbmap_path, "r", encoding="utf-8") as _f:
                    entry = _json.load(_f).get(_token)
            except (OSError, ValueError):
                entry = None
            if not isinstance(entry, dict):
                await query.message.reply_text(
                    "❌ visrev butonu süresi doldu (cbmap entry missing)."
                )
                return

            visrev_kind = entry.get("kind")

            if visrev_kind == "approve":
                step_id_v = str(entry.get("step_id") or "")
                frame_filename = str(entry.get("frame") or "")
                try:
                    import shutil as _shutil
                    captured_path = os.path.join(
                        _WS, f"mission_{mid}", ".visual", "captured",
                        step_id_v, frame_filename,
                    )
                    baseline_dir = os.path.join(
                        _WS, f"mission_{mid}", ".visual", "baseline",
                    )
                    os.makedirs(baseline_dir, exist_ok=True)
                    dst = os.path.join(baseline_dir, frame_filename)
                    _shutil.copy2(captured_path, dst)
                except Exception as e:
                    await query.message.reply_text(
                        f"❌ Baseline approval failed: {e}"
                    )
                    return
                # Promote to the cross-mission baseline store so a stable
                # design system reuses this baseline in future missions.
                cross_msg = ""
                try:
                    _th_path = os.path.join(
                        _WS, f"mission_{mid}", ".visual", ".token_hash"
                    )
                    with open(_th_path, "r", encoding="utf-8") as _f:
                        _thash = _f.read().strip()
                    if _thash:
                        from mr_roboto.visual_baseline import (
                            cross_mission_baseline_dir,
                            promote_to_cross_mission,
                        )
                        # repo root = src/app/telegram_bot.py → climb 3
                        _repo_root = os.path.dirname(os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))
                        ))
                        _cross = cross_mission_baseline_dir(_repo_root, _thash)
                        promote_to_cross_mission(dst, _cross)
                        cross_msg = " + cross-mission store"
                except Exception as e:
                    logger.debug("visrev cross-mission promote skipped: %s", e)
                await self._reply(
                    update,
                    f"✅ Approved `{frame_filename}` as new baseline "
                    f"(mission #{mid}){cross_msg}.",
                    parse_mode="Markdown",
                )
                return

            if visrev_kind == "cal":
                cal_verdict = str(entry.get("verdict") or "")
                lesson_pattern = str(entry.get("lesson_pattern") or "")
                try:
                    from src.infra.mission_lessons import upsert_mission_lesson
                    fix_msg = (
                        "Founder marked this visual pattern as acceptable — suppress future alerts."
                        if cal_verdict == "fine"
                        else "Founder confirmed this visual pattern is genuinely broken."
                    )
                    severity = "info" if cal_verdict == "fine" else "blocker"
                    await upsert_mission_lesson(
                        stack="frontend",
                        domain="visual",
                        pattern=lesson_pattern,
                        fix=fix_msg,
                        severity=severity,
                        source_kind="visrev_calibration",
                        source_ref={"mission_id": mid, "verdict": cal_verdict},
                    )
                except Exception as e:
                    await query.message.reply_text(
                        f"❌ Calibration failed: {e}"
                    )
                    return
                label = "🟢 Fine" if cal_verdict == "fine" else "🔴 Broken"
                await self._reply(
                    update,
                    f"{label} — `{lesson_pattern}` recorded as *{cal_verdict}* "
                    f"for mission #{mid}.",
                    parse_mode="Markdown",
                )
                return

            await query.message.reply_text("❌ Unknown visrev action.")
            return

        # ── Z1 Tier 4B: propagate inline button ─────────────────────
        # Format: `propagate:<mission_id>:<artifact_path>`
        # On tap, stash a `_pending_action` so the next message becomes
        # the change description (mirrors the /propagate <path> flow).
        if data.startswith("propagate:"):
            try:
                _, mid_s, asset_path = data.split(":", 2)
                mid = int(mid_s)
            except (ValueError, IndexError):
                await query.message.reply_text("❌ Bozuk propagate butonu.")
                return
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {
                "command": "propagate",
                "stage": "ask_change",
                "mission_id": mid,
                "asset_path": asset_path,
                "ts": _time.time(),
            }
            await query.message.reply_text(
                f"🎯 Describe the change to propagate from `{asset_path}` "
                f"(mission #{mid}):",
                parse_mode="Markdown",
            )
            return

        # ── Z1 Tier 4A: regen inline button ─────────────────────────
        # Format: `regen:<mission_id>:<base64-or-encoded-artifact-path>`
        # On tap, stash a `_pending_action` so the next message becomes
        # the change description (mirrors the /regen <mid> <path> flow).
        if data.startswith("regen:"):
            try:
                _, mid_s, art_path = data.split(":", 2)
                mid = int(mid_s)
            except (ValueError, IndexError):
                await query.message.reply_text("❌ Bozuk regen butonu.")
                return
            chat_id = update.effective_chat.id
            self._pending_action[chat_id] = {
                "command": "regen",
                "stage": "ask_change",
                "mission_id": mid,
                "artifact_path": art_path,
                "ts": _time.time(),
            }
            await query.message.reply_text(
                f"📝 Describe the change for `{art_path}` (mission #{mid}):",
                parse_mode="Markdown",
            )
            return

        # ── Deep Research Intent Fork ─────────────────────────────
        if data.startswith("shop:"):
            sub = data.split(":", 1)[1]
            chat_id = update.effective_chat.id
            import time as _time
            self._pending_action[chat_id] = {
                "command": "research_product",
                "ts": _time.time(),
            }
            self._pending_shop_subintent[chat_id] = sub
            prompt = (
                "🎯 Hangi ürün? (marka + model yazın)"
                if sub == "specific" else
                "🏷 Hangi kategori? (örn. 'kahve makinesi 5000 TL altı')"
            )
            await query.message.reply_text(prompt)
            return

        # ── Z1 T4C Preview Share Button ───────────────────────────
        if data.startswith("preview:share:"):
            try:
                mission_id = int(data.split(":")[-1])
            except (ValueError, IndexError):
                await query.message.reply_text("Bad mission id in preview share.")
                return
            try:
                from mr_roboto.emit_preview_url import emit_preview_url
                res = await emit_preview_url(mission_id=mission_id)
            except Exception as e:
                await query.message.reply_text(f"Preview emit failed: {e}")
                return
            if res.get("pending"):
                await query.message.reply_text(
                    f"Preview pending for mission #{mission_id} — hosting "
                    f"deferred to Z2 (path: {res.get('path')})"
                )
            else:
                await query.message.reply_text(
                    f"📡 Preview ready for mission #{mission_id}\n"
                    f"{res.get('url')}"
                )
            return

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

        if data == "m:dlq:retry_all":
            try:
                from ..infra.dead_letter import get_dlq_tasks, retry_dlq_task
                entries = await get_dlq_tasks(unresolved_only=True)
                retried = 0
                failed: list[tuple[int, str]] = []
                for entry in entries:
                    tid = entry["task_id"]
                    try:
                        if await retry_dlq_task(tid):
                            retried += 1
                    except Exception as exc:
                        failed.append((tid, str(exc)[:60]))
                lines = [f"🔄 Tümü yeniden denendi: {retried}/{len(entries)} görev kuyruğa eklendi."]
                if failed:
                    lines.append(f"\n⚠️ {len(failed)} başarısız:")
                    for tid, err in failed[:5]:
                        lines.append(f"  #{tid}: {err}")
                await query.message.reply_text("\n".join(lines))
            except Exception as e:
                await query.message.reply_text(f"❌ {e}")
            return

        # ── DLQ Analyst actions ──
        if data.startswith("dlqa:"):
            parts = data.split(":", 2)
            if len(parts) < 3:
                await query.answer("Invalid action")
                return

            action = parts[1]
            payload = parts[2]

            if action == "retry":
                task_ids = [int(t) for t in payload.split(",") if t.isdigit()]
                from src.infra.dead_letter import retry_dlq_task
                retried = 0
                for tid in task_ids:
                    if await retry_dlq_task(tid):
                        retried += 1
                await query.answer(f"Retried {retried}/{len(task_ids)} tasks")
                await query.edit_message_text(
                    query.message.text + f"\n\nRetried {retried} tasks.",
                )

            elif action == "drop":
                task_ids = [int(t) for t in payload.split(",") if t.isdigit()]
                from src.infra.dead_letter import resolve_dlq_task
                dropped = 0
                for tid in task_ids:
                    if await resolve_dlq_task(tid, resolution="discarded"):
                        dropped += 1
                await query.answer(f"Dropped {dropped}/{len(task_ids)} tasks")
                await query.edit_message_text(
                    query.message.text + f"\n\nDropped {dropped} tasks.",
                )

            elif action == "pause":
                pattern_key = payload
                try:
                    from general_beckman import paused_patterns as _pp
                    _pp.pause(pattern_key)
                    await query.answer(f"Paused: {pattern_key}")
                    await query.edit_message_text(
                        query.message.text + f"\n\nPaused pattern: {pattern_key}. Use /dlq unpause to lift.",
                    )
                except Exception as e:
                    await query.answer(f"Pause failed: {e}")

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
            for _attempt in range(3):
                try:
                    await db.execute("DELETE FROM tasks")
                    await db.execute("DELETE FROM missions")
                    await db.execute("DELETE FROM dead_letter_tasks")
                    await db.execute("DELETE FROM workflow_checkpoints")
                    await db.execute("DELETE FROM blackboards")
                    await db.execute("DELETE FROM approval_requests")
                    await db.execute("DELETE FROM scheduled_tasks")
                    await db.commit()
                    break
                except Exception as _db_err:
                    if _attempt < 2:
                        await asyncio.sleep(1)
                    else:
                        raise
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

    async def send_notification(self, text: str, retries: int = 2, reply_markup=None):
        """Send a notification message. Returns the sent Message or None.

        When ``reply_markup`` is None, attach the persistent ``REPLY_KEYBOARD``.
        When the caller passes an explicit markup (e.g. ``InlineKeyboardMarkup``
        for artifact-emit buttons), use that instead — Telegram only allows one
        reply_markup per message and inline + reply markups are mutually
        exclusive. The persistent keyboard remains attached to prior messages.
        """
        import asyncio as _asyncio

        # Phase 8.3: Redact secrets from outgoing messages
        try:
            from ..security.sensitivity import redact_secrets
            text = redact_secrets(text)
        except Exception:
            pass

        markup = reply_markup if reply_markup is not None else REPLY_KEYBOARD

        for attempt in range(retries + 1):
            try:
                msg = await self.app.bot.send_message(
                    chat_id=TELEGRAM_ADMIN_CHAT_ID,
                    text=text,
                    parse_mode="Markdown",
                    reply_markup=markup,
                )
                return msg
            except Exception as e:
                # First fallback: retry without markdown
                try:
                    msg = await self.app.bot.send_message(
                        chat_id=TELEGRAM_ADMIN_CHAT_ID, text=text,
                        reply_markup=markup,
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
            from src.tools.workspace import WORKSPACE_DIR as _ws_dir
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


    async def _recover_question_from_child(self, db, source_task_id: int) -> str:
        """Pull the clarify question from a mechanical child task's payload.

        Used by the restore path when the source task's context has no
        saved clarification text. Older clarify executors persisted the
        question on the mechanical child rather than the source.
        """
        import json as _json
        try:
            cur = await db.execute(
                """SELECT context FROM tasks
                   WHERE parent_task_id = ?
                     AND agent_type = 'mechanical'
                   ORDER BY id DESC LIMIT 3""",
                (source_task_id,),
            )
            rows = await cur.fetchall()
        except Exception:
            return ""
        for row in rows or []:
            # aiosqlite.Row supports index access but NOT .get(). Use
            # positional indexing since the SELECT has a single column.
            try:
                raw_ctx = row[0]
            except Exception:
                raw_ctx = None
            if not raw_ctx:
                continue
            try:
                ctx = _json.loads(raw_ctx)
            except (ValueError, TypeError):
                continue
            payload = (ctx or {}).get("payload") or {}
            if payload.get("action") != "clarify":
                continue
            q = payload.get("question") or ""
            if isinstance(q, str) and q.strip():
                return q
        return ""

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

    async def send_variant_keyboard(
        self,
        chat_id: int,
        mission_id: int,
        task_id: int,
        base_label: str,
        options: list,
    ) -> None:
        """Send an inline-keyboard asking the user to pick a variant.

        callback_data encodes mission_id + task_id so taps survive bot restart
        (in-memory _pending_action is lost on restart and would otherwise leave
        buttons silently non-responsive).
        """
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        buttons = [
            [InlineKeyboardButton(
                opt["label"],
                callback_data=f"vc:{mission_id}:{task_id}:{opt['group_id']}",
            )]
            for opt in options
        ]
        buttons.append([InlineKeyboardButton(
            "📊 Hepsini karşılaştır",
            callback_data=f"vc:{mission_id}:{task_id}:all",
        )])
        markup = InlineKeyboardMarkup(buttons)
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=f"*{base_label}* için hangi model?",
            reply_markup=markup,
            parse_mode="Markdown",
        )
        # In-memory cache (fast path / contains options for compare-all UX);
        # callback handler also reconstructs from DB when missing.
        self._pending_action[chat_id] = {
            "kind": "variant_choice",
            "mission_id": mission_id,
            "task_id": task_id,
            "options": options,
            "base_label": base_label,
        }

    async def send_artifact_confirm_keyboard(
        self,
        *,
        chat_id: int,
        mission_id: int,
        task_id: int,
        kind: str,
        question: str,
        files: list,
        regenerate_step_id: str = "",
    ) -> None:
        """Inline artifact preview + [OK / Regenerate / Edit] keyboard.

        Founder never has to leave Telegram. callback_data shape:
            rpc:<verb>:<mission_id>:<task_id>
        with verb in {OK, RE, ED}. step_id stays in _pending_action so
        Regenerate can reset the right step.
        """
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        TELEGRAM_LIMIT = 3900
        header = f"❓ *{kind.replace('_',' ').title()}* — Task #{task_id}\n\n{question}\n"
        body_blocks: list[str] = []
        for rel, content in (files or []):
            content = content or ""
            label = f"\n📄 `{rel}`\n"
            block = f"{label}```markdown\n{content}\n```\n"
            body_blocks.append(block)
        full = header + "".join(body_blocks)
        # Truncate if over Telegram limit — keep header + truncated marker.
        if len(full) > TELEGRAM_LIMIT:
            avail = TELEGRAM_LIMIT - len(header) - 100
            joined = "".join(body_blocks)
            full = header + joined[:avail] + "\n…(truncated — open file to read full)\n"
        buttons = [[
            InlineKeyboardButton("✅ OK", callback_data=f"rpc:OK:{mission_id}:{task_id}"),
            InlineKeyboardButton("♻️ Regenerate", callback_data=f"rpc:RE:{mission_id}:{task_id}"),
            InlineKeyboardButton("✏️ Edit", callback_data=f"rpc:ED:{mission_id}:{task_id}"),
        ]]
        try:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=full,
                reply_markup=InlineKeyboardMarkup(buttons),
                parse_mode="Markdown",
            )
        except Exception as exc:
            # Markdown parse can fail on backticks inside content; retry plain.
            logger.warning(f"send_artifact_confirm_keyboard markdown send failed: {exc}; retrying plain")
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=full,
                reply_markup=InlineKeyboardMarkup(buttons),
            )
        self._pending_action[chat_id] = {
            "kind": "artifact_confirm",
            "confirm_kind": kind,
            "mission_id": mission_id,
            "task_id": task_id,
            "regenerate_step_id": regenerate_step_id,
            "files": [rel for rel, _ in (files or [])],
        }

    async def _handle_artifact_confirm_callback(self, update, context):
        """Dispatch rpc:OK / rpc:RE / rpc:ED for artifact-confirm flow."""
        query = update.callback_query
        try:
            await query.answer()
        except Exception:
            pass
        data = query.data or ""
        parts = data.split(":")
        if len(parts) < 4 or parts[0] != "rpc":
            return
        verb = parts[1]
        try:
            mission_id = int(parts[2])
            task_id = int(parts[3])
        except ValueError:
            return
        chat_id = query.message.chat_id

        pending = self._pending_action.get(chat_id) or {}
        regen_step = pending.get("regenerate_step_id") or ""
        files = pending.get("files") or []

        from src.infra.db import get_db, update_task
        db = await get_db()

        if verb == "OK":
            await update_task(task_id, status="completed", result='{"confirmed": true}')
            await query.edit_message_reply_markup(reply_markup=None)
            await self.app.bot.send_message(chat_id=chat_id, text=f"✅ Confirmed task #{task_id}. Mission advancing.")
            self._pending_action.pop(chat_id, None)
            return

        if verb == "RE":
            # Reset the originating writer step (if known) AND this confirm
            # step so the workflow re-runs draft + verify + confirm.
            try:
                if regen_step:
                    await db.execute(
                        "UPDATE tasks SET status='pending', worker_attempts=0, error=NULL, "
                        "error_category=NULL, started_at=NULL, completed_at=NULL "
                        "WHERE mission_id=? AND json_extract(context,'$.workflow_step_id')=?",
                        (mission_id, regen_step),
                    )
                # Reset the verify sibling too if present (`<regen>.verify`).
                if regen_step:
                    await db.execute(
                        "UPDATE tasks SET status='pending', worker_attempts=0, error=NULL, "
                        "error_category=NULL, started_at=NULL, completed_at=NULL "
                        "WHERE mission_id=? AND json_extract(context,'$.workflow_step_id')=?",
                        (mission_id, regen_step + ".verify"),
                    )
                # Reset this confirm task back to pending so it re-fires
                # after regeneration.
                await db.execute(
                    "UPDATE tasks SET status='pending', worker_attempts=0, error=NULL, "
                    "error_category=NULL, started_at=NULL, completed_at=NULL "
                    "WHERE id=?",
                    (task_id,),
                )
                await db.commit()
            except Exception as exc:
                logger.exception(f"rpc:RE failed: {exc}")
                await self.app.bot.send_message(chat_id=chat_id, text=f"❌ Regenerate failed: {exc}")
                return
            await query.edit_message_reply_markup(reply_markup=None)
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=f"♻️ Regenerating step `{regen_step or '?'}` for mission #{mission_id}.",
                parse_mode="Markdown",
            )
            self._pending_action.pop(chat_id, None)
            return

        if verb == "ED":
            # Stash edit intent. Next message from this chat → write content
            # to the (first) attached file, then reset writer for regenerate.
            self._pending_action[chat_id] = {
                "kind": "artifact_edit_inline",
                "mission_id": mission_id,
                "task_id": task_id,
                "regenerate_step_id": regen_step,
                "files": files,
            }
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=(
                    "✏️ Paste your edited version of the artifact as the NEXT message.\n"
                    "I'll overwrite the file with what you send and re-run the step.\n"
                    "Send `/cancel` to abort."
                ),
            )
            return

    # ─── Z10 T2B (D3): mission-event reaction handler ────────────────
    async def _handle_mission_event_callback(self, update, context):
        """Dispatch ``confirm:*`` and ``event:*`` callback presses.

        Patterns handled:
          * ``confirm:approve:<id>``  → db.resolve_confirmation(id, 'approved')
          * ``confirm:reject:<id>``   → db.resolve_confirmation(id, 'rejected')
          * ``event:approve:<event_id>`` / ``event:reject:<event_id>``
          * ``event:answer:<event_id>:<idx>``  ('asking' option pressed)
        """
        from src.infra.db import resolve_confirmation
        from .mission_events import resolve_event, post_event
        query = update.callback_query
        try:
            await query.answer()
        except Exception:  # noqa: BLE001
            pass
        data = (query.data or "")

        try:
            if data.startswith("confirm:approve:") or data.startswith("confirm:reject:"):
                _, verdict_kw, cid_s = data.split(":", 2)
                cid = int(cid_s)
                verdict = "approved" if verdict_kw == "approve" else "rejected"
                await resolve_confirmation(cid, verdict)
                # Update the message in-place to show resolved state.
                try:
                    new_text = (
                        (query.message.text or "")
                        + f"\n\n✅ Resolved: *{verdict}*"
                    )
                    await query.edit_message_text(
                        text=new_text, parse_mode="Markdown",
                    )
                except Exception:  # noqa: BLE001
                    pass
                logger.info(
                    "confirmation resolved via reaction",
                    confirmation_id=cid, verdict=verdict,
                )
                return

            if data.startswith("event:approve:") or data.startswith("event:reject:"):
                _, verdict_kw, eid_s = data.split(":", 2)
                event_id = int(eid_s)
                resolution = "approve" if verdict_kw == "approve" else "reject"
                await resolve_event(event_id, resolution)
                # If 'asking' rejected → post a [blocker] follow-up so an agent
                # sees the deadlock without polling.
                if verdict_kw == "reject":
                    # Lookup the event row to find mission_id.
                    db = await get_db()
                    cur = await db.execute(
                        "SELECT mission_id, payload FROM mission_events WHERE id = ?",
                        (event_id,),
                    )
                    row = await cur.fetchone()
                    if row:
                        import json as _json
                        try:
                            pl = _json.loads(row[1] or "{}")
                        except Exception:
                            pl = {}
                        await post_event(
                            self.app.bot, int(row[0]), "blocker",
                            {
                                "reason": (
                                    f"Asking event #{event_id} rejected: "
                                    f"{pl.get('question', '')}"
                                )[:300],
                            },
                        )
                try:
                    await query.edit_message_text(
                        text=(query.message.text or "")
                        + f"\n\n✅ Resolved: *{resolution}*",
                        parse_mode="Markdown",
                    )
                except Exception:  # noqa: BLE001
                    pass
                logger.info(
                    "mission_event resolved via reaction",
                    event_id=event_id, resolution=resolution,
                )
                return

            if data.startswith("event:answer:"):
                # event:answer:<event_id>:<option_idx>
                _, _, eid_s, idx_s = data.split(":", 3)
                event_id = int(eid_s)
                # Persist resolution as 'answer'; the agent polling this event
                # can read payload to map idx→option.
                await resolve_event(event_id, f"answer:{idx_s}")
                try:
                    await query.edit_message_text(
                        text=(query.message.text or "")
                        + f"\n\n✅ Answered: option #{idx_s}",
                    )
                except Exception:  # noqa: BLE001
                    pass
                logger.info(
                    "mission_event answer recorded",
                    event_id=event_id, option_idx=idx_s,
                )
                return
        except Exception as e:  # noqa: BLE001
            logger.exception("mission_event callback failed: %s", e)

    # ─── Z10 T2B (D5): thread + cost commands ────────────────────────
    async def cmd_mission_thread(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Surface the Telegram thread link for a mission. /mission_thread <id>"""
        if not context.args:
            await self._reply(update, "Usage: /mission_thread <mission_id>")
            return
        try:
            mid = int(context.args[0])
        except (TypeError, ValueError):
            await self._reply(update, "❌ mission_id must be an integer.")
            return
        mission = await get_mission(mid)
        if not mission:
            await self._reply(update, f"❌ Mission #{mid} not found.")
            return
        thread_id = mission.get("telegram_thread_id")
        if not thread_id:
            await self._reply(
                update,
                f"📭 Mission #{mid}: no thread (flat-mode fallback).",
            )
            return
        chat_id = update.effective_chat.id
        # Telegram deep-link: t.me/c/<chat_id without -100 prefix>/<thread_id>
        c_id = str(chat_id)
        if c_id.startswith("-100"):
            c_id = c_id[4:]
        link = f"https://t.me/c/{c_id}/{int(thread_id)}"
        await self._reply(
            update,
            f"🧵 Mission #{mid} thread:\n{link}",
        )

    async def cmd_missions_active(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List currently-active missions with their thread links."""
        missions = await get_active_missions()
        if not missions:
            await self._reply(update, "📭 No active missions.")
            return
        chat_id = update.effective_chat.id
        c_id = str(chat_id)
        if c_id.startswith("-100"):
            c_id = c_id[4:]
        lines = ["🎯 *Active missions:*"]
        for m in missions:
            title = (m.get("title") or "")[:60]
            mid = m.get("id")
            tid = m.get("telegram_thread_id")
            if tid:
                link = f"https://t.me/c/{c_id}/{int(tid)}"
                lines.append(f"• #{mid} {title} — [thread]({link})")
            else:
                lines.append(f"• #{mid} {title} — (flat)")
        await self._reply(
            update, "\n".join(lines), parse_mode="Markdown",
        )

    async def cmd_rollback_mission(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Roll a mission back to its last green checkpoint.

        Usage: ``/rollback_mission <mission_id> [task_id]``

        Reversibility: ``irreversible`` — verb is registered in
        VERB_REVERSIBILITY so the dispatcher's confirmation flow gates on
        founder approval when ``require_confirmation=True`` is passed.
        """
        if not context.args:
            await self._reply(
                update,
                "Usage: /rollback_mission <mission_id> [task_id]"
            )
            return
        try:
            mid = int(context.args[0])
        except (TypeError, ValueError):
            await self._reply(update, "mission_id must be an integer.")
            return
        target_task = None
        if len(context.args) >= 2:
            try:
                target_task = int(context.args[1])
            except (TypeError, ValueError):
                await self._reply(update, "task_id must be an integer.")
                return

        try:
            import mr_roboto
            task = {
                "id": 0,
                "mission_id": mid,
                "title": f"rollback_mission {mid}",
                "payload": {
                    "action": "rollback_mission",
                    "mission_id": mid,
                    "target_task_id": target_task,
                    "require_confirmation": True,
                },
            }
            action = await mr_roboto.run(task)
        except Exception as e:
            await self._reply(update, f"rollback failed: {e}")
            return

        if action.status != "completed":
            await self._reply(
                update,
                f"rollback {action.status}: {action.error or ''}"
            )
            return
        res = action.result or {}
        ledger = res.get("ledger") or {}
        lines = [
            f"Rollback OK for mission #{mid}",
            f"git tag: {ledger.get('git_tag')}",
            f"db restored: {res.get('db', {}).get('counts')}",
            f"chroma restored: {len(res.get('chroma', {}).get('restored') or [])}",
            f"schema rewind: rewound={len(res.get('schema_rewind', {}).get('rewound') or [])} "
            f"skipped={len(res.get('schema_rewind', {}).get('skipped') or [])}",
        ]
        await self._reply(update, "\n".join(lines))

    async def _handle_variant_choice(self, update, context):
        """Parse callback_data directly — survives bot restart."""
        chat_id = update.effective_chat.id
        data = (update.callback_query.data or "")

        mission_id: int | None = None
        task_id: int | None = None
        choice: str = ""
        # New format: "vc:{mission_id}:{task_id}:{choice}"
        if data.startswith("vc:"):
            parts = data.split(":")
            if len(parts) != 4:
                await update.callback_query.answer()
                return
            try:
                mission_id = int(parts[1])
                task_id = int(parts[2])
            except ValueError:
                await update.callback_query.answer()
                return
            choice = parts[3]
            await update.callback_query.answer()
        # Legacy format: "variant_choice:{choice}" — fall back to in-memory pending
        elif data.startswith("variant_choice:"):
            pending = self._pending_action.get(chat_id)
            if not pending or pending.get("kind") != "variant_choice":
                await update.callback_query.answer()
                return
            mission_id = pending["mission_id"]
            task_id = pending["task_id"]
            choice = data.split(":", 1)[1]
            await update.callback_query.answer()
        else:
            await update.callback_query.answer()
            return

        # Refresh / hydrate in-memory pending so compare-all has options to render
        pending = self._pending_action.get(chat_id) or {}
        if not pending or pending.get("task_id") != task_id:
            pending = await self._hydrate_variant_pending(chat_id, mission_id, task_id)
            if pending:
                self._pending_action[chat_id] = pending

        if choice == "all" or choice == "compare_all":
            await self._run_compare_all_and_reply(chat_id, mission_id, task_id, pending)
            return
        try:
            gid = int(choice)
        except ValueError:
            return
        self._pending_action.pop(chat_id, None)
        await self._resume_mission_at_step(
            mission_id=mission_id,
            after_task_id=task_id,
            clarify_choice={"kind": "variant", "group_id": gid},
        )

    async def _hydrate_variant_pending(
        self, chat_id: int, mission_id: int, task_id: int,
    ) -> dict | None:
        """Reconstruct variant_choice pending state from DB after restart."""
        try:
            from src.workflows.engine.artifacts import ArtifactStore
            store = ArtifactStore()
            await store.warm_cache(mission_id)
            gate_raw = await store.retrieve(mission_id, "gate_result") or "{}"
            import json as _json
            payload = _json.loads(gate_raw) if isinstance(gate_raw, str) else gate_raw
            options = payload.get("clarify_options") or []
            base_label = payload.get("base_label") or "Ürün"
            if not options:
                return None
            return {
                "kind": "variant_choice",
                "mission_id": mission_id,
                "task_id": task_id,
                "options": options,
                "base_label": base_label,
            }
        except Exception as exc:
            self.logger.debug("hydrate_variant_pending failed: %s", exc) if hasattr(self, "logger") else None
            return None

    async def _resume_mission_at_step(
        self,
        *,
        mission_id: int,
        after_task_id: int,
        clarify_choice: dict,
    ) -> None:
        """Write clarify_choice artifact + mark clarify_variant task completed so the
        workflow advances. Implementation uses the existing db + workflow advance helpers."""
        import json as _json
        from src.infra.db import update_task  # lazy import keeps module load cheap
        from src.workflows.engine.artifacts import ArtifactStore
        store = ArtifactStore()
        await store.store(
            mission_id,
            "clarify_choice",
            _json.dumps(clarify_choice, ensure_ascii=False),
        )
        await update_task(after_task_id, status="completed")

    async def _run_compare_all_and_reply(
        self,
        chat_id: int,
        mission_id: int,
        task_id: int,
        pending: dict | None = None,
    ) -> None:
        """Render category-style comparison then re-attach line buttons for user pick."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        from telegram.constants import ChatAction
        from src.workflows.engine.artifacts import ArtifactStore
        from src.workflows.shopping.pipeline_v2 import _handler_format_compare

        # Interim feedback — N synth calls can take 10-30s
        try:
            await self.app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        progress_msg = None
        try:
            progress_msg = await self.app.bot.send_message(
                chat_id=chat_id,
                text="🔍 Tüm seçenekler için inceleme özetleri hazırlanıyor…",
            )
        except Exception:
            progress_msg = None

        store = ArtifactStore()
        await store.warm_cache(mission_id)
        gate_raw = await store.retrieve(mission_id, "gate_result") or "{}"
        out = await _handler_format_compare(
            task={"id": task_id},
            artifacts={"gate_result": gate_raw},
            ctx={"mission_id": mission_id},
        )

        # Drop progress message — final cards replace it
        if progress_msg is not None:
            try:
                await self.app.bot.delete_message(chat_id=chat_id, message_id=progress_msg.message_id)
            except Exception:
                pass
        text = out.get("formatted_text") or "Bilgi yok."

        options = (pending or {}).get("options") or []
        markup = None
        if options:
            buttons = [
                [InlineKeyboardButton(
                    opt["label"],
                    callback_data=f"variant_choice:{opt['group_id']}",
                )]
                for opt in options
            ]
            markup = InlineKeyboardMarkup(buttons)

        # Telegram caps text at 4096 chars; category compare can exceed it
        MAX_LEN = 3800
        if len(text) > MAX_LEN:
            chunks: list[str] = []
            remaining = text
            while len(remaining) > MAX_LEN:
                cut = remaining.rfind("\n", 0, MAX_LEN)
                if cut <= 0:
                    cut = MAX_LEN
                chunks.append(remaining[:cut])
                remaining = remaining[cut:].lstrip("\n")
            chunks.append(remaining)
            for chunk in chunks[:-1]:
                await self.app.bot.send_message(
                    chat_id=chat_id, text=chunk, parse_mode="Markdown",
                )
            await self.app.bot.send_message(
                chat_id=chat_id, text=chunks[-1],
                reply_markup=markup, parse_mode="Markdown",
            )
        else:
            await self.app.bot.send_message(
                chat_id=chat_id, text=text,
                reply_markup=markup, parse_mode="Markdown",
            )

    # ─── Z6 T1D: founder_actions surface ────────────────────────────────
    async def cmd_actions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ):
        """List pending founder_actions.

        Usage:
          ``/actions`` — pending across all active missions.
          ``/actions <mission_id>`` — pending for that mission.
        """
        import src.founder_actions as fa
        from src.app.founder_action_render import render_action_card

        mission_id: int | None = None
        if context.args:
            try:
                mission_id = int(context.args[0])
            except (TypeError, ValueError):
                await self._reply(update, "mission_id must be an integer.")
                return

        if mission_id is not None:
            rows = await fa.list_by_mission(
                mission_id, status_filter=["pending", "in_progress"],
            )
        else:
            rows = await fa.list_pending()

        if not rows:
            # Z6 T7B — show "last resolved" tail so the founder knows the
            # surface is alive (vs. silent because the bot crashed).
            tail = ""
            try:
                from src.infra.db import get_db
                db = await get_db()
                if mission_id is not None:
                    cur = await db.execute(
                        "SELECT title, resolved_at FROM founder_actions "
                        "WHERE mission_id = ? AND status = 'done' "
                        "AND resolved_at IS NOT NULL "
                        "ORDER BY resolved_at DESC LIMIT 1",
                        (mission_id,),
                    )
                else:
                    cur = await db.execute(
                        "SELECT title, resolved_at FROM founder_actions "
                        "WHERE status = 'done' AND resolved_at IS NOT NULL "
                        "ORDER BY resolved_at DESC LIMIT 1"
                    )
                row = await cur.fetchone()
                if row:
                    last_title = str(row[0])[:60]
                    last_ts = str(row[1])
                    tail = (
                        f"\nLast resolved: _{last_title}_ at {last_ts}"
                    )
            except Exception:
                pass
            await self._reply(
                update,
                f"✅ All clear — no pending founder_actions.{tail}",
                parse_mode="Markdown",
            )
            return

        # First message: index. Each action gets its own card below.
        index_lines = [f"📋 *Pending founder_actions* ({len(rows)})"]
        for r in rows[:20]:
            index_lines.append(
                f"  #{r.id} [{r.kind}] m={r.mission_id} — {r.title[:50]}"
            )
        await self._reply(
            update, "\n".join(index_lines), parse_mode="Markdown",
        )

        # Send the top 5 as inline cards so the founder can act inline.
        for r in rows[:5]:
            text, kb = render_action_card(r.to_dict())
            try:
                await update.message.reply_text(
                    text, reply_markup=kb, parse_mode="Markdown",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"actions: card send failed #{r.id}: {e}")

    async def cmd_action_done(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ):
        """Resolve a founder_action.

        Usage: ``/action_done <id> [json_payload]``
        """
        if not context.args:
            await self._reply(
                update, "Usage: /action_done <id> [json_payload]",
            )
            return
        try:
            aid = int(context.args[0])
        except (TypeError, ValueError):
            await self._reply(update, "action id must be an integer.")
            return
        payload: dict | None = None
        if len(context.args) > 1:
            import json as _json
            try:
                payload = _json.loads(" ".join(context.args[1:]))
                if not isinstance(payload, dict):
                    payload = {"raw": payload}
            except Exception:
                payload = {"raw_text": " ".join(context.args[1:])}
        import src.founder_actions as fa
        try:
            action = await fa.resolve(aid, payload)
        except ValueError as e:
            await self._reply(update, f"❌ {e}")
            return
        # Z7 FAQ flywheel (#2): a resolved _faq_approval_pending card routes
        # its drafted FAQ entry into the support-docs collections. Without
        # this hook the only writer of those collections had zero callers,
        # so support_docs_* stayed permanently empty.
        extra = await self._apply_faq_approval_if_pending(action, payload)
        extra += await self._dispatch_outreach_if_pending(action, payload)
        await self._reply(
            update,
            f"✅ founder_action #{action.id} marked done.{extra}",
        )

    async def _dispatch_outreach_if_pending(self, action, payload) -> str:
        """If `action` is an outreach batch-approval card, draft + dispatch
        every pending prospect on its list. Returns a reply suffix."""
        schema = getattr(action, "expected_output_schema", None) or {}
        if not schema.get("_outreach_approval_pending"):
            return ""
        if isinstance(payload, dict) and (
            payload.get("reject") or payload.get("approved") is False
        ):
            return "\nOutreach batch rejected — nothing dispatched."
        list_id = schema.get("list_id")
        product_id = schema.get("product_id")
        if not list_id or not product_id:
            return "\n⚠️ Outreach card missing list_id/product_id."
        try:
            from src.infra.db import get_db
            import general_beckman
            db = await get_db()
            cur = await db.execute(
                "SELECT prospect_id, target_email, name FROM outreach_prospects "
                "WHERE product_id=? AND list_id=? AND status='pending'",
                (product_id, list_id),
            )
            prospects = await cur.fetchall()
            if not prospects:
                return "\nNo pending prospects on that list."
            mid = getattr(action, "mission_id", None)
            dispatched = 0
            for prospect_id, email, name in prospects:
                await general_beckman.enqueue(
                    {"agent_type": "mechanical",
                     "title": f"Draft outreach: {email}",
                     "mission_id": mid,
                     "payload": {"action": "outreach/draft",
                                 "product_id": product_id,
                                 "list_id": list_id,
                                 "template_id": "cold",
                                 "prospect_data": {"email": email,
                                                   "name": name or ""}}},
                    lane="oneshot",
                )
                dispatched += 1
            await db.execute(
                "UPDATE outreach_prospects SET status='approved' "
                "WHERE product_id=? AND list_id=? AND status='pending'",
                (product_id, list_id),
            )
            await db.commit()
            return f"\nOutreach: {dispatched} prospect(s) queued for draft + send."
        except Exception as e:
            logger.warning(f"outreach dispatch failed: {e}")
            return f"\n⚠️ Outreach dispatch failed: {e}"

    async def _apply_faq_approval_if_pending(self, action, payload) -> str:
        """If `action` is a FAQ-approval card, route its entry into the
        support-docs collections. Returns a status suffix for the reply."""
        schema = getattr(action, "expected_output_schema", None) or {}
        if not schema.get("_faq_approval_pending"):
            return ""
        # An explicit reject in the payload discards the draft.
        if isinstance(payload, dict) and (
            payload.get("reject") or payload.get("approved") is False
        ):
            return "\nFAQ draft rejected — not indexed."
        entry = schema.get("faq_entry")
        if not isinstance(entry, dict) or not entry.get("question"):
            return "\n⚠️ FAQ card had no usable entry — nothing indexed."
        try:
            from src.app.jobs.faq_regen import _apply_faq_approval
            await _apply_faq_approval(entry)
            return "\nFAQ entry appended to support docs + re-indexed."
        except Exception as e:
            logger.warning(f"faq approval apply failed: {e}")
            return f"\n⚠️ FAQ apply failed: {e}"

    async def cmd_ops_log(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ):
        """Z8 T4D — show recent on-call agent actions for a mission.

        Usage: ``/ops_log <mission_id>``

        Reads from ``registry_events`` (scope='action' rows written by the
        per-action audit path). The registry stores ``payload_json`` as
        ``{"payload": ..., "status": ...}`` — we extract ``status`` as the
        per-row outcome cell.
        """
        if not context.args:
            await self._reply(update, "Usage: /ops_log <mission_id>")
            return
        try:
            mid = int(context.args[0])
        except (TypeError, ValueError):
            await self._reply(update, "mission_id must be an integer.")
            return
        from src.infra.db import get_db
        import json as _json
        db = await get_db()
        async with db.execute(
            "SELECT verb, reversibility, payload_json, timestamp "
            "FROM registry_events "
            "WHERE scope = 'action' AND mission_id = ? "
            "ORDER BY id DESC LIMIT 20",
            (mid,),
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            await self._reply(update, f"No on-call actions for mission {mid}.")
            return
        lines = [f"📒 *Ops log mission {mid}* (last {len(rows)})"]
        for verb, rev, payload_json, ts in rows:
            outcome = ""
            try:
                pj = _json.loads(payload_json) if payload_json else {}
                outcome = str(pj.get("status") or "")
            except Exception:
                outcome = ""
            parts = [f"`{verb or '?'}`"]
            if rev:
                parts.append(str(rev))
            if outcome:
                parts.append(outcome)
            parts.append(str(ts))
            lines.append("  " + " · ".join(parts))
        await self._reply(update, "\n".join(lines), parse_mode="Markdown")

    # --- Z7 A0 briefing commands ---

    async def cmd_founder_hours_saved(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ):
        """Z7 A0 — /founder_hours_saved [period_days].

        Sums mission_events.founder_minutes_saved over the last N days
        (default 7) and replies with the total in hours + minutes.

        Usage: /founder_hours_saved [period_days]
        """
        period_days = 7
        if context.args:
            try:
                period_days = max(1, int(context.args[0]))
            except (TypeError, ValueError):
                await self._reply(
                    update,
                    "Usage: /founder_hours_saved [period_days] — period_days must be an integer.",
                )
                return

        try:
            from src.app.jobs.daily_briefing import sum_founder_minutes_saved
            total_minutes = await sum_founder_minutes_saved(period_days=period_days)
        except Exception as exc:
            await self._reply(
                update, f"Failed to compute founder hours saved: {exc}"
            )
            return

        hours = total_minutes // 60
        mins = total_minutes % 60
        if total_minutes == 0:
            msg = (
                f"No founder time saved recorded in the last {period_days} day(s).\n"
                "_(briefing_compose posthook must have run at least once)_"
            )
        else:
            msg = (
                f"*Founder time saved (last {period_days} day(s)):* "
                f"{total_minutes} min = {hours}h {mins}m"
            )
        await self._reply(update, msg, parse_mode="Markdown")

    # --- Z9 growth (stubs) ---
    # Reserve the founder-track command surface so Z9 T2-T5 can fill the
    # bodies without colliding with an existing command name. Each stub
    # just replies via _reply() naming the tier that delivers it.

    async def cmd_northstar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T4B — show the mission's north-star metric + recent trend.

        Usage: ``/northstar [mission_id]`` — defaults to the most recently
        created active mission. Reads ``mission.context['north_star']``
        (injected by the inject_north_star verb at Phase 8) and the latest
        ``metric_emit`` / ``hypothesis_recorded`` growth_events for trend.
        """
        logger.info("/northstar invoked")
        import json
        try:
            from src.infra.db import (
                get_active_missions,
                get_growth_events,
                get_mission,
            )
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ {_friendly_error(str(e))}")
            return

        # Resolve target mission — explicit arg, else newest active.
        mission = None
        args = context.args or []
        if args and args[0].isdigit():
            mission = await get_mission(int(args[0]))
            if mission is None:
                await self._reply(
                    update, f"❌ Mission {args[0]} not found.",
                )
                return
        else:
            actives = await get_active_missions()
            if actives:
                mission = max(actives, key=lambda m: m.get("id") or 0)

        if mission is None:
            await self._reply(
                update,
                "🧭 *North-star metric*\n\nNo active mission. Run "
                "`/northstar <mission_id>` to inspect a specific mission.",
                parse_mode="Markdown",
            )
            return

        mission_id = mission.get("id")
        ctx_raw = mission.get("context") or "{}"
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
        ns = ctx.get("north_star") or {}
        nsm = ns.get("north_star_metric") or {}
        aarrr = ns.get("aarrr_metrics") or []

        lines = [f"🧭 *North-star metric — mission #{mission_id}*\n"]
        if not nsm.get("name"):
            lines.append(
                "_Not configured yet._ The success_metrics artifact is "
                "created at step 2.9 and injected into mission context at "
                "Phase 8 (`inject_north_star`). Check back once the mission "
                "reaches Phase 8."
            )
        else:
            lines.append(f"*{nsm.get('name')}*")
            just = (nsm.get("justification") or "").strip()
            if just:
                lines.append(f"_{just[:300]}_")
            if aarrr:
                lines.append(f"\n*AARRR metrics* ({len(aarrr)}):")
                for m in aarrr[:8]:
                    tgt = m.get("target_value")
                    tgt_s = f" → target `{tgt}`" if tgt is not None else ""
                    lines.append(
                        f"  • `{m.get('name','?')}`{tgt_s} "
                        f"({m.get('measurement_frequency','?')})"
                    )

        # Latest measured value + trend from growth_events.
        try:
            emits = await get_growth_events(
                mission_id=mission_id, kind="metric_emit", limit=5,
            )
        except Exception:  # noqa: BLE001
            emits = []
        if emits:
            lines.append("\n*Recent values:*")
            for ev in emits:
                p = ev.get("properties") or {}
                val = p.get("value", p.get("metric_value", "?"))
                when = (ev.get("occurred_at") or "")[:16]
                lines.append(f"  • `{val}` — {when}")
        else:
            lines.append(
                "\n_No measured values yet — metric_emit fires once the "
                "feature is live._"
            )

        await self._reply(update, "\n".join(lines), parse_mode="Markdown")

    async def cmd_hypothesis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T4E — list pending hypotheses + recent verdicts.

        Pending rows show the predicted metric impact and when their
        measurement window closes; recent verdicts show confirmed /
        refuted / inconclusive outcomes. The verdict loop runs
        mechanically (daily verdict_window_sweep cron) — this command is
        the founder's read surface for the next-iteration /approve gate.
        """
        logger.info("/hypothesis invoked (Z9 T4E)")
        try:
            from datetime import datetime, timedelta
            from src.infra.db import get_pending_hypotheses, get_growth_events
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ {_friendly_error(str(e))}")
            return

        pending = await get_pending_hypotheses() or []
        verdicts = await get_growth_events(kind="verdict", limit=10) or []

        def _fmt_pred(pred):
            if not isinstance(pred, dict):
                return "?"
            metric = pred.get("metric", "?")
            direction = pred.get("direction", "?")
            mag = pred.get("magnitude")
            mag_s = f" {mag}" if mag is not None else ""
            return f"{metric} {direction}{mag_s}"

        def _window_close(h):
            created = h.get("created_at")
            window = h.get("window_seconds")
            if not created or not window:
                return "unknown"
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(str(created)[:19], fmt)
                    close = dt + timedelta(seconds=int(window))
                    return close.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    continue
            return "unknown"

        lines = ["🔬 *Hypotheses*\n"]
        lines.append("*Pending* — predictions awaiting their verdict window:")
        if pending:
            for h in pending[:10]:
                lines.append(
                    f"• *#{h.get('id')}* `{h.get('feature','?')}` — "
                    f"predicts {_fmt_pred(h.get('predicted_json'))}; "
                    f"window closes {_window_close(h)}"
                )
        else:
            lines.append("  _none — no open predictions._")

        lines.append("\n*Recent verdicts:*")
        if verdicts:
            icon = {"confirmed": "✅", "refuted": "❌", "inconclusive": "➖"}
            for v in verdicts:
                p = v.get("properties") or {}
                vd = p.get("verdict", "?")
                lines.append(
                    f"{icon.get(vd, '•')} *{vd}* — `{p.get('feature','?')}` / "
                    f"`{p.get('metric','?')}` "
                    f"(measured {p.get('observed_lift', 0.0):+.1%}, "
                    f"P held {p.get('p_held', 0.0):.2f})"
                )
        else:
            lines.append("  _none recorded yet._")

        lines.append(
            "\n_Confirmed verdicts reinforce the winning model; refuted "
            "pairs are suppressed 90 days. The next iteration stays "
            "founder-gated — use /approve._"
        )
        await self._reply(update, "\n".join(lines), parse_mode="Markdown")

    async def cmd_backlog(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T3C — list top-N scored backlog candidates.

        Each candidate carries the inspectable score formula breakdown.
        Founder promotes one to a mission with ``/approve <id>``.
        """
        try:
            from src.infra.db import get_growth_events
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ {_friendly_error(str(e))}")
            return

        rows = await get_growth_events(kind="backlog_candidate", limit=50)
        # Hide consumed / superseded candidates — only the live ranking.
        live = [
            r for r in rows
            if not (r.get("properties") or {}).get("consumed")
            and not (r.get("properties") or {}).get("superseded")
        ]
        if not live:
            await self._reply(
                update,
                "📋 *Growth backlog*\n\nNo scored candidates yet. The "
                "weekly digest classifies and scores signals — check back "
                "after the next cycle.",
                parse_mode="Markdown",
            )
            return

        live.sort(
            key=lambda r: (r.get("properties") or {}).get("score", 0.0),
            reverse=True,
        )
        lines = ["📋 *Growth backlog — top candidates*\n"]
        for r in live[:10]:
            p = r.get("properties") or {}
            f = p.get("formula") or {}
            lines.append(
                f"*#{r['id']}* `{p.get('label','?')}` / `{p.get('domain','?')}` "
                f"— score *{p.get('score', 0.0):.3f}*"
            )
            expr = f.get("expression")
            if expr:
                lines.append(f"  freq×rev×ns×age/cost: `{expr}`")
            excerpt = (p.get("sample_excerpt") or "").strip()
            if excerpt:
                lines.append(f"  _{excerpt[:120]}_")
            lines.append(f"  → `/approve {r['id']}`")
        await self._reply(
            update, "\n".join(lines), parse_mode="Markdown",
        )

    async def cmd_sunset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T5C — list current feature sunset candidates.

        Each candidate carries the inspectable usage/cost breakdown produced
        by the weekly ``sunset_score_recompute`` cron. Founder approves a
        deprecation mission with ``/approve_sunset <id>``.
        """
        logger.info("/sunset invoked")
        try:
            from src.infra.db import get_growth_events
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ {_friendly_error(str(e))}")
            return

        rows = await get_growth_events(kind="sunset_candidate", limit=100)
        live = [
            r for r in rows
            if not (r.get("properties") or {}).get("consumed")
            and not (r.get("properties") or {}).get("superseded")
        ]
        if not live:
            await self._reply(
                update,
                "🌅 *Feature sunset*\n\nNo sunset candidates. The weekly "
                "scorer flags features with usage below the threshold that "
                "still cost money to maintain — check back after the next "
                "cycle.",
                parse_mode="Markdown",
            )
            return

        live.sort(
            key=lambda r: (r.get("properties") or {}).get("sunset_score", 0.0),
            reverse=True,
        )
        lines = ["🌅 *Feature sunset candidates*\n"]
        for r in live[:15]:
            p = r.get("properties") or {}
            usage = p.get("usage_rate", 0.0)
            lines.append(
                f"*#{r['id']}* `{p.get('feature','?')}` "
                f"— sunset score *{p.get('sunset_score', 0.0):.2f}*"
            )
            lines.append(
                f"  usage *{usage * 100:.2f}%* "
                f"({p.get('distinct_users', 0)}/{p.get('active_users', 0)} users)"
                f" · cost `{p.get('cost_band','?')}`"
            )
            why = (p.get("why") or "").strip()
            if why:
                lines.append(f"  _{why[:160]}_")
            lines.append(f"  → `/approve_sunset {r['id']}`")
        await self._reply(
            update, "\n".join(lines), parse_mode="Markdown",
        )

    async def cmd_approve_sunset(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ):
        """Z9 T5C — approve a feature sunset candidate, spawning ONE
        deprecation mission.

        Mirrors ``/approve``: the founder gate is the ONLY path from a
        sunset candidate to a deprecation mission — sunsets are NEVER
        auto-executed. Marks the candidate consumed so it can't be approved
        twice.
        """
        if not context.args or not context.args[0].lstrip("#").isdigit():
            await self._reply(
                update,
                "Usage: `/approve_sunset <candidate_id>` — see `/sunset` "
                "for ids.",
                parse_mode="Markdown",
            )
            return

        cand_id = int(context.args[0].lstrip("#"))
        try:
            import json as _json
            from src.infra.db import (
                get_growth_events, insert_growth_event, get_db, add_mission,
            )
            import general_beckman
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ {_friendly_error(str(e))}")
            return

        rows = await get_growth_events(kind="sunset_candidate", limit=200)
        match = next(
            (r for r in rows if int(r.get("id") or 0) == cand_id), None
        )
        if match is None:
            await self._reply(
                update, f"❌ No sunset candidate #{cand_id}.",
            )
            return
        props = match.get("properties") or {}
        if props.get("consumed"):
            await self._reply(
                update,
                f"⚠️ Sunset candidate #{cand_id} was already approved — no "
                f"new mission spawned.",
            )
            return

        feature = props.get("feature", "feature")
        domain = props.get("domain", "general")
        usage = props.get("usage_rate", 0.0)
        cost_band = props.get("cost_band", "?")
        title = f"Deprecate feature: {feature}"[:80]
        description = (
            f"Approved feature sunset candidate #{cand_id}.\n"
            f"Feature: {feature} | Domain: {domain}\n"
            f"Usage: {usage * 100:.2f}% of active users "
            f"({props.get('distinct_users', 0)}/{props.get('active_users', 0)})"
            f" | Maintenance cost: {cost_band}\n"
            f"Why flagged: {props.get('why') or 'n/a'}\n\n"
            f"Plan and execute the safe removal of this feature's code and "
            f"infrastructure: feature flags, routes/endpoints, UI surfaces, "
            f"database columns/tables, scheduled jobs, docs. Verify nothing "
            f"else depends on it before deleting; ship behind a staged "
            f"rollout where reversible.\n"
        )

        # Spawn exactly ONE deprecation mission via Beckman — mirror /approve.
        mission_id = await add_mission(
            title=title,
            description=description,
            priority=5,
            context={
                "source": "growth_sunset",
                "sunset_candidate_id": cand_id,
                "sunset_feature": feature,
                "sunset_domain": domain,
            },
        )
        task_id = await general_beckman.enqueue({
            "title": f"[plan] {title}",
            "description": description,
            "agent_type": "planner",
            "kind": "main_work",
            "priority": 5,
            "mission_id": mission_id,
            "context": {
                "source": "growth_sunset",
                "sunset_candidate_id": cand_id,
            },
        })

        # Mark the candidate consumed (append-only flag flip).
        props["consumed"] = True
        props["approved_mission_id"] = mission_id
        try:
            db = await get_db()
            await db.execute(
                "UPDATE growth_events SET properties_json = ? WHERE id = ?",
                (_json.dumps(props), cand_id),
            )
            await db.commit()
        except Exception as e:  # noqa: BLE001
            logger.warning("cmd_approve_sunset: consume flag failed", error=str(e))
        # Audit row — sunset_approved keeps the lifecycle loop inspectable.
        try:
            await insert_growth_event(
                mission_id,
                "sunset_approved",
                {
                    "sunset_candidate_id": cand_id,
                    "mission_id": mission_id,
                    "planner_task_id": task_id,
                    "feature": feature,
                    "domain": domain,
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("cmd_approve_sunset: audit row failed: %s", e)

        await self._reply(
            update,
            f"✅ Sunset candidate #{cand_id} approved.\n"
            f"🗑️ Deprecation mission #{mission_id} created — _{title}_\n"
            f"Planning task #{task_id} queued.",
            parse_mode="Markdown",
        )

    async def cmd_experiment(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T5D — list A/B experiments + per-variant status + posterior.

        Shows every mission with ``experiment_variants`` rows: the control
        / treatment arms, their lifecycle status, and the most recent
        ``ab_result`` Bayesian posterior. Confident winners surface the
        founder gate (``/experiment_ship`` / ``/experiment_rollback``) —
        nothing auto-ships.
        """
        logger.info("/experiment invoked (Z9 T5D)")
        try:
            from src.infra.db import get_variants, get_growth_events
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ {_friendly_error(str(e))}")
            return

        variants = await get_variants() or []
        if not variants:
            await self._reply(
                update,
                "🧪 *A/B experiments*\n\nNo experiments yet. Phase 8+ "
                "feature missions assign variants automatically — check "
                "back after the next build.",
                parse_mode="Markdown",
            )
            return

        results = await get_growth_events(kind="ab_result", limit=100) or []
        result_by_mission: dict = {}
        for r in results:
            mid = r.get("mission_id")
            if mid is not None and mid not in result_by_mission:
                result_by_mission[mid] = r.get("properties") or {}

        # Group variants by mission.
        by_mission: dict = {}
        for v in variants:
            by_mission.setdefault(v.get("mission_id"), []).append(v)

        icon = {"active": "🟢", "winner": "🏆", "loser": "🔻",
                "stopped": "⏹️"}
        lines = ["🧪 *A/B experiments*\n"]
        for mid, vs in list(by_mission.items())[:15]:
            lines.append(f"*Mission #{mid}*")
            for v in vs:
                st = v.get("status", "?")
                lines.append(
                    f"  {icon.get(st, '•')} `{v.get('variant_name','?')}` "
                    f"— {st} (variant #{v.get('id')})"
                )
            res = result_by_mission.get(mid)
            if res:
                w = res.get("winner", "?")
                pt = res.get("p_treatment_better", 0.0)
                conf = res.get("confident")
                lines.append(
                    f"  → posterior: treatment better *{pt:.2f}* — "
                    f"winner *{w}* "
                    f"({'confident' if conf else 'inconclusive'})"
                )
                if conf:
                    lines.append(
                        f"  ⚖️ founder gate: `/experiment_ship {mid}` or "
                        f"`/experiment_rollback {mid}`"
                    )
            else:
                lines.append("  → no verdict yet")
            lines.append("")
        lines.append(
            "_A/B is default-on for Phase 8+ features. "
            "`/experiment_disable <mission_id>` opts a mission out._"
        )
        await self._reply(update, "\n".join(lines), parse_mode="Markdown")

    async def cmd_experiment_ship(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T5D — founder promotes the A/B winner to 100% rollout.

        Usage: ``/experiment_ship <mission_id> [winner]`` (winner defaults
        to ``treatment``). Runs the ``retire_variant`` mechanical: marks
        winner / loser and flips the PostHog flag. Founder-gated — there is
        no auto-ship even for a statistically confident winner.
        """
        if not context.args or not context.args[0].lstrip("#").isdigit():
            await self._reply(
                update,
                "Usage: `/experiment_ship <mission_id> [winner]`",
                parse_mode="Markdown",
            )
            return
        mid = int(context.args[0].lstrip("#"))
        winner = (context.args[1] if len(context.args) > 1
                  else "treatment").lower()
        try:
            import mr_roboto
            task = {
                "id": 0,
                "mission_id": mid,
                "title": f"experiment_ship {mid}",
                "payload": {
                    "action": "retire_variant",
                    "mission_id": mid,
                    "winner": winner,
                    "decision": "ship",
                },
            }
            action = await mr_roboto.run(task)
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ experiment_ship failed: {e}")
            return
        if action.status != "completed":
            await self._reply(
                update,
                f"⚠️ experiment_ship {action.status}: {action.error or ''}",
            )
            return
        res = action.result or {}
        await self._reply(
            update,
            f"🏆 Mission #{mid} — `{winner}` shipped to 100%.\n"
            f"Retired {res.get('retired', 0)} variant(s); loser arm → 0%.",
            parse_mode="Markdown",
        )

    async def cmd_experiment_rollback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T5D — founder force-rolls-back an A/B experiment.

        Usage: ``/experiment_rollback <mission_id>``. Crowns *control* the
        winner regardless of stats — the treatment arm goes to 0% rollout.
        Founder-gated; auto-rollback of a confident loser is never silent.
        """
        if not context.args or not context.args[0].lstrip("#").isdigit():
            await self._reply(
                update,
                "Usage: `/experiment_rollback <mission_id>`",
                parse_mode="Markdown",
            )
            return
        mid = int(context.args[0].lstrip("#"))
        try:
            import mr_roboto
            task = {
                "id": 0,
                "mission_id": mid,
                "title": f"experiment_rollback {mid}",
                "payload": {
                    "action": "retire_variant",
                    "mission_id": mid,
                    "winner": "control",
                    "decision": "rollback",
                },
            }
            action = await mr_roboto.run(task)
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ experiment_rollback failed: {e}")
            return
        if action.status != "completed":
            await self._reply(
                update,
                f"⚠️ experiment_rollback {action.status}: "
                f"{action.error or ''}",
            )
            return
        res = action.result or {}
        await self._reply(
            update,
            f"🔻 Mission #{mid} rolled back — `control` → 100%, "
            f"treatment → 0%.\nRetired {res.get('retired', 0)} variant(s).",
            parse_mode="Markdown",
        )

    async def cmd_experiment_disable(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T5D — opt a mission out of the default-on A/B harness.

        Usage: ``/experiment_disable <mission_id>``. Flips
        ``mission.context['use_ab']`` false so the Phase-8 ``assign_variant``
        step ships the feature at 100% rather than splitting traffic.
        """
        if not context.args or not context.args[0].lstrip("#").isdigit():
            await self._reply(
                update,
                "Usage: `/experiment_disable <mission_id>`",
                parse_mode="Markdown",
            )
            return
        mid = int(context.args[0].lstrip("#"))
        try:
            import json as _json
            from src.infra.db import get_mission, update_mission
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ {_friendly_error(str(e))}")
            return
        row = await get_mission(mid)
        if not row:
            await self._reply(update, f"❌ No mission #{mid}.")
            return
        ctx = row.get("context")
        if isinstance(ctx, str) and ctx.strip():
            try:
                ctx = _json.loads(ctx)
            except Exception:  # noqa: BLE001
                ctx = {}
        if not isinstance(ctx, dict):
            ctx = {}
        ctx["use_ab"] = False
        await update_mission(mid, context=_json.dumps(ctx))
        await self._reply(
            update,
            f"🚫 Mission #{mid} opted out of A/B. The Phase-8 "
            f"`assign_variant` step will ship the feature at 100% rollout; "
            f"the hypothesis is still recorded.",
            parse_mode="Markdown",
        )

    async def cmd_confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T5E — full-params typed confirmation for irreversible actions.

        Pricing changes move real revenue, so a bare ``/confirm`` is NOT
        accepted. The founder must re-type the full parameter set:

            ``/confirm pricing <amount> <interval> <window>``

        e.g. ``/confirm pricing 19.99 month 14d``. Only when every param
        echoes a pending pricing experiment does the Stripe pricing variant
        get created. This mirrors the irreversible-money rule from
        [[project_z10_complete]] — the typed echo is the gate.
        """
        args = context.args or []
        if not args or args[0].lower() != "pricing":
            await self._reply(
                update,
                "⚠️ *Typed confirmation required.*\n\n"
                "A bare `/confirm` is not enough for an irreversible "
                "pricing change. Re-type the full parameter set:\n"
                "`/confirm pricing <amount> <interval> <window>`\n"
                "e.g. `/confirm pricing 19.99 month 14d`",
                parse_mode="Markdown",
            )
            return
        if len(args) < 4:
            await self._reply(
                update,
                "❌ Incomplete. Pricing confirmation needs ALL params:\n"
                "`/confirm pricing <amount> <interval> <window>`\n"
                "e.g. `/confirm pricing 19.99 month 14d`",
                parse_mode="Markdown",
            )
            return
        amount_s, interval, window = args[1], args[2].lower(), args[3]
        try:
            amount = float(amount_s)
            if amount <= 0:
                raise ValueError("amount must be positive")
        except (TypeError, ValueError):
            await self._reply(
                update, "❌ `<amount>` must be a positive number "
                "(e.g. 19.99).", parse_mode="Markdown",
            )
            return
        if interval not in ("month", "year", "week", "day"):
            await self._reply(
                update,
                "❌ `<interval>` must be one of: month, year, week, day.",
                parse_mode="Markdown",
            )
            return

        chat_id = update.effective_chat.id if update.effective_chat else 0
        pending = self._pending_action.get(chat_id) or {}
        if pending.get("command") != "_pricing_confirm":
            await self._reply(
                update,
                "⚠️ No pending pricing experiment to confirm. Start one "
                "via the pricing-A/B flow first, then re-type "
                "`/confirm pricing <amount> <interval> <window>`.",
                parse_mode="Markdown",
            )
            return
        # Full-params echo MUST match the pending experiment exactly.
        exp = pending.get("params") or {}
        mismatches = []
        if abs(float(exp.get("amount", -1)) - amount) > 1e-6:
            mismatches.append(
                f"amount (expected {exp.get('amount')}, got {amount})"
            )
        if str(exp.get("interval", "")).lower() != interval:
            mismatches.append(
                f"interval (expected {exp.get('interval')}, got {interval})"
            )
        if str(exp.get("window", "")).lower() != str(window).lower():
            mismatches.append(
                f"window (expected {exp.get('window')}, got {window})"
            )
        if mismatches:
            await self._reply(
                update,
                "❌ Typed params do not match the pending pricing "
                "experiment:\n• " + "\n• ".join(mismatches) + "\n\n"
                "Re-type the EXACT params to confirm.",
                parse_mode="Markdown",
            )
            return

        # Confirmed — consume the pending action and create the pricing
        # variant (T5E). assign_variant with variant_kind='pricing'.
        self._pending_action.pop(chat_id, None)
        mid = exp.get("mission_id")
        try:
            import mr_roboto
            task = {
                "id": 0,
                "mission_id": mid,
                "title": f"pricing_ab confirm m{mid}",
                "payload": {
                    "action": "assign_variant",
                    "mission_id": mid,
                    "variant_kind": "pricing",
                    "feature": "pricing",
                    "control_price_id": exp.get("control_price_id"),
                    "treatment_price_id": exp.get("treatment_price_id"),
                },
            }
            action = await mr_roboto.run(task)
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ pricing confirm failed: {e}")
            return
        if action.status != "completed":
            await self._reply(
                update,
                f"⚠️ pricing confirm {action.status}: {action.error or ''}",
            )
            return
        await self._reply(
            update,
            f"💳 Pricing A/B confirmed for mission #{mid} — "
            f"${amount:.2f}/{interval}, {window} window.\n"
            f"Control + treatment pricing variants created. The "
            f"statistical-significance gate still applies before a winner "
            f"ships.",
            parse_mode="Markdown",
        )

    async def cmd_approve(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z9 T3C — approve a backlog candidate, spawning exactly one mission.

        This is the ONLY path from a backlog candidate to a mission — there
        is no auto-spawn. Marks the candidate consumed so it can't be
        approved twice.
        """
        if not context.args or not context.args[0].lstrip("#").isdigit():
            await self._reply(
                update,
                "Usage: `/approve <candidate_id>` — see `/backlog` for ids.",
                parse_mode="Markdown",
            )
            return

        cand_id = int(context.args[0].lstrip("#"))
        try:
            import json as _json
            from src.infra.db import get_growth_events, insert_growth_event, get_db
            import general_beckman
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"❌ {_friendly_error(str(e))}")
            return

        rows = await get_growth_events(kind="backlog_candidate", limit=200)
        match = next((r for r in rows if int(r.get("id") or 0) == cand_id), None)
        if match is None:
            await self._reply(
                update, f"❌ No backlog candidate #{cand_id}.",
            )
            return
        props = match.get("properties") or {}
        if props.get("consumed"):
            await self._reply(
                update,
                f"⚠️ Candidate #{cand_id} was already approved — no new "
                f"mission spawned.",
            )
            return

        label = props.get("label", "feature_request")
        domain = props.get("domain", "general")
        excerpt = (props.get("sample_excerpt") or "").strip()
        freq = props.get("frequency", 0)
        verb = {
            "bug": "Fix",
            "churn_signal": "Address churn driver in",
            "pricing_feedback": "Revisit pricing for",
        }.get(label, "Build")
        title = f"{verb} {domain} ({label})"[:80]
        description = (
            f"Approved growth backlog candidate #{cand_id}.\n"
            f"Label: {label} | Domain: {domain} | Signal frequency: {freq}\n"
            f"Score formula: {((props.get('formula') or {}).get('expression')) or 'n/a'}\n"
        )
        if excerpt:
            description += f"Representative signal: {excerpt}\n"

        # Spawn exactly one mission via Beckman. mr_roboto routes a
        # mechanical workflow_advance after planning; here we enqueue a
        # planner task scoped to the new mission so the founder gate stays
        # the single entry point.
        from src.infra.db import add_mission

        mission_id = await add_mission(
            title=title,
            description=description,
            priority=6,
            context={
                "source": "growth_backlog",
                "backlog_candidate_id": cand_id,
                "growth_label": label,
                "growth_domain": domain,
            },
        )
        task_id = await general_beckman.enqueue({
            "title": f"[plan] {title}",
            "description": description,
            "agent_type": "planner",
            "kind": "main_work",
            "priority": 6,
            "mission_id": mission_id,
            "context": {
                "source": "growth_backlog",
                "backlog_candidate_id": cand_id,
            },
        })

        # Mark the candidate consumed (append-only: flip the flag in place).
        props["consumed"] = True
        props["approved_mission_id"] = mission_id
        try:
            db = await get_db()
            await db.execute(
                "UPDATE growth_events SET properties_json = ? WHERE id = ?",
                (_json.dumps(props), cand_id),
            )
            await db.commit()
        except Exception as e:  # noqa: BLE001
            logger.warning("cmd_approve: consume flag failed", error=str(e))
        # Audit row — backlog_approved keeps the loop inspectable.
        try:
            await insert_growth_event(
                mission_id,
                "backlog_approved",
                {
                    "backlog_candidate_id": cand_id,
                    "mission_id": mission_id,
                    "planner_task_id": task_id,
                    "label": label,
                    "domain": domain,
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("cmd_approve: audit row failed: %s", e)

        await self._reply(
            update,
            f"✅ Candidate #{cand_id} approved.\n"
            f"🎯 Mission #{mission_id} created — _{title}_\n"
            f"Planning task #{task_id} queued.",
            parse_mode="Markdown",
        )
    # --- end Z9 growth (stubs) ---

    async def cmd_ask(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ):
        """Z8 T5E — tier-1 support inlet.

        Usage: ``/ask <question>``

        Pipeline:
        1. Retrieve top-3 ``support_docs`` via RAG.
        2. Detect angry/urgent sentiment via keyword classifier.
        3. Save an ``open`` ticket immediately so escalation has an anchor.
        4. (v1) Skip the agent loop in the inlet — saving + escalating
           directly gives the user a same-turn ack. The agent runs from a
           background mission via ``support_tier1`` on the ``ongoing`` lane;
           when it completes, ``support_rag.update_ticket`` patches answer +
           confidence and re-runs ``escalate_if_needed``.
        """
        if not context.args:
            await self._reply(update, "Usage: /ask <question>")
            return
        question = " ".join(context.args).strip()
        if not question:
            await self._reply(update, "Usage: /ask <question>")
            return
        user_id = str(update.effective_user.id) if update.effective_user else "anon"

        from src.ops.support_rag import (
            detect_sentiment,
            escalate_if_needed,
            retrieve_docs,
            save_ticket,
        )

        sentiment = detect_sentiment(question)
        docs = await retrieve_docs(question, top_k=3)
        ticket_id = await save_ticket(
            user_id=user_id,
            question=question,
            sentiment=sentiment,
            status="open",
        )

        if sentiment in ("angry", "urgent"):
            await escalate_if_needed(
                ticket_id,
                mission_id=None,
                user_id=user_id,
                question=question,
                answer=None,
                confidence=0.0,
                sentiment=sentiment,
            )
            await self._reply(
                update,
                f"📨 Ticket #{ticket_id} opened — escalated to the founder "
                "directly. You'll hear back shortly.",
            )
            return

        if not docs:
            await escalate_if_needed(
                ticket_id,
                mission_id=None,
                user_id=user_id,
                question=question,
                answer=None,
                confidence=0.0,
                sentiment=sentiment,
            )
            await self._reply(
                update,
                f"📨 Ticket #{ticket_id} opened — no matching docs, "
                "escalated to the founder.",
            )
            return

        # We have RAG hits; queue a tier-1 task and ack with hits inline.
        snippet = (docs[0].get("text") or "")[:300]
        # Z8 P1 (2026-05-18 sweep) — enqueue support_tier1 so the answer
        # actually arrives. The previous version saved the ticket, acked
        # "Full answer coming." and exited; the agent never ran. Tickets
        # piled up status='open' forever.
        try:
            from general_beckman import enqueue as _enqueue
            await _enqueue({
                "title": f"support tier-1: ticket #{ticket_id}",
                "description": (
                    f"Answer the founder support ticket #{ticket_id} using "
                    "the retrieved support_docs as grounding. Update the "
                    "tickets row with answer + confidence + sentiment via "
                    "src.ops.support_rag.update_ticket; escalate via "
                    "escalate_if_needed when confidence < 0.6 or the "
                    "answer doesn't address the question."
                ),
                "agent_type": "support_tier1",
                "context": {
                    "ticket_id": ticket_id,
                    "user_id": user_id,
                    "question": question,
                    "sentiment": sentiment,
                    "rag_docs": docs,
                },
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("cmd_ask: support_tier1 enqueue failed: %s", exc)
        await self._reply(
            update,
            f"📨 Ticket #{ticket_id} opened — closest doc:\n\n{snippet}\n\n"
            "Full answer coming.",
        )

    async def _handle_founder_action_callback(self, update, context):
        """Inline-button handler for founder_action cards.

        callback_data shape: ``fa_<verb>_<id>`` where verb ∈ {done,
        inprogress, block}.
        """
        query = update.callback_query
        if not query:
            return
        await query.answer()
        data = query.data or ""
        # fa_done_42, fa_inprogress_42, fa_block_42
        parts = data.split("_")
        if len(parts) != 3 or parts[0] != "fa":
            return
        verb = parts[1]
        try:
            aid = int(parts[2])
        except (TypeError, ValueError):
            return
        verb_to_status = {
            "done": "done",
            "inprogress": "in_progress",
            "block": "blocked",
        }
        new_status = verb_to_status.get(verb)
        if new_status is None:
            return
        import src.founder_actions as fa
        try:
            action = await fa.update_status(aid, new_status)
        except ValueError as e:
            try:
                await query.edit_message_text(f"❌ {e}")
            except Exception:
                pass
            return
        # Try to edit the card to reflect the new state.
        emoji = {"done": "✅", "in_progress": "▶️", "blocked": "⛔"}[new_status]
        new_text = (
            f"{emoji} founder_action #{action.id} → *{new_status}*\n"
            f"_{action.title}_"
        )
        try:
            await query.edit_message_text(new_text, parse_mode="Markdown")
        except Exception:
            pass

    async def _notify_founder_action(self, action) -> None:
        """Posted as a callback registered with founder_actions.create().

        Z6 polish P1: when action.urgent is truthy, bypass the mission
        thread and DM the admin chat directly (disputes, expired creds,
        security incidents). Regular path uses the mission thread.
        """
        try:
            from src.app.founder_action_render import render_action_card
            from src.app.telegram_topics import post_to_mission_thread
        except Exception as e:  # noqa: BLE001
            logger.debug(f"founder_action notify imports failed: {e}")
            return
        text, kb = render_action_card(action.to_dict())
        urgent = bool(getattr(action, "urgent", False))
        if urgent:
            # DM bypass — admin chat directly, prefixed for visibility.
            if not TELEGRAM_ADMIN_CHAT_ID:
                logger.debug("urgent founder_action: no admin chat configured")
                return
            urgent_text = f"🚨 *URGENT* — {text}"
            try:
                await self.app.bot.send_message(
                    chat_id=int(TELEGRAM_ADMIN_CHAT_ID),
                    text=urgent_text,
                    reply_markup=kb,
                    parse_mode="Markdown",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"urgent founder_action DM failed: {e}")
            return
        try:
            await post_to_mission_thread(
                self.app.bot, action.mission_id, text,
                reply_markup=kb, parse_mode="Markdown",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"founder_action thread post failed: {e}")

    # ── Z7 T4 A10 — CRM-as-interaction-log + A10.r1 consent ledger ──────────

    async def cmd_contact(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T4 A10 — Add or update a CRM contact.

        Usage:
          /contact add @handle [category]   — add/update a contact
          /contact @handle                  — show contact details

        Categories: customer, prospect, investor, journalist, partner,
                    advisor, candidate, vendor, other

        Examples:
          /contact add @alice_smith customer
          /contact add @bob_investor investor
          /contact @alice_smith
        """
        args = context.args or []

        if not args:
            await self._reply(
                update,
                (
                    "*CRM — Add contact*\n\n"
                    "Usage: `/contact add @handle [category]`\n"
                    "Categories: customer, prospect, investor, journalist, "
                    "partner, advisor, candidate, vendor, other\n\n"
                    "To view: `/contact @handle`"
                ),
                parse_mode="Markdown",
            )
            return

        if args[0] == "add":
            if len(args) < 2:
                await self._reply(
                    update,
                    "Usage: `/contact add @handle [category]`",
                    parse_mode="Markdown",
                )
                return
            handle = args[1] if args[1].startswith("@") else f"@{args[1]}"
            category = args[2] if len(args) >= 3 else "other"
            product_id = "default"
            try:
                from src.app.crm import add_contact
                contact_id = await add_contact(
                    product_id=product_id,
                    handle=handle,
                    display_name=handle.lstrip("@"),
                    category=category,
                )
                await self._reply(
                    update,
                    f"Contact added: {handle} (category: {category}, id: {contact_id})\n"
                    f"Use `/log {handle} <summary>` to log interactions.",
                )
            except Exception as exc:
                await self._reply(update, f"Error adding contact: {exc}")
            return

        # View contact by handle
        handle = args[0] if args[0].startswith("@") else f"@{args[0]}"
        product_id = "default"
        try:
            from src.app.crm import get_contact_by_handle, list_interactions
            contact = await get_contact_by_handle(product_id=product_id, handle=handle)
            if contact is None:
                await self._reply(update, f"Contact {handle} not found. Use `/contact add {handle}` to create.")
                return
            interactions = await list_interactions(product_id=product_id, contact_id=contact["contact_id"], limit=5)
            lines = [
                f"*Contact: {handle}*",
                f"Name: {contact['display_name']}",
                f"Category: {contact['category']}",
            ]
            if contact.get("email"):
                lines.append(f"Email: {contact['email']}")
            if interactions:
                lines.append(f"\nLast {len(interactions)} interactions:")
                for i in interactions:
                    lines.append(f"  [{i['kind']}] {i['summary'][:60]} ({i['logged_at'][:10]})")
            else:
                lines.append("\nNo interactions logged yet.")
            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
        except Exception as exc:
            await self._reply(update, f"Error: {exc}")

    async def cmd_crm_log(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T4 A10 — Log an interaction with a contact.

        Usage:
          /log @handle <summary> [follow-up: Nd|Nw|Nm]

        The optional follow-up window sets a reminder:
          2d = 2 days, 1w = 1 week, 3m = 3 months

        Examples:
          /log @alice Discussed pricing. follow-up: 2w
          /log @bob Demo call — interested in enterprise plan
          /log @carol Signed NDA. follow-up: 1m
        """
        args = context.args or []

        if not args:
            await self._reply(
                update,
                (
                    "*CRM — Log interaction*\n\n"
                    "Usage: `/log @handle <summary> [follow-up: Nd|Nw|Nm]`\n\n"
                    "Examples:\n"
                    "  `/log @alice Discussed pricing. follow-up: 2w`\n"
                    "  `/log @bob Demo call — interested`"
                ),
                parse_mode="Markdown",
            )
            return

        if len(args) < 2:
            await self._reply(update, "Usage: `/log @handle <summary>`", parse_mode="Markdown")
            return

        handle = args[0] if args[0].startswith("@") else f"@{args[0]}"
        rest = " ".join(args[1:])

        # Parse optional follow-up: "follow-up: 2w" or "follow-up:2w"
        import re as _re
        follow_up: str | None = None
        fu_match = _re.search(r"follow-?up:\s*(\S+)", rest, _re.IGNORECASE)
        if fu_match:
            follow_up = fu_match.group(1)
            rest = rest[:fu_match.start()].strip()

        summary = rest.strip()
        if not summary:
            await self._reply(update, "Please provide a summary after the handle.")
            return

        product_id = "default"
        try:
            from src.app.crm import get_contact_by_handle, log_interaction
            contact = await get_contact_by_handle(product_id=product_id, handle=handle)
            if contact is None:
                await self._reply(
                    update,
                    f"Contact {handle} not found. Create with `/contact add {handle}` first.",
                )
                return
            iid = await log_interaction(
                product_id=product_id,
                contact_id=contact["contact_id"],
                kind="other",
                summary=summary,
                follow_up=follow_up,
            )
            fu_text = f" Follow-up set: {follow_up}." if follow_up else ""
            await self._reply(
                update,
                f"Logged interaction #{iid} with {handle}.{fu_text}",
            )
        except Exception as exc:
            await self._reply(update, f"Error logging interaction: {exc}")

    async def cmd_contacts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T4 A10 — List CRM contacts with last interaction.

        Usage:
          /contacts              — list all contacts
          /contacts [category]   — filter by category

        Categories: customer, prospect, investor, journalist, partner,
                    advisor, candidate, vendor, other

        Examples:
          /contacts
          /contacts investor
          /contacts customer
        """
        args = context.args or []
        category = args[0] if args else None
        product_id = "default"

        try:
            from src.app.crm import list_contacts
            contacts = await list_contacts(product_id=product_id, category=category)
            if not contacts:
                label = f"category '{category}'" if category else "any category"
                await self._reply(
                    update,
                    f"No contacts found for {label}. Use `/contact add @handle` to add one.",
                )
                return
            header = f"*Contacts ({category or 'all'})*" if category else "*All Contacts*"
            lines = [header]
            for c in contacts:
                last = c.get("last_interaction")
                last_str = f" — last: {last[:10]}" if last else ""
                lines.append(
                    f"• {c['handle']} ({c['display_name']}) [{c['category']}]{last_str}"
                )
            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
        except Exception as exc:
            await self._reply(update, f"Error listing contacts: {exc}")

    async def cmd_follow_ups(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T4 A10 — Show pending follow-ups due within 7 days.

        Usage:
          /follow_ups          — pending within 7 days
          /follow_ups [days]   — pending within N days (e.g. /follow_ups 14)

        Examples:
          /follow_ups
          /follow_ups 14
        """
        args = context.args or []
        within_days = 7
        if args:
            try:
                within_days = int(args[0])
            except ValueError:
                pass

        product_id = "default"
        try:
            from src.app.crm import get_pending_follow_ups
            items = await get_pending_follow_ups(product_id=product_id, within_days=within_days)
            if not items:
                await self._reply(
                    update,
                    f"No pending follow-ups in the next {within_days} days.",
                )
                return
            lines = [f"*Follow-ups due within {within_days} days* ({len(items)} pending):"]
            for item in items:
                handle = item.get("handle") or f"contact#{item['contact_id']}"
                display = item.get("display_name") or handle
                fu_date = item.get("follow_up_at", "?")[:10]
                kind = item.get("kind") or "other"
                summary = (item.get("summary") or "")[:60]
                lines.append(
                    f"• {display} ({handle}) [{kind}] — due {fu_date}\n"
                    f"  {summary}"
                )
            lines.append(
                f"\nUse `/log @handle <summary>` to log outcome, or mark done via crm."
            )
            await self._reply(update, "\n".join(lines), parse_mode="Markdown")
        except Exception as exc:
            await self._reply(update, f"Error fetching follow-ups: {exc}")

    async def cmd_consent(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T4 A10.r1 — Manage contact consent records.

        Usage:
          /consent grant @handle <purpose>    — grant consent
          /consent revoke @handle <purpose>   — revoke consent
          /consent check @handle <purpose>    — check consent status

        Purposes: quote_use, data_processing, marketing_email,
                  interview_recording, case_study

        Examples:
          /consent grant @alice quote_use
          /consent revoke @bob marketing_email
          /consent check @carol data_processing
        """
        args = context.args or []

        if len(args) < 3:
            await self._reply(
                update,
                (
                    "*CRM — Consent ledger*\n\n"
                    "Usage:\n"
                    "  `/consent grant @handle <purpose>`\n"
                    "  `/consent revoke @handle <purpose>`\n"
                    "  `/consent check @handle <purpose>`\n\n"
                    "Purposes: `quote_use`, `data_processing`, `marketing_email`, "
                    "`interview_recording`, `case_study`"
                ),
                parse_mode="Markdown",
            )
            return

        action_word = args[0].lower()
        handle = args[1] if args[1].startswith("@") else f"@{args[1]}"
        purpose = args[2].lower()
        product_id = "default"

        VALID_ACTIONS = {"grant", "revoke", "check"}
        if action_word not in VALID_ACTIONS:
            await self._reply(
                update,
                f"Unknown action '{action_word}'. Use: grant, revoke, or check.",
            )
            return

        try:
            from src.app.crm import (
                get_contact_by_handle,
                grant_consent,
                revoke_consent,
                check_consent,
                has_consent,
            )
            contact = await get_contact_by_handle(product_id=product_id, handle=handle)
            if contact is None:
                await self._reply(
                    update,
                    f"Contact {handle} not found. Create with `/contact add {handle}` first.",
                )
                return

            cid = contact["contact_id"]

            if action_word == "grant":
                await grant_consent(
                    product_id=product_id,
                    contact_id=cid,
                    purpose=purpose,
                    source_evidence_url="telegram:manual",
                )
                await self._reply(
                    update,
                    f"Consent granted: {handle} → {purpose}\n"
                    f"Source: manual (Telegram). `/consent check {handle} {purpose}` to verify.",
                )

            elif action_word == "revoke":
                await revoke_consent(
                    product_id=product_id,
                    contact_id=cid,
                    purpose=purpose,
                )
                await self._reply(
                    update,
                    f"Consent revoked: {handle} → {purpose}",
                )

            elif action_word == "check":
                valid = await has_consent(
                    product_id=product_id,
                    contact_id=cid,
                    purpose=purpose,
                )
                record = await check_consent(
                    product_id=product_id,
                    contact_id=cid,
                    purpose=purpose,
                )
                if record is None:
                    await self._reply(update, f"No consent record for {handle} / {purpose}.")
                    return
                status = "VALID" if valid else "INVALID (revoked or expired)"
                lines = [
                    f"*Consent check: {handle} / {purpose}*",
                    f"Status: {status}",
                    f"Granted: {record.get('granted_at', '?')}",
                ]
                if record.get("expires_at"):
                    lines.append(f"Expires: {record['expires_at']}")
                if record.get("revoked_at"):
                    lines.append(f"Revoked: {record['revoked_at']}")
                if record.get("source_evidence_url"):
                    lines.append(f"Source: {record['source_evidence_url']}")
                await self._reply(update, "\n".join(lines), parse_mode="Markdown")

        except ValueError as ve:
            await self._reply(update, f"Validation error: {ve}")
        except Exception as exc:
            await self._reply(update, f"Error: {exc}")

    async def cmd_meeting(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T4 B4 — Schedule a meeting and manage meeting briefs.

        Usage:
          /meeting @handle YYYY-MM-DD HH:MM [purpose...]
                             — schedule a meeting with a contact
          /meeting list      — list upcoming meetings
          /meeting list @handle — list meetings for a specific contact

        Examples:
          /meeting @alice 2026-05-17 14:00 Product demo
          /meeting @investor 2026-06-01 10:30 Q2 investor update
          /meeting list
          /meeting list @alice

        A meeting brief is automatically generated 30 minutes before the
        scheduled time and delivered here. You'll also receive a prompt to
        log the meeting outcome 30 minutes after.
        """
        args = context.args or []
        product_id = "default"

        # /meeting list [@handle]
        if args and args[0].lower() == "list":
            try:
                from src.app.meetings import list_meetings
                from src.app.crm import get_contact_by_handle

                contact_filter = None
                if len(args) >= 2:
                    handle = args[1] if args[1].startswith("@") else f"@{args[1]}"
                    contact = await get_contact_by_handle(product_id=product_id, handle=handle)
                    if contact is None:
                        await self._reply(update, f"Contact {handle} not found.")
                        return
                    contact_filter = contact["contact_id"]

                meetings = await list_meetings(product_id, contact_id=contact_filter)
                if not meetings:
                    await self._reply(update, "No meetings scheduled.")
                    return

                lines = ["*Scheduled Meetings*\n"]
                for m in meetings[:20]:
                    cid = m.get("contact_id")
                    purpose = m.get("purpose") or "(no purpose)"
                    scheduled = m.get("scheduled_for") or "?"
                    brief_icon = "📋" if m.get("brief_md") else "⏳"
                    outcome_icon = "✅" if m.get("outcome_logged_interaction_id") else "🔲"
                    lines.append(
                        f"{brief_icon}{outcome_icon} `{scheduled}` — {purpose} "
                        f"(contact#{cid})"
                    )
                await self._reply(update, "\n".join(lines), parse_mode="Markdown")
            except Exception as exc:
                await self._reply(update, f"Error listing meetings: {exc}")
            return

        # /meeting @handle YYYY-MM-DD HH:MM [purpose...]
        if len(args) < 3:
            await self._reply(
                update,
                (
                    "*Meeting Brief — Auto-generation*\n\n"
                    "Schedule a meeting:\n"
                    "  `/meeting @handle YYYY-MM-DD HH:MM [purpose]`\n\n"
                    "List meetings:\n"
                    "  `/meeting list`\n"
                    "  `/meeting list @handle`\n\n"
                    "*Example:*\n"
                    "  `/meeting @alice 2026-05-17 14:00 Product demo`\n\n"
                    "A brief is auto-generated 30min before; an outcome prompt "
                    "is sent 30min after."
                ),
                parse_mode="Markdown",
            )
            return

        handle = args[0] if args[0].startswith("@") else f"@{args[0]}"
        date_str = args[1]
        time_str = args[2]
        purpose = " ".join(args[3:]) if len(args) > 3 else ""
        scheduled_for = f"{date_str} {time_str}"

        try:
            from src.app.crm import get_contact_by_handle
            from src.app.meetings import create_meeting

            contact = await get_contact_by_handle(product_id=product_id, handle=handle)
            if contact is None:
                await self._reply(
                    update,
                    f"Contact {handle} not found. Add them first with:\n"
                    f"`/contact add {handle}`",
                    parse_mode="Markdown",
                )
                return

            contact_id = contact["contact_id"]
            meeting_id = await create_meeting(
                product_id=product_id,
                contact_id=contact_id,
                scheduled_for=scheduled_for,
                purpose=purpose or "(no purpose stated)",
            )

            display = contact.get("display_name") or handle
            await self._reply(
                update,
                (
                    f"Meeting scheduled\n\n"
                    f"*Contact:* {display} ({handle})\n"
                    f"*Time:* {scheduled_for}\n"
                    f"*Purpose:* {purpose or '(none)'}\n"
                    f"*ID:* #{meeting_id}\n\n"
                    f"A brief will be generated 30min before. "
                    f"You'll be prompted to log the outcome 30min after."
                ),
                parse_mode="Markdown",
            )
        except ValueError as ve:
            await self._reply(
                update,
                f"Invalid date/time: {ve}\n\nExpected format: `YYYY-MM-DD HH:MM`",
                parse_mode="Markdown",
            )
        except Exception as exc:
            await self._reply(update, f"Error: {exc}")

    # ── Z7 T5 B1 — Lifecycle email engine ────────────────────────────────────

    async def cmd_lifecycle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Z7 T5 B1 — Lifecycle email engine commands.

        Subcommands:
          /lifecycle trigger <product_id> <user_id> <sequence_id>
              Manually trigger a sequence for a user (fallback until Z6
              event stream is live).
          /lifecycle status <product_id>
              Show sequence counts and recent send stats for the product.

        Examples:
          /lifecycle trigger prod-1 user@example.com 3
          /lifecycle status prod-1
        """
        chat_id = update.effective_chat.id
        args = context.args or []

        # Owner-only — /lifecycle triggers real email sends. Reject any chat
        # that is not the configured admin chat. When no admin chat is
        # configured the bot is single-tenant and the gate is a no-op.
        if not self._is_admin_chat(chat_id):
            await self._reply(
                update,
                "Not authorized — /lifecycle is restricted to the bot owner.",
            )
            return

        if not args:
            await self._reply(
                update,
                (
                    "*Lifecycle Email Engine (B1)*\n\n"
                    "Subcommands:\n"
                    "  `/lifecycle trigger <product> <user_id> <sequence_id>`\n"
                    "      Manually trigger a sequence for a user.\n\n"
                    "  `/lifecycle status <product>`\n"
                    "      Show sequence counts + recent send stats.\n\n"
                    "*Example:*\n"
                    "  `/lifecycle trigger prod-1 user@example.com 3`\n"
                    "  `/lifecycle status prod-1`"
                ),
                parse_mode="Markdown",
            )
            return

        sub = args[0].lower()

        if sub == "trigger":
            if len(args) < 4:
                await self._reply(
                    update,
                    "Usage: `/lifecycle trigger <product_id> <user_id> <sequence_id>`",
                    parse_mode="Markdown",
                )
                return
            product_id = args[1]
            user_id = args[2]
            try:
                sequence_id = int(args[3])
            except ValueError:
                await self._reply(update, "sequence_id must be an integer.")
                return
            try:
                from src.app.lifecycle_email import trigger_sequence
                result = await trigger_sequence(
                    product_id=product_id,
                    user_id=user_id,
                    sequence_id=sequence_id,
                )
                if result.get("ok"):
                    await self._reply(
                        update,
                        (
                            f"Sequence triggered.\n\n"
                            f"*Product:* {product_id}\n"
                            f"*User:* {user_id}\n"
                            f"*Sequence ID:* {sequence_id}\n"
                            f"*Sends scheduled:* {result.get('sends_created', 0)}"
                        ),
                        parse_mode="Markdown",
                    )
                else:
                    await self._reply(
                        update,
                        f"Failed to trigger sequence: {result.get('reason', 'unknown error')}",
                    )
            except Exception as exc:
                await self._reply(update, f"Error triggering sequence: {exc}")
            return

        if sub == "status":
            if len(args) < 2:
                await self._reply(
                    update,
                    "Usage: `/lifecycle status <product_id>`",
                    parse_mode="Markdown",
                )
                return
            product_id = args[1]
            try:
                from src.infra.db import get_db
                db = await get_db()

                # Count sequences
                cur = await db.execute(
                    "SELECT COUNT(*) FROM email_sequences WHERE product_id=?",
                    (product_id,),
                )
                row = await cur.fetchone()
                seq_total = row[0] if row else 0

                cur = await db.execute(
                    "SELECT COUNT(*) FROM email_sequences "
                    "WHERE product_id=? AND enabled=1",
                    (product_id,),
                )
                row = await cur.fetchone()
                seq_enabled = row[0] if row else 0

                # Count templates
                cur = await db.execute(
                    "SELECT COUNT(*) FROM email_templates WHERE product_id=?",
                    (product_id,),
                )
                row = await cur.fetchone()
                tmpl_total = row[0] if row else 0

                cur = await db.execute(
                    "SELECT COUNT(*) FROM email_templates "
                    "WHERE product_id=? AND status='approved'",
                    (product_id,),
                )
                row = await cur.fetchone()
                tmpl_approved = row[0] if row else 0

                # Recent sends
                cur = await db.execute(
                    "SELECT COUNT(*) FROM email_sends "
                    "WHERE product_id=? AND sent_at IS NOT NULL "
                    "AND scheduled_for >= strftime('%Y-%m-%d %H:%M:%S', 'now', '-7 days')",
                    (product_id,),
                )
                row = await cur.fetchone()
                sent_7d = row[0] if row else 0

                cur = await db.execute(
                    "SELECT COUNT(*) FROM email_sends "
                    "WHERE product_id=? AND sent_at IS NULL",
                    (product_id,),
                )
                row = await cur.fetchone()
                pending = row[0] if row else 0

                await self._reply(
                    update,
                    (
                        f"*Lifecycle Email Status — {product_id}*\n\n"
                        f"*Sequences:* {seq_enabled}/{seq_total} enabled\n"
                        f"*Templates:* {tmpl_approved}/{tmpl_total} approved\n"
                        f"*Sends (7d):* {sent_7d} sent\n"
                        f"*Pending:* {pending} scheduled"
                    ),
                    parse_mode="Markdown",
                )
            except Exception as exc:
                await self._reply(update, f"Error fetching lifecycle status: {exc}")
            return

        await self._reply(
            update,
            f"Unknown subcommand: {sub!r}\nUsage: /lifecycle trigger | /lifecycle status",
        )

    # ── Z7 T4 B7 — Customer interview / call notes pipeline ──────────────────

    async def cmd_interview(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /interview command.

        Subcommands:
          /interview start @handle [audio_path]  — begin a new interview session
          /interview stop [note_id] [audio=path] — mark interview stopped; set audio_path
          /interview list [@handle]              — list recent interview notes

        Examples:
          /interview start @alice
          /interview stop 42 audio=/tmp/call.mp3
          /interview list
          /interview list @alice
        """
        chat_id = update.effective_chat.id
        product_id = str(chat_id)

        args = (context.args or [])
        if not args:
            await self._reply(
                update,
                "Usage:\n"
                "  `/interview start @handle [audio_path]`\n"
                "  `/interview stop [note_id] [audio=/path/to/file.mp3]`\n"
                "  `/interview list [@handle]`",
                parse_mode="Markdown",
            )
            return

        sub = args[0].lower()

        # /interview list [@handle]
        if sub == "list":
            handle = args[1].lstrip("@") if len(args) > 1 else None
            try:
                from src.app.interview import list_interviews
                notes = await list_interviews(product_id, handle=handle, limit=10)
                if not notes:
                    target = f"@{handle}" if handle else "any contact"
                    await self._reply(update, f"No interview notes found for {target}.")
                    return
                lines = []
                for n in notes:
                    display = n.get("display_name") or n.get("handle") or f"contact#{n['contact_id']}"
                    dur = f" ({n['duration_minutes']:.0f}min)" if n.get("duration_minutes") else ""
                    lines.append(
                        f"• #{n['note_id']} — {display}{dur} — {n['started_at'] or 'no date'}"
                    )
                await self._reply(update, "Interview notes:\n" + "\n".join(lines))
            except Exception as exc:
                await self._reply(update, f"Error listing interviews: {exc}")
            return

        # /interview start @handle [audio_path]
        if sub == "start":
            if len(args) < 2:
                await self._reply(update, "Usage: `/interview start @handle [audio_path]`", parse_mode="Markdown")
                return
            handle = args[1].lstrip("@")
            audio_path = args[2] if len(args) > 2 else None
            try:
                from src.app.interview import start_interview
                result = await start_interview(product_id, handle, audio_path=audio_path)
                note_id = result["note_id"]
                consent_hint = ""
                try:
                    from src.app import crm
                    if not await crm.has_consent(product_id, result["contact_id"], "interview_recording"):
                        consent_hint = (
                            "\n\nReminder: recording consent is not on file for this contact. "
                            "Use `/consent grant @handle interview_recording <evidence_url>` "
                            "to record consent before uploading audio."
                        )
                except Exception:
                    pass
                await self._reply(
                    update,
                    f"Interview started — note #{note_id} for @{handle}.{consent_hint}\n\n"
                    f"When done, use `/interview stop {note_id} audio=/path/to/file.mp3` "
                    "to mark the interview complete and upload the audio.",
                )
            except Exception as exc:
                await self._reply(update, f"Error starting interview: {exc}")
            return

        # /interview stop [note_id] [audio=/path]
        if sub == "stop":
            note_id = int(args[1]) if len(args) > 1 and args[1].isdigit() else None
            audio_path = None
            for arg in args[2:]:
                if arg.startswith("audio="):
                    audio_path = arg[len("audio="):]
            if note_id is None:
                await self._reply(
                    update,
                    "Usage: `/interview stop <note_id> [audio=/path/to/file.mp3]`",
                    parse_mode="Markdown",
                )
                return
            try:
                from src.app.interview import stop_interview
                await stop_interview(note_id, product_id, audio_path=audio_path)
                next_steps = f"Interview #{note_id} stopped."
                if audio_path:
                    next_steps += f"\nAudio path set: `{audio_path}`"
                next_steps += (
                    "\n\nRun the pipeline:\n"
                    f"  1. `interview/transcribe` note_id={note_id}\n"
                    f"  2. `interview/summarize` note_id={note_id}\n"
                    f"  3. `interview/cross_link` note_id={note_id}\n\n"
                    "(These steps run automatically when dispatched via mr_roboto.)"
                )
                await self._reply(update, next_steps, parse_mode="Markdown")
            except Exception as exc:
                await self._reply(update, f"Error stopping interview: {exc}")
            return

        await self._reply(
            update,
            f"Unknown /interview subcommand: {sub!r}\n\n"
            "Available: `start`, `stop`, `list`",
            parse_mode="Markdown",
        )

    # ── Z7 T6 A7 — Cold outreach + deliverability spine ──────────────────────

    async def cmd_outreach(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /outreach command.

        Subcommands:
          /outreach upload <list_id>   — upload a prospect list + request batch approval
          /outreach status [list_id]   — show warmup status + recent sends + bounce rate
          /outreach verify <domain>    — verify SPF/DKIM/DMARC for outreach domain
          /outreach resume <list_id>   — clear a deliverability pause and resume sending

        Examples:
          /outreach upload prospects_q2
          /outreach status
          /outreach status prospects_q2
          /outreach verify outreach.example.com
          /outreach resume prospects_q2
        """
        import os as _os
        chat_id = update.effective_chat.id
        product_id = str(chat_id)

        args = (context.args or [])

        # Feature flag check
        enabled = _os.getenv("OUTREACH_ENABLED", "0").strip().lower() in ("1", "true", "yes")
        if not enabled and args and args[0].lower() not in ("status", "verify"):
            await self._reply(
                update,
                "Outreach is currently disabled (OUTREACH_ENABLED not set to 1).\n"
                "Ask the operator to enable it or use `/outreach status` to check config.",
            )
            return

        if not args:
            await self._reply(
                update,
                "Usage:\n"
                "  `/outreach upload <list_id>` — upload list + get approval card\n"
                "  `/outreach status [list_id]` — show warmup + send metrics\n"
                "  `/outreach verify <domain>` — check SPF/DKIM/DMARC\n"
                "  `/outreach resume <list_id>` — clear deliverability pause + resume sending\n\n"
                f"Feature flag: {'ON' if enabled else 'OFF'} (OUTREACH_ENABLED)",
                parse_mode="Markdown",
            )
            return

        sub = args[0].lower()

        if sub == "upload":
            list_id = args[1] if len(args) > 1 else None
            if not list_id:
                await self._reply(
                    update,
                    "Usage: `/outreach upload <list_id> <email> [email ...]`",
                    parse_mode="Markdown",
                )
                return
            # Parse the prospect emails (args after the list_id).
            emails = [e.strip().lower() for e in args[2:]
                      if "@" in e and "." in e.split("@")[-1]]
            if not emails:
                await self._reply(
                    update,
                    "No valid prospect emails.\n"
                    "Usage: `/outreach upload <list_id> <email> [email ...]`",
                    parse_mode="Markdown",
                )
                return
            from src.infra.db import get_db
            db = await get_db()
            inserted = 0
            for em in emails:
                try:
                    cur = await db.execute(
                        "INSERT OR IGNORE INTO outreach_prospects "
                        "(product_id, list_id, target_email, status) "
                        "VALUES (?, ?, ?, 'pending')",
                        (product_id, list_id, em),
                    )
                    inserted += cur.rowcount or 0
                except Exception as e:
                    logger.warning(f"outreach prospect insert failed: {e}")
            await db.commit()
            if inserted == 0:
                await self._reply(
                    update,
                    f"All {len(emails)} prospect(s) already on list `{list_id}` "
                    "— nothing new to approve.",
                    parse_mode="Markdown",
                )
                return
            # Founder approval card — approving it drafts + dispatches.
            try:
                import src.founder_actions as fa
                preview = ", ".join(emails[:3])
                mid = await self._resolve_or_create_outreach_mission(product_id)
                await fa.create(
                    mission_id=mid,
                    kind="generic",
                    title=f"Approve cold-outreach batch: list {list_id}",
                    why=(
                        f"{inserted} new prospect(s) uploaded to list "
                        f"'{list_id}'. Approve to draft + dispatch outreach "
                        f"to each (gated by warmup quota + suppression)."
                    ),
                    instructions=[
                        f"Prospects ({inserted} new): {preview}"
                        + (" ..." if len(emails) > 3 else ""),
                        "Approve via /action_done <id> to draft + send.",
                        "Reject via /action_done <id> {\"reject\": true}.",
                    ],
                    expected_output_kind="ack_only",
                    expected_output_schema={
                        "_outreach_approval_pending": True,
                        "list_id": list_id,
                        "product_id": product_id,
                    },
                    notify_telegram=True,
                )
            except Exception as e:
                logger.warning(f"outreach approval card failed: {e}")
                await self._reply(
                    update,
                    f"{inserted} prospect(s) saved to `{list_id}` but the "
                    f"approval card failed to surface: {e}",
                    parse_mode="Markdown",
                )
                return
            await self._reply(
                update,
                f"{inserted} prospect(s) saved to list `{list_id}`. A founder "
                "approval card has surfaced — approve it to draft + dispatch.",
                parse_mode="Markdown",
            )
            return

        if sub == "status":
            list_id = args[1] if len(args) > 1 else None
            try:
                from src.infra.db import get_db
                db = await get_db()
                where = "WHERE product_id=?"
                params: list = [product_id]
                if list_id:
                    where += " AND list_id=?"
                    params.append(list_id)

                cur = await db.execute(
                    f"SELECT COUNT(*) FROM outreach_sends {where} AND sent_at IS NOT NULL",
                    params,
                )
                total_row = await cur.fetchone()
                total_sent = total_row[0] if total_row else 0

                cur = await db.execute(
                    f"SELECT COUNT(*) FROM outreach_sends {where} AND bounced_at IS NOT NULL",
                    params,
                )
                bounce_row = await cur.fetchone()
                bounced = bounce_row[0] if bounce_row else 0

                bounce_pct = f"{bounced / total_sent:.1%}" if total_sent > 0 else "N/A"

                cur = await db.execute(
                    "SELECT domain, day, sent_count, target_count "
                    "FROM outreach_warmup WHERE product_id=? ORDER BY domain, day",
                    (product_id,),
                )
                warmup_rows = await cur.fetchall()
                warmup_lines = [
                    f"  {r[0]} day{r[1]}: {r[2]}/{r[3]}"
                    for r in warmup_rows
                ]
                warmup_section = "\n".join(warmup_lines) if warmup_lines else "  (no warmup data)"

                scope = f"list `{list_id}`" if list_id else "all lists"
                await self._reply(
                    update,
                    f"Outreach status ({scope}):\n"
                    f"  Total sent: {total_sent}\n"
                    f"  Bounced: {bounced} ({bounce_pct})\n\n"
                    f"Warmup:\n{warmup_section}\n\n"
                    f"Feature flag: {'ON' if enabled else 'OFF'}",
                    parse_mode="Markdown",
                )
            except Exception as exc:
                await self._reply(update, f"Error fetching outreach status: {exc}")
            return

        if sub == "verify":
            domain = args[1] if len(args) > 1 else None
            if not domain:
                await self._reply(update, "Usage: `/outreach verify <domain>`", parse_mode="Markdown")
                return
            try:
                from mr_roboto.outreach_domain_verify import run_domain_verify
                result = await run_domain_verify(product_id=product_id, mission_id=0, domain=domain)
                if result["status"] == "ok":
                    records = result.get("records", {})
                    lines = [f"  {'✓' if v else '✗'} {k.upper()}" for k, v in records.items()]
                    await self._reply(
                        update,
                        f"Domain `{domain}` DNS check PASSED:\n" + "\n".join(lines),
                        parse_mode="Markdown",
                    )
                else:
                    missing = result.get("missing", [])
                    await self._reply(
                        update,
                        f"Domain `{domain}` is missing: {', '.join(missing)}\n"
                        "A setup card has been queued in your founder_actions.",
                        parse_mode="Markdown",
                    )
            except Exception as exc:
                await self._reply(update, f"Error verifying domain: {exc}")
            return

        if sub == "resume":
            list_id = args[1] if len(args) > 1 else None
            if not list_id:
                await self._reply(update, "Usage: `/outreach resume <list_id>`", parse_mode="Markdown")
                return
            try:
                from mr_roboto.outreach_deliverability_check import clear_pause
                result = await clear_pause(product_id=product_id, list_id=list_id)
                if result["status"] == "cleared":
                    await self._reply(
                        update,
                        f"Campaign for list `{list_id}` has been resumed.\n"
                        "The deliverability pause has been cleared — outreach/send will "
                        "allow sends for this list again.",
                        parse_mode="Markdown",
                    )
                else:
                    await self._reply(
                        update,
                        f"No active pause found for list `{list_id}`.\n"
                        "The campaign is not currently paused.",
                        parse_mode="Markdown",
                    )
            except Exception as exc:
                await self._reply(update, f"Error resuming campaign: {exc}")
            return

        await self._reply(
            update,
            f"Unknown /outreach subcommand: {sub!r}\n\n"
            "Available: `upload`, `status`, `verify`, `resume`",
            parse_mode="Markdown",
        )

    # ─── Yalayut Phase 4 — /yalayut command group ───────────────────────

    async def cmd_yalayut(self, update, context):
        """`/yalayut <subcommand> ...` — catalog ops surface.

        Subcommands (spec Telegram UX): (no args) overview · sources pending ·
        review <source> · pending · policy add|review · disable|enable|requeue
        <id> · source promote <id> <tier> · owner promote <name> · stats ·
        discover "<intent>" · scout <url> · auth missing · auth set <K>=<V> ·
        mcp status|restart|kill <id>.
        """
        # Owner-only — /yalayut admin subcommands (enable / requeue / promote)
        # are by-design founder overrides that can flip a T3-quarantined
        # (unsafe) artifact to enabled with no re-vetting. Gate the whole
        # command at the boundary. Single-tenant bots (no admin chat set)
        # treat the gate as a no-op.
        chat_id = update.effective_chat.id
        if not self._is_admin_chat(chat_id):
            await self._reply(
                update,
                "Not authorized — /yalayut is restricted to the bot owner.",
            )
            return

        from yalayut import admin
        from yalayut.discovery import demand as _demand
        args = list(getattr(context, "args", []) or [])
        sub = args[0] if args else ""

        try:
            if not sub:
                st = await admin.stats()
                lines = ["📚 *Yalayut catalog*"]
                lines.append(f"Tiers: {st['tier_counts']}")
                lines.append(f"Types: {st['type_counts']}")
                lines.append(f"Vet queue: {st['vet_queue_depth']}")
                lines.append(
                    f"Source candidates: {st['source_candidate_queue_depth']}")
                lines.append(
                    f"Demand backlog: {st['demand_signal_backlog']}")
                await self._reply(update, "\n".join(lines))
                return

            if sub == "sources" and len(args) > 1 and args[1] == "pending":
                pend = await admin.pending_sources()
                if not pend:
                    await self._reply(update, "No pending source candidates.")
                    return
                for p in pend[:10]:
                    text = (f"🔎 *Source candidate*\n`{p['candidate_source_id']}`"
                            f"\ntype: {p['source_type']}\n{p['metadata']}")
                    kb = self._yalayut_source_keyboard(p["id"])
                    await self._reply(update, text, reply_markup=kb)
                return

            if sub == "review" and len(args) > 1:
                # collapsed per-source digest = pending artifacts of a source.
                src = args[1]
                pend = [a for a in await admin.pending_artifacts()
                        if a["source"] == src]
                if not pend:
                    await self._reply(update, f"No pending artifacts for {src}.")
                    return
                names = "\n".join(f"• {a['name']} (id {a['id']})"
                                  for a in pend)
                await self._reply(update, f"Pending in {src}:\n{names}")
                return

            if sub == "pending":
                pend = await admin.pending_artifacts()
                if not pend:
                    await self._reply(update, "No T2 escalations pending.")
                    return
                for a in pend[:10]:
                    text = (f"📦 *{a['name']}*\nsource: {a['source']}\n"
                            f"kind: {a['kind']} · tier T{a['vet_tier']}")
                    kb = self._yalayut_vet_keyboard(a["id"])
                    await self._reply(update, text, reply_markup=kb)
                return

            if sub == "policy":
                if len(args) > 1 and args[1] == "review":
                    props = await admin.policy_proposals()
                    if not props:
                        await self._reply(update, "No policy proposals.")
                        return
                    for p in props[:10]:
                        text = (f"⚙️ *Policy proposal*\n"
                                f"{p['check_name']} → `{p['key']}` "
                                f"= {p['proposed_value']}\n{p['evidence']}")
                        kb = self._yalayut_policy_keyboard(p["id"])
                        await self._reply(update, text, reply_markup=kb)
                    return
                if len(args) > 3 and args[1] == "add":
                    await admin.propose_policy(args[2], args[3])
                    await self._reply(
                        update, f"Policy proposal queued: {args[2]}/{args[3]}")
                    return
                await self._reply(update,
                                  "Usage: /yalayut policy add <check> <key> "
                                  "| /yalayut policy review")
                return

            if sub in ("disable", "enable", "requeue") and len(args) > 1:
                aid = int(args[1])
                fn = {"disable": admin.disable, "enable": admin.enable,
                      "requeue": admin.requeue}[sub]
                await fn(aid)
                await self._reply(update, f"Artifact {aid}: {sub} done.")
                return

            if sub == "source" and len(args) > 3 and args[1] == "promote":
                await admin.promote_source(args[2], int(args[3]))
                await self._reply(update,
                                  f"Source {args[2]} promoted to T{args[3]}.")
                return

            if sub == "owner" and len(args) > 2 and args[1] == "promote":
                await admin.promote_owner(args[2])
                await self._reply(update, f"Owner {args[2]} promoted.")
                return

            if sub == "stats":
                st = await admin.stats()
                lines = ["📊 *Yalayut stats*"]
                for cls, ab in (st.get("exposure_ab") or {}).items():
                    tot = ab["total"] or 1
                    rate = 100.0 * ab["succeeded"] / tot
                    lines.append(f"{cls}: {ab['succeeded']}/{ab['total']} "
                                 f"({rate:.0f}%)")
                await self._reply(update, "\n".join(lines))
                return

            if sub == "discover" and len(args) > 1:
                intent = " ".join(args[1:]).strip('"')
                await _demand.record_signal(_demand.DemandSignal(
                    source_step_pattern=f"founder:{intent[:40]}",
                    intent_keywords=intent.split(),
                    signal_type="founder", confidence=0.8))
                # immediately enqueue an on-demand discovery for this intent.
                import general_beckman
                await general_beckman.enqueue(
                    {"agent_type": "mechanical",
                     "title": f"Yalayut discover: {intent[:40]}",
                     "payload": {"action": "yalayut_discovery",
                                 "mode": "on_demand",
                                 "demand": {
                                     "source_step_pattern":
                                         f"founder:{intent[:40]}",
                                     "intent_keywords": intent.split(),
                                     "stacked_confidence": 0.8}}},
                    lane="oneshot")
                await self._reply(update,
                                  f"Discovery queued for: {intent}")
                return

            if sub == "scout" and len(args) > 1:
                res = await admin.queue_scout_url(args[1])
                await self._reply(update,
                                  f"Scout URL queued: {res['candidate_source_id']}")
                return

            if sub == "auth":
                if len(args) > 1 and args[1] == "missing":
                    miss = await admin.missing_auth()
                    if not miss:
                        await self._reply(update,
                                          "No artifacts blocked on auth.")
                        return
                    txt = "\n".join(f"• {m['name']}: {m['env_status']}"
                                    for m in miss)
                    await self._reply(update, f"Missing auth:\n{txt}")
                    return
                if len(args) > 2 and args[1] == "set":
                    kv = args[2]
                    if "=" not in kv:
                        await self._reply(update,
                                          "Usage: /yalayut auth set KEY=VALUE")
                        return
                    k, v = kv.split("=", 1)
                    await admin.set_secret(k.strip(), v.strip())
                    await self._reply(update, f"Secret {k.strip()} stored.")
                    return
                await self._reply(update,
                                  "Usage: /yalayut auth missing "
                                  "| /yalayut auth set KEY=VALUE")
                return

            if sub == "mcp":
                if len(args) > 1 and args[1] == "status":
                    rows = await admin.mcp_status()
                    if not rows:
                        await self._reply(update, "No MCP servers running.")
                        return
                    txt = "\n".join(
                        f"• {r['name']}: {r['health']} (pid {r['pid']})"
                        for r in rows)
                    await self._reply(update, f"MCP servers:\n{txt}")
                    return
                if len(args) > 2 and args[1] in ("restart", "kill"):
                    aid = int(args[2])
                    fn = (admin.mcp_restart if args[1] == "restart"
                          else admin.mcp_kill)
                    res = await fn(aid)
                    await self._reply(update,
                                      f"MCP {args[1]} {aid}: "
                                      f"{'ok' if res.get('ok') else res}")
                    return
                await self._reply(update,
                                  "Usage: /yalayut mcp status|restart|kill")
                return

            await self._reply(update,
                              "Unknown subcommand. Try /yalayut for overview.")
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"⚠️ /yalayut error: {e}")

    # ── inline keyboards ────────────────────────────────────────────────

    def _yalayut_vet_keyboard(self, artifact_id: int):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "✅ Approve", callback_data=f"yal:vet_approve:{artifact_id}"),
            InlineKeyboardButton(
                "❌ Reject", callback_data=f"yal:vet_reject:{artifact_id}"),
        ]])

    def _yalayut_source_keyboard(self, candidate_id: int):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "Trust", callback_data=f"yal:src_approve_trusted:{candidate_id}"),
            InlineKeyboardButton(
                "Untrust",
                callback_data=f"yal:src_approve_untrusted:{candidate_id}"),
        ], [
            InlineKeyboardButton(
                "Reject", callback_data=f"yal:src_reject:{candidate_id}"),
        ]])

    def _yalayut_policy_keyboard(self, proposal_id: int):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "Approve", callback_data=f"yal:pol_approve:{proposal_id}"),
            InlineKeyboardButton(
                "Reject", callback_data=f"yal:pol_reject:{proposal_id}"),
        ]])

    async def handle_yalayut_callback(self, update, context):
        """Route `yal:<action>:<id>` inline-button taps."""
        from yalayut import admin
        query = update.callback_query

        # Owner-only — these taps drive admin overrides (approve / reject /
        # promote) with no re-vetting. Reject any non-admin chat.
        chat_id = update.effective_chat.id
        if not self._is_admin_chat(chat_id):
            await query.answer("Not authorized — owner only.")
            return

        data = query.data or ""
        parts = data.split(":")
        if len(parts) != 3:
            await query.answer("Bad callback")
            return
        _, action, raw_id = parts
        try:
            target_id = int(raw_id)
        except ValueError:
            await query.answer("Bad id")
            return

        try:
            if action == "vet_approve":
                await admin.approve_artifact(target_id)
                msg = f"Artifact {target_id} approved."
            elif action == "vet_reject":
                await admin.reject_artifact(target_id)
                msg = f"Artifact {target_id} rejected."
            elif action == "src_approve_trusted":
                await admin.approve_source(target_id, trusted=True)
                msg = f"Source candidate {target_id} approved (trusted)."
            elif action == "src_approve_untrusted":
                await admin.approve_source(target_id, trusted=False)
                msg = f"Source candidate {target_id} approved (untrusted)."
            elif action == "src_reject":
                await admin.reject_source(target_id)
                msg = f"Source candidate {target_id} rejected."
            elif action == "pol_approve":
                await admin.decide_policy(target_id, approve=True)
                msg = f"Policy proposal {target_id} approved."
            elif action == "pol_reject":
                await admin.decide_policy(target_id, approve=False)
                msg = f"Policy proposal {target_id} rejected."
            else:
                msg = f"Unknown yalayut action: {action}"
            await query.answer("Done")
            await query.edit_message_text(msg)
        except Exception as e:  # noqa: BLE001
            await query.answer("Error")
            await query.edit_message_text(f"⚠️ {e}")
