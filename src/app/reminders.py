# reminders.py
"""Todo reminder notifications sent via Telegram."""

from datetime import datetime, timezone

from ..infra.db import get_todos
from ..infra.logging_config import get_logger

logger = get_logger("app.reminders")

_PRIORITY_ICONS = {
    "high": "🔴",
    "normal": "🟡",
    "low": "⚪",
}


def _format_age(created_at, now=None):
    """Return compact relative time: '2m ago', '3h ago', '5d ago', '2w ago'.

    ``created_at`` comes from SQLite's CURRENT_TIMESTAMP which is UTC.
    We compare against UTC now to avoid the 3-hour Turkey (UTC+3) offset.
    """
    if isinstance(created_at, str):
        # SQLite CURRENT_TIMESTAMP format: "2026-03-28 10:00:00" (no tz info, UTC)
        created_at = datetime.fromisoformat(created_at.replace(" ", "T"))
    # Ensure both sides are naive UTC (strip tzinfo if present so subtraction works)
    if created_at.tzinfo is not None:
        created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
    if now is None:
        now = datetime.utcnow()
    elif now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    diff = now - created_at
    minutes = int(diff.total_seconds() / 60)
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 14:
        return f"{days}d ago"
    weeks = days // 7
    return f"{weeks}w ago"


async def build_todo_list_message(suggestions=None):
    """Build the todo list text and inline keyboard markup.

    Returns:
        (text, markup) tuple, or (None, None) if no pending todos.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    todos = await get_todos(status="pending")
    if not todos:
        return None, None

    suggestions = suggestions or {}
    lines = ["📝 *Pending Todos*\n"]
    done_buttons = []
    help_buttons = []

    for todo in todos:
        tid = todo["id"]
        title = todo["title"]
        priority = todo.get("priority", "normal")
        p_icon = _PRIORITY_ICONS.get(priority, "🟡")
        age = _format_age(todo["created_at"])

        lines.append(f"  {p_icon} *#{tid}* — {title} ({age})")

        raw_suggestion = suggestions.get(tid)
        # Support both old format (str) and new format (text, agent_type) tuple
        if isinstance(raw_suggestion, tuple):
            suggestion, _agent = raw_suggestion
        else:
            suggestion = raw_suggestion
        if suggestion:
            lines.append(f"   💡 {suggestion}")

        done_buttons.append(
            InlineKeyboardButton(f"✅ #{tid}", callback_data=f"todo_toggle:{tid}")
        )
        if suggestion:
            # Encode agent type in callback data for the help handler
            if isinstance(raw_suggestion, tuple):
                agent_type = raw_suggestion[1]
            else:
                agent_type = "researcher"
            help_buttons.append(
                InlineKeyboardButton(f"🤖 #{tid}", callback_data=f"todo_help:{tid}:{agent_type}")
            )

    text = "\n".join(lines)
    rows = []
    # Pack done buttons, max 4 per row
    for i in range(0, len(done_buttons), 4):
        rows.append(done_buttons[i:i + 4])
    if help_buttons:
        for i in range(0, len(help_buttons), 4):
            rows.append(help_buttons[i:i + 4])
    rows.append([InlineKeyboardButton("🔙 Close", callback_data="todo_close")])

    markup = InlineKeyboardMarkup(rows)
    return text, markup


async def send_todo_reminder(telegram, suggestions=None):
    """Fetch pending todos, format with inline buttons, send to Telegram.

    Args:
        telegram: TelegramBot instance.
        suggestions: Optional dict {todo_id: suggestion_str} from suggestion tasks.
    """
    try:
        text, markup = await build_todo_list_message(suggestions)
        if not text:
            return

        await telegram.app.bot.send_message(
            chat_id=_get_admin_chat_id(),
            text=text,
            parse_mode="Markdown",
            reply_markup=markup,
        )
        logger.info("todo reminder sent", count=text.count("*#"), suggestions=len(suggestions or {}))

    except Exception as e:
        logger.error("failed to send todo reminder", error=str(e))


def _get_admin_chat_id():
    """Import admin chat ID lazily to avoid circular imports."""
    from .config import TELEGRAM_ADMIN_CHAT_ID
    return TELEGRAM_ADMIN_CHAT_ID
