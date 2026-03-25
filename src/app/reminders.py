# reminders.py
"""Todo reminder notifications sent via Telegram."""

from datetime import datetime

from ..infra.db import get_todos
from ..infra.logging_config import get_logger

logger = get_logger("app.reminders")

_PRIORITY_ICONS = {
    "high": "🔴",
    "normal": "🟡",
    "low": "⚪",
}


def _format_age(created_at, now=None):
    """Return compact relative time: '2m ago', '3h ago', '5d ago', '2w ago'."""
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    if now is None:
        now = datetime.now()
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


async def send_todo_reminder(telegram, suggestions=None):
    """Fetch pending todos, format with inline buttons, send to Telegram.

    Args:
        telegram: TelegramBot instance.
        suggestions: Optional dict {todo_id: suggestion_str} from suggestion tasks.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    try:
        todos = await get_todos(status="pending")
        if not todos:
            return

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

            suggestion = suggestions.get(tid)
            if suggestion:
                lines.append(f"   💡 {suggestion}")

            done_buttons.append(
                InlineKeyboardButton(f"✅ #{tid}", callback_data=f"todo_toggle:{tid}")
            )
            if suggestion:
                help_buttons.append(
                    InlineKeyboardButton(f"🤖 #{tid}", callback_data=f"todo_help:{tid}")
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

        await telegram.app.bot.send_message(
            chat_id=_get_admin_chat_id(),
            text=text,
            parse_mode="Markdown",
            reply_markup=markup,
        )
        logger.info("todo reminder sent", count=len(todos), suggestions=len(suggestions))

    except Exception as e:
        logger.error("failed to send todo reminder", error=str(e))


def _get_admin_chat_id():
    """Import admin chat ID lazily to avoid circular imports."""
    from .config import TELEGRAM_ADMIN_CHAT_ID
    return TELEGRAM_ADMIN_CHAT_ID
