# reminders.py
"""Todo reminder notifications sent via Telegram."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ..infra.db import get_todos
from ..infra.logging_config import get_logger

logger = get_logger("app.reminders")

# Heuristic: if title contains these words, KutAI can likely help
_AI_HELPABLE_KEYWORDS = [
    "research", "find", "search", "compare", "check",
    "analyze", "summarize", "look up", "investigate",
    "ara", "bul", "karsilastir", "kontrol",
]

_PRIORITY_ICONS = {
    "high": "🔴",
    "normal": "🟡",
    "low": "⚪",
}


async def send_todo_reminder(telegram):
    """Fetch pending todos, format with inline buttons, send to Telegram."""
    try:
        todos = await get_todos(status="pending")
        if not todos:
            return  # Nothing pending — skip reminder

        lines = ["📝 *Pending Todos*\n"]
        buttons = []

        for todo in todos:
            tid = todo["id"]
            title = todo["title"]
            priority = todo.get("priority", "normal")
            p_icon = _PRIORITY_ICONS.get(priority, "🟡")

            lines.append(f"  {p_icon} *#{tid}* — {title}")

            # Toggle done button
            buttons.append(
                [InlineKeyboardButton(
                    f"✅ Done: {title[:25]}",
                    callback_data=f"todo_toggle:{tid}",
                )]
            )

            # Check if AI can help
            title_lower = title.lower()
            if any(kw in title_lower for kw in _AI_HELPABLE_KEYWORDS):
                buttons.append(
                    [InlineKeyboardButton(
                        f"🤖 Help: {title[:20]}",
                        callback_data=f"todo_ai:{tid}",
                    )]
                )

        text = "\n".join(lines)
        markup = InlineKeyboardMarkup(buttons) if buttons else None

        await telegram.app.bot.send_message(
            chat_id=telegram.app.bot.defaults and telegram.app.bot.defaults.chat_id
            or _get_admin_chat_id(),
            text=text,
            parse_mode="Markdown",
            reply_markup=markup,
        )
        logger.info("todo reminder sent", count=len(todos))

    except Exception as e:
        logger.error("failed to send todo reminder", error=str(e))


def _get_admin_chat_id():
    """Import admin chat ID lazily to avoid circular imports."""
    from .config import TELEGRAM_ADMIN_CHAT_ID
    return TELEGRAM_ADMIN_CHAT_ID
