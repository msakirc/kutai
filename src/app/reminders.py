# reminders.py
"""Todo reminder notifications sent via Telegram."""

from datetime import datetime, timezone

from ..infra.db import get_todos
from ..infra.times import utc_now
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
        now = utc_now().replace(tzinfo=None)
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


async def build_todo_list_message():
    """Build the numbered todo list overview with suggestion hints.

    Returns:
        (text, markup) tuple, or (None, None) if no pending todos AND no DLQ tasks.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from ..infra.dead_letter import get_dlq_tasks

    todos = await get_todos(status="pending")
    dlq_tasks = await get_dlq_tasks(unresolved_only=True)

    if not todos and not dlq_tasks:
        return None, None

    lines = ["📝 *Pending Todos*\n"]
    for i, todo in enumerate(todos):
        num = i + 1
        title = todo["title"]
        suggestion = todo.get("suggestion")
        if suggestion:
            hint = suggestion[:40] + ("..." if len(suggestion) > 40 else "")
            lines.append(f"{num}. {title} — 💡 _{hint}_")
        else:
            lines.append(f"{num}. {title}")

    text = "\n".join(lines)

    # Numbered buttons, max 5 per row
    buttons = []
    row = []
    for i, todo in enumerate(todos):
        row.append(InlineKeyboardButton(
            str(i + 1), callback_data=f"todo_detail:{todo['id']}",
        ))
        if len(row) >= 5:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    # DLQ section — up to 5 most recent unresolved tasks
    if dlq_tasks:
        text += "\n\n⚠️ *Görev Hataları (DLQ)*"
        for dlq in dlq_tasks[:5]:
            task_id = dlq["task_id"]
            error = dlq.get("error") or ""
            quarantined_at = dlq.get("quarantined_at", "")
            age = _format_age(quarantined_at) if quarantined_at else "?"
            text += f"\n❌ #{task_id} — {error[:60]} ({age})"
            buttons.append([
                InlineKeyboardButton("🔁 Retry", callback_data=f"m:dlq:retry:{task_id}"),
                InlineKeyboardButton("⏭ Skip", callback_data=f"m:dlq:discard:{task_id}"),
            ])

    buttons.append([InlineKeyboardButton("🔙 Close", callback_data="todo_close")])
    markup = InlineKeyboardMarkup(buttons)
    return text, markup


def build_todo_detail_message(todo: dict):
    """Build the detail view for a single todo item.

    Returns:
        (text, markup) tuple.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    tid = todo["id"]
    title = todo["title"]
    priority = todo.get("priority", "normal")
    p_icon = _PRIORITY_ICONS.get(priority, "🟡")
    age = _format_age(todo["created_at"])
    suggestion = todo.get("suggestion")
    agent_type = todo.get("suggestion_agent", "researcher")

    lines = [f"📝 *#{tid}* — {title}"]
    lines.append(f"Priority: {p_icon} {priority} | Added: {age}")

    if todo.get("description"):
        lines.append(f"\n_{todo['description']}_")

    if suggestion:
        lines.append(f"\n💡 {suggestion}")

    text = "\n".join(lines)

    # Action buttons
    action_row = [
        InlineKeyboardButton("✅ Done", callback_data=f"todo_toggle:{tid}"),
        InlineKeyboardButton("✏️ Edit", callback_data=f"todo_edit:{tid}"),
    ]
    if suggestion:
        action_row.append(
            InlineKeyboardButton("🤖 Help", callback_data=f"todo_help:{tid}:{agent_type}")
        )
    action_row.append(
        InlineKeyboardButton("❌ Cancel", callback_data=f"todo_delete:{tid}")
    )

    buttons = [action_row]
    buttons.append([InlineKeyboardButton("🔙 Back", callback_data="todo_list")])

    markup = InlineKeyboardMarkup(buttons)
    return text, markup


async def send_todo_reminder(telegram):
    """Fetch pending todos and send the overview list to Telegram."""
    try:
        text, markup = await build_todo_list_message()
        if not text:
            return

        try:
            await telegram.app.bot.send_message(
                chat_id=_get_admin_chat_id(),
                text=text,
                parse_mode="Markdown",
                reply_markup=markup,
            )
        except Exception:
            # Markdown parse failure (unescaped chars) — retry plain
            await telegram.app.bot.send_message(
                chat_id=_get_admin_chat_id(),
                text=text,
                reply_markup=markup,
            )
        logger.info("todo reminder sent")

    except Exception as e:
        logger.error("failed to send todo reminder", error=str(e))


def _get_admin_chat_id():
    """Import admin chat ID lazily to avoid circular imports."""
    from .config import TELEGRAM_ADMIN_CHAT_ID
    return TELEGRAM_ADMIN_CHAT_ID
