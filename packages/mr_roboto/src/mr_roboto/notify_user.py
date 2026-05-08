"""Mechanical notify_user executor: plain status Telegram send."""
from __future__ import annotations

from src.app.telegram_bot import get_telegram


async def notify_user(task: dict) -> dict:
    payload = task.get("payload") or {}
    # Accept either 'message' (beckman/cron callers) or 'text' (legacy).
    text = payload.get("message") or payload.get("text")
    if not text:
        raise ValueError("notify_user payload requires 'message' or 'text'")
    tg = get_telegram()
    chat_id = payload.get("chat_id")
    if chat_id is None:
        # Default to admin chat via the interface's helper.
        await tg.send_notification(text)
    else:
        await tg.app.bot.send_message(chat_id=chat_id, text=text)
    return {"sent": True}
