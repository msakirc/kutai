"""Mechanical notify_user executor: plain status Telegram send."""
from __future__ import annotations

from src.app.telegram_bot import get_telegram


async def notify_user(task: dict) -> dict:
    payload = task.get("payload") or {}
    # Accept either 'message' (beckman/cron callers) or 'text' (legacy).
    text = payload.get("message") or payload.get("text")
    if not text:
        raise ValueError("notify_user payload requires 'message' or 'text'")
    chat_id = payload.get("chat_id")
    if chat_id is None:
        from src.app.config import TELEGRAM_ADMIN_CHAT_ID
        chat_id = TELEGRAM_ADMIN_CHAT_ID
    tg = get_telegram()
    await tg.send_message(chat_id, text)
    return {"sent": True}
