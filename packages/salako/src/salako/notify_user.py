"""Mechanical notify_user executor: plain status Telegram send."""
from __future__ import annotations

from src.app.telegram_bot import get_telegram


async def notify_user(task: dict) -> dict:
    payload = task.get("payload") or {}
    chat_id = payload.get("chat_id")
    text = payload.get("text")
    if not text or chat_id is None:
        raise ValueError("notify_user payload requires 'chat_id' and 'text'")
    tg = get_telegram()
    await tg.send_message(chat_id, text)
    return {"sent": True}
