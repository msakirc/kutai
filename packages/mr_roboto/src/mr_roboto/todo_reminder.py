"""Mechanical todo_reminder executor: sends pending-todo overview to Telegram."""
from __future__ import annotations

from src.app.reminders import send_todo_reminder
from src.app.telegram_bot import get_telegram


async def run(task: dict) -> dict:
    tg = get_telegram()
    await send_todo_reminder(tg)
    return {"sent": True}
