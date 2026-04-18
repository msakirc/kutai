"""Mechanical clarify executor: sends clarification prompt via Telegram."""
from __future__ import annotations

from src.infra.db import update_task
from src.app.telegram_bot import get_telegram


async def clarify(task: dict) -> dict:
    payload = task.get("payload") or {}
    question = payload.get("question")
    if not question:
        raise ValueError("clarify payload requires 'question'")
    tg = get_telegram()
    await tg.request_clarification(task["id"], task.get("title", ""), question)
    await update_task(task["id"], status="waiting_human")
    return {"sent": True, "question": question}
