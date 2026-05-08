"""Mechanical price_watch_check executor: re-scrapes watched products and notifies on drops."""
from __future__ import annotations

from src.app.price_watch_checker import check_price_watches
from src.app.telegram_bot import get_telegram


async def run(task: dict) -> dict:
    tg = get_telegram()
    result = await check_price_watches(tg)
    return result or {"checked": 0}
