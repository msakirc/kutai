"""Telegram Bot API helpers — send, edit, poll."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("yasar_usta.telegram")


class TelegramAPI:
    """Minimal Telegram Bot API client using aiohttp.

    Args:
        token: Bot token.
        chat_id: Admin chat ID.
    """

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{token}"

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(
        self,
        text: str,
        reply_markup: dict | None = None,
        parse_mode: str = "Markdown",
    ) -> dict | None:
        """Send a message to the admin chat."""
        if not self.enabled:
            return None
        import aiohttp
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/sendMessage",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    return await resp.json()
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
            return None

    async def edit(
        self,
        message_id: int,
        text: str,
        reply_markup: dict | None = None,
        parse_mode: str = "Markdown",
    ) -> dict | None:
        """Edit an existing message."""
        if not self.enabled:
            return None
        import aiohttp
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/editMessageText",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    return await resp.json()
        except Exception as e:
            logger.warning("Telegram edit failed: %s", e)
            return None

    async def answer_callback(self, callback_query_id: str) -> None:
        """Answer a callback query (removes loading spinner)."""
        if not self.enabled:
            return
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{self._base_url}/answerCallbackQuery",
                    json={"callback_query_id": callback_query_id},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception:
            pass

    async def get_updates(self, offset: int = 0, timeout: int = 5) -> list[dict]:
        """Long-poll for updates."""
        if not self.enabled:
            return []
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/getUpdates",
                    params={"offset": offset, "timeout": timeout},
                    timeout=aiohttp.ClientTimeout(total=timeout + 10),
                ) as resp:
                    data = await resp.json()
            return data.get("result", [])
        except Exception as e:
            logger.warning("Telegram poll error: %s", e)
            return []

    async def flush_updates(self) -> None:
        """Confirm all pending updates (used before self-restart)."""
        if not self.enabled:
            return
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                await session.get(
                    f"{self._base_url}/getUpdates",
                    params={"offset": -1, "timeout": 0},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception:
            pass
