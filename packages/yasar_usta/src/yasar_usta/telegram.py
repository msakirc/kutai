"""Telegram Bot API helpers — send, edit, poll."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

logger = logging.getLogger("yasar_usta.telegram")


class TelegramAPI:
    """Minimal Telegram Bot API client using aiohttp.

    Uses a single persistent ClientSession so DNS results and TCP
    connections are reused across requests.  This avoids a fresh DNS
    lookup on every poll — the root cause of the 4 944-error burst on
    2026-04-15 when a brief DNS blip took out Yaşar Usta's poller while
    the main bot (python-telegram-bot, which pools connections) was fine.

    Args:
        token: Bot token.
        chat_id: Admin chat ID.
    """

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._session: aiohttp.ClientSession | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def _get_session(self) -> aiohttp.ClientSession:
        """Return (and lazily create) a persistent session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def send(
        self,
        text: str,
        reply_markup: dict | None = None,
        parse_mode: str = "Markdown",
    ) -> dict | None:
        """Send a message to the admin chat."""
        if not self.enabled:
            return None
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            session = self._get_session()
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
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            session = self._get_session()
            async with session.post(
                f"{self._base_url}/editMessageText",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return await resp.json()
        except Exception as e:
            logger.warning("Telegram edit failed: %s", e)
            return None

    async def delete(self, message_id: int) -> None:
        """Delete a message."""
        if not self.enabled:
            return
        payload = {"chat_id": self.chat_id, "message_id": message_id}
        try:
            session = self._get_session()
            await session.post(
                f"{self._base_url}/deleteMessage",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            )
        except Exception:
            pass

    async def answer_callback(self, callback_query_id: str, text: str | None = None) -> None:
        """Answer a callback query (removes loading spinner, optionally shows toast)."""
        if not self.enabled:
            return
        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        try:
            session = self._get_session()
            await session.post(
                f"{self._base_url}/answerCallbackQuery",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            )
        except Exception:
            pass

    async def get_updates(self, offset: int = 0, timeout: int = 5) -> list[dict]:
        """Long-poll for updates."""
        if not self.enabled:
            return []
        try:
            session = self._get_session()
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

    async def flush_updates(self) -> int:
        """Confirm all pending updates. Returns the next offset to use."""
        if not self.enabled:
            return 0
        next_offset = 0
        try:
            session = self._get_session()
            async with session.get(
                f"{self._base_url}/getUpdates",
                params={"offset": 0, "timeout": 0},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                pending = data.get("result", [])
                if pending:
                    max_id = max(u["update_id"] for u in pending)
                    next_offset = max_id + 1
                    logger.info("Flushing %d pending updates (next_offset=%d)", len(pending), next_offset)
                    # Confirm by fetching with offset past the last one
                    await session.get(
                        f"{self._base_url}/getUpdates",
                        params={"offset": next_offset, "timeout": 0},
                        timeout=aiohttp.ClientTimeout(total=5),
                    )
                else:
                    logger.info("No pending updates to flush")
        except Exception as e:
            logger.warning("flush_updates failed: %s", e)
        return next_offset
