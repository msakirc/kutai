"""Per-mission Telegram forum topic provisioning (Z10 T2B / D1).

Single-bot, per-mission threading. Each mission gets its own Telegram
forum topic so two concurrent missions don't interleave in one flat
chat. If the chat is not a supergroup with topics enabled,
``create_forum_topic`` raises and this module falls back to flat
posting with a ``[Mission {id}]`` prefix — the rest of the codebase
keeps working unchanged.

Public API
----------
``ensure_mission_topic(bot, mission_id, mission_title, chat_id) -> int | None``
``post_to_mission_thread(bot, mission_id, text, **kwargs)``
``close_mission_topic(bot, mission_id)``

All functions are idempotent and tolerate missing thread_id (NULL in
``missions.telegram_thread_id``) — in fallback mode they prefix the
text with ``[Mission {id}]`` and post to the chat without a thread.
"""
from __future__ import annotations

from typing import Any

from src.infra.db import get_db, get_mission
from src.infra.logging_config import get_logger

logger = get_logger("app.telegram_topics")


async def _store_thread_id(mission_id: int, thread_id: int | None) -> None:
    from general_beckman import update_mission_fields
    await update_mission_fields(mission_id, telegram_thread_id=thread_id)


async def _store_thread_archived(mission_id: int, archived: bool) -> None:
    from general_beckman import update_mission_fields
    await update_mission_fields(mission_id, telegram_thread_archived=(1 if archived else 0))


async def ensure_mission_topic(
    bot: Any,
    mission_id: int,
    mission_title: str,
    chat_id: int,
) -> int | None:
    """Return the thread_id for ``mission_id``, creating it if absent.

    - If ``missions.telegram_thread_id`` is already set, returns it unchanged
      (idempotent).
    - Otherwise calls ``bot.create_forum_topic(chat_id, name=...)``,
      persists the new thread_id, and returns it.
    - On error (chat is not a forum supergroup, bot lacks permission,
      etc.) logs a warning and returns ``None`` — callers must handle
      ``None`` by falling back to flat posting (``post_to_mission_thread``
      does this automatically).
    """
    mission = await get_mission(mission_id)
    if mission is None:
        logger.warning("ensure_mission_topic: mission missing", mission_id=mission_id)
        return None
    existing = mission.get("telegram_thread_id")
    if existing:
        return int(existing)

    name = f"Mission {mission_id}: {mission_title}"[:128]  # Telegram caps name
    try:
        result = await bot.create_forum_topic(chat_id=chat_id, name=name)
        # PTB returns a ForumTopic object with `.message_thread_id`
        thread_id = getattr(result, "message_thread_id", None)
        if thread_id is None and isinstance(result, dict):
            thread_id = result.get("message_thread_id")
        if thread_id is None:
            logger.warning(
                "create_forum_topic returned no thread_id; "
                "falling back to flat mode",
                mission_id=mission_id,
                result=str(result),
            )
            return None
        await _store_thread_id(mission_id, int(thread_id))
        logger.info(
            "mission topic provisioned",
            mission_id=mission_id,
            thread_id=int(thread_id),
        )
        return int(thread_id)
    except Exception as e:
        # Forum topics not enabled / not a supergroup / permission missing.
        # Log once and fall back. Callers will see None and prefix-flat.
        logger.warning(
            "create_forum_topic failed — falling back to flat mode",
            mission_id=mission_id,
            error=str(e),
        )
        return None


async def post_to_mission_thread(
    bot: Any,
    mission_id: int,
    text: str,
    chat_id: int | None = None,
    **kwargs,
) -> Any:
    """Send ``text`` to the mission's thread or flat-prefix fallback.

    Returns the sent ``Message``.

    - If ``missions.telegram_thread_id`` is set, sends with
      ``message_thread_id=thread_id``.
    - If unset (fallback mode), prefixes text with ``[Mission {id}]``
      and sends without a thread.
    - ``chat_id`` defaults to TELEGRAM_ADMIN_CHAT_ID when None.
    """
    if chat_id is None:
        from .config import TELEGRAM_ADMIN_CHAT_ID
        if not TELEGRAM_ADMIN_CHAT_ID:
            raise RuntimeError(
                "post_to_mission_thread: chat_id and TELEGRAM_ADMIN_CHAT_ID both unset"
            )
        chat_id = int(TELEGRAM_ADMIN_CHAT_ID)

    mission = await get_mission(mission_id)
    thread_id = mission.get("telegram_thread_id") if mission else None

    if thread_id:
        return await bot.send_message(
            chat_id=chat_id,
            text=text,
            message_thread_id=int(thread_id),
            **kwargs,
        )
    # Fallback: flat prefix
    prefixed = f"[Mission {mission_id}] {text}"
    return await bot.send_message(chat_id=chat_id, text=prefixed, **kwargs)


async def close_mission_topic(
    bot: Any,
    mission_id: int,
    chat_id: int | None = None,
) -> bool:
    """Close the forum topic for ``mission_id`` and mark archived.

    Best-effort: errors (already closed, fallback mode, permission)
    are logged and swallowed. Returns True if a close call was issued
    successfully, False on any failure or no thread to close.
    """
    if chat_id is None:
        from .config import TELEGRAM_ADMIN_CHAT_ID
        if not TELEGRAM_ADMIN_CHAT_ID:
            logger.warning("close_mission_topic: no chat_id available")
            return False
        chat_id = int(TELEGRAM_ADMIN_CHAT_ID)

    mission = await get_mission(mission_id)
    thread_id = mission.get("telegram_thread_id") if mission else None
    if not thread_id:
        logger.debug(
            "close_mission_topic: no thread to close",
            mission_id=mission_id,
        )
        return False
    try:
        await bot.close_forum_topic(
            chat_id=chat_id, message_thread_id=int(thread_id),
        )
        await _store_thread_archived(mission_id, True)
        logger.info("mission topic closed", mission_id=mission_id)
        return True
    except Exception as e:
        logger.warning(
            "close_forum_topic failed",
            mission_id=mission_id,
            error=str(e),
        )
        return False
