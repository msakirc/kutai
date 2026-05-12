"""Mechanical notify_user executor: plain status Telegram send.

Optional payload field `inline_buttons` attaches a single-row inline keyboard:

    "inline_buttons": [
        {"label": "🔄 Regen", "callback_data": "regen:42:mission_42/charter.md"},
        {"label": "🎯 Propagate", "callback_data": "propagate:42:mission_42/charter.md"},
    ]

Telegram caps callback_data at 64 bytes; entries exceeding that are dropped
silently so a bad button never blocks the notification.
"""
from __future__ import annotations

from src.app.telegram_bot import get_telegram


def _build_reply_markup(inline_buttons):
    """Return an InlineKeyboardMarkup or None. Drops malformed entries silently."""
    if not isinstance(inline_buttons, list) or not inline_buttons:
        return None
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    except Exception:
        return None
    row = []
    for b in inline_buttons:
        if not isinstance(b, dict):
            continue
        label = b.get("label")
        cb = b.get("callback_data")
        if not isinstance(label, str) or not isinstance(cb, str):
            continue
        if len(cb.encode("utf-8")) > 64:
            continue  # Telegram limit
        row.append(InlineKeyboardButton(label, callback_data=cb))
    if not row:
        return None
    return InlineKeyboardMarkup([row])


async def notify_user(task: dict) -> dict:
    payload = task.get("payload") or {}
    # Accept either 'message' (beckman/cron callers) or 'text' (legacy).
    text = payload.get("message") or payload.get("text")
    if not text:
        raise ValueError("notify_user payload requires 'message' or 'text'")
    tg = get_telegram()
    chat_id = payload.get("chat_id")
    reply_markup = _build_reply_markup(payload.get("inline_buttons"))
    if chat_id is None:
        # Default to admin chat via the interface's helper.
        await tg.send_notification(text, reply_markup=reply_markup)
    else:
        await tg.app.bot.send_message(
            chat_id=chat_id, text=text, reply_markup=reply_markup
        )
    return {"sent": True, "with_buttons": reply_markup is not None}
