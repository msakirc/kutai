"""Typed mission-event API + formatters (Z10 T2B / D2).

Every cross-cutting signal a mission emits to Telegram routes through
``post_event`` so we get:

- A single rendering surface per ``kind`` (one terse line + reply markup
  appropriate to the kind).
- Persistence in ``mission_events`` so reactions, comments, and reply-
  to-message can be resolved back to the originating event later.

Kinds
-----
- ``milestone``: payload['summary']
- ``blocker``:   payload['reason']                     + reply_keyboard
- ``asking``:    payload['question'], payload['options'?: list[str]]
- ``confirmation_required``:
      payload['confirmation_id'], payload['verb'],
      payload['reversibility'],   payload['payload_summary'?]
- ``cost_alert``:
      payload['threshold_pct'], payload['total'], payload['ceiling']

Public API
----------
``post_event(bot, mission_id, kind, payload) -> int``
returns the mission_events row id.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from src.infra.db import get_db
from src.infra.logging_config import get_logger

from .telegram_topics import post_to_mission_thread

logger = get_logger("app.mission_events")

EventKind = Literal[
    "milestone", "blocker", "asking",
    "confirmation_required", "cost_alert",
]

_VALID_KINDS = {
    "milestone", "blocker", "asking",
    "confirmation_required", "cost_alert",
}


# ── Per-kind formatters ──────────────────────────────────────────────

def _fmt_milestone(payload: dict) -> tuple[str, Any]:
    return f"🎯 [milestone] {payload.get('summary', '(no summary)')}", None


def _fmt_blocker(payload: dict) -> tuple[str, Any]:
    # REPLY_KEYBOARD belongs to TelegramInterface; we don't import it
    # here to keep mission_events test-friendly. The bot.send_message
    # call inside post_to_mission_thread keeps user's existing reply
    # keyboard intact when reply_markup is omitted.
    return f"🚧 [blocker] {payload.get('reason', '(no reason)')}", None


def _fmt_asking(payload: dict) -> tuple[str, Any]:
    text = f"❓ [asking] {payload.get('question', '(no question)')}"
    options = payload.get("options") or []
    if not isinstance(options, list) or not options:
        return text, None
    # event_id is filled in by caller after row is created; mark the
    # callback_data with a placeholder that post_event rewrites.
    rows = []
    for idx, opt in enumerate(options):
        label = opt if isinstance(opt, str) else str(opt)
        rows.append([InlineKeyboardButton(
            label[:60],
            callback_data=f"event:answer:__EID__:{idx}",
        )])
    return text, InlineKeyboardMarkup(rows)


def _fmt_confirmation_required(payload: dict) -> tuple[str, Any]:
    verb = payload.get("verb", "?")
    rev = payload.get("reversibility", "?")
    summary = payload.get("payload_summary") or ""
    confirmation_id = payload.get("confirmation_id")
    text = (
        f"⚠️ [confirmation_required] {verb} ({rev})\n"
        f"{summary}".rstrip()
    )
    if confirmation_id is None:
        return text, None
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton(
            "👍 Approve",
            callback_data=f"confirm:approve:{int(confirmation_id)}",
        ),
        InlineKeyboardButton(
            "👎 Reject",
            callback_data=f"confirm:reject:{int(confirmation_id)}",
        ),
    ]])
    return text, kb


def _fmt_cost_alert(payload: dict) -> tuple[str, Any]:
    threshold = payload.get("threshold_pct", "?")
    total = payload.get("total", "?")
    ceiling = payload.get("ceiling", "?")
    mid = payload.get("mission_id", "?")
    return (
        f"💸 [cost_alert] Mission {mid} hit {threshold}%: "
        f"${total} / ${ceiling}",
        None,
    )


_FORMATTERS = {
    "milestone": _fmt_milestone,
    "blocker": _fmt_blocker,
    "asking": _fmt_asking,
    "confirmation_required": _fmt_confirmation_required,
    "cost_alert": _fmt_cost_alert,
}


def format_event(kind: str, payload: dict) -> tuple[str, Any]:
    """Render ``(text, reply_markup)`` for ``kind`` + ``payload``.

    Exposed for tests; ``post_event`` calls it internally.
    """
    if kind not in _FORMATTERS:
        raise ValueError(f"unknown mission_event kind: {kind!r}")
    return _FORMATTERS[kind](payload)


# ── Persistence + send ──────────────────────────────────────────────

async def post_event(
    bot: Any,
    mission_id: int,
    kind: EventKind,
    payload: dict,
    chat_id: int | None = None,
) -> int:
    """Persist + post a typed mission event. Returns mission_events.id.

    Flow:
      1. INSERT pre-send so a Telegram failure still leaves a trail.
      2. Render via per-kind formatter.
      3. Send to mission thread (or flat fallback).
      4. UPDATE telegram_message_id with the sent message's id.

    The 'asking' kind has its callback_data rewritten post-INSERT to
    embed the new event_id (placeholder ``__EID__`` is replaced).
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"unknown mission_event kind: {kind!r}")

    payload_json = json.dumps(payload or {})
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO mission_events (mission_id, kind, payload) "
        "VALUES (?, ?, ?)",
        (mission_id, kind, payload_json),
    )
    await db.commit()
    event_id = cur.lastrowid or 0

    text, markup = format_event(kind, payload)

    # Rewrite the 'asking' option callbacks with the real event_id.
    if kind == "asking" and markup is not None:
        new_rows = []
        for row in markup.inline_keyboard:
            new_row = []
            for btn in row:
                data = btn.callback_data
                if data and "__EID__" in data:
                    data = data.replace("__EID__", str(event_id))
                new_row.append(InlineKeyboardButton(btn.text, callback_data=data))
            new_rows.append(new_row)
        markup = InlineKeyboardMarkup(new_rows)

    try:
        msg = await post_to_mission_thread(
            bot, mission_id, text,
            chat_id=chat_id,
            reply_markup=markup,
        )
        msg_id = getattr(msg, "message_id", None)
        if msg_id is None and isinstance(msg, dict):
            msg_id = msg.get("message_id")
        if msg_id is not None:
            await db.execute(
                "UPDATE mission_events SET telegram_message_id = ? "
                "WHERE id = ?",
                (int(msg_id), event_id),
            )
            await db.commit()
    except Exception as e:
        logger.warning(
            "post_event send failed (row kept)",
            event_id=event_id, kind=kind, error=str(e),
        )

    return event_id


async def get_event_by_message_id(message_id: int) -> dict | None:
    """Reverse-lookup for reply-to handler. Returns dict or None."""
    db = await get_db()
    cur = await db.execute(
        "SELECT * FROM mission_events WHERE telegram_message_id = ?",
        (int(message_id),),
    )
    row = await cur.fetchone()
    return dict(row) if row else None


async def resolve_event(
    event_id: int,
    resolution: str,
) -> None:
    """Stamp resolution + resolved_at. ``resolution`` is one of
    approve / reject / comment / answer.
    """
    db = await get_db()
    await db.execute(
        "UPDATE mission_events SET resolution = ?, "
        "resolved_at = CURRENT_TIMESTAMP WHERE id = ?",
        (resolution, int(event_id)),
    )
    await db.commit()
