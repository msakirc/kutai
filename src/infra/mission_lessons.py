"""Shim — mission_lessons relocated to ``kara_kutu`` (durable cross-mission
history; the events/history observability pillar).

Kept as a thin re-export because ``src`` core still imports it
(``src/app/telegram_bot.py`` → ``upsert_mission_lesson``). New code should
import from ``kara_kutu`` directly. The ``src``→``src`` cleanup of this shim
is out of scope (tied to the telegram_bot split).

CLI moved to: ``python -m kara_kutu.mission_lessons emit-dlq``.
"""
from __future__ import annotations

from kara_kutu.mission_lessons import (  # noqa: F401
    emit_lessons_from_dlq_patterns,
    suppress_mission_lesson,
    top_mission_lessons,
    upsert_mission_lesson,
)

__all__ = [
    "emit_lessons_from_dlq_patterns",
    "suppress_mission_lesson",
    "top_mission_lessons",
    "upsert_mission_lesson",
]
