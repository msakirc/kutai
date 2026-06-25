"""kara_kutu — flight recorder.

Durable, append-only execution history. Two pillars of agent observability that
nerd_herd (live system metrics) and yazbunu (log lines) do not cover:

  * audit   — who did what, when (forensic trail, append-only events)
  * tracing — what a single task did, step by step, at what cost (per-task trace)

Both write to SQLite via dabidabi and expose human-facing markdown formatters.
"""
from __future__ import annotations

from .audit import (
    audit,
    get_audit_log,
    format_audit_log,
    ACTOR_AGENT,
    ACTOR_SYSTEM,
    ACTOR_HUMAN,
    ACTION_TOOL_EXEC,
    ACTION_MODEL_CALL,
    ACTION_STATE_CHANGE,
    ACTION_FILE_MODIFY,
    ACTION_HUMAN_APPROVE,
    ACTION_MISSION_CREATE,
    ACTION_MISSION_COMPLETE,
    ACTION_TASK_CREATE,
    ACTION_TASK_COMPLETE,
)
from .tracing import append_trace, get_trace, format_trace
from .admission_forensics import record_admission_violation
from .mission_lessons import (
    upsert_mission_lesson,
    top_mission_lessons,
    suppress_mission_lesson,
    emit_lessons_from_dlq_patterns,
)

__all__ = [
    # audit
    "audit",
    "get_audit_log",
    "format_audit_log",
    "ACTOR_AGENT",
    "ACTOR_SYSTEM",
    "ACTOR_HUMAN",
    "ACTION_TOOL_EXEC",
    "ACTION_MODEL_CALL",
    "ACTION_STATE_CHANGE",
    "ACTION_FILE_MODIFY",
    "ACTION_HUMAN_APPROVE",
    "ACTION_MISSION_CREATE",
    "ACTION_MISSION_COMPLETE",
    "ACTION_TASK_CREATE",
    "ACTION_TASK_COMPLETE",
    # tracing
    "append_trace",
    "get_trace",
    "format_trace",
    # admission forensics
    "record_admission_violation",
    # mission lessons
    "upsert_mission_lesson",
    "top_mission_lessons",
    "suppress_mission_lesson",
    "emit_lessons_from_dlq_patterns",
]
