"""Per-task progress heartbeat.

Agents call :func:`bump` at iteration boundaries (or other natural
progress points). The orchestrator's dispatch watchdog reads
:func:`stale_seconds` to decide whether a running task has wedged.

This replaces the per-agent-type wall-clock dispatch timeout. A long
but progressing task no longer gets killed; a hung task no longer hides
behind a still-running budget.

Async contextvar carries the active task_id through the call stack so
inner layers (dispatcher, hallederiz_kadir streaming) can bump without
plumbing the id explicitly. The orchestrator sets the var on dispatch
and clears it after.
"""
from __future__ import annotations

import time
from contextvars import ContextVar

current_task_id: ContextVar[int | None] = ContextVar("current_task_id", default=None)

_HEARTBEATS: dict[int, float] = {}

# How long without a progress signal before the dispatch watchdog
# considers the task stuck. Tunable; doctrine says "few minutes is
# usually enough" — set generous default, override per-task only when
# justified.
PROGRESS_TIMEOUT_SECONDS: float = 300.0  # 5 minutes


def bump(task_id: int | str | None = None) -> None:
    """Record a progress event for ``task_id``.

    If ``task_id`` is omitted, falls back to the contextvar set by the
    orchestrator at dispatch time. Inner layers (dispatcher, streaming
    accumulator) call ``bump()`` with no arg.
    """
    if task_id is None:
        task_id = current_task_id.get()
    if task_id is None or task_id == "?":
        return
    try:
        key = int(task_id)
    except (TypeError, ValueError):
        return
    _HEARTBEATS[key] = time.monotonic()


def stale_seconds(task_id: int) -> float:
    """Seconds since the last :func:`bump` for ``task_id``.

    Returns 0.0 if no heartbeat has been recorded yet (treat as
    just-started, not yet stuck).
    """
    last = _HEARTBEATS.get(task_id)
    if last is None:
        return 0.0
    return time.monotonic() - last


def clear(task_id: int | str | None) -> None:
    """Drop the heartbeat entry — call when the dispatch ends."""
    if task_id is None:
        return
    try:
        key = int(task_id)
    except (TypeError, ValueError):
        return
    _HEARTBEATS.pop(key, None)
