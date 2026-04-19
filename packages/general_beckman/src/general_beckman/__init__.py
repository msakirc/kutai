"""General Beckman — the task master.

Public API (everything else is internal):
  - next_task() -> Task | None
  - on_task_finished(task_id, result) -> None
  - enqueue(spec) -> int
"""
from __future__ import annotations

from general_beckman.types import Task, AgentResult

__all__ = ["next_task", "on_task_finished", "enqueue", "Task", "AgentResult"]


def _capacity_snapshot():
    """Best-effort capacity snapshot. Returns None if nerd_herd isn't wired."""
    try:
        import nerd_herd
        nh = getattr(nerd_herd, "_singleton", None)
        if nh is None:
            return None
        return nh.snapshot()
    except Exception:
        return None


def _saturated_lanes(snap) -> set[str]:
    """Transitional: kept until Task 4 deletes lanes entirely."""
    saturated: set[str] = set()
    if snap is None:
        return saturated
    try:
        if int(getattr(snap, "vram_available_mb", 0)) < 500 and \
           getattr(snap, "local", None) is not None:
            saturated.add("local_llm")
    except Exception:
        pass
    return saturated


async def next_task():
    """Cycle: sweep (throttled) + fire due crons + pick one.

    Called by orchestrator on its ~3s cycle.
    """
    from general_beckman.cron import fire_due
    from general_beckman.queue import pick_ready_task

    # Cron processor internally seeds and throttles sweep.
    await fire_due()

    snap = _capacity_snapshot()
    saturated = _saturated_lanes(snap)
    return await pick_ready_task(saturated)


async def on_task_finished(task_id: int, result: dict) -> None:
    # Kept as-is for now — Task 6 rewrites this to use rewrite+apply.
    from general_beckman.lifecycle import on_task_finished as _legacy
    await _legacy(task_id, result)


async def enqueue(spec: dict) -> int:
    """Single external write path for user-/bot-initiated tasks."""
    from src.infra.db import add_task
    return await add_task(**spec)


from general_beckman.lifecycle import set_orchestrator  # noqa: F401, transitional
