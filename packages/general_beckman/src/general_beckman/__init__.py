"""General Beckman — the task master."""
from __future__ import annotations

from general_beckman.types import Task, AgentResult, Lane
from general_beckman.lifecycle import on_task_finished, set_orchestrator
from general_beckman.queue import pick_ready_task, classify_lane

__all__ = [
    "next_task", "on_task_finished", "tick", "set_orchestrator",
    "Task", "AgentResult", "Lane",
]


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
    saturated: set[str] = set()
    if snap is None:
        return saturated
    # Local LLM is saturated when VRAM headroom drops below a safe floor.
    try:
        if int(getattr(snap, "vram_available_mb", 0)) < 500 and getattr(snap, "local", None) is not None:
            saturated.add("local_llm")
    except Exception:
        pass
    return saturated


async def next_task() -> Task | None:
    """Return one task ready to dispatch, or None if nothing should be released.

    Consults :func:`nerd_herd.snapshot` (if wired) for lane saturation, then
    claims the first eligible task via :mod:`general_beckman.queue`.
    """
    snap = _capacity_snapshot()
    saturated = _saturated_lanes(snap)
    task = await pick_ready_task(saturated)
    return task


async def tick() -> None:
    """Stub — filled in by Task 12."""
    return None
