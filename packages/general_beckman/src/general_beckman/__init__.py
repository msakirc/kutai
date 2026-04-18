"""General Beckman — the task master."""
from __future__ import annotations

from general_beckman.types import Task, AgentResult, Lane
from general_beckman.lifecycle import on_task_finished, set_orchestrator
from general_beckman.queue import pick_ready_task, classify_lane, count_pending_cloud_tasks, unclaim
from general_beckman.lookahead import should_hold_back

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
    if task is None:
        return None
    try:
        queue_depth = await count_pending_cloud_tasks()
    except Exception:
        queue_depth = 0
    if should_hold_back(task, snap, queue_depth):
        await unclaim(task)
        return None
    return task


async def tick() -> None:
    """Periodic maintenance. Called every 3s by the orchestrator main loop.

    Invokes the watchdog and the registered orchestrator's scheduled-jobs
    tick entry points. Each subroutine is guarded: an exception from one
    must not abort the rest.
    """
    from src.infra.logging_config import get_logger
    from general_beckman.watchdog import check_stuck_tasks
    from general_beckman import lifecycle
    log = get_logger("general_beckman.tick")

    async def _safe(coro, name):
        try:
            await coro
        except Exception as e:
            log.warning("tick subroutine failed", fn=name, error=str(e))

    await _safe(check_stuck_tasks(), "check_stuck_tasks")
    try:
        orch = lifecycle.get_orchestrator()
    except RuntimeError:
        orch = None
    sj = getattr(orch, "scheduled_jobs", None) if orch is not None else None
    if sj is None:
        return
    for name in (
        "tick_todos",
        "tick_api_discovery",
        "tick_digest",
        "tick_price_watches",
        "tick_benchmark_refresh",
        "check_scheduled_tasks",
    ):
        fn = getattr(sj, name, None)
        if fn is None:
            continue
        await _safe(fn(), name)
