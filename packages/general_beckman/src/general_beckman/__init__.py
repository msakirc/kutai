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


def _system_busy(snap) -> bool:
    """Return True when VRAM is too low to start a new LLM task."""
    if snap is None:
        return False
    try:
        if int(getattr(snap, "vram_available_mb", 0)) < 500:
            return True
    except Exception:
        pass
    return False


async def next_task():
    """Cycle: sweep (throttled) + fire due crons + pick one.

    Called by orchestrator on its ~3s cycle.
    """
    from general_beckman.cron import fire_due
    from general_beckman.queue import pick_ready_task

    # Cron processor internally seeds and throttles sweep.
    await fire_due()

    snap = _capacity_snapshot()
    return await pick_ready_task(_system_busy(snap))


async def on_task_finished(task_id: int, result: dict) -> None:
    """Mark terminal + create any follow-up tasks the result implies.

    Pipeline: route_result -> rewrite_actions -> apply_actions.
    No delegation to Orchestrator. Mission-task completions produce a
    MissionAdvance action which spawns a salako workflow_advance task.
    """
    from general_beckman.result_router import route_result
    from general_beckman.rewrite import rewrite_actions
    from general_beckman.apply import apply_actions
    from general_beckman.task_context import parse_context
    from src.infra.db import get_task
    from src.infra.logging_config import get_logger

    log = get_logger("beckman.on_task_finished")
    task = await get_task(task_id)
    if task is None:
        log.warning("on_task_finished: missing task", task_id=task_id)
        return
    task_ctx = parse_context(task)
    # Workflow-step post-hook runs synchronously before routing — stores
    # artifacts and may flip status (degenerate output, schema validation,
    # disguised failures, human-gate clarifications). Deferring this to
    # the workflow_advance mechanical task caused a race: dependent tasks
    # became ready and picked up empty blackboards before the advance
    # task ran.
    try:
        from src.workflows.engine.hooks import (
            is_workflow_step, post_execute_workflow_step,
        )
        if is_workflow_step(task_ctx):
            await post_execute_workflow_step(task, result)
    except Exception as e:
        log.warning("post_execute_workflow_step raised", task_id=task_id, error=str(e))
    actions = route_result(task, result)
    if actions is None:
        return
    if not isinstance(actions, (list, tuple)):
        actions = [actions]
    actions = rewrite_actions(task, task_ctx, actions)
    await apply_actions(task, actions)


async def enqueue(spec: dict) -> int:
    """Single external write path for user-/bot-initiated tasks."""
    from src.infra.db import add_task
    return await add_task(**spec)

