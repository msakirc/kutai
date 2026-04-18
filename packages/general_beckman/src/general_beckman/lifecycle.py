"""Task lifecycle handlers.

Phase 2b intermediate shape: `on_task_finished(task_id, result)` is the drain
entry point invoked by the orchestrator after each dispatched task returns.
It routes through the existing `result_router` + guards + per-action handlers.

During the intermediate Task 9 state, action dispatch delegates to a
registered orchestrator singleton (`set_orchestrator(instance)`) so the heavy
per-handler logic that still lives on `Orchestrator` stays callable. Task 13
will move the handler bodies here and delete the orchestrator stubs.

`handle_clarification` is the one handler that is already rewritten to emit a
mechanical `salako clarify` task instead of calling Telegram directly (per
spec §6).
"""
from __future__ import annotations

from typing import Any

from src.infra.db import add_task, get_task, update_task
from src.infra.logging_config import get_logger

from general_beckman.task_context import parse_context
from general_beckman.result_router import (
    route_result,
    Complete,
    SpawnSubtasks,
    RequestClarification,
    RequestReview,
    Exhausted,
    Failed,
)

logger = get_logger("general_beckman.lifecycle")

_ORCH_INSTANCE: Any = None


def set_orchestrator(instance: Any) -> None:
    """Register the orchestrator singleton that owns the legacy _handle_*
    methods. Called from Orchestrator.__init__."""
    global _ORCH_INSTANCE
    _ORCH_INSTANCE = instance


def get_orchestrator() -> Any:
    if _ORCH_INSTANCE is None:
        raise RuntimeError("orchestrator not registered with general_beckman.lifecycle")
    return _ORCH_INSTANCE


# ────────────────────────────────────────────────────────────────────────
# Standalone handlers — Task 9 keeps these minimal; full extraction happens
# in Task 13 when the orchestrator stubs are deleted.
# ────────────────────────────────────────────────────────────────────────


async def handle_complete(task: dict, result: dict) -> None:
    """Delegate to orchestrator._handle_complete (legacy). Task 13 inlines this."""
    orch = get_orchestrator()
    await orch._handle_complete(task, result)


async def handle_subtasks(task: dict, result: dict) -> None:
    orch = get_orchestrator()
    await orch._handle_subtasks(task, result)


async def handle_clarification(task: dict, result: dict) -> None:
    """Emit a mechanical salako `clarify` task instead of calling Telegram.

    This is the lifecycle handler already rewritten per the new pattern.
    """
    task_id = task["id"]
    question = result.get("clarification") or result.get("question") or "Need more information"
    await update_task(task_id, status="waiting_human")
    await add_task(
        title=f"Clarify: {task.get('title','')[:40]}",
        description=question,
        mission_id=task.get("mission_id"),
        parent_task_id=task_id,
        agent_type="mechanical",
        payload={
            "action": "clarify",
            "question": question,
            "chat_id": task.get("chat_id"),
        },
        depends_on=[],
    )
    logger.info("emitted clarify mechanical task", task_id=task_id)


async def handle_review(task: dict, result: dict) -> None:
    orch = get_orchestrator()
    await orch._handle_review(task, result)


async def handle_exhausted(task: dict, result: dict) -> None:
    orch = get_orchestrator()
    await orch._handle_exhausted(task, result)


async def handle_failed(task: dict, result: dict) -> None:
    orch = get_orchestrator()
    await orch._handle_failed(task, result)


async def _dispatch_action(action, task: dict) -> None:
    if isinstance(action, Complete):
        await handle_complete(task, action.raw)
    elif isinstance(action, SpawnSubtasks):
        await handle_subtasks(task, action.raw)
    elif isinstance(action, RequestClarification):
        await handle_clarification(task, action.raw)
    elif isinstance(action, RequestReview):
        await handle_review(task, action.raw)
    elif isinstance(action, Exhausted):
        await handle_exhausted(task, action.raw)
    elif isinstance(action, Failed):
        await handle_failed(task, action.raw)
    else:
        logger.warning("unknown action type", action=type(action).__name__)


async def on_task_finished(task_id: int, result: dict) -> None:
    """Lifecycle drain entry point.

    Fetches the task row, routes the result through result_router, and
    dispatches the resulting actions. Called by the orchestrator main loop
    after every dispatched task returns.
    """
    task = await get_task(task_id)
    if task is None:
        logger.warning("on_task_finished called for missing task", task_id=task_id)
        return
    # task_ctx is parsed for future guard wiring; not yet consumed.
    parse_context(task)
    actions = route_result(task, result)
    # route_result returns a single Action today; loop defensively in case that
    # changes.
    if actions is None:
        return
    if not isinstance(actions, (list, tuple)):
        actions = [actions]
    for action in actions:
        await _dispatch_action(action, task)
