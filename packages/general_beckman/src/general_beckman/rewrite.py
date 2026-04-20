"""Pure action-rewriting rules. Replaces the old result_guards.py.

Runs between result_router.route_result() and apply._apply_actions().
No I/O — pure transformation of the action list given (task, task_ctx).

Rules (order matters — earlier rules can short-circuit):
  1. Mission-task completion → inject MissionAdvance
  2. Workflow step emitted subtasks → replace with Failed (quality)
  3. Silent task requested clarification → replace with Failed
  4. may_need_clarification=False + clarify request → Failed
  5. Existing clarification_history + clarify request → CompleteWithReusedAnswer
"""
from __future__ import annotations

from typing import Iterable

from general_beckman.result_router import (
    Action, Complete, SpawnSubtasks, RequestClarification,
    Failed, MissionAdvance, CompleteWithReusedAnswer,
)


def _is_workflow_step(task_ctx: dict) -> bool:
    return bool(task_ctx.get("workflow_step") or task_ctx.get("is_workflow_step"))


def _format_history(history: list) -> str:
    parts = []
    for entry in history:
        if isinstance(entry, dict):
            q = entry.get("question", "")
            a = entry.get("answer", "")
        else:
            q, a = "", str(entry)
        if q or a:
            parts.append(f"**Q:** {q}\n**A:** {a}")
    return "\n\n".join(parts)


def rewrite_actions(
    task: dict, task_ctx: dict, actions: Iterable[Action]
) -> list[Action]:
    out: list[Action] = []
    for a in actions:
        out.extend(_rewrite_one(task, task_ctx, a))
    return out


def _rewrite_one(task: dict, task_ctx: dict, a: Action) -> list[Action]:
    # Rule 1: mission-task clean completion → also emit MissionAdvance
    if isinstance(a, Complete) and task.get("mission_id"):
        return [
            a,
            MissionAdvance(
                task_id=a.task_id,
                mission_id=task["mission_id"],
                completed_task_id=a.task_id,
                raw=a.raw,
            ),
        ]
    # Rule 2: workflow step tried to decompose
    if isinstance(a, SpawnSubtasks) and _is_workflow_step(task_ctx):
        return [Failed(
            task_id=a.parent_task_id,
            error="Workflow step tried to decompose instead of producing artifact",
            raw=a.raw,
        )]
    # Rules 3–5: clarification rewrites
    if isinstance(a, RequestClarification):
        if task_ctx.get("silent"):
            return [Failed(
                task_id=a.task_id,
                error="Insufficient info (silent task, no clarification)",
                raw=a.raw,
            )]
        if task_ctx.get("may_need_clarification") is False:
            return [Failed(
                task_id=a.task_id,
                error="Agent requested clarification on no-clarification step",
                raw=a.raw,
            )]
        history = task_ctx.get("clarification_history")
        if history:
            body = _format_history(history) or task_ctx.get("user_clarification", "")
            return [CompleteWithReusedAnswer(
                task_id=a.task_id, result=body, raw=a.raw,
            )]
    return [a]
