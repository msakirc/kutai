"""Inject sibling-task results + workspace snapshot into a task's context.

Extracted from `Orchestrator._inject_chain_context` during the Task 13 trim
to keep `orchestrator.py` focused on the dispatch pump.
"""
from __future__ import annotations

import json

from src.app.config import MAX_CONTEXT_CHAIN_LENGTH
from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.tools.workspace import (
    get_file_tree, get_mission_workspace_relative,
)

from .task_context import parse_context, set_context

logger = get_logger("core.context_injection")

# A result truncated by _cap_prior_steps lands at this floor: the first 500
# chars plus the marker suffix. Re-truncating an already-floored string is a
# no-op, so the cap loop can only shrink `total` further by DROPPING steps.
_TRUNC_MARKER = "\n... [heavily truncated]"
_TRUNC_FLOOR = 500 + len(_TRUNC_MARKER)


def _is_raw_dispatch(ctx) -> bool:
    """True if ``ctx`` is a self-contained raw_dispatch LLM-call envelope.

    raw_dispatch tasks (inline grader/reviewer LLM calls, see grading.py) carry
    their full prompt in ``context.llm_call.messages`` — they consume no
    prior_steps. They are also parented to the task being graded, so every such
    child is a sibling of every other; injecting siblings made each grade child
    read all OTHER grade verdicts, the volume that drove the 2026-05-26
    context_injection infinite-loop crash. Used to (a) skip injection for these
    tasks and (b) exclude them from any other task's prior_steps.
    """
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            return False
    return (
        isinstance(ctx, dict)
        and isinstance(ctx.get("llm_call"), dict)
        and bool(ctx["llm_call"].get("raw_dispatch"))
    )


def _cap_prior_steps(prior_steps: list[dict], max_len: int) -> None:
    """Bound the combined length of ``prior_steps`` results to ``max_len``, in place.

    Each iteration makes strict progress: it either truncates the longest
    result (when that result is still above the truncation floor) or, once
    every result is already at the floor, drops the oldest step. This
    guarantees termination.

    The earlier version only ever truncated the longest result to a ~524-char
    floor and never dropped a step. When a task's parent had enough completed
    siblings that ``N * floor > max_len`` (production: 23 reviewer siblings ×
    524 = 12052 > MAX_CONTEXT_CHAIN_LENGTH 12000), re-truncating already-floored
    results never reduced ``total`` below ``max_len`` — an INFINITE LOOP that
    spun the orchestrator's event-loop thread at 100% CPU, starved the Yaşar
    Usta heartbeat, and produced the 2026-05-26 watchdog crash loop (task
    #178354 under parent #166114). The drop arm below is the fix.
    """
    def _total() -> int:
        return sum(len(s["result"]) for s in prior_steps)

    total = _total()
    while total > max_len and prior_steps:
        i = max(range(len(prior_steps)),
                key=lambda x: len(prior_steps[x]["result"]))
        longest = prior_steps[i]["result"]
        if len(longest) > _TRUNC_FLOOR:
            prior_steps[i]["result"] = longest[:500] + _TRUNC_MARKER
        else:
            # Everything is already floored — truncation can no longer shrink
            # the total. Drop the oldest step (list is ordered completed_at
            # ASC) so the loop makes progress and terminates.
            prior_steps.pop(0)
        total = _total()


async def inject_chain_context(task: dict) -> dict:
    """Return a new task dict with sibling results + workspace snapshot injected.

    Reads completed sibling tasks (same parent_task_id) and appends their
    (truncated) results under ``prior_steps``. For code-oriented agents,
    also attaches a short workspace file tree so the agent sees the
    workspace state before acting.
    """
    task_context = parse_context(task)

    # raw_dispatch tasks are self-contained — no chain context to inject, and
    # injecting it is the dangerous path (see _is_raw_dispatch). Return early
    # before touching the DB.
    if _is_raw_dispatch(task_context):
        return task

    parent_id = task.get("parent_task_id")
    prior_steps: list[dict] = []

    if parent_id:
        db = await get_db()
        cursor = await db.execute(
            """SELECT id, title, result, agent_type, status, context
               FROM tasks WHERE parent_task_id = ? AND status = 'completed'
               AND id != ? ORDER BY completed_at ASC""",
            (parent_id, task["id"])
        )
        for sib in [dict(r) for r in await cursor.fetchall()]:
            # Skip raw_dispatch overhead children (inline grade/review LLM
            # calls). They are not meaningful "prior steps" and they pile up
            # unbounded under a heavily-graded parent (one per grade re-run).
            if _is_raw_dispatch(sib.get("context")):
                continue
            rt = sib.get("result", "")
            prior_steps.append({
                "title": sib["title"],
                "agent_type": sib.get("agent_type", "?"),
                "status": sib["status"],
                "result": rt[:1500] + "\n... [truncated]" if len(rt) > 1500 else rt,
            })

    _cap_prior_steps(prior_steps, MAX_CONTEXT_CHAIN_LENGTH)
    if prior_steps:
        task_context["prior_steps"] = prior_steps

    agent_type = task.get("agent_type", "executor")
    mission_id = task.get("mission_id")
    if agent_type in ("coder", "reviewer", "writer", "planner"):
        try:
            tree_path = (
                get_mission_workspace_relative(mission_id) if mission_id else ""
            )
            tree = await get_file_tree(path=tree_path, max_depth=3)
            if tree and "File not found" not in tree and len(tree.split("\n")) > 1:
                task_context["workspace_snapshot"] = tree
                if mission_id:
                    task_context["workspace_path"] = (
                        get_mission_workspace_relative(mission_id)
                    )
        except Exception as e:
            logger.debug(f"workspace snapshot failed: {e}")

    return set_context(task, task_context)
