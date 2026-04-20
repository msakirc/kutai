"""Inject sibling-task results + workspace snapshot into a task's context.

Extracted from `Orchestrator._inject_chain_context` during the Task 13 trim
to keep `orchestrator.py` focused on the dispatch pump.
"""
from __future__ import annotations

from src.app.config import MAX_CONTEXT_CHAIN_LENGTH
from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.tools.workspace import (
    get_file_tree, get_mission_workspace_relative,
)

from .task_context import parse_context, set_context

logger = get_logger("core.context_injection")


async def inject_chain_context(task: dict) -> dict:
    """Return a new task dict with sibling results + workspace snapshot injected.

    Reads completed sibling tasks (same parent_task_id) and appends their
    (truncated) results under ``prior_steps``. For code-oriented agents,
    also attaches a short workspace file tree so the agent sees the
    workspace state before acting.
    """
    task_context = parse_context(task)
    parent_id = task.get("parent_task_id")
    prior_steps: list[dict] = []

    if parent_id:
        db = await get_db()
        cursor = await db.execute(
            """SELECT id, title, result, agent_type, status
               FROM tasks WHERE parent_task_id = ? AND status = 'completed'
               AND id != ? ORDER BY completed_at ASC""",
            (parent_id, task["id"])
        )
        for sib in [dict(r) for r in await cursor.fetchall()]:
            rt = sib.get("result", "")
            prior_steps.append({
                "title": sib["title"],
                "agent_type": sib.get("agent_type", "?"),
                "status": sib["status"],
                "result": rt[:1500] + "\n... [truncated]" if len(rt) > 1500 else rt,
            })

    total = sum(len(s["result"]) for s in prior_steps)
    while total > MAX_CONTEXT_CHAIN_LENGTH and prior_steps:
        i = max(range(len(prior_steps)),
                key=lambda x: len(prior_steps[x]["result"]))
        prior_steps[i]["result"] = (
            prior_steps[i]["result"][:500] + "\n... [heavily truncated]"
        )
        total = sum(len(s["result"]) for s in prior_steps)
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
