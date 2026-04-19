"""Workflow engine: advance one mission by consuming a completed step's result.

Delegates to src/workflows/engine primitives until/unless they are migrated
wholesale into this package. Minimal surface: one function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdvanceResult:
    status: str = "completed"   # 'completed' | 'needs_clarification' | 'failed'
    error: str = ""
    next_subtasks: list[dict] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


async def advance(mission_id: int, completed_task_id: int,
                  previous_result: dict) -> AdvanceResult:
    """Post-step hook + artifact capture + next-phase subtask emission."""
    from src.workflows.engine.hooks import (
        is_workflow_step, post_execute_workflow_step, get_artifact_store,
    )
    from src.workflows.engine.pipeline_artifacts import extract_pipeline_artifacts
    from src.tools.workspace import get_mission_workspace
    from src.infra.db import get_task

    out = AdvanceResult()
    task = await get_task(completed_task_id)
    if task is None:
        out.status = "failed"
        out.error = f"completed_task_id {completed_task_id} not found"
        return out
    task_ctx = _parse_ctx(task)
    if not is_workflow_step(task_ctx):
        # Not a workflow step; nothing to advance. Callers should guard,
        # but we defend here too.
        return out

    # 1. Artifact capture (from guard_pipeline_artifacts).
    try:
        ws = None
        if task.get("mission_id"):
            try:
                ws = get_mission_workspace(task["mission_id"])
            except Exception:
                ws = None
        extra = await extract_pipeline_artifacts(task, previous_result, ws)
        if extra:
            store = get_artifact_store()
            for name, content in extra.items():
                await store.store(mission_id, name, content)
            out.artifacts = dict(extra)
    except Exception:
        pass

    # 2. Post-hook: may flip status.
    try:
        await post_execute_workflow_step(task, previous_result)
    except Exception as e:
        out.status = "failed"
        out.error = str(e)[:300]
        return out

    flipped = previous_result.get("status")
    if flipped == "needs_clarification":
        out.status = "needs_clarification"
        out.error = previous_result.get("question", "")
        return out
    if flipped == "failed":
        out.status = "failed"
        out.error = previous_result.get("error", "Post-hook failed")
        return out

    # 3. Next-phase subtasks (if engine emits them).
    try:
        from src.workflows.engine.recipe import advance_recipe
        next_subs = await advance_recipe(mission_id, completed_task_id,
                                         previous_result)
        out.next_subtasks = list(next_subs or [])
    except ImportError:
        # No recipe-advance primitive yet — no-op. Phase transition logic
        # stays in _handle_complete until migrated.
        pass
    except Exception as e:
        out.status = "failed"
        out.error = f"advance_recipe: {e}"[:300]
    return out


def _parse_ctx(task: dict) -> dict:
    import json
    raw = task.get("context") or "{}"
    if isinstance(raw, dict):
        return dict(raw)
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}
