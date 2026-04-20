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

    # 4. Mission completion — if no next subtasks queued AND every other
    # mission task is terminal, mark the mission done. Replaces
    # _check_mission_completion that used to live in the old orchestrator's
    # _handle_complete (deleted in Task 13).
    if not out.next_subtasks and out.status == "completed":
        try:
            await _maybe_complete_mission(mission_id, completed_task_id)
        except Exception as e:
            # Non-fatal: mission stays open, but advance result stands.
            import logging
            logging.getLogger("workflow_engine.advance").warning(
                "mission completion check failed: %s", e,
            )
    return out


async def _maybe_complete_mission(mission_id: int, completed_task_id: int) -> None:
    """Mark a mission 'completed' when no non-terminal tasks remain.

    Excludes the workflow_advance task currently running (caller's
    completed_task_id points at the source task, not the advance task, so
    we also skip any pending workflow_advance mechanical rows on the same
    mission — they are bookkeeping, not work.)
    """
    from src.infra.db import get_tasks_for_mission, update_mission, get_mission
    from src.infra.times import db_now
    import json as _json

    mission = await get_mission(mission_id)
    if mission is None:
        return
    if mission.get("status") in ("completed", "cancelled", "failed"):
        return

    tasks = await get_tasks_for_mission(mission_id)
    terminal = {"completed", "skipped", "cancelled", "failed"}
    for t in tasks:
        status = t.get("status")
        if status in terminal:
            continue
        # Skip pending/in_progress workflow_advance rows — they are the
        # machinery that drove us here, not pending work.
        ctx_raw = t.get("context") or "{}"
        try:
            ctx = _json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except Exception:
            ctx = {}
        payload_action = (ctx.get("payload") or {}).get("action") if isinstance(ctx, dict) else None
        if payload_action == "workflow_advance":
            continue
        # Some non-terminal, non-advance task still pending — mission not done.
        return

    await update_mission(mission_id, status="completed", completed_at=db_now())

    # Best-effort Telegram notification — get_telegram() and send are both
    # optional; failing here must not break advance().
    try:
        from src.app.telegram_bot import get_telegram
        tg = get_telegram()
        title = mission.get("title") or f"mission #{mission_id}"
        n_completed = sum(1 for t in tasks if t.get("status") == "completed")
        n_failed = sum(1 for t in tasks if t.get("status") == "failed")
        msg = f"\u2705 Mission #{mission_id} complete\n**{title[:80]}**\n{n_completed} done, {n_failed} failed"
        from src.app.config import TELEGRAM_ADMIN_CHAT_ID
        if tg and TELEGRAM_ADMIN_CHAT_ID:
            await tg.send_message(TELEGRAM_ADMIN_CHAT_ID, msg)
    except Exception:
        pass


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
