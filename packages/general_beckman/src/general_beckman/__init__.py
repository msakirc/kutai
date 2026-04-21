"""General Beckman — the task master.

Public API (everything else is internal):
  - next_task() -> Task | None
  - on_task_finished(task_id, result) -> None
  - enqueue(spec) -> int
"""
from __future__ import annotations

from general_beckman.types import Task, AgentResult

__all__ = ["next_task", "on_task_finished", "enqueue", "on_model_swap", "Task", "AgentResult"]


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


async def _currently_dispatched_count() -> int:
    """Count tasks currently being processed (status='processing')."""
    import os
    import aiosqlite
    db_path = os.environ.get("DB_PATH", "kutai.db")
    try:
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'processing'"
            )
            row = await cur.fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


async def _claim_task(task_id: int) -> bool:
    """Claim a single task by id via the existing DB claim primitive."""
    from src.infra.db import claim_task
    return await claim_task(task_id)


async def next_task():
    """Admission loop: pick one ready task whose pool pressure clears its urgency threshold.

    Called by orchestrator on its pump cycle. Iterates top-K ready tasks
    by urgency; for each, asks Fatih Hoca for a Pick, then gates on
    ``snapshot.pressure_for(pick.model) >= threshold(urgency)``. First
    candidate to clear is claimed, tagged with ``preselected_pick``, and
    returned. Non-admitted candidates remain in the queue untouched.
    """
    import os
    from general_beckman import queue as _queue
    from general_beckman.admission import compute_urgency, threshold
    from general_beckman.cron import fire_due
    from general_beckman import posthook_migration

    hard_cap = int(os.environ.get("BECKMAN_HARD_CAP", "4"))
    top_k = int(os.environ.get("BECKMAN_TOP_K", "5"))

    # Hard cap first — cheap DB read.
    if await _currently_dispatched_count() >= hard_cap:
        return None

    await posthook_migration.run()  # one-shot; no-op after first success
    await fire_due()

    # One snapshot per tick.
    try:
        import nerd_herd
        snap = nerd_herd.snapshot()
    except Exception:
        snap = None

    try:
        import fatih_hoca
    except Exception:
        fatih_hoca = None  # type: ignore

    candidates = await _queue.pick_ready_top_k(k=top_k)
    for task in candidates:
        agent_type = task.get("agent_type") or ""
        difficulty = task.get("difficulty", 5)
        pick = None
        if fatih_hoca is not None:
            try:
                pick = fatih_hoca.select(
                    task=agent_type,
                    agent_type=agent_type,
                    difficulty=difficulty,
                )
            except Exception:
                pick = None
        if pick is None:
            continue

        if snap is not None:
            try:
                pressure = snap.pressure_for(pick.model)
            except Exception:
                pressure = 1.0  # fail-open: admit rather than starve
        else:
            pressure = 1.0

        urgency = compute_urgency(task)
        if pressure < threshold(urgency):
            continue

        if not await _claim_task(task["id"]):
            continue

        task["preselected_pick"] = pick
        task["status"] = "processing"
        return task

    return None


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

    # Progress ping: terse per-step notification for workflow-step tasks so
    # the user sees a mission moving forward rather than 2+ minutes of
    # silence. Bookkeeping tasks (mechanical / grader / artifact_summarizer)
    # are skipped — they're internal machinery, not user progress.
    try:
        _bookkeeping = task.get("agent_type") in (
            "mechanical", "grader", "artifact_summarizer",
        )
        if task.get("mission_id") and not _bookkeeping:
            status = (result or {}).get("status", "completed")
            if status in ("completed", "failed", "needs_clarification"):
                await _send_step_progress(task, status, result)
    except Exception as e:
        log.debug("progress ping failed", task_id=task_id, error=str(e))

    try:
        from general_beckman.queue_profile_push import build_and_push
        await build_and_push()
    except Exception as e:
        log.debug("queue_profile push failed", task_id=task_id, error=str(e))


async def _send_step_progress(task: dict, status: str, result: dict) -> None:
    """Send a one-line Telegram progress update when a mission step finishes."""
    if task.get("agent_type") in ("mechanical", "grader", "artifact_summarizer"):
        return
    from src.app.telegram_bot import get_telegram
    tg = get_telegram()
    if tg is None:
        return
    # Title is typically "[1.1] enrich_product_results"; reuse it verbatim.
    title = (task.get("title") or "").strip() or f"task #{task['id']}"
    icon = {"completed": "\u2705", "failed": "\u274c", "needs_clarification": "\u2753"}.get(status, "\u2139\ufe0f")
    msg = f"{icon} {title}"
    if status == "failed":
        err = (result or {}).get("error") or "error"
        msg += f"\n  {str(err)[:140]}"
    await tg.send_notification(msg)


async def enqueue(spec: dict) -> int:
    """Single external write path for user-/bot-initiated tasks."""
    from src.infra.db import add_task
    from general_beckman.queue_profile_push import build_and_push
    task_id = await add_task(**spec)
    await build_and_push()
    return task_id


async def on_model_swap(old_model: str | None, new_model: str | None) -> None:
    """Called by the local model manager when a model swap completes.

    Wakes tasks whose retries were delayed waiting for *any* model to
    load. Grading is no longer triggered here — it's a regular task
    flowing through next_task().
    """
    try:
        from src.infra.db import accelerate_retries
        await accelerate_retries("model_swap")
    except Exception as e:
        from src.infra.logging_config import get_logger
        get_logger("beckman.on_model_swap").debug(
            f"accelerate_retries failed: {e}",
        )

