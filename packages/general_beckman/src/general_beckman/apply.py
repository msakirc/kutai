"""Apply Beckman actions to the DB. One branch per action type.

Every function returns None. Side-effects: insert rows, update task status.
Retry / DLQ decisions come from `general_beckman.retry`. Clarify and notify
tasks are created as mechanical salako rows — salako executors do the
actual Telegram I/O at dispatch time.

NOTE: The tasks table has no 'payload' column. Mechanical task payloads are
stored in the 'context' JSON column with the shape:

    {"executor": "mechanical", "payload": {"action": <name>, **kwargs}}

The orchestrator's `_dispatch` copies `ctx["payload"]` onto `task["payload"]`
before calling `salako.run`, which routes on `payload["action"]`. Use
``_mechanical_context(action, **kwargs)`` to build this consistently.
"""
from __future__ import annotations

import json
from datetime import timedelta
from typing import Iterable

from src.infra.logging_config import get_logger
from src.infra.times import to_db, utc_now

from general_beckman.result_router import (
    Action, Complete, SpawnSubtasks, RequestClarification, RequestReview,
    Exhausted, Failed, MissionAdvance, CompleteWithReusedAnswer,
    RequestPostHook, PostHookVerdict,
)
from general_beckman.retry import decide_retry, DLQAction, RetryDecision


def _mechanical_context(action: str, **payload_fields) -> dict:
    """Build the canonical context shape for a mechanical salako task.

    The workflow engine's `expand_steps_to_tasks` emits the same shape
    (see tests/workflows/test_mechanical_step_materializes_with_executor_tag.py).
    """
    return {
        "executor": "mechanical",
        "payload": {"action": action, **payload_fields},
    }

logger = get_logger("beckman.apply")


async def apply_actions(task: dict, actions: Iterable[Action]) -> None:
    for a in actions:
        await _apply_one(task, a)


async def _apply_one(task: dict, a: Action) -> None:
    if isinstance(a, (Complete, CompleteWithReusedAnswer)):
        await _apply_complete(task, a)
    elif isinstance(a, SpawnSubtasks):
        await _apply_subtasks(task, a)
    elif isinstance(a, RequestClarification):
        await _apply_clarify(task, a)
    elif isinstance(a, RequestReview):
        await _apply_review(task, a)
    elif isinstance(a, Exhausted):
        await _apply_exhausted(task, a)
    elif isinstance(a, Failed):
        await _apply_failed(task, a)
    elif isinstance(a, MissionAdvance):
        await _apply_mission_advance(task, a)
    elif isinstance(a, RequestPostHook):
        await _apply_request_posthook(task, a)
    elif isinstance(a, PostHookVerdict):
        await _apply_posthook_verdict(task, a)
    else:
        logger.warning("unknown action type", action=type(a).__name__)


async def _apply_complete(task: dict, a) -> None:
    """Handles both Complete and CompleteWithReusedAnswer."""
    from src.infra.db import update_task
    await update_task(
        a.task_id, status="completed",
        completed_at=to_db(utc_now()),
        result=a.result,
    )


async def _apply_subtasks(task: dict, a: SpawnSubtasks) -> None:
    from src.infra.db import add_task, update_task
    for sub in a.subtasks:
        await add_task(
            title=sub.get("title", ""),
            description=sub.get("description", ""),
            agent_type=sub.get("agent_type", "coder"),
            parent_task_id=a.parent_task_id,
            mission_id=task.get("mission_id"),
            depends_on=sub.get("depends_on", []),
            context=sub.get("context", {}),
            priority=sub.get("priority", task.get("priority", 5)),
        )
    await update_task(a.parent_task_id, status="waiting_subtasks")


async def _apply_clarify(task: dict, a: RequestClarification) -> None:
    from src.infra.db import add_task, update_task
    await update_task(a.task_id, status="waiting_human")
    await add_task(
        title=f"Clarify: {task.get('title','')[:40]}",
        description=a.question,
        mission_id=task.get("mission_id"),
        parent_task_id=a.task_id,
        agent_type="mechanical",
        context=_mechanical_context(
            "clarify",
            question=a.question,
            chat_id=a.chat_id,
        ),
        depends_on=[],
    )


async def _apply_review(task: dict, a: RequestReview) -> None:
    from src.infra.db import add_task, get_db
    # Dedup: if a review task already exists for this parent, skip.
    conn = await get_db()
    cursor = await conn.execute(
        """SELECT id FROM tasks
           WHERE parent_task_id = ? AND agent_type = 'reviewer'
             AND status IN ('pending', 'processing', 'ungraded')""",
        (a.task_id,),
    )
    if await cursor.fetchone():
        logger.info("review task deduped", parent=a.task_id)
        return
    await add_task(
        title=f"Review: {task.get('title','')[:40]}",
        description=a.summary,
        mission_id=task.get("mission_id"),
        parent_task_id=a.task_id,
        agent_type="reviewer",
        depends_on=[],
    )


async def _apply_exhausted(task: dict, a: Exhausted) -> None:
    await _retry_or_dlq(task, category="exhausted", error=a.error)


async def _apply_failed(task: dict, a: Failed) -> None:
    await _retry_or_dlq(task, category=task.get("error_category") or "worker",
                        error=a.error)


async def _apply_mission_advance(task: dict, a: MissionAdvance) -> None:
    from src.infra.db import add_task
    # Pass the completed task's result dict (a.raw) through to the
    # workflow_advance executor — post_execute_workflow_step needs it
    # to extract output_value and persist artifacts. Omitting it made
    # every mission step produce an empty artifact downstream.
    await add_task(
        title=f"Workflow advance: mission #{a.mission_id}",
        description="",
        agent_type="mechanical",
        mission_id=a.mission_id,
        depends_on=[],
        context=_mechanical_context(
            "workflow_advance",
            mission_id=a.mission_id,
            completed_task_id=a.completed_task_id,
            previous_result=a.raw or {},
        ),
    )


async def _retry_or_dlq(task: dict, *, category: str, error: str) -> None:
    """Shared retry/DLQ path for Failed and Exhausted."""
    from src.infra.db import update_task
    attempts = int(task.get("worker_attempts") or 0) + 1
    max_attempts = int(task.get("max_worker_attempts") or 3)
    progress = _parse_progress(task)
    ctx = _parse_ctx(task)
    bonus_count = int(ctx.get("_bonus_count", 0))

    decision = decide_retry(
        {
            "category": category,
            "worker_attempts": attempts,
            "max_worker_attempts": max_attempts,
            "model": task.get("model", ""),
            "error": error,
        },
        progress=progress,
        bonus_count=bonus_count,
    )

    if isinstance(decision, DLQAction):
        await _dlq_write(task, error=error, category=category, attempts=attempts)
        return

    if decision.bonus_used:
        ctx["_bonus_count"] = bonus_count + 1
        max_attempts += 1

    next_retry_at = None
    if decision.action == "delayed":
        next_retry_at = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))

    await update_task(
        task["id"],
        status="pending",
        error=error[:500],
        worker_attempts=attempts,
        max_worker_attempts=max_attempts,
        error_category=category,
        next_retry_at=next_retry_at,
        context=json.dumps(ctx),
    )


async def _dlq_write(task: dict, *, error: str, category: str, attempts: int) -> None:
    from src.infra.db import add_task, update_task
    from src.infra.dead_letter import quarantine_task
    await update_task(
        task["id"], status="failed",
        error=error[:500],
        failed_in_phase=task.get("failed_in_phase") or "worker",
    )
    try:
        await quarantine_task(
            task_id=task["id"],
            mission_id=task.get("mission_id"),
            error=error[:500],
            error_category=category,
            original_agent=task.get("agent_type", "executor"),
            attempts_snapshot=attempts,
        )
    except Exception as exc:
        logger.warning("DLQ write failed", task_id=task["id"], error=str(exc))
    # Telegram DLQ notification → mechanical salako task (no inline send).
    await add_task(
        title=f"Notify: DLQ task #{task['id']}",
        description="",
        agent_type="mechanical",
        mission_id=task.get("mission_id"),
        context=_mechanical_context(
            "notify_user",
            message=(
                f"\u274c Task #{task['id']} \u2192 DLQ\n"
                f"**{(task.get('title') or '')[:60]}**\n"
                f"Reason: {error[:100]}"
            ),
        ),
        depends_on=[],
    )


async def _apply_request_posthook(task: dict, a: RequestPostHook) -> None:
    """Park the source in `ungraded`, enqueue a post-hook task row."""
    import json as _json
    from src.infra.db import add_task, get_task, update_task

    source = await get_task(a.source_task_id)
    if source is None:
        logger.warning("posthook: source missing", source_id=a.source_task_id)
        return

    ctx = _parse_ctx(source)
    pending = list(ctx.get("_pending_posthooks") or [])
    if a.kind not in pending:
        pending.append(a.kind)
    ctx["_pending_posthooks"] = pending

    await update_task(
        a.source_task_id,
        status="ungraded",
        context=_json.dumps(ctx),
    )

    agent_type, payload = _posthook_agent_and_payload(a, source, ctx)
    await add_task(
        title=_posthook_title(a, source),
        description="",
        agent_type=agent_type,
        mission_id=source.get("mission_id"),
        depends_on=[],
        context=payload,
    )


def _posthook_agent_and_payload(
    a: RequestPostHook, source: dict, source_ctx: dict,
) -> tuple[str, dict]:
    if a.kind == "grade":
        return ("grader", {
            "source_task_id": a.source_task_id,
            "generating_model": source_ctx.get("generating_model", ""),
            "excluded_models": list(source_ctx.get("grade_excluded_models") or []),
        })
    if a.kind.startswith("summary:"):
        artifact_name = a.kind.split(":", 1)[1]
        return ("artifact_summarizer", {
            "source_task_id": a.source_task_id,
            "artifact_name": artifact_name,
        })
    raise ValueError(f"unknown posthook kind: {a.kind!r}")


def _posthook_title(a: RequestPostHook, source: dict) -> str:
    if a.kind == "grade":
        return f"Grade task #{a.source_task_id}"
    if a.kind.startswith("summary:"):
        name = a.kind.split(":", 1)[1]
        return f"Summarize '{name}' for #{a.source_task_id}"
    return f"Posthook {a.kind} for #{a.source_task_id}"


async def _apply_posthook_verdict(task: dict, a: PostHookVerdict) -> None:
    """Placeholder; implemented in Task 5."""
    raise NotImplementedError("_apply_posthook_verdict lands in Task 5")


def _parse_ctx(task: dict) -> dict:
    raw = task.get("context") or "{}"
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
        # Handle double-encoded context (string stored as JSON string).
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _parse_progress(task: dict) -> float | None:
    ctx = _parse_ctx(task)
    p = ctx.get("_last_progress")
    if isinstance(p, (int, float)):
        return float(p)
    return None
