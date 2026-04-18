"""Pre-handler guards: code that sits between `route_result` and `_handle_*`.

Each guard is an `async def guard_<name>(self, ...)` taking the orchestrator
instance as its first argument (so it can reach `self.telegram`,
`self._validate_clarification`, etc.) plus the task/result context it needs.

Each guard returns either:
- `None` — orchestrator should continue (fall through to the handler)
- `GuardHandled(reason)` — orchestrator should stop processing this task now
  (DB side-effects were already applied by the guard)

Phase 1: verbatim port of the inline guard code that used to sit in the
giant `if status == "completed" / ungraded / needs_subtasks / needs_clarification`
chain inside `process_task`.  Phase 2b will clean up the self-coupling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from ..infra.db import get_task, update_task
from ..infra.logging_config import get_logger
from ..infra.times import to_db, utc_now
from ..tools.workspace import get_mission_workspace

logger = get_logger("core.result_guards")


@dataclass(frozen=True)
class GuardHandled:
    """Marker: guard fully handled this task — caller must return now."""
    reason: str


GuardOutcome = Optional[GuardHandled]


# ─── Helpers ────────────────────────────────────────────────────────────

async def _refresh_task_ctx_from_db(task_id: int, task_ctx: dict) -> None:
    """Re-read context from DB so post-hook side-effects land in task_ctx."""
    try:
        _fresh = await get_task(task_id)
        if _fresh:
            _fc = _fresh.get("context", "{}")
            if isinstance(_fc, str):
                _fc = json.loads(_fc)
            if isinstance(_fc, dict):
                task_ctx.update(_fc)
    except Exception:
        pass


async def _quality_retry_flow(
    self, task: dict, task_ctx: dict, result: dict, error_msg: str
) -> GuardHandled:
    """Shared quality-retry path used by ungraded/completed post-hook guards.
    Applies bonus-attempt logic, persists the retry, and sends DLQ notification.
    """
    task_id = task["id"]
    from src.core.retry import RetryContext
    retry_ctx = RetryContext.from_task(task)
    decision = retry_ctx.record_failure("quality", model=result.get("model", ""))

    # Bonus attempt for quality failures with real progress
    _MAX_BONUS = 2
    if decision.action == "terminal":
        bonus_count = task_ctx.get("_bonus_count", 0)
        if bonus_count < _MAX_BONUS:
            try:
                progress = await self._assess_timeout_progress(task_id, task_ctx)
                if progress >= 0.5:
                    task_ctx["_bonus_count"] = bonus_count + 1
                    retry_ctx.max_worker_attempts += 1
                    decision = retry_ctx.record_failure(
                        "quality", model=result.get("model", "")
                    )
                    logger.info(
                        f"[Task #{task_id}] Quality bonus attempt "
                        f"({bonus_count + 1}/{_MAX_BONUS}, "
                        f"progress={progress:.0%})"
                    )
            except Exception:
                pass

    if decision.action == "terminal":
        task_ctx.update(retry_ctx.to_context_patch())
        await update_task(
            task_id, status="failed",
            error=f"Disguised failure exhausted: {error_msg[:300]}",
            context=json.dumps(task_ctx),
            **retry_ctx.to_db_fields(),
        )
        try:
            from src.infra.dead_letter import quarantine_task
            await quarantine_task(
                task_id=task_id,
                mission_id=task.get("mission_id"),
                error=f"Disguised failure after {retry_ctx.worker_attempts} attempts: {error_msg[:300]}",
                error_category="quality",
                original_agent=task.get("agent_type", "executor"),
                attempts_snapshot=retry_ctx.worker_attempts,
            )
        except Exception:
            pass
        await self.telegram.send_notification(
            f"❌ Task #{task_id} disguised failure → DLQ\n"
            f"**{task.get('title', '')[:60]}**\n"
            f"Reason: {error_msg[:100]}"
        )
        return GuardHandled("quality_terminal")

    next_retry = None
    if decision.action == "delayed":
        next_retry = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))
    retry_ctx.next_retry_at = next_retry
    task_ctx.update(retry_ctx.to_context_patch())
    # Inject previous output so next attempt can continue
    result_text = result.get("result", "")
    if result_text:
        task_ctx["_prev_output"] = str(result_text)[:6000]
        task_ctx["_retry_hint"] = (
            "Your previous attempt's output failed quality checks. "
            "Your partial work is shown in context. Build on it — "
            "do NOT start over."
        )
    await update_task(
        task_id, status="pending",
        error=error_msg[:500],
        context=json.dumps(task_ctx),
        **retry_ctx.to_db_fields(),
    )
    return GuardHandled("quality_retry")


# ─── Guards ─────────────────────────────────────────────────────────────

async def guard_pipeline_artifacts(
    self, task: dict, task_ctx: dict, result: dict, agent_type: str
) -> GuardOutcome:
    """Extract pipeline artifacts for workflow-step pipeline tasks."""
    from ..workflows.engine.hooks import is_workflow_step
    if agent_type != "pipeline" or not is_workflow_step(task_ctx):
        return None
    task_id = task["id"]
    try:
        from ..workflows.engine.pipeline_artifacts import extract_pipeline_artifacts
        from ..workflows.engine.hooks import get_artifact_store

        ws_path = None
        if task.get("mission_id"):
            try:
                ws_path = get_mission_workspace(task["mission_id"])
            except Exception:
                pass

        extra_artifacts = await extract_pipeline_artifacts(task, result, ws_path)
        if extra_artifacts:
            store = get_artifact_store()
            mission_id = task.get("mission_id")
            for name, content in extra_artifacts.items():
                await store.store(mission_id, name, content)
            logger.info(f"[Task #{task_id}] Stored {len(extra_artifacts)} pipeline artifacts")
    except Exception as e:
        logger.debug(f"[Task #{task_id}] Pipeline artifact extraction failed: {e}")
    return None


async def guard_workflow_step_post_hook(
    self, task: dict, task_ctx: dict, result: dict
) -> GuardOutcome:
    """Run workflow step post-hook; handle post-hook status flips
    (needs_clarification, failed).  Returns GuardHandled if the post-hook's
    status flip fully consumed the task.
    """
    from ..workflows.engine.hooks import is_workflow_step, post_execute_workflow_step
    if not is_workflow_step(task_ctx):
        return None
    task_id = task["id"]
    await post_execute_workflow_step(task, result)
    await _refresh_task_ctx_from_db(task_id, task_ctx)

    # Post-hook may override status
    if result.get("status") == "needs_clarification":
        if not await self._validate_clarification(task_id, task, task_ctx, result):
            # Validation failed — flip to failed; caller will see quality path.
            result["status"] = "failed"
        else:
            await self._handle_clarification(task, result)
            return GuardHandled("clarification_after_post_hook")

    if result.get("status") == "failed":
        error_msg = result.get("error", "Disguised failure detected")
        return await _quality_retry_flow(self, task, task_ctx, result, error_msg)

    return None


async def guard_ungraded_post_hook(
    self, task: dict, task_ctx: dict, result: dict
) -> GuardOutcome:
    """For ungraded results: run post-hook and handle post-hook status flips.
    Returns GuardHandled if post-hook consumed the task; None → caller stores
    ungraded result.
    """
    from ..workflows.engine.hooks import is_workflow_step, post_execute_workflow_step
    if not is_workflow_step(task_ctx):
        return None
    task_id = task["id"]
    await post_execute_workflow_step(task, result)
    await _refresh_task_ctx_from_db(task_id, task_ctx)

    if result.get("status") == "needs_clarification":
        if not await self._validate_clarification(task_id, task, task_ctx, result):
            result["status"] = "failed"
        else:
            await self._handle_clarification(task, result)
            return GuardHandled("clarification_after_ungraded")

    if result.get("status") == "failed":
        error_msg = result.get("error", "Disguised failure detected")
        return await _quality_retry_flow(self, task, task_ctx, result, error_msg)

    return None


async def guard_subtasks_blocked_for_workflow(
    self, task: dict, task_ctx: dict, result: dict
) -> GuardOutcome:
    """Workflow steps must not decompose into subtasks — treat as quality failure."""
    from ..workflows.engine.hooks import is_workflow_step
    if not is_workflow_step(task_ctx):
        return None
    task_id = task["id"]
    logger.warning(
        f"[Task #{task_id}] Blocked subtask creation for "
        f"workflow step — retrying"
    )
    from src.core.retry import RetryContext
    retry_ctx = RetryContext.from_task(task)
    retry_ctx.record_failure("quality", model=result.get("model", ""))
    task_ctx.update(retry_ctx.to_context_patch())
    await update_task(
        task_id, status="pending",
        error="Workflow step tried to decompose instead of producing artifact",
        context=json.dumps(task_ctx),
        **retry_ctx.to_db_fields(),
    )
    return GuardHandled("subtasks_blocked_for_workflow")


async def guard_clarification_suppression(
    self, task: dict, task_ctx: dict, result: dict
) -> GuardOutcome:
    """Suppress clarification for silent tasks / no-clarification workflow
    steps / tasks with existing clarification history.  Also validates
    schema for workflow clarifications.
    """
    from ..workflows.engine.hooks import is_workflow_step, post_execute_workflow_step
    task_id = task["id"]

    if task_ctx.get("silent"):
        logger.info(f"[Task #{task_id}] Suppressed clarification (silent task)")
        await update_task(
            task_id, status="failed",
            error="Insufficient info (silent task, no clarification)",
            failed_in_phase="worker",
        )
        return GuardHandled("clarification_silent")

    if task_ctx.get("may_need_clarification") is False:
        logger.warning(
            f"[Task #{task_id}] Suppressed clarification "
            f"(may_need_clarification=false), retrying"
        )
        from src.core.retry import RetryContext
        retry_ctx = RetryContext.from_task(task)
        decision = retry_ctx.record_failure("quality")
        if decision.action == "terminal":
            await update_task(
                task_id, status="failed",
                error="Agent requested clarification on no-clarification step",
                **retry_ctx.to_db_fields(),
            )
        else:
            next_retry = None
            if decision.action == "delayed":
                next_retry = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))
            retry_ctx.next_retry_at = next_retry
            await update_task(
                task_id, status="pending",
                error="Suppressed clarification, retrying",
                **retry_ctx.to_db_fields(),
            )
        return GuardHandled("clarification_not_allowed")

    if task_ctx.get("clarification_history"):
        logger.info(
            f"[Task #{task_id}] Suppressed repeat clarification "
            f"(clarification_history already exists)"
        )
        history = task_ctx["clarification_history"]
        qa_parts = []
        for entry in history:
            if isinstance(entry, dict):
                q = entry.get("question", "")
                a = entry.get("answer", "")
            else:
                q, a = "", str(entry)
            if q or a:
                qa_parts.append(f"**Q:** {q}\n**A:** {a}")
        qa_result = "\n\n".join(qa_parts) if qa_parts else task_ctx.get("user_clarification", "")
        result["status"] = "completed"
        result["result"] = qa_result or result.get("result", "")
        if is_workflow_step(task_ctx):
            await post_execute_workflow_step(task, result)
        await self._handle_complete(task, result)
        return GuardHandled("clarification_history_reused")

    # Workflow clarification validation — if the clarification schema is
    # wrong, drop into quality-retry; otherwise let caller dispatch to
    # _handle_clarification as normal.
    if is_workflow_step(task_ctx) and not await self._validate_clarification(
        task_id, task, task_ctx, result
    ):
        from src.core.retry import RetryContext
        retry_ctx = RetryContext.from_task(task)
        decision = retry_ctx.record_failure("quality", model=result.get("model", ""))
        if decision.action == "terminal":
            task_ctx.update(retry_ctx.to_context_patch())
            await update_task(
                task_id, status="failed",
                error=f"Clarification schema failed: {result.get('error', '')[:300]}",
                context=json.dumps(task_ctx),
                **retry_ctx.to_db_fields(),
            )
        else:
            next_retry = None
            if decision.action == "delayed":
                next_retry = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))
            retry_ctx.next_retry_at = next_retry
            task_ctx.update(retry_ctx.to_context_patch())
            await update_task(
                task_id, status="pending",
                error=f"Clarification schema failed: {result.get('error', '')[:200]}",
                context=json.dumps(task_ctx),
                **retry_ctx.to_db_fields(),
            )
        return GuardHandled("clarification_schema_failed")

    return None
