"""Apply Beckman actions to the DB. One branch per action type.

Every function returns None. Side-effects: insert rows, update task status.
Retry / DLQ decisions come from `general_beckman.retry`. Clarify and notify
tasks are created as mechanical mr_roboto rows — mr_roboto executors do the
actual Telegram I/O at dispatch time.

NOTE: The tasks table has no 'payload' column. Mechanical task payloads are
stored in the 'context' JSON column with the shape:

    {"executor": "mechanical", "payload": {"action": <name>, **kwargs}}

The orchestrator's `_dispatch` copies `ctx["payload"]` onto `task["payload"]`
before calling `mr_roboto.run`, which routes on `payload["action"]`. Use
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
    """Build the canonical context shape for a mechanical mr_roboto task.

    The workflow engine's `expand_steps_to_tasks` emits the same shape
    (see tests/workflows/test_mechanical_step_materializes_with_executor_tag.py).
    """
    return {
        "executor": "mechanical",
        "payload": {"action": action, **payload_fields},
    }

logger = get_logger("beckman.apply")


async def _record_and_resolve_confidence(
    task_id: int, correct: bool, source: str,
    reviewer_verdict_id: int | None = None,
) -> None:
    """Z10 T4B — record + immediately resolve a confidence claim.

    Skips silently if the source task has no confidence signal (record
    returns None) or already resolved. Safe in mechanical/skipped paths.
    """
    from src.infra.db import (
        record_confidence_claim, resolve_confidence_outcome,
    )
    claim_id = await record_confidence_claim(task_id)
    if claim_id is None:
        return
    await resolve_confidence_outcome(
        claim_id, correct=correct, source=source,
        reviewer_verdict_id=reviewer_verdict_id,
    )


def _task_phase_label(task: dict, ctx: dict | None = None) -> str:
    """Return a phase label for B10 rework telemetry.

    Prefers ``workflow_step_id`` (e.g. "8.3") which is the granular step
    address, falls back to ``workflow_phase`` (e.g. "phase_8") which is
    the band, falls back to "" when neither is set (non-workflow task).
    """
    if ctx is None:
        try:
            raw = task.get("context") or "{}"
            ctx = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            ctx = {}
    return str(
        ctx.get("workflow_step_id")
        or ctx.get("step_id")
        or ctx.get("workflow_phase")
        or ""
    )


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
    # result_router accepts dict/list as a non-empty result (some agents
    # return structured payloads e.g. {"formatted_text": "..."}). The
    # tasks.result column is TEXT — sqlite InterfaceError fires when a
    # dict is passed as bind parameter (mission 46 task 2949 hit this
    # 2026-04-26: shopping format_response shape leaked through). JSON-
    # stringify non-str payloads before writing so the column always
    # holds a parseable text body.
    result_value = a.result
    if not isinstance(result_value, (str, bytes, type(None))):
        try:
            result_value = json.dumps(result_value, ensure_ascii=False)
        except (TypeError, ValueError):
            result_value = str(result_value)
    # Clear stale failure metadata that the prior failed attempt left
    # in the row. Without this, a row that failed attempt N with error X
    # then succeeded at N+1 still shows X in tasks.error — misleads
    # post-mortems and /queue UI. Mission 46 task 2921 (4.14) wore a
    # ghost error column for hours after a clean retry succeeded.
    # error_category, next_retry_at, retry_reason, failed_in_phase are
    # paired stale signals — clear together. (Handoff item B.)
    await update_task(
        a.task_id, status="completed",
        completed_at=to_db(utc_now()),
        result=result_value,
        error=None,
        error_category=None,
        next_retry_at=None,
        retry_reason=None,
        failed_in_phase=None,
    )

    # Z2 T5C — pin recipes from artifact when the completing task was a
    # pick_recipe mechanical step with pin_recipes=true in its context.
    # Best-effort: failure must never break the completion path.
    try:
        ctx = _parse_ctx(task)
        payload = ctx.get("payload") or {}
        if (
            payload.get("action") == "pick_recipe"
            and ctx.get("pin_recipes")
        ):
            recipe_picks_path = str(
                payload.get("recipe_picks_path")
                or ctx.get("recipe_picks_path")
                or ""
            )
            mission_id = task.get("mission_id")
            if recipe_picks_path and mission_id:
                from src.infra.recipes import pin_recipes_from_artifact
                count = await pin_recipes_from_artifact(
                    mission_id=int(mission_id),
                    recipe_picks_path=recipe_picks_path,
                )
                logger.info(
                    "_apply_complete: pin_recipes_from_artifact: mission_id=%s count=%d",
                    mission_id, count,
                )
    except Exception:
        logger.debug(
            "_apply_complete: pin_recipes_from_artifact failed", exc_info=True
        )


async def _apply_subtasks(task: dict, a: SpawnSubtasks) -> None:
    """Spawn N subtasks atomically.

    Was a loop of add_task() calls, each opening its own aux connection
    + BEGIN/COMMIT — N writer-slot acquisitions per batch. Migrated to
    add_subtasks_atomically which does ONE connection + ONE tx for the
    whole batch, dropping write-lock contention on bursty mission
    advances.
    """
    from src.infra.db import add_subtasks_atomically
    subtasks = [
        {
            "title": sub.get("title", ""),
            "description": sub.get("description", ""),
            "agent_type": sub.get("agent_type", "coder"),
            "tier": sub.get("tier", "auto"),
            "priority": sub.get("priority", task.get("priority", 5)),
            "depends_on": sub.get("depends_on", []),
            "context": sub.get("context", {}),
        }
        for sub in a.subtasks
    ]
    await add_subtasks_atomically(
        parent_task_id=a.parent_task_id,
        subtasks=subtasks,
        mission_id=task.get("mission_id"),
        parent_status="waiting_subtasks",
    )


async def _apply_clarify(task: dict, a: RequestClarification) -> None:
    from src.infra.db import add_task, update_task
    # Model sometimes signals needs_clarification without a usable
    # question (returned empty string or missing field). Spawning a
    # clarify task with empty payload DLQs the whole step, confusing
    # the user. Treat empty-question as a soft failure: mark the
    # source failed with a clear reason so the retry/DLQ path can
    # react, and do not create an orphan clarify task.
    question = (a.question or "").strip()
    if not question:
        logger.warning(
            "clarify skipped: agent returned needs_clarification without a "
            "question (task_id=%s)", a.task_id,
        )
        # Route to DLQ instead of silent fail so user can see + /retry.
        from src.infra.db import get_task as _get
        src = await _get(a.task_id)
        if src:
            await _dlq_write(
                dict(src, failed_in_phase="worker"),
                error="agent signalled needs_clarification with empty question",
                category="quality",
                attempts=int(src.get("worker_attempts") or 0),
            )
        return
    await update_task(a.task_id, status="waiting_human")
    await add_task(
        title=f"Clarify: {task.get('title','')[:40]}",
        description=question,
        mission_id=task.get("mission_id"),
        parent_task_id=a.task_id,
        agent_type="mechanical",
        context=_mechanical_context(
            "clarify",
            question=question,
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
    # Category precedence: this attempt's result wins over the stale row.
    # The orchestrator stamps `error_category=availability` on the result
    # dict when a ModelCallFailed bubbles up (rate-limited / no-model /
    # provider-down). Without preferring it here, the task carried the
    # PRIOR attempt's category — typically `quality` from a grader-FAIL —
    # and decide_retry took the immediate-retry path that's correct for
    # quality failures but wrong for availability ones. That burned
    # worker_attempts in seconds when waiting was the right move
    # (production triage 2026-04-30: task #4457 hit 5/6 attempts in
    # under a minute against rate-limited gemini quota).
    raw = a.raw or {}
    category = (
        raw.get("error_category")
        or task.get("error_category")
        or "worker"
    )
    await _retry_or_dlq(task, category=category, error=a.error)


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
    from src.infra.db import get_task as _get_task, update_task
    # Refetch after post_execute_workflow_step — the hook writes
    # _schema_error and _prev_output into the DB context so the next
    # attempt knows what to fix. on_task_finished keeps a snapshot
    # taken BEFORE the hook, so reading task.context here returns the
    # stale pre-hook view, and writing it back below would erase the
    # hook's contributions. Observed mission 46 task 2867: context
    # reset after each retry, agent had no idea why 0.6 kept failing.
    fresh = await _get_task(task["id"])
    if fresh:
        task = fresh
    attempts = int(task.get("worker_attempts") or 0) + 1
    max_attempts = int(task.get("max_worker_attempts") or 15)
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

    # Drop carry-over retry feedback that wasn't refreshed for this attempt.
    # Hooks.py writes _schema_error+_prev_output during post_execute when a
    # schema validation fails — and stamps _schema_error_for_attempt = attempts.
    # If this retry was triggered by a non-schema path (availability, exhausted),
    # the stamp won't match and we drop the stale feedback so the next prompt
    # doesn't replay an unrelated failure as "your last output failed".
    _drop_stale_retry_feedback(ctx, attempts)

    # Persist the failed model so the NEXT pick excludes it. Without
    # this, fatih_hoca.requirements_builder.get_model_constraints reads
    # an empty failed_models list and the same model is re-picked on
    # every retry — defeating model rotation. Production triage
    # 2026-05-08 task #11930: 5 schema-validation retries, all on the
    # same Qwen3.5-9B output, no exclusion grew because update_task
    # below only persisted the new context (without failed_models)
    # and update_exclusions_on_failure was wired into design but not
    # called anywhere in production code.
    #
    # Source the model that just failed from task_state.used_model
    # (coulson writes this after every llm call) with fallback to
    # ctx.generating_model (older shim path). Don't fail the retry on
    # any error here — exclusion is a quality-of-life nudge, the gate
    # is decide_retry/DLQ which already runs.
    try:
        from src.core.retry import update_exclusions_on_failure
        _failed_model = ""
        try:
            _ts_raw = task.get("task_state")
            if isinstance(_ts_raw, str):
                _ts_raw = json.loads(_ts_raw or "{}")
            if isinstance(_ts_raw, dict):
                _failed_model = _ts_raw.get("used_model") or ""
        except Exception:
            _failed_model = ""
        if not _failed_model:
            _failed_model = ctx.get("generating_model") or ""
        if _failed_model:
            update_exclusions_on_failure(ctx, _failed_model, attempts)
    except Exception as _exc:
        logger.debug(
            "update_exclusions_on_failure skipped",
            task_id=task.get("id"),
            error=str(_exc),
        )

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

    # B10 telemetry: a quality-category retry is a schema_failure rework
    # signal. Today the retry replays the SAME step (no cross-band
    # rollback) so the counter only bumps when ctx already records the
    # original (pre-retry) step as phase>=7 AND the retry forces a
    # re-entry to phase<=6. The helper handles that test internally —
    # we just hand it both phases. For same-step retries the helper
    # logs the event but skips the counter bump per is_phase_7_rework.
    if category == "quality" and task.get("mission_id"):
        try:
            from src.telemetry.rework import record_rollback
            phase_label = _task_phase_label(task, ctx)
            if phase_label:
                await record_rollback(
                    mission_id=int(task["mission_id"]),
                    from_phase=phase_label,
                    # to_phase from ctx if a rollback target was set by
                    # the workflow engine; otherwise same-step retry
                    to_phase=str(ctx.get("rollback_to_phase") or phase_label),
                    reason="schema_failure",
                    triggered_by=str(task.get("agent_type") or "worker"),
                )
        except Exception as _exc:
            logger.debug("rework telemetry skipped",
                         task_id=task.get("id"), error=str(_exc))


async def _maybe_emit_lesson_from_posthook_fail(
    source: dict,
    kind: str,
    error_str: str,
    feedback: str,
    attempts: int,
) -> None:
    """Side-effect: upsert a mission_lessons row on posthook exhaustion.

    Wraps import + upsert in try/except — NEVER lets lesson-emit failure
    cascade into the verdict path. Idempotent via dedup_key.
    """
    try:
        from src.infra.mission_lessons import upsert_mission_lesson
        from src.infra.db import get_db
        import json as _json

        mission_id = source.get("mission_id")
        stack = "unknown"
        if mission_id:
            try:
                _db = await get_db()
                _cur = await _db.execute(
                    "SELECT context FROM missions WHERE id = ?", (mission_id,)
                )
                _row = await _cur.fetchone()
                if _row:
                    _ctx = _json.loads(_row[0] or "{}")
                    stack = str(_ctx.get("tech_stack_detected") or "unknown")
            except Exception:
                pass

        pattern = (error_str or "").strip()[:120]
        fix = (feedback or "").strip()[:300]

        await upsert_mission_lesson(
            stack=stack,
            domain=kind,
            pattern=pattern or f"{kind} gate exhausted",
            fix=fix,
            severity="blocker",
            source_kind="posthook_fail",
            source_ref={
                "source_task_id": source.get("id"),
                "kind": kind,
                "attempts": attempts,
            },
        )
    except Exception as _exc:
        logger.debug(
            "lesson emit skipped (posthook_fail)",
            kind=kind,
            task_id=source.get("id"),
            error=str(_exc),
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
    # Post-hook DLQ cascade: a posthook task exhausting its retries means
    # the source task can no longer advance through the verdict path.
    # Synthesise the terminal outcome instead of leaving the source stuck
    # in 'ungraded' forever. Detected by ctx.source_task_id presence
    # rather than agent_type, so mechanical posthooks (verify_artifacts)
    # cascade without listing every agent_type that might run a posthook.
    _ctx_for_cascade = _parse_ctx(task)
    if (
        task.get("agent_type") in ("grader", "artifact_summarizer")
        or (_ctx_for_cascade.get("source_task_id") is not None
            and _ctx_for_cascade.get("posthook_kind"))
    ):
        try:
            await _posthook_dlq_cascade(task, error)
        except Exception as exc:
            logger.warning("posthook DLQ cascade failed",
                           task_id=task["id"], error=str(exc))
    # Telegram DLQ notification → mechanical mr_roboto task (no inline send).
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
                f"Reason: {_humanize_error(error)}"
            ),
        ),
        depends_on=[],
    )



_NULLISH_STRINGS = {"", "none", "null", "nil", "n/a", "na", "-"}


def _stamp_retry_feedback(ctx: dict, next_attempt: int) -> None:
    """Tag freshly-written ``_schema_error``/``_prev_output`` with the attempt
    number they were written FOR. Readers gate on this so stale feedback from
    earlier failure modes (e.g. schema reject 2 attempts ago, then availability
    bounce that re-queues without rewriting) doesn't leak into the next prompt
    as ``"your last output failed: <unrelated>"``.
    """
    if "_schema_error" in ctx or "_prev_output" in ctx:
        ctx["_schema_error_for_attempt"] = int(next_attempt)


def _drop_stale_retry_feedback(ctx: dict, current_attempts: int) -> None:
    """Pop ``_schema_error``/``_prev_output`` if not stamped for this attempt.

    ``current_attempts`` is the value of ``worker_attempts`` for the row about
    to execute. A fresh feedback write tags `_schema_error_for_attempt` to that
    same number; anything else is left over from an earlier lifecycle stage
    and must not be replayed.
    """
    stamp = ctx.get("_schema_error_for_attempt")
    if stamp is None:
        # Untagged + present means it was written before the staleness scheme
        # existed. Conservative: drop, since we can't prove it's fresh.
        if "_schema_error" in ctx or "_prev_output" in ctx:
            ctx.pop("_schema_error", None)
            ctx.pop("_prev_output", None)
        return
    if int(stamp) != int(current_attempts):
        ctx.pop("_schema_error", None)
        ctx.pop("_prev_output", None)
        ctx.pop("_schema_error_for_attempt", None)


def _is_meaningful_text(val) -> bool:
    """Return True if ``val`` is a non-empty string carrying real content.

    Grader LLMs sometimes populate optional fields with the literal text
    ``"None"`` / ``"NONE"`` / ``"n/a"`` — Python-truthy strings but
    semantically missing. Callers that key off these fields must reject
    them to avoid leaking the sentinel back to the user.
    """
    if not isinstance(val, str):
        return False
    stripped = val.strip()
    if not stripped:
        return False
    return stripped.lower() not in _NULLISH_STRINGS


def _is_title_echo(val: str, source_title: str) -> bool:
    """True if a grader free-text field is just the source task title echoed back.

    Thinking-model graders sometimes skip evaluation and re-emit the prompt's
    ``Task: <title>`` content as their SITUATION/INSIGHT. Surfacing that as the
    "reason" leaks the title into the error column and DLQ notification (e.g.
    a generic "MVP scope definition task" reason on task 2889). Callers must
    treat such echoes as nullish so the cascade falls through to the next
    candidate field or "grader verdict unavailable".
    """
    if not source_title or not isinstance(val, str):
        return False
    a = val.strip().lower()
    b = source_title.strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    # Substantial overlap: candidate is a substring of title (or vice versa)
    # and short enough that it's clearly an echo, not analysis. Cap at 80
    # chars so longer free-form text that happens to mention the title isn't
    # discarded.
    if len(a) <= 80 and (a in b or b in a):
        return True
    return False


def _grader_verdict_text(raw, *, source_title: str = "") -> str:
    """Extract the most useful human sentence from a grader verdict payload.

    ``raw`` may be a dict (common), a stringified dict (legacy), or free text.
    Prefers ``insight`` → ``strategy`` → ``situation`` → ``message``/``error``;
    falls back to a failed-axes summary, or a final "verdict unavailable"
    string when every candidate field is missing / nullish. Callers use this
    as the error column upstream so downstream consumers (Telegram DLQ
    notice, logs) never see raw dict reprs or "None" sentinels.

    ``source_title`` enables title-echo rejection: when a grader echoes the
    input task title in a free-text field instead of evaluating, we treat it
    as nullish and continue the cascade.
    """
    import ast
    candidate = raw
    if isinstance(raw, str):
        text = raw.strip()
        start = text.find("{")
        if start != -1:
            depth, end = 0, -1
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                try:
                    candidate = ast.literal_eval(text[start:end + 1])
                except (ValueError, SyntaxError):
                    candidate = raw
            else:
                candidate = raw
    if isinstance(candidate, dict):
        for key in ("insight", "strategy", "situation", "message", "error"):
            val = candidate.get(key)
            if _is_meaningful_text(val) and not _is_title_echo(val, source_title):
                return val.strip()
        failed_axes = [
            k for k in ("relevant", "complete", "well_formed", "coherent")
            if candidate.get(k) is False
        ]
        if failed_axes:
            return "grader rejected: " + ", ".join(failed_axes)
        return "grader verdict unavailable"
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in _NULLISH_STRINGS):
        return "grader verdict unavailable"
    return str(raw)[:140]


def _humanize_error(raw: str) -> str:
    """Turn internal error payloads into one-line user-facing text.

    Grader verdicts arrive as stringified Python dicts like
    ``{'passed': False, 'relevant': False, 'insight': '...'}``. Showing
    the raw dict to Telegram users leaks internals and wastes display
    budget on field names. Prefer the grader's insight when present,
    else strategy, else a short verbatim head.

    Parser accepts the dict even when surrounded by prose or truncated
    off a wrapper — the strict ``startswith("{") and endswith("}")``
    check used to fail on strings like ``"grader said: {'passed': ...}
    more stuff"``, leaking the raw repr straight to Telegram.
    """
    import ast
    import re
    if not raw:
        return "unknown"
    text = raw.strip()
    parsed = None
    # Find the largest balanced {...} substring. Works for leading/
    # trailing prose, and for the case the outer dict itself is intact.
    start = text.find("{")
    if start != -1:
        depth = 0
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            candidate = text[start:end + 1]
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                parsed = None
    if isinstance(parsed, dict):
        for key in ("insight", "strategy", "situation", "message", "error"):
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:140]
        failed_axes = [
            k for k in ("relevant", "complete", "well_formed", "coherent")
            if parsed.get(k) is False
        ]
        if failed_axes:
            return "grader rejected: " + ", ".join(failed_axes)
    # Strip obvious prefixes like "Reason: " or "Grader verdict: "
    text = re.sub(r"^\s*(?:reason|grader[^:]*|error)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text[:140]


async def _posthook_dlq_cascade(task: dict, error: str) -> None:
    """Propagate a grader/summarizer DLQ to its source task.

    - Grader DLQ → source permanently failed ("no grader succeeded").
      Distinct from a legitimate reject verdict, which retries the source.
    - Summary DLQ → remove that summary kind from pending_posthooks;
      structural fallback (already stored by post_execute_workflow_step)
      stands. If the pending list drains to empty, flip source to completed.
    """
    import json as _json
    from src.infra.db import get_task, update_task

    ctx_raw = task.get("context") or "{}"
    try:
        ctx = _json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        if isinstance(ctx, str):
            try:
                ctx = _json.loads(ctx)
            except (_json.JSONDecodeError, TypeError):
                ctx = {}
    except (_json.JSONDecodeError, TypeError):
        ctx = {}
    if not isinstance(ctx, dict):
        return

    source_id = ctx.get("source_task_id")
    if source_id is None:
        return

    source = await get_task(source_id)
    if source is None or source.get("status") != "ungraded":
        return

    source_ctx = _parse_ctx(source)
    pending = list(source_ctx.get("_pending_posthooks") or [])
    agent_type = task.get("agent_type")

    if agent_type == "grader":
        # Permanent failure: no grader could produce a verdict.
        source_ctx["_pending_posthooks"] = []
        source_ctx["_grade_dlq_reason"] = error[:300]
        await update_task(
            source_id,
            status="failed",
            error=f"grade DLQ: {error[:400]}",
            failed_in_phase="grading",
            context=_json.dumps(source_ctx),
        )
        logger.warning("grader DLQ cascaded source to failed",
                       source_id=source_id, grader_task_id=task["id"])
        return

    posthook_kind = ctx.get("posthook_kind")
    if posthook_kind == "code_review":
        # Code reviewer task itself DLQ'd (e.g. all reviewer models failed,
        # not a fail-verdict outcome which would have completed normally).
        # Source can't advance — mark failed so depends_on cascade blocks
        # downstream rather than leaving it stuck in 'ungraded'.
        source_ctx["_pending_posthooks"] = [
            k for k in (source_ctx.get("_pending_posthooks") or [])
            if k != "code_review"
        ]
        source_ctx["_review_dlq_reason"] = error[:300]
        await update_task(
            source_id,
            status="failed",
            error=f"code_review DLQ: {error[:400]}",
            failed_in_phase="code_review",
            context=_json.dumps(source_ctx),
        )
        logger.warning("code_review DLQ cascaded source to failed",
                       source_id=source_id, reviewer_task_id=task["id"])
        return

    if posthook_kind == "verify_artifacts":
        # Mechanical verifier itself DLQ'd (e.g. workspace permission denied,
        # not a missing-files outcome which would have completed normally).
        # Source can't advance — mark failed so depends_on cascade blocks
        # downstream rather than leaving it stuck in 'ungraded'.
        source_ctx["_pending_posthooks"] = [
            k for k in (source_ctx.get("_pending_posthooks") or [])
            if k != "verify_artifacts"
        ]
        source_ctx["_verify_dlq_reason"] = error[:300]
        await update_task(
            source_id,
            status="failed",
            error=f"verify_artifacts DLQ: {error[:400]}",
            failed_in_phase="verify",
            context=_json.dumps(source_ctx),
        )
        logger.warning("verify_artifacts DLQ cascaded source to failed",
                       source_id=source_id, verifier_task_id=task["id"])
        return

    if posthook_kind == "pattern_lint":
        # Mechanical semgrep runner DLQ'd (e.g. workspace permission denied,
        # unexpected semgrep crash).  pattern_lint is v1-warning-only so a
        # DLQ here does NOT cascade the source to failed — we soft-drop the
        # pending kind and let the source advance.  This mirrors the
        # soft-skip path used when semgrep is not installed.
        source_ctx["_pending_posthooks"] = [
            k for k in (source_ctx.get("_pending_posthooks") or [])
            if k != "pattern_lint"
        ]
        source_ctx["_pattern_lint_dlq_reason"] = error[:300]
        new_pending = source_ctx["_pending_posthooks"]
        if not new_pending:
            await update_task(
                source_id,
                status="completed",
                context=_json.dumps(source_ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            try:
                await _spawn_workflow_advance_if_mission(source, {})
            except Exception:
                pass
        else:
            await update_task(source_id, context=_json.dumps(source_ctx))
        logger.warning(
            "pattern_lint DLQ soft-dropped (warning-only kind)",
            source_id=source_id, linter_task_id=task["id"],
        )
        return

    if posthook_kind == "design_system_check":
        # Z2 T3C — design_system_check is v1-warning-only; same soft-drop
        # behaviour as pattern_lint.  Never cascade source to failed.
        source_ctx["_pending_posthooks"] = [
            k for k in (source_ctx.get("_pending_posthooks") or [])
            if k != "design_system_check"
        ]
        source_ctx["_design_system_check_dlq_reason"] = error[:300]
        new_pending = source_ctx["_pending_posthooks"]
        if not new_pending:
            await update_task(
                source_id,
                status="completed",
                context=_json.dumps(source_ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            try:
                await _spawn_workflow_advance_if_mission(source, {})
            except Exception:
                pass
        else:
            await update_task(source_id, context=_json.dumps(source_ctx))
        logger.warning(
            "design_system_check DLQ soft-dropped (warning-only kind)",
            source_id=source_id, linter_task_id=task["id"],
        )
        return

    if posthook_kind in ("openapi_sync", "typescript_sync"):
        # Z2 T3B — blocker kinds: DLQ cascades source to failed so it doesn't
        # get stuck in 'ungraded' when the regen_and_diff executor itself DLQs.
        source_ctx["_pending_posthooks"] = [
            k for k in (source_ctx.get("_pending_posthooks") or [])
            if k != posthook_kind
        ]
        source_ctx[f"_{posthook_kind}_dlq_reason"] = error[:300]
        await update_task(
            source_id,
            status="failed",
            error=f"{posthook_kind} DLQ: {error[:400]}",
            failed_in_phase="posthook",
            context=_json.dumps(source_ctx),
        )
        logger.warning(
            "%s DLQ cascaded source to failed",
            posthook_kind, source_id=source_id, posthook_task_id=task["id"],
        )
        return

    # Z3 R1 — review-density blocker kinds. DLQ cascades source to failed.
    if posthook_kind in (
        "security_review", "accessibility_review", "contract_review",
        "performance_review", "adr_drift_check", "integration_replay",
        "integration_review",
    ):
        source_ctx["_pending_posthooks"] = [
            k for k in (source_ctx.get("_pending_posthooks") or [])
            if k != posthook_kind
        ]
        source_ctx[f"_{posthook_kind}_dlq_reason"] = error[:300]
        await update_task(
            source_id,
            status="failed",
            error=f"{posthook_kind} DLQ: {error[:400]}",
            failed_in_phase="posthook",
            context=_json.dumps(source_ctx),
        )
        logger.warning(
            "%s DLQ cascaded source to failed",
            posthook_kind, source_id=source_id, posthook_task_id=task["id"],
        )
        return

    if posthook_kind in _Z1_BLOCKER_KINDS:
        # Z1 blocker post-hook (compliance_template_present, etc.) DLQ'd
        # — cascade source to failed so it doesn't get stuck in 'ungraded'.
        source_ctx["_pending_posthooks"] = [
            k for k in (source_ctx.get("_pending_posthooks") or [])
            if k != posthook_kind
        ]
        source_ctx[f"_{posthook_kind}_dlq_reason"] = error[:300]
        await update_task(
            source_id,
            status="failed",
            error=f"{posthook_kind} DLQ: {error[:400]}",
            failed_in_phase="posthook",
            context=_json.dumps(source_ctx),
        )
        logger.warning(
            "%s DLQ cascaded source to failed",
            posthook_kind, source_id=source_id, posthook_task_id=task["id"],
        )
        return

    if posthook_kind in _Z1_WARNING_KINDS:
        # Z1 warning post-hook (find_similar_missions, etc.) DLQ'd — soft-drop
        # the pending kind; advance source if no others remain.
        new_pending = [
            k for k in (source_ctx.get("_pending_posthooks") or [])
            if k != posthook_kind
        ]
        source_ctx["_pending_posthooks"] = new_pending
        source_ctx[f"_{posthook_kind}_dlq_reason"] = error[:300]
        if not new_pending:
            await update_task(
                source_id, status="completed",
                context=_json.dumps(source_ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            try:
                await _spawn_workflow_advance_if_mission(source, {})
            except Exception:
                pass
        else:
            await update_task(source_id, context=_json.dumps(source_ctx))
        logger.warning(
            "%s DLQ soft-dropped (Z1 warning kind)",
            posthook_kind, source_id=source_id, posthook_task_id=task["id"],
        )
        return

    if agent_type == "artifact_summarizer":
        artifact_name = ctx.get("artifact_name") or ""
        kind = f"summary:{artifact_name}"
        pending = [k for k in pending if k != kind]
        source_ctx["_pending_posthooks"] = pending
        if not pending:
            await update_task(
                source_id, status="completed",
                context=_json.dumps(source_ctx),
            )
        else:
            await update_task(
                source_id, context=_json.dumps(source_ctx),
            )
        logger.info("summary DLQ falls back to structural",
                    source_id=source_id, artifact=artifact_name)


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

    # Z3 T2C: For integration_review, run the mechanical AST pre-check here
    # (async context) and inject the result into the payload before enqueue.
    if a.kind == "integration_review":
        payload = await _enrich_integration_review_payload(payload)

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
    if a.kind == "verify_artifacts":
        # Mechanical post-hook: mr_roboto resolves declared ``produces`` paths
        # under the mission workspace, checks file exists + non-empty +
        # optional compile/parse. Failure → source retries with feedback.
        produces = list(source_ctx.get("produces") or [])
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "verify_artifacts",
            "executor": "mechanical",
            "payload": {
                "action": "verify_artifacts",
                "paths": produces,
                "min_bytes": 1,
                "compile_check": True,
            },
        })
    if a.kind == "grounding":
        # Mechanical post-hook: mr_roboto matches the source task's tool_calls
        # audit log against declared ``produces`` paths. Pass = at least one
        # successful write_file call per produces slot. Fail = the agent
        # narrated completion without ever calling write_file. Floor for
        # the L1 sub-iter grounding guard — catches anything that escaped
        # in-loop (suppress_guards path, exhausted sub-iter budget).
        produces = list(source_ctx.get("produces") or [])
        tool_calls = list(source_ctx.get("tool_calls") or [])
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "grounding",
            "executor": "mechanical",
            "payload": {
                "action": "check_grounding",
                "produces": produces,
                "tool_calls": tool_calls,
            },
        })
    if a.kind == "code_review":
        # LLM post-hook: a code-review-flavoured reviewer judges the source's
        # emitted code. Its verdict (PASS/FAIL) drives the same retry-with-
        # feedback path as verify_artifacts. Issue list is fed back via
        # _schema_error so the source's next attempt sees what to fix.
        produces = list(source_ctx.get("produces") or [])
        return ("code_reviewer", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "code_review",
            "produces": produces,
            "review_excluded_models": list(source_ctx.get("review_excluded_models") or []),
        })
    if a.kind == "imports_check":
        produces = list(source_ctx.get("produces") or [])
        workspace_path = source_ctx.get("workspace_path") or ""
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "imports_check",
            "executor": "mechanical",
            "payload": {
                "action": "check_imports",
                "target_files": produces,
                "workspace_path": workspace_path,
            },
        })
    if a.kind == "test_run":
        produces = list(source_ctx.get("produces") or [])
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "test_run",
            "executor": "mechanical",
            "payload": {
                "action": "run_tests",
                "target_files": produces,
                "stack_hint": source_ctx.get("stack_hint") or "",
            },
        })
    if a.kind == "pattern_lint":
        # Z2 T2C — semgrep with forbidden-patterns rule pack. Warning-only
        # in v1; soft-skips when semgrep not installed.
        from mr_roboto.run_semgrep import DEFAULT_RULE_PACK
        produces = list(source_ctx.get("produces") or [])
        rule_pack = str(source_ctx.get("rule_pack_path") or DEFAULT_RULE_PACK)
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "pattern_lint",
            "executor": "mechanical",
            "payload": {
                "action": "run_semgrep",
                "target_files": produces,
                "rule_pack_path": rule_pack,
            },
        })
    if a.kind == "openapi_sync":
        # Z2 T3B — regenerate OpenAPI spec; diff vs committed.
        default_cmd = [
            "python", "-c",
            "from app.main import app; import json; print(json.dumps(app.openapi()))",
        ]
        generator_cmd = list(source_ctx.get("regen_cmd") or default_cmd)
        target_path = str(source_ctx.get("openapi_target_path") or "openapi.json")
        workspace_path = source_ctx.get("workspace_path") or ""
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "openapi_sync",
            "executor": "mechanical",
            "payload": {
                "action": "regen_and_diff",
                "generator_cmd": generator_cmd,
                "target_path": target_path,
                "workspace_path": workspace_path,
            },
        })
    if a.kind == "typescript_sync":
        # Z2 T3B — regenerate frontend API types; diff vs committed.
        default_cmd = ["npx", "openapi-typescript", "openapi.json"]
        generator_cmd = list(source_ctx.get("regen_cmd") or default_cmd)
        target_path = str(source_ctx.get("types_target_path") or "types/api.ts")
        workspace_path = source_ctx.get("workspace_path") or ""
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "typescript_sync",
            "executor": "mechanical",
            "payload": {
                "action": "regen_and_diff",
                "generator_cmd": generator_cmd,
                "target_path": target_path,
                "workspace_path": workspace_path,
            },
        })
    if a.kind == "design_system_check":
        # Z2 T3C — semgrep with design-system rule pack; warning-only.
        from pathlib import Path as _Path
        _DS_RULE_PACK = str(
            _Path(__file__).parent.parent.parent.parent.parent
            / "mr_roboto" / "src" / "mr_roboto" / "rule_packs" / "design_system.yml"
        )
        try:
            import importlib.resources as _ir
            import mr_roboto.rule_packs as _rp_pkg  # type: ignore[import]
            _DS_RULE_PACK = str(_ir.files(_rp_pkg).joinpath("design_system.yml"))
        except Exception:
            pass
        produces = list(source_ctx.get("produces") or [])
        rule_pack = str(source_ctx.get("design_system_rule_pack") or _DS_RULE_PACK)
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "design_system_check",
            "executor": "mechanical",
            "payload": {
                "action": "run_semgrep",
                "target_files": produces,
                "rule_pack_path": rule_pack,
            },
        })
    if a.kind == "migration_apply":
        # Z2 T3A — apply migration to ephemeral DB. Stack-aware.
        produces = list(source_ctx.get("produces") or [])
        workspace_path = source_ctx.get("workspace_path") or ""
        stack_hint = str(source_ctx.get("stack_hint") or "")
        enable_tc = bool(source_ctx.get("enable_testcontainers", False))
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "migration_apply",
            "executor": "mechanical",
            "payload": {
                "action": "apply_migration",
                "target_files": produces,
                "workspace_path": workspace_path,
                "stack_hint": stack_hint,
                "enable_testcontainers": enable_tc,
            },
        })
    if a.kind == "compliance_template_present":
        # Z1 T5A (P6) — overlay_path defaults to produces[0]
        # (mission_<id>/compliance_overlay.json). Handler reads file +
        # extracts required_documents → template_ids → walks
        # compliance_templates/.
        produces = list(source_ctx.get("produces") or [])
        overlay_path = (
            source_ctx.get("overlay_path")
            or (produces[0] if produces else None)
        )
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "compliance_template_present",
            "executor": "mechanical",
            "payload": {
                "action": "compliance_template_present",
                "overlay_path": overlay_path,
                "template_ids": source_ctx.get("template_ids"),
                "template_root": source_ctx.get("template_root"),
                "workspace_path": source_ctx.get("workspace_path"),
            },
        })
    if a.kind == "compliance_blocker_check":
        # Z1 T5A (P6) — phase-boundary check on step 6.6 project_plan_review.
        # current_phase defaults to 6 (matches step ID phase_6).
        current_phase = (
            source_ctx.get("current_phase")
            or source_ctx.get("workflow_phase_index")
            or 6
        )
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "compliance_blocker_check",
            "executor": "mechanical",
            "payload": {
                "action": "compliance_blocker_check",
                "current_phase": int(current_phase) if isinstance(current_phase, (int, str)) and str(current_phase).isdigit() else 6,
                "workspace_path": source_ctx.get("workspace_path"),
            },
        })
    if a.kind == "find_similar_missions":
        # Z1 T6A (A7) — handler resolves idea_summary from workspace
        # (.charter/product_charter.md, idea_brief.md) when None.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "find_similar_missions",
            "executor": "mechanical",
            "payload": {
                "action": "find_similar_missions",
                "idea_summary": source_ctx.get("idea_summary"),
                "workspace_path": source_ctx.get("workspace_path"),
                "top_k": int(source_ctx.get("similar_top_k") or 3),
                "threshold": source_ctx.get("similar_threshold"),
            },
        })
    if a.kind == "index_idea_fingerprint":
        # Z1 T6A (A7) — siblings find_similar_missions on step 0.1.
        # Indexes idea fingerprint into mission_ideas ChromaDB collection
        # so future missions can dedup. Title from source title or ctx.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "index_idea_fingerprint",
            "executor": "mechanical",
            "payload": {
                "action": "index_idea_fingerprint",
                "idea_summary": source_ctx.get("idea_summary"),
                "workspace_path": source_ctx.get("workspace_path"),
                "title": str(
                    source_ctx.get("title")
                    or source_ctx.get("mission_title")
                    or source.get("title")
                    or ""
                ),
                "final_status_note": str(
                    source_ctx.get("final_status_note") or ""
                ),
            },
        })
    if a.kind == "surface_prior_mission_hints":
        # Z1 T6A (P9) — advisory cross-mission ADR + compliance hints on
        # step 0.5 clarification_questions. Handler always completes.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "surface_prior_mission_hints",
            "executor": "mechanical",
            "payload": {
                "action": "surface_prior_mission_hints",
                "workspace_path": source_ctx.get("workspace_path"),
                "founder_id": str(source_ctx.get("founder_id") or "default"),
                "top_k": int(source_ctx.get("hints_top_k") or 3),
                "jaccard_threshold": float(
                    source_ctx.get("hints_jaccard_threshold") or 0.3
                ),
            },
        })
    if a.kind == "prior_art_min_coverage":
        # Z1 T6B (P5) — handler reads report_path; defaults to produces[0]
        # (mission_<id>/.research/prior_art_report.json).
        produces = list(source_ctx.get("produces") or [])
        report_path = (
            source_ctx.get("report_path")
            or (produces[0] if produces else None)
        )
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "prior_art_min_coverage",
            "executor": "mechanical",
            "payload": {
                "action": "prior_art_min_coverage",
                "report_path": report_path,
                "report": source_ctx.get("report"),
            },
        })
    if a.kind == "verify_falsification_present":
        # Z1 T2 (P4) — falsification triple check on phase-3 commitments.
        # Resolve artifacts from source.result. Most phase-3 steps emit a
        # single output artifact (functional_requirements, etc.); we wrap
        # the parsed result under its declared output_artifacts[0] name.
        # legacy_pre_falsification flag flows through source_ctx when the
        # mission predates the Z1 P4 reshape.
        source_result = source.get("result") or ""
        parsed: object = {}
        if isinstance(source_result, str) and source_result.strip():
            try:
                parsed = json.loads(source_result)
            except (ValueError, TypeError):
                parsed = {}
        elif isinstance(source_result, (list, dict)):
            parsed = source_result
        output_names = list(source_ctx.get("output_artifacts") or [])
        artifacts: dict = {}
        if isinstance(parsed, dict):
            # Result is already a {name: value} mapping.
            artifacts = parsed
        elif output_names and parsed:
            artifacts = {output_names[0]: parsed}
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "verify_falsification_present",
            "executor": "mechanical",
            "payload": {
                "action": "verify_falsification_present",
                "artifacts": artifacts,
                "legacy_pre_falsification": bool(
                    source_ctx.get("legacy_pre_falsification", False)
                ),
            },
        })
    if a.kind == "critic_gate":
        # Z1 T5C (B4) — standalone critic-gate post-hook. action_name +
        # target_payload come from source_ctx so any step that declares
        # `post_hooks: ["critic_gate"]` can pass the payload the critic
        # should judge. mission_id flows through the source row.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "critic_gate",
            "executor": "mechanical",
            "payload": {
                "action": "critic_gate",
                "action_name": str(
                    source_ctx.get("critic_action_name")
                    or source_ctx.get("step_id")
                    or "unknown"
                ),
                "target_payload": source_ctx.get("critic_target_payload"),
                "mission_id": source.get("mission_id"),
            },
        })
    if a.kind == "integration_review":
        # Z3 T2C — cross-file consistency review after multi-file expansion.
        # The mechanical pre-check (extract_signatures) is run async in
        # _enrich_integration_review_payload, called from _apply_request_posthook
        # AFTER this function returns. Signatures start empty here; enriched
        # before the reviewer task row is enqueued.
        all_produces = list(
            source_ctx.get("all_sub_task_produces")
            or source_ctx.get("produces")
            or []
        )
        workspace_path = source_ctx.get("workspace_path") or ""
        sub_task_ids = list(source_ctx.get("sub_task_ids") or [])

        return ("integration_reviewer", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "integration_review",
            "sub_task_ids": sub_task_ids,
            "sub_task_titles": list(source_ctx.get("sub_task_titles") or []),
            "all_sub_task_produces": all_produces,
            "workspace_path": workspace_path,
            # Enriched by _enrich_integration_review_payload (async):
            "signatures": {},
            "mismatches": [],
        })
    if a.kind == "security_review":
        # Z3 T3A — composite mechanical: semgrep + bandit + npm audit.
        produces = list(source_ctx.get("produces") or [])
        workspace_path = source_ctx.get("workspace_path") or ""
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "security_review",
            "executor": "mechanical",
            "payload": {
                "action": "security_review",
                "target_files": produces,
                "workspace_path": workspace_path,
            },
        })
    if a.kind == "accessibility_review":
        # Z3 T3B — axe-core scan against tunneled preview URL.
        # URL resolved from .preview/last_preview_url.txt (T3B follow-up in
        # emit_preview_url) with source_ctx fallback.
        # "pending:" marker (hosting deferred) suppresses URL — axe soft-skips.
        import os as _os
        workspace_path = source_ctx.get("workspace_path") or ""
        preview_url = source_ctx.get("preview_url") or ""
        if workspace_path and not preview_url:
            _last_url_path = _os.path.join(workspace_path, ".preview", "last_preview_url.txt")
            try:
                with open(_last_url_path, "r", encoding="utf-8") as _f:
                    _read = _f.read().strip()
                # Filter pending markers — only real URLs propagate downstream.
                if _read.startswith("http://") or _read.startswith("https://"):
                    preview_url = _read
            except (OSError, FileNotFoundError):
                pass
        produces = list(source_ctx.get("produces") or [])
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "accessibility_review",
            "executor": "mechanical",
            "payload": {
                "action": "run_axe",
                "preview_url": preview_url,
                "target_paths": produces,
            },
        })
    if a.kind == "contract_review":
        # Z3 T3C — schemathesis contract fuzz against running app.
        import os as _os
        workspace_path = source_ctx.get("workspace_path") or ""
        spec_path = source_ctx.get("openapi_spec_path") or source_ctx.get("spec_path") or ""
        if not spec_path and workspace_path:
            _candidate = _os.path.join(workspace_path, "openapi.json")
            if _os.path.exists(_candidate):
                spec_path = _candidate
        base_url = source_ctx.get("preview_url") or source_ctx.get("base_url") or ""
        if workspace_path and not base_url:
            _last_url_path = _os.path.join(workspace_path, ".preview", "last_preview_url.txt")
            try:
                with open(_last_url_path, "r", encoding="utf-8") as _f:
                    base_url = _f.read().strip()
            except (OSError, FileNotFoundError):
                pass
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "contract_review",
            "executor": "mechanical",
            "payload": {
                "action": "run_schemathesis",
                "spec_path": spec_path,
                "base_url": base_url,
            },
        })
    if a.kind == "performance_review":
        # Z3 T3C — opt-in lighthouse (web) or k6 (api).
        import os as _os
        workspace_path = source_ctx.get("workspace_path") or ""
        mode = source_ctx.get("perf_mode") or "web"
        preview_url = source_ctx.get("preview_url") or ""
        if workspace_path and not preview_url:
            _last_url_path = _os.path.join(workspace_path, ".preview", "last_preview_url.txt")
            try:
                with open(_last_url_path, "r", encoding="utf-8") as _f:
                    preview_url = _f.read().strip()
            except (OSError, FileNotFoundError):
                pass
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "performance_review",
            "executor": "mechanical",
            "payload": {
                "action": "performance_review",
                "mode": mode,
                "preview_url": preview_url,
                "script_path": source_ctx.get("perf_script_path") or "",
                "thresholds": source_ctx.get("perf_thresholds") or {},
            },
        })
    if a.kind == "integration_replay":
        # Z3 T5 — rerun suite against shuffled prior commits. mode from dial
        # (quick|standard|strict); commits + shuffle_seed default from mission_id.
        workspace_path = source_ctx.get("workspace_path") or ""
        mission_id = source.get("mission_id") or 0
        mode = source_ctx.get("integration_replay_mode") or "standard"
        commits = list(source_ctx.get("integration_replay_commits") or [])
        seed = int(source_ctx.get("shuffle_seed") or mission_id or 0)
        suite_glob = source_ctx.get("integration_suite_glob") or "tests/integration/**"
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "integration_replay",
            "executor": "mechanical",
            "payload": {
                "action": "integration_replay",
                "commits": commits,
                "suite_glob": suite_glob,
                "shuffle_seed": seed,
                "mode": mode,
                "workspace_path": workspace_path,
            },
        })
    if a.kind == "adr_drift_check":
        # Z3 T4B — mechanical ADR drift gate.
        workspace_path = source_ctx.get("workspace_path") or ""
        register_path = source_ctx.get("adr_register_path") or ""
        if workspace_path and not register_path:
            # Use forward slash regardless of platform — downstream consumers
            # treat the path as a POSIX-style artifact reference.
            wp = workspace_path.rstrip("/\\")
            register_path = f"{wp}/.adr/register.md"
        produces = list(source_ctx.get("produces") or [])
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "adr_drift_check",
            "executor": "mechanical",
            "payload": {
                "action": "check_adr_drift",
                "adr_register_path": register_path,
                "produced_files": produces,
                "workspace_path": workspace_path,
            },
        })
    raise ValueError(f"unknown posthook kind: {a.kind!r}")


async def _enrich_integration_review_payload(payload: dict) -> dict:
    """Z3 T2C — run extract_signatures and inject the result into *payload*.

    Called from ``_apply_request_posthook`` in the async path so we can
    properly await the mr_roboto verb without wrapping in sync hacks.
    Best-effort: any failure returns the original payload unchanged.
    """
    all_produces = list(payload.get("all_sub_task_produces") or [])
    workspace_path = payload.get("workspace_path") or None
    if not all_produces:
        return payload
    try:
        from mr_roboto.extract_signatures import extract_signatures as _es
        sig_result = await _es(
            target_files=all_produces,
            workspace_path=workspace_path,
        )
        return {
            **payload,
            "signatures": sig_result.get("signatures") or {},
            "mismatches": sig_result.get("mismatches") or [],
        }
    except Exception:
        # Best-effort — never block the reviewer dispatch
        return payload


async def _apply_grounding_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Apply a grounding post-hook verdict.

    Layer 2 of G: catches narration-as-completion that escaped the L1
    sub-iter guard (suppress_guards path, exhausted sub-iter budget).

    Pass: drop ``grounding`` from pending; if no other post-hooks remain,
    mark source completed.
    Fail: retry source with the ungrounded paths in ``_schema_error`` so
    the agent's next prompt sees what was never written. Honors worker
    attempt cap + bonus-progress budget the same way verify_artifacts
    does. Does NOT bump model exclusions — narration is an agent-behaviour
    failure, not a model-quality verdict.
    """
    import json as _json
    from src.infra.db import update_task

    if verdict.passed:
        new_pending = [k for k in pending if k != "grounding"]
        ctx["_pending_posthooks"] = new_pending
        if not new_pending:
            await update_task(
                verdict.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, verdict.raw)
            try:
                from general_beckman import _send_step_progress
                from src.infra.db import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", verdict.raw or {})
            except Exception:
                pass
        else:
            await update_task(
                verdict.source_task_id, context=_json.dumps(ctx),
            )
        # Z10 wire-fix F8 — grounding-PASS resolves confidence claim true.
        try:
            await _record_and_resolve_confidence(
                task_id=verdict.source_task_id, correct=True,
                source="grounding",
            )
        except Exception as _e:
            logger.debug("confidence claim record failed (grounding pass)",
                         task_id=verdict.source_task_id, error=str(_e))
        return

    raw = verdict.raw or {}
    missing = raw.get("missing") or []
    written = raw.get("written") or []
    error_str = (
        f"check_grounding: {len(missing)} produces slot(s) ungrounded. "
        f"missing={missing[:8]} written={written[:8]}"
    )[:500]

    attempts = int(source.get("worker_attempts") or 0) + 1
    max_attempts = int(source.get("max_worker_attempts") or 15)
    ctx["_pending_posthooks"] = []

    feedback = (
        "You declared this task done but never called write_file (or "
        "edit_file/patch_file) for the path(s) you were supposed to "
        f"produce. ungrounded={missing}; you wrote={written}. "
        "On retry: actually call the write_file tool for each declared "
        "path before final_answer. Do NOT just narrate the file contents."
    )

    if attempts >= max_attempts:
        from general_beckman.retry import _MAX_BONUS
        bonus_count = int(ctx.get("_bonus_count", 0))
        progress = _parse_progress(source)
        can_bonus = (
            progress is not None
            and progress >= 0.5
            and bonus_count < _MAX_BONUS
        )
        if can_bonus:
            ctx["_bonus_count"] = bonus_count + 1
            max_attempts += 1
            ctx["_schema_error"] = feedback
            prev_output = source.get("result") or ""
            if isinstance(prev_output, str) and prev_output.strip():
                ctx["_prev_output"] = prev_output[:6000]
            _stamp_retry_feedback(ctx, attempts)
            await update_task(
                verdict.source_task_id,
                status="pending",
                worker_attempts=attempts,
                max_worker_attempts=max_attempts,
                error=error_str,
                error_category="quality",
                next_retry_at=None,
                context=_json.dumps(ctx),
            )
            return
        await _dlq_write(
            source, error=error_str or "grounding gate exhausted",
            category="quality", attempts=attempts,
        )
        await _maybe_emit_lesson_from_posthook_fail(
            source=source, kind="grounding",
            error_str=error_str, feedback=feedback, attempts=attempts,
        )
        return

    ctx["_schema_error"] = feedback
    prev_output = source.get("result") or ""
    if isinstance(prev_output, str) and prev_output.strip():
        ctx["_prev_output"] = prev_output[:6000]
    _stamp_retry_feedback(ctx, attempts)
    await update_task(
        verdict.source_task_id,
        status="pending",
        worker_attempts=attempts,
        error=error_str,
        error_category="quality",
        next_retry_at=None,
        context=_json.dumps(ctx),
    )

    # Z10 wire-fix F8 — grounding-FAIL resolves confidence claim false.
    try:
        await _record_and_resolve_confidence(
            task_id=verdict.source_task_id, correct=False,
            source="grounding",
        )
    except Exception as _e:
        logger.debug("confidence claim record (grounding reject) failed",
                     task_id=verdict.source_task_id, error=str(_e))


async def _apply_verify_artifacts_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Apply a verify_artifacts post-hook verdict back to the source task.

    Pass: drop the kind from pending; if no other post-hooks remain, mark
    source completed.
    Fail: retry source with the verifier's missing/failed paths in
    ``_schema_error`` so the agent's next prompt sees what was missing.
    Honors worker attempt cap + bonus-progress budget the same way the
    grade-fail path does, but does NOT bump model exclusions — missing
    files are an agent-behaviour failure, not a model-quality one.
    """
    import json as _json
    from src.infra.db import update_task

    if verdict.passed:
        new_pending = [k for k in pending if k != "verify_artifacts"]
        ctx["_pending_posthooks"] = new_pending
        if not new_pending:
            await update_task(
                verdict.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, verdict.raw)
            try:
                from general_beckman import _send_step_progress
                from src.infra.db import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", verdict.raw or {})
            except Exception:
                pass
        else:
            await update_task(
                verdict.source_task_id, context=_json.dumps(ctx),
            )
        return

    # Fail path: retry source with the verifier's findings as feedback.
    raw = verdict.raw or {}
    missing = raw.get("missing") or []
    failed = raw.get("failed") or []
    error_str = (
        f"verify_artifacts: {len(missing)} missing path(s), "
        f"{len(failed)} failed check(s). "
        f"missing={missing[:8]} failed={failed[:8]}"
    )[:500]

    attempts = int(source.get("worker_attempts") or 0) + 1
    max_attempts = int(source.get("max_worker_attempts") or 15)
    ctx["_pending_posthooks"] = []

    feedback = (
        "Files you said you wrote are missing or empty. "
        f"missing={missing}; failed={failed}. "
        "On retry: actually call the write_file tool for each declared path. "
        "Do not just emit JSON describing the file."
    )

    if attempts >= max_attempts:
        from general_beckman.retry import _MAX_BONUS
        bonus_count = int(ctx.get("_bonus_count", 0))
        progress = _parse_progress(source)
        can_bonus = (
            progress is not None
            and progress >= 0.5
            and bonus_count < _MAX_BONUS
        )
        if can_bonus:
            ctx["_bonus_count"] = bonus_count + 1
            max_attempts += 1
            ctx["_schema_error"] = feedback
            prev_output = source.get("result") or ""
            if isinstance(prev_output, str) and prev_output.strip():
                ctx["_prev_output"] = prev_output[:6000]
            _stamp_retry_feedback(ctx, attempts)
            await update_task(
                verdict.source_task_id,
                status="pending",
                worker_attempts=attempts,
                max_worker_attempts=max_attempts,
                error=error_str,
                error_category="quality",
                next_retry_at=None,
                context=_json.dumps(ctx),
            )
            return
        await _dlq_write(
            source, error=error_str or "verify_artifacts gate exhausted",
            category="quality", attempts=attempts,
        )
        await _maybe_emit_lesson_from_posthook_fail(
            source=source, kind="verify_artifacts",
            error_str=error_str, feedback=feedback, attempts=attempts,
        )
        return

    ctx["_schema_error"] = feedback
    prev_output = source.get("result") or ""
    if isinstance(prev_output, str) and prev_output.strip():
        ctx["_prev_output"] = prev_output[:6000]
    _stamp_retry_feedback(ctx, attempts)
    await update_task(
        verdict.source_task_id,
        status="pending",
        worker_attempts=attempts,
        error=error_str,
        error_category="quality",
        next_retry_at=None,
        context=_json.dumps(ctx),
    )


async def _apply_code_review_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Apply a code_review post-hook verdict back to the source task.

    Pass: drop the kind from pending; if no other post-hooks remain, mark
    source completed.
    Fail: retry source with the reviewer's issues list as feedback. Honors
    worker attempt cap + bonus-progress budget. Bumps the review-side
    exclusion list (so the same reviewer model isn't rerun on the same
    output) but does NOT add to ctx.failed_models — code style / coverage
    feedback is reviewer-judgment, not a model-quality verdict on the
    coder. The next coder attempt may still legitimately use the same
    coder model with the issues list as steering.
    """
    import json as _json
    from src.infra.db import update_task

    if verdict.passed:
        new_pending = [k for k in pending if k != "code_review"]
        ctx["_pending_posthooks"] = new_pending
        if not new_pending:
            await update_task(
                verdict.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, verdict.raw)
            try:
                from general_beckman import _send_step_progress
                from src.infra.db import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", verdict.raw or {})
            except Exception:
                pass
        # Z10 T4B — record + resolve confidence claim against this reviewer
        # verdict. record+resolve in one go: the verdict IS the resolution
        # signal, no need for a separate pending row.
        try:
            await _record_and_resolve_confidence(
                task_id=verdict.source_task_id, correct=True,
                source="reviewer_verdict",
            )
        except Exception as _e:
            logger.debug("confidence claim record failed",
                         task_id=verdict.source_task_id, error=str(_e))
        return

    # Fail path
    raw = verdict.raw or {}
    issues = raw.get("issues") or []
    error_str = (
        f"code_review: {len(issues)} issue(s) found. "
        f"first: {(issues[0] if issues else '<no detail>')}"
    )[:500]

    attempts = int(source.get("worker_attempts") or 0) + 1
    max_attempts = int(source.get("max_worker_attempts") or 15)
    ctx["_pending_posthooks"] = []

    bullet_block = "\n".join(f"- {i}" for i in issues[:30]) or "- (no detail provided)"
    feedback = (
        "Code review rejected your output. Fix these issues on retry, "
        "then re-emit:\n" + bullet_block
    )

    if attempts >= max_attempts:
        from general_beckman.retry import _MAX_BONUS
        bonus_count = int(ctx.get("_bonus_count", 0))
        progress = _parse_progress(source)
        can_bonus = (
            progress is not None
            and progress >= 0.5
            and bonus_count < _MAX_BONUS
        )
        if can_bonus:
            ctx["_bonus_count"] = bonus_count + 1
            max_attempts += 1
            ctx["_schema_error"] = feedback
            prev_output = source.get("result") or ""
            if isinstance(prev_output, str) and prev_output.strip():
                ctx["_prev_output"] = prev_output[:6000]
            _stamp_retry_feedback(ctx, attempts)
            await update_task(
                verdict.source_task_id,
                status="pending",
                worker_attempts=attempts,
                max_worker_attempts=max_attempts,
                error=error_str,
                error_category="quality",
                next_retry_at=None,
                context=_json.dumps(ctx),
            )
            return
        await _dlq_write(
            source, error=error_str or "code review gate exhausted",
            category="quality", attempts=attempts,
        )
        await _maybe_emit_lesson_from_posthook_fail(
            source=source, kind="code_review",
            error_str=error_str, feedback=feedback, attempts=attempts,
        )
        return

    ctx["_schema_error"] = feedback
    prev_output = source.get("result") or ""
    if isinstance(prev_output, str) and prev_output.strip():
        ctx["_prev_output"] = prev_output[:6000]
    _stamp_retry_feedback(ctx, attempts)
    await update_task(
        verdict.source_task_id,
        status="pending",
        worker_attempts=attempts,
        error=error_str,
        error_category="quality",
        next_retry_at=None,
        context=_json.dumps(ctx),
    )

    # Z10 T4B — reviewer rejection: confidence claim resolves as incorrect
    try:
        await _record_and_resolve_confidence(
            task_id=verdict.source_task_id, correct=False,
            source="reviewer_verdict",
        )
    except Exception as _e:
        logger.debug("confidence claim record (reject) failed",
                     task_id=verdict.source_task_id, error=str(_e))

    # B10 telemetry: code-reviewer rejection is a reviewer_reject rework
    # signal. See _retry_or_dlq for the same-step / cross-band rationale.
    if source.get("mission_id"):
        try:
            from src.telemetry.rework import record_rollback
            phase_label = _task_phase_label(source, ctx)
            if phase_label:
                await record_rollback(
                    mission_id=int(source["mission_id"]),
                    from_phase=phase_label,
                    to_phase=str(ctx.get("rollback_to_phase") or phase_label),
                    reason="reviewer_reject",
                    triggered_by="code_reviewer",
                )
        except Exception as _exc:
            logger.debug("rework telemetry skipped (review)",
                         task_id=source.get("id"), error=str(_exc))


async def _apply_test_run_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Apply a test_run post-hook verdict back to the source task."""
    import json as _json
    from src.infra.db import update_task

    if verdict.passed:
        new_pending = [k for k in pending if k != "test_run"]
        ctx["_pending_posthooks"] = new_pending
        # Surface slow-suite warning into context without blocking.
        raw = verdict.raw or {}
        if raw.get("warning"):
            ctx["_test_run_warning"] = raw["warning"]
        if not new_pending:
            await update_task(
                verdict.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, verdict.raw)
            try:
                from general_beckman import _send_step_progress
                from src.infra.db import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", verdict.raw or {})
            except Exception:
                pass
        else:
            await update_task(
                verdict.source_task_id, context=_json.dumps(ctx),
            )
        return

    # Fail path: blocker on any red result (failed/errors > 0, timed_out,
    # zero-collected = import error, spawn error).
    raw = verdict.raw or {}
    failed_n = int(raw.get("failed") or 0)
    errors_n = int(raw.get("errors") or 0)
    total_n = int(raw.get("total") or 0)
    timed_out = bool(raw.get("timed_out"))
    spawn_err = raw.get("error") or ""
    stdout_tail = (raw.get("stdout_tail") or "")[:400]

    if timed_out:
        error_str = "test_run: suite timed out"
    elif spawn_err:
        error_str = f"test_run: spawn error: {spawn_err}"[:500]
    elif total_n == 0:
        error_str = "test_run: 0 tests collected (possible import error)"
    else:
        error_str = (
            f"test_run: {failed_n} failed, {errors_n} errors, "
            f"{total_n} total."
        )
    if stdout_tail:
        error_str = (error_str + f" output={stdout_tail!r}")[:500]

    attempts = int(source.get("worker_attempts") or 0) + 1
    max_attempts = int(source.get("max_worker_attempts") or 15)
    ctx["_pending_posthooks"] = []

    feedback = (
        "Tests you wrote are failing. Fix the implementation (or the test "
        "if it is wrong) so the suite goes green. "
        f"Details: {error_str}"
    )

    if attempts >= max_attempts:
        from general_beckman.retry import _MAX_BONUS
        bonus_count = int(ctx.get("_bonus_count", 0))
        progress = _parse_progress(source)
        can_bonus = (
            progress is not None
            and progress >= 0.5
            and bonus_count < _MAX_BONUS
        )
        if can_bonus:
            ctx["_bonus_count"] = bonus_count + 1
            max_attempts += 1
            ctx["_schema_error"] = feedback
            prev_output = source.get("result") or ""
            if isinstance(prev_output, str) and prev_output.strip():
                ctx["_prev_output"] = prev_output[:6000]
            _stamp_retry_feedback(ctx, attempts)
            await update_task(
                verdict.source_task_id,
                status="pending",
                worker_attempts=attempts,
                max_worker_attempts=max_attempts,
                error=error_str,
                error_category="quality",
                next_retry_at=None,
                context=_json.dumps(ctx),
            )
            return
        await _dlq_write(
            source, error=error_str or "test_run gate exhausted",
            category="quality", attempts=attempts,
        )
        return

    ctx["_schema_error"] = feedback
    prev_output = source.get("result") or ""
    if isinstance(prev_output, str) and prev_output.strip():
        ctx["_prev_output"] = prev_output[:6000]
    _stamp_retry_feedback(ctx, attempts)
    await update_task(
        verdict.source_task_id,
        status="pending",
        worker_attempts=attempts,
        error=error_str,
        error_category="quality",
        next_retry_at=None,
        context=_json.dumps(ctx),
    )


async def _apply_semgrep_warning_verdict(
    kind: str,
    findings_ctx_key: str,
    dlq_reason_ctx_key: str,
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Shared implementation for warning-only semgrep post-hook kinds.

    Used by both pattern_lint (T2C) and design_system_check (T3C).  The two
    callers differ only in the ``kind`` name used to remove the pending entry
    and the ctx keys used to surface findings.

    Parameters
    ----------
    kind:
        The post-hook kind string (e.g. ``"pattern_lint"``).
    findings_ctx_key:
        Context key under which findings list is stored
        (e.g. ``"_pattern_lint_findings"``).
    dlq_reason_ctx_key:
        Context key for DLQ soft-drop reason (unused here; present for
        symmetry with ``_posthook_dlq_cascade``).
    """
    import json as _json
    from src.infra.db import update_task

    raw = verdict.raw or {}
    findings = raw.get("findings") or []
    skipped = bool(raw.get("skipped"))

    if findings:
        ctx[findings_ctx_key] = findings[:50]
        blocker_count = int(raw.get("blocker_count") or 0)
        warning_count = int(raw.get("warning_count") or 0)
        logger.info(
            f"{kind}: findings surfaced (warning-only, not retrying)",
            source_id=verdict.source_task_id,
            findings=len(findings),
            blockers=blocker_count,
            warnings=warning_count,
            skipped=skipped,
        )
    elif skipped:
        logger.debug(
            f"{kind}: semgrep not installed, soft-skipped",
            source_id=verdict.source_task_id,
        )

    new_pending = [k for k in pending if k != kind]
    ctx["_pending_posthooks"] = new_pending

    if not new_pending:
        await update_task(
            verdict.source_task_id,
            status="completed",
            context=_json.dumps(ctx),
            error=None,
            error_category=None,
            next_retry_at=None,
            retry_reason=None,
            failed_in_phase=None,
        )
        await _spawn_workflow_advance_if_mission(source, verdict.raw)
        try:
            from general_beckman import _send_step_progress
            from src.infra.db import get_task
            fresh = await get_task(verdict.source_task_id)
            if fresh:
                await _send_step_progress(fresh, "completed", verdict.raw or {})
        except Exception:
            pass
    else:
        await update_task(
            verdict.source_task_id,
            context=_json.dumps(ctx),
        )


async def _apply_semgrep_blocker_verdict(
    kind: str,
    findings_ctx_key: str,
    dlq_reason_ctx_key: str,
    blocker_threshold: str,
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Blocker-severity semgrep post-hook verdict.

    Mirrors _apply_test_run_verdict shape: retry-with-feedback on findings,
    bonus-budget + DLQ-on-exhaust. Soft-skip preserved when semgrep absent.

    Parameters
    ----------
    kind:
        Post-hook kind string (e.g. ``"pattern_lint"``).
    findings_ctx_key:
        Context key for findings list.
    dlq_reason_ctx_key:
        Context key for DLQ reason.
    blocker_threshold:
        Severity level that triggers blocking (e.g. ``"ERROR"``).
        Findings with severity >= threshold cause a retry.
    """
    import json as _json
    from src.infra.db import update_task

    raw = verdict.raw or {}
    findings = raw.get("findings") or []
    skipped = bool(raw.get("skipped"))

    # Soft-skip: semgrep not installed → advance as if passed.
    if skipped:
        logger.debug(
            f"{kind}: semgrep not installed, soft-skipped (blocker severity)",
            source_id=verdict.source_task_id,
        )
        new_pending = [k for k in pending if k != kind]
        ctx["_pending_posthooks"] = new_pending
        if not new_pending:
            await update_task(
                verdict.source_task_id,
                status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, verdict.raw)
            try:
                from general_beckman import _send_step_progress
                from src.infra.db import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", verdict.raw or {})
            except Exception:
                pass
        else:
            await update_task(verdict.source_task_id, context=_json.dumps(ctx))
        return

    # Separate blocker-level findings from sub-threshold findings.
    #
    # blocker_threshold is a semgrep severity string (ERROR / WARNING / INFO).
    # A finding blocks the source when its severity is >= threshold.
    #
    # When the threshold value is the posthook severity word "blocker" or
    # "warning" (i.e. caller passed `default_severity` straight through), it
    # is reinterpreted: "blocker" means "this kind gates aggressively" → use
    # the LOWEST semgrep severity (INFO) so ANY finding blocks. "warning"
    # means "non-fatal" → use ERROR so only the worst findings block.
    # Step context can override via `semgrep_blocker_threshold`.
    _POSTHOOK_TO_SEMGREP = {"blocker": "WARNING", "warning": "ERROR"}
    _SEVERITY_ORDER = {"ERROR": 3, "WARNING": 2, "INFO": 1}
    normalised_threshold = _POSTHOOK_TO_SEMGREP.get(
        blocker_threshold.lower(), blocker_threshold.upper()
    )
    threshold_level = _SEVERITY_ORDER.get(normalised_threshold.upper(), 2)
    blocker_findings = [
        f for f in findings
        if _SEVERITY_ORDER.get((f.get("severity") or "WARNING").upper(), 2) >= threshold_level
    ]

    if not blocker_findings:
        # All findings below threshold → warning surface + advance (old behaviour).
        if findings:
            ctx[findings_ctx_key] = findings[:50]
            logger.info(
                f"{kind}: {len(findings)} finding(s) below blocker threshold, warning only",
                source_id=verdict.source_task_id,
                threshold=blocker_threshold,
            )
        new_pending = [k for k in pending if k != kind]
        ctx["_pending_posthooks"] = new_pending
        if not new_pending:
            await update_task(
                verdict.source_task_id,
                status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, verdict.raw)
            try:
                from general_beckman import _send_step_progress
                from src.infra.db import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", verdict.raw or {})
            except Exception:
                pass
        else:
            await update_task(verdict.source_task_id, context=_json.dumps(ctx))
        return

    # Blocker path: retry-with-feedback.
    ctx[findings_ctx_key] = blocker_findings[:50]

    finding_strs = [
        f"{f.get('path','?')}:{f.get('line','?')} {f.get('rule_id','?')} {f.get('message','')}"
        for f in blocker_findings[:10]
    ]
    error_str = (
        f"{kind}: {len(blocker_findings)} blocker finding(s). "
        f"Fix: {'; '.join(finding_strs)}"
    )[:500]

    feedback = (
        f"Semgrep found {len(blocker_findings)} pattern violation(s) that must be fixed. "
        f"Details: {error_str}"
    )

    attempts = int(source.get("worker_attempts") or 0) + 1
    max_attempts = int(source.get("max_worker_attempts") or 15)
    ctx["_pending_posthooks"] = []

    if attempts >= max_attempts:
        from general_beckman.retry import _MAX_BONUS
        bonus_count = int(ctx.get("_bonus_count", 0))
        progress = _parse_progress(source)
        can_bonus = (
            progress is not None
            and progress >= 0.5
            and bonus_count < _MAX_BONUS
        )
        if can_bonus:
            ctx["_bonus_count"] = bonus_count + 1
            max_attempts += 1
            ctx["_schema_error"] = feedback
            prev_output = source.get("result") or ""
            if isinstance(prev_output, str) and prev_output.strip():
                ctx["_prev_output"] = prev_output[:6000]
            _stamp_retry_feedback(ctx, attempts)
            await update_task(
                verdict.source_task_id,
                status="pending",
                worker_attempts=attempts,
                max_worker_attempts=max_attempts,
                error=error_str,
                error_category="quality",
                next_retry_at=None,
                context=_json.dumps(ctx),
            )
            return
        await _dlq_write(
            source, error=error_str or f"{kind} blocker gate exhausted",
            category="quality", attempts=attempts,
        )
        return

    ctx["_schema_error"] = feedback
    prev_output = source.get("result") or ""
    if isinstance(prev_output, str) and prev_output.strip():
        ctx["_prev_output"] = prev_output[:6000]
    _stamp_retry_feedback(ctx, attempts)
    await update_task(
        verdict.source_task_id,
        status="pending",
        worker_attempts=attempts,
        error=error_str,
        error_category="quality",
        next_retry_at=None,
        context=_json.dumps(ctx),
    )


async def _apply_pattern_lint_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Z2 T2C — routes by registry default_severity (blocker or warning)."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY.get("pattern_lint")
    default_severity = (spec.default_severity if spec else "warning")
    step_threshold = ctx.get("semgrep_blocker_threshold") or default_severity
    if default_severity == "blocker":
        await _apply_semgrep_blocker_verdict(
            kind="pattern_lint",
            findings_ctx_key="_pattern_lint_findings",
            dlq_reason_ctx_key="_pattern_lint_dlq_reason",
            blocker_threshold=step_threshold,
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )
    else:
        await _apply_semgrep_warning_verdict(
            kind="pattern_lint",
            findings_ctx_key="_pattern_lint_findings",
            dlq_reason_ctx_key="_pattern_lint_dlq_reason",
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )


async def _apply_design_system_check_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Z2 T3C — routes by registry default_severity (blocker or warning)."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY.get("design_system_check")
    default_severity = (spec.default_severity if spec else "warning")
    step_threshold = ctx.get("semgrep_blocker_threshold") or default_severity
    if default_severity == "blocker":
        await _apply_semgrep_blocker_verdict(
            kind="design_system_check",
            findings_ctx_key="_design_system_findings",
            dlq_reason_ctx_key="_design_system_check_dlq_reason",
            blocker_threshold=step_threshold,
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )
    else:
        await _apply_semgrep_warning_verdict(
            kind="design_system_check",
            findings_ctx_key="_design_system_findings",
            dlq_reason_ctx_key="_design_system_check_dlq_reason",
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )


async def _apply_type_sync_verdict(
    kind: str,
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Z2 T3B — openapi_sync / typescript_sync shared verdict.

    Drift detected → blocker retry with feedback. Skipped → soft-advance.
    No drift → advance.
    """
    import json as _json
    from src.infra.db import update_task

    raw = verdict.raw or {}
    skipped = bool(raw.get("skipped"))
    diff_present = bool(raw.get("diff_present"))
    target_path = str(raw.get("target_path") or "")
    generator_cmd = list(raw.get("generator_cmd") or [])

    # Slow-regen warning — surface into ctx but never block.
    if raw.get("warning"):
        ctx[f"_{kind}_warning"] = raw["warning"]

    if not verdict.passed or diff_present:
        # Soft-skip: generator not installed → advance (v1 ramp).
        if skipped:
            logger.debug(
                "%s: generator not installed, soft-skipped",
                kind, source_id=verdict.source_task_id,
            )
            new_pending = [k for k in pending if k != kind]
            ctx["_pending_posthooks"] = new_pending
            if not new_pending:
                await update_task(
                    verdict.source_task_id,
                    status="completed",
                    context=_json.dumps(ctx),
                    error=None,
                    error_category=None,
                    next_retry_at=None,
                    retry_reason=None,
                    failed_in_phase=None,
                )
                await _spawn_workflow_advance_if_mission(source, verdict.raw)
                try:
                    from general_beckman import _send_step_progress
                    from src.infra.db import get_task
                    fresh = await get_task(verdict.source_task_id)
                    if fresh:
                        await _send_step_progress(fresh, "completed", verdict.raw or {})
                except Exception:
                    pass
            else:
                await update_task(verdict.source_task_id, context=_json.dumps(ctx))
            return

        # Drift or internal failure → blocker retry with feedback.
        diff_excerpt = str(raw.get("diff_excerpt") or "")
        error_str = (
            f"{kind}: Drift detected in {target_path!r}. "
            f"Regenerate via {generator_cmd!r}."
        )
        if diff_excerpt:
            error_str = (error_str + f" Diff:\n{diff_excerpt}")[:500]

        attempts = int(source.get("worker_attempts") or 0) + 1
        max_attempts = int(source.get("max_worker_attempts") or 15)
        ctx["_pending_posthooks"] = []

        feedback = (
            f"The committed {target_path!r} is out of sync with the generated "
            f"output. Run: {' '.join(generator_cmd)!r} and commit the updated file. "
            f"Details: {error_str}"
        )

        if attempts >= max_attempts:
            from general_beckman.retry import _MAX_BONUS
            bonus_count = int(ctx.get("_bonus_count", 0))
            progress = _parse_progress(source)
            can_bonus = (
                progress is not None
                and progress >= 0.5
                and bonus_count < _MAX_BONUS
            )
            if can_bonus:
                ctx["_bonus_count"] = bonus_count + 1
                max_attempts += 1
                ctx["_schema_error"] = feedback
                prev_output = source.get("result") or ""
                if isinstance(prev_output, str) and prev_output.strip():
                    ctx["_prev_output"] = prev_output[:6000]
                _stamp_retry_feedback(ctx, attempts)
                await update_task(
                    verdict.source_task_id,
                    status="pending",
                    worker_attempts=attempts,
                    max_worker_attempts=max_attempts,
                    error=error_str[:500],
                    error_category="quality",
                    next_retry_at=None,
                    context=_json.dumps(ctx),
                )
                return
            await _dlq_write(
                source, error=error_str[:500] or f"{kind} gate exhausted",
                category="quality", attempts=attempts,
            )
            return

        ctx["_schema_error"] = feedback
        prev_output = source.get("result") or ""
        if isinstance(prev_output, str) and prev_output.strip():
            ctx["_prev_output"] = prev_output[:6000]
        _stamp_retry_feedback(ctx, attempts)
        await update_task(
            verdict.source_task_id,
            status="pending",
            worker_attempts=attempts,
            error=error_str[:500],
            error_category="quality",
            next_retry_at=None,
            context=_json.dumps(ctx),
        )
        return

    # Pass — no drift.
    new_pending = [k for k in pending if k != kind]
    ctx["_pending_posthooks"] = new_pending
    if not new_pending:
        await update_task(
            verdict.source_task_id,
            status="completed",
            context=_json.dumps(ctx),
            error=None,
            error_category=None,
            next_retry_at=None,
            retry_reason=None,
            failed_in_phase=None,
        )
        await _spawn_workflow_advance_if_mission(source, verdict.raw)
        try:
            from general_beckman import _send_step_progress
            from src.infra.db import get_task
            fresh = await get_task(verdict.source_task_id)
            if fresh:
                await _send_step_progress(fresh, "completed", verdict.raw or {})
        except Exception:
            pass
    else:
        await update_task(verdict.source_task_id, context=_json.dumps(ctx))


async def _apply_migration_apply_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Z2 T3A — migration_apply verdict.

    Apply error → blocker retry. Slow migration → warning surfaced, no
    block. Skipped (testcontainers absent) → pass through.
    """
    import json as _json
    from src.infra.db import update_task

    raw = verdict.raw or {}
    skipped = bool(raw.get("skipped"))
    warning = raw.get("warning") or ""

    # Skipped or slow-warning pass-through.
    if verdict.passed:
        new_pending = [k for k in pending if k != "migration_apply"]
        ctx["_pending_posthooks"] = new_pending
        if warning:
            ctx["_migration_apply_warning"] = warning
        if skipped:
            logger.debug(
                "migration_apply: soft-skipped",
                source_id=verdict.source_task_id,
                reason=raw.get("reason") or "",
            )
        elif warning:
            logger.info(
                "migration_apply: slow migration warning",
                source_id=verdict.source_task_id,
                warning=warning,
                duration_s=raw.get("duration_s"),
            )
        if not new_pending:
            await update_task(
                verdict.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, verdict.raw)
            try:
                from general_beckman import _send_step_progress
                from src.infra.db import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", verdict.raw or {})
            except Exception:
                pass
        else:
            await update_task(
                verdict.source_task_id, context=_json.dumps(ctx),
            )
        return

    # Fail path — blocker: migration apply error.
    error_detail = raw.get("error") or ""
    stack_used = raw.get("stack_used") or ""
    stderr_tail = (raw.get("stderr_tail") or "")[:300]
    error_str = (
        f"migration_apply: apply error (stack={stack_used}). "
        f"{error_detail}"
        + (f" stderr={stderr_tail!r}" if stderr_tail else "")
    )[:500]

    attempts = int(source.get("worker_attempts") or 0) + 1
    max_attempts = int(source.get("max_worker_attempts") or 15)
    ctx["_pending_posthooks"] = []

    feedback = (
        "Migration apply failed. Fix the migration file so it applies "
        "cleanly to an empty database. "
        f"Error: {error_str}"
    )

    if attempts >= max_attempts:
        from general_beckman.retry import _MAX_BONUS
        bonus_count = int(ctx.get("_bonus_count", 0))
        progress = _parse_progress(source)
        can_bonus = (
            progress is not None
            and progress >= 0.5
            and bonus_count < _MAX_BONUS
        )
        if can_bonus:
            ctx["_bonus_count"] = bonus_count + 1
            max_attempts += 1
            ctx["_schema_error"] = feedback
            prev_output = source.get("result") or ""
            if isinstance(prev_output, str) and prev_output.strip():
                ctx["_prev_output"] = prev_output[:6000]
            _stamp_retry_feedback(ctx, attempts)
            await update_task(
                verdict.source_task_id,
                status="pending",
                worker_attempts=attempts,
                max_worker_attempts=max_attempts,
                error=error_str,
                error_category="quality",
                next_retry_at=None,
                context=_json.dumps(ctx),
            )
            return
        await _dlq_write(
            source, error=error_str or "migration_apply gate exhausted",
            category="quality", attempts=attempts,
        )
        return

    ctx["_schema_error"] = feedback
    prev_output = source.get("result") or ""
    if isinstance(prev_output, str) and prev_output.strip():
        ctx["_prev_output"] = prev_output[:6000]
    _stamp_retry_feedback(ctx, attempts)
    await update_task(
        verdict.source_task_id,
        status="pending",
        worker_attempts=attempts,
        error=error_str,
        error_category="quality",
        next_retry_at=None,
        context=_json.dumps(ctx),
    )


# Z1 mechanical post-hook kinds — handled by the shared
# _apply_z1_mechanical_verdict below. Blocker kinds DLQ the source on
# fail; warning kinds soft-drop the pending kind and let source advance.
_Z1_BLOCKER_KINDS: frozenset[str] = frozenset({
    "compliance_template_present",
    "compliance_blocker_check",
    "prior_art_min_coverage",
    "verify_falsification_present",
    "critic_gate",  # Z1 T5C — veto fails source
})

_Z1_WARNING_KINDS: frozenset[str] = frozenset({
    "find_similar_missions",
    "index_idea_fingerprint",
    "surface_prior_mission_hints",
})

_Z1_MECHANICAL_KINDS: frozenset[str] = _Z1_BLOCKER_KINDS | _Z1_WARNING_KINDS


async def _apply_z1_mechanical_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Z1 mechanical post-hook verdict.

    Pass: drop ``kind`` from pending; if no other post-hooks remain,
    mark source completed and advance the workflow.

    Fail (blocker kinds): DLQ source with the error string.  No retry —
    these are deterministic shape checks against artifacts that are
    already on disk; a retry of the source step would re-emit the same
    artifact unless the founder intervenes.  We surface DLQ so the
    founder reviews the report path (e.g. similar_missions.md,
    prior_art_report.json) and either fixes the artifact directly or
    branches the mission.

    Fail (warning kinds): soft-drop the pending kind and advance source.
    The handler's own side-effects (write report + Telegram notify)
    surface the issue; the source step itself shouldn't block.
    """
    import json as _json
    from src.infra.db import update_task

    kind = verdict.kind

    if verdict.passed:
        new_pending = [k for k in pending if k != kind]
        ctx["_pending_posthooks"] = new_pending
        if not new_pending:
            await update_task(
                verdict.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, verdict.raw)
            try:
                from general_beckman import _send_step_progress
                from src.infra.db import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", verdict.raw or {})
            except Exception:
                pass
        else:
            await update_task(
                verdict.source_task_id, context=_json.dumps(ctx),
            )
        return

    raw = verdict.raw or {}

    # Warning kind — soft-drop pending and advance regardless.
    if kind in _Z1_WARNING_KINDS:
        new_pending = [k for k in pending if k != kind]
        ctx["_pending_posthooks"] = new_pending
        ctx[f"_{kind}_warning"] = (
            raw.get("error") or raw.get("summary") or "needs_review"
        )[:300]
        if not new_pending:
            await update_task(
                verdict.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            try:
                await _spawn_workflow_advance_if_mission(source, raw)
            except Exception:
                pass
            try:
                from general_beckman import _send_step_progress
                from src.infra.db import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", raw)
            except Exception:
                pass
        else:
            await update_task(verdict.source_task_id, context=_json.dumps(ctx))
        logger.info(
            "Z1 warning post-hook soft-dropped",
            kind=kind, source_id=verdict.source_task_id,
        )
        return

    # Blocker kind — DLQ source. No retry-with-feedback: these checks
    # are deterministic against on-disk artifacts; founder must intervene.
    error_detail = (
        raw.get("error")
        or str(raw.get("problems"))
        or str(raw.get("missing"))
        or str(raw.get("pending"))
        or "blocker post-hook failed"
    )
    error_str = f"{kind}: {error_detail}"[:500]
    ctx["_pending_posthooks"] = []
    ctx[f"_{kind}_dlq_reason"] = error_str[:300]
    attempts = int(source.get("worker_attempts") or 0) + 1
    await _dlq_write(
        source, error=error_str, category="quality", attempts=attempts,
    )
    await _maybe_emit_lesson_from_posthook_fail(
        source=source, kind=kind,
        error_str=error_str, feedback=error_str, attempts=attempts,
    )


def _posthook_title(a: RequestPostHook, source: dict) -> str:
    if a.kind == "grade":
        return f"Grade task #{a.source_task_id}"
    if a.kind.startswith("summary:"):
        name = a.kind.split(":", 1)[1]
        return f"Summarize '{name}' for #{a.source_task_id}"
    if a.kind == "verify_artifacts":
        return f"Verify artifacts for #{a.source_task_id}"
    if a.kind == "code_review":
        return f"Code review for #{a.source_task_id}"
    if a.kind == "grounding":
        return f"Grounding check for #{a.source_task_id}"
    if a.kind == "test_run":
        return f"Test run for #{a.source_task_id}"
    if a.kind == "pattern_lint":
        return f"Pattern lint for #{a.source_task_id}"
    if a.kind == "design_system_check":
        return f"Design system check for #{a.source_task_id}"
    if a.kind == "openapi_sync":
        return f"OpenAPI sync check for #{a.source_task_id}"
    if a.kind == "typescript_sync":
        return f"TypeScript sync check for #{a.source_task_id}"
    if a.kind == "migration_apply":
        return f"Migration apply for #{a.source_task_id}"
    if a.kind == "compliance_template_present":
        return f"Compliance template presence for #{a.source_task_id}"
    if a.kind == "compliance_blocker_check":
        return f"Compliance blocker check for #{a.source_task_id}"
    if a.kind == "find_similar_missions":
        return f"Find similar missions for #{a.source_task_id}"
    if a.kind == "index_idea_fingerprint":
        return f"Index idea fingerprint for #{a.source_task_id}"
    if a.kind == "surface_prior_mission_hints":
        return f"Surface prior mission hints for #{a.source_task_id}"
    if a.kind == "prior_art_min_coverage":
        return f"Prior art min coverage for #{a.source_task_id}"
    if a.kind == "verify_falsification_present":
        return f"Verify falsification present for #{a.source_task_id}"
    return f"Posthook {a.kind} for #{a.source_task_id}"


async def _apply_simple_blocker_verdict(
    kind: str,
    source: dict,
    ctx: dict,
    pending: list[str],
    verdict: PostHookVerdict,
    feedback_prefix: str,
) -> None:
    """Z3 T3 — generic pass/fail handler for mechanical security/access/contract/perf hooks.

    Pass: drop *kind* from pending; mark source completed when nothing left.
    Fail: retry source with feedback; DLQ on attempts exhausted (bonus path honored).
    """
    import json as _json
    from src.infra.db import update_task

    raw = verdict.raw or {}
    skipped = bool(raw.get("skipped"))

    if verdict.passed:
        new_pending = [k for k in pending if k != kind]
        ctx["_pending_posthooks"] = new_pending
        if skipped:
            logger.debug(f"{kind}: soft-skipped",
                         source_id=verdict.source_task_id,
                         reason=raw.get("reason") or "")
        if not new_pending:
            await update_task(
                verdict.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None, error_category=None, next_retry_at=None,
                retry_reason=None, failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, raw)
            try:
                from general_beckman import _send_step_progress
                from src.infra.db import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", raw)
            except Exception:
                pass
        else:
            await update_task(verdict.source_task_id, context=_json.dumps(ctx))
        return

    # Fail path.
    findings = raw.get("findings") or []
    findings_summary = "; ".join(
        (f.get("why") or f.get("kind") or str(f))[:80]
        for f in findings[:3]
    )
    error_str = f"{feedback_prefix}: {findings_summary}"[:500]
    feedback = (
        f"{feedback_prefix} failed. Fix the issues and re-emit. "
        f"Findings: {findings_summary}"
    )

    attempts = int(source.get("worker_attempts") or 0) + 1
    max_attempts = int(source.get("max_worker_attempts") or 15)
    ctx["_pending_posthooks"] = []

    if attempts >= max_attempts:
        from general_beckman.retry import _MAX_BONUS
        bonus_count = int(ctx.get("_bonus_count", 0))
        progress = _parse_progress(source)
        can_bonus = (
            progress is not None and progress >= 0.5 and bonus_count < _MAX_BONUS
        )
        if can_bonus:
            ctx["_bonus_count"] = bonus_count + 1
            max_attempts += 1
            ctx["_schema_error"] = feedback
            prev_output = source.get("result") or ""
            if isinstance(prev_output, str) and prev_output.strip():
                ctx["_prev_output"] = prev_output[:6000]
            _stamp_retry_feedback(ctx, attempts)
            await update_task(
                verdict.source_task_id, status="pending",
                worker_attempts=attempts, max_worker_attempts=max_attempts,
                error=error_str, error_category="quality",
                next_retry_at=None, context=_json.dumps(ctx),
            )
            return
        await _dlq_write(source, error=error_str or f"{kind} gate exhausted",
                         category="quality", attempts=attempts)
        return

    ctx["_schema_error"] = feedback
    prev_output = source.get("result") or ""
    if isinstance(prev_output, str) and prev_output.strip():
        ctx["_prev_output"] = prev_output[:6000]
    _stamp_retry_feedback(ctx, attempts)
    await update_task(
        verdict.source_task_id, status="pending",
        worker_attempts=attempts, error=error_str, error_category="quality",
        next_retry_at=None, context=_json.dumps(ctx),
    )


# Z3 T3A — security_review verdict is a thin wrapper around the generic helper
# above. Tests import this name directly.
async def _apply_security_review_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    await _apply_simple_blocker_verdict(
        kind="security_review", source=source, ctx=ctx, pending=pending,
        verdict=verdict, feedback_prefix="security_review gate",
    )


async def _apply_posthook_verdict(task: dict, a: PostHookVerdict) -> None:
    """Apply a post-hook verdict back to the source task."""
    import json as _json
    from src.infra.db import get_task, update_task, add_task
    from src.workflows.engine.artifacts import ArtifactStore

    source = await get_task(a.source_task_id)
    if source is None:
        logger.debug("posthook verdict: source missing",
                     source_id=a.source_task_id)
        return
    if source.get("status") != "ungraded":
        logger.debug(
            "posthook verdict: source no longer ungraded, dropping",
            source_id=a.source_task_id, status=source.get("status"),
        )
        return

    ctx = _parse_ctx(source)
    pending = list(ctx.get("_pending_posthooks") or [])

    if a.kind == "grade" and not a.passed:
        # Reject: retry the source with updated exclude list — but honor
        # the retry cap. Without this, a grader that keeps failing drives
        # the source through unbounded reruns (observed: 51 retries on
        # shopping_pipeline_v2 before manual kill, 2026-04-22).
        attempts = int(source.get("worker_attempts") or 0) + 1
        max_attempts = int(source.get("max_worker_attempts") or 15)
        # Extract the grader's human-readable text up front so the error
        # column is never a truncated dict repr. str(a.raw)[:500] used to
        # leave unterminated braces that _humanize_error couldn't parse,
        # leaking `{'passed': False, ...}` head to Telegram DLQ notices.
        error_str = _grader_verdict_text(
            a.raw, source_title=source.get("title", "") or "",
        )[:500]
        excluded = list(ctx.get("grade_excluded_models") or [])
        gen_model = ctx.get("generating_model") or ""
        if gen_model and gen_model not in excluded:
            excluded.append(gen_model)
        ctx["grade_excluded_models"] = excluded
        # Also add to worker-side exclusion list so the source task doesn't
        # re-pick the same model that just produced FAIL'd output.
        # Previously `grade_excluded_models` only fed the next grader pick;
        # the worker kept drawing Qwen3.5-9B-thinking attempt after attempt
        # and producing the same truncated output (2888
        # feature_prioritization looped 6+ attempts on the same model
        # 2026-04-24). `failed_models` is what `src.core.retry.
        # get_model_constraints` feeds into the selector exclusion at
        # attempts >= 3 — adding gen_model here lets the worker escalate
        # to a larger / non-thinking / cloud model.
        worker_failed = list(ctx.get("failed_models") or [])
        if gen_model and gen_model not in worker_failed:
            worker_failed.append(gen_model)
        ctx["failed_models"] = worker_failed
        ctx["_pending_posthooks"] = []

        if attempts >= max_attempts:
            # Bonus: mirror _retry_or_dlq — if task made progress and
            # bonus budget remains, grant one more attempt. Otherwise DLQ.
            from general_beckman.retry import _MAX_BONUS
            bonus_count = int(ctx.get("_bonus_count", 0))
            progress = _parse_progress(source)
            can_bonus = (
                progress is not None
                and progress >= 0.5
                and bonus_count < _MAX_BONUS
            )
            if can_bonus:
                ctx["_bonus_count"] = bonus_count + 1
                max_attempts += 1
                # Same feedback injection as the ordinary retry branch —
                # see the detailed comment further down.
                ctx["_schema_error"] = f"Grader rejected output: {error_str}"
                prev_output = source.get("result") or ""
                if isinstance(prev_output, str) and prev_output.strip():
                    ctx["_prev_output"] = prev_output[:6000]
                _stamp_retry_feedback(ctx, attempts)
                # error_category=quality so the next retry decision
                # (decide_retry) takes the immediate path, not the
                # availability backoff ladder. Grader rejection is a
                # quality failure by definition.
                await update_task(
                    a.source_task_id,
                    status="pending",
                    worker_attempts=attempts,
                    max_worker_attempts=max_attempts,
                    error=error_str,
                    error_category="quality",
                    next_retry_at=None,
                    context=_json.dumps(ctx),
                )
                return
            # Terminal — write to DLQ, transition to failed.
            await _dlq_write(
                source, error=error_str or "quality gate exhausted",
                category="quality", attempts=attempts,
            )
            await _maybe_emit_lesson_from_posthook_fail(
                source=source, kind="grade",
                error_str=error_str,
                feedback=f"Grader rejected output: {error_str}",
                attempts=attempts,
            )
            # Z10 wire-fix F8 — grade-FAIL/DLQ resolves confidence claim false.
            try:
                await _record_and_resolve_confidence(
                    task_id=a.source_task_id, correct=False, source="grade",
                )
            except Exception as _e:
                logger.debug(
                    "confidence claim record (grade DLQ) failed",
                    task_id=a.source_task_id, error=str(_e),
                )
            return

        # Feed the grader's rejection text + previous output back into the
        # task context so the agent's retry prompt sees it via the same
        # mechanism that post_execute_workflow_step uses for schema
        # validation failures. Without this, grader-FAIL retries went into
        # the next attempt blind — model produced the same truncated/
        # misclassified output over and over until the attempt cap hit DLQ
        # (2888 feature_prioritization observed 2026-04-24 at 5/6 attempts).
        ctx["_schema_error"] = f"Grader rejected output: {error_str}"
        prev_output = source.get("result") or ""
        if isinstance(prev_output, str) and prev_output.strip():
            try:
                from src.workflows.engine.hooks import (
                    canonicalize_for_retry, _unwrap_envelope,
                )
                # Unwrap final_answer envelope first so _prev_output
                # carries the bare artifact JSON, not the agent's
                # ``{"action":"final_answer","result":"..."}`` wrapper.
                # Without this, downstream parsers (per-artifact
                # checklist in base.py::_build_context) see the wrapper
                # keys instead of the artifact keys and falsely mark
                # every required field as missing.
                prev_output = _unwrap_envelope(prev_output)
                prev_output = canonicalize_for_retry(prev_output)
            except Exception:
                pass
            ctx["_prev_output"] = prev_output[:6000]
        _stamp_retry_feedback(ctx, attempts)
        # error_category=quality so decide_retry takes the immediate
        # path, not the availability backoff ladder. Grader rejection
        # is a quality failure by definition. Mission 46 task 4047
        # observed 2026-04-26: grader rejected, retry path defaulted
        # to error_category='worker' (left over from result_router),
        # next decide_retry call applied 60s+ exponential backoff.
        await update_task(
            a.source_task_id,
            status="pending",
            worker_attempts=attempts,
            error=error_str,
            error_category="quality",
            next_retry_at=None,
            context=_json.dumps(ctx),
        )
        # Z10 wire-fix F8 — grade-FAIL resolves confidence claim false.
        try:
            await _record_and_resolve_confidence(
                task_id=a.source_task_id, correct=False, source="grade",
            )
        except Exception as _e:
            logger.debug("confidence claim record (grade reject) failed",
                         task_id=a.source_task_id, error=str(_e))
        return

    if a.kind == "verify_artifacts":
        await _apply_verify_artifacts_verdict(
            source=source, ctx=ctx, pending=pending, verdict=a,
        )
        return

    # Z3 T3 + T4B + T5 — security/accessibility/contract/performance/adr_drift_check/
    # integration_replay share the simple blocker-or-pass pattern (mechanical
    # verb produces {findings, verdict}).
    if a.kind in (
        "security_review", "accessibility_review", "contract_review",
        "performance_review", "adr_drift_check", "integration_replay",
    ):
        await _apply_simple_blocker_verdict(
            kind=a.kind, source=source, ctx=ctx, pending=pending, verdict=a,
            feedback_prefix=f"{a.kind} gate",
        )
        return

    if a.kind == "grounding":
        await _apply_grounding_verdict(
            source=source, ctx=ctx, pending=pending, verdict=a,
        )
        return

    if a.kind == "code_review":
        await _apply_code_review_verdict(
            source=source, ctx=ctx, pending=pending, verdict=a,
        )
        return

    if a.kind == "test_run":
        await _apply_test_run_verdict(
            source=source, ctx=ctx, pending=pending, verdict=a,
        )
        return

    if a.kind == "pattern_lint":
        await _apply_pattern_lint_verdict(
            source=source, ctx=ctx, pending=pending, verdict=a,
        )
        return

    if a.kind == "design_system_check":
        await _apply_design_system_check_verdict(
            source=source, ctx=ctx, pending=pending, verdict=a,
        )
        return

    if a.kind == "migration_apply":
        await _apply_migration_apply_verdict(
            source=source, ctx=ctx, pending=pending, verdict=a,
        )
        return

    if a.kind in ("openapi_sync", "typescript_sync"):
        await _apply_type_sync_verdict(
            kind=a.kind, source=source, ctx=ctx, pending=pending, verdict=a,
        )
        return

    if a.kind in _Z1_MECHANICAL_KINDS:
        await _apply_z1_mechanical_verdict(
            source=source, ctx=ctx, pending=pending, verdict=a,
        )
        return


    if a.kind == "grade" and a.passed:
        # Remove "grade" from pending; spawn summary tasks for large artifacts.
        pending = [k for k in pending if k != "grade"]
        new_summary_kinds = await _summary_kinds_for_source(source, ctx)
        for kind in new_summary_kinds:
            pending.append(kind)
            await add_task(
                title=f"Summarize '{kind.split(':',1)[1]}' for #{a.source_task_id}",
                description="",
                agent_type="artifact_summarizer",
                mission_id=source.get("mission_id"),
                depends_on=[],
                context={
                    "source_task_id": a.source_task_id,
                    "artifact_name": kind.split(":", 1)[1],
                },
            )
        ctx["_pending_posthooks"] = pending
        if not pending:
            # Clear stale failure metadata on grade-PASS completion (B).
            await update_task(
                a.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            # Grader-mediated completion still needs to drive the workflow
            # forward. Without this, the final mission step passes grading
            # but no workflow_advance is ever spawned, so the engine never
            # runs _maybe_complete_mission — mission stays 'active' and
            # the shopping_response is never delivered to Telegram.
            await _spawn_workflow_advance_if_mission(source, a.raw)
            # Now that grader approved, surface the step completion to
            # the user. _send_step_progress (in general_beckman.__init__)
            # gates on live DB status, so it's safe to call with the
            # updated source.
            try:
                from general_beckman import _send_step_progress
                fresh = await get_task(a.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", a.raw or {})
            except Exception:
                pass
        else:
            await update_task(
                a.source_task_id, context=_json.dumps(ctx),
            )
        # Z10 wire-fix F8 — grade-PASS resolves confidence claim true. Wire
        # fires once whether or not summary post-hooks are still pending —
        # the grader verdict is the resolution signal, summaries are
        # orthogonal artefact extraction.
        try:
            await _record_and_resolve_confidence(
                task_id=a.source_task_id, correct=True, source="grade",
            )
        except Exception as _e:
            logger.debug("confidence claim record (grade pass) failed",
                         task_id=a.source_task_id, error=str(_e))
        return

    if a.kind.startswith("summary:"):
        artifact_name = a.kind.split(":", 1)[1]
        if a.passed:
            summary_text = a.raw.get("summary", "") if isinstance(a.raw, dict) else ""
            if summary_text:
                store = ArtifactStore()
                await store.store(
                    source.get("mission_id"),
                    f"{artifact_name}_summary",
                    summary_text,
                )
        # On fail: structural summary already stored by post_execute; nothing to do.
        pending = [k for k in pending if k != a.kind]
        ctx["_pending_posthooks"] = pending
        if not pending:
            # Clear stale failure metadata on summary-PASS completion (B).
            await update_task(
                a.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None,
                error_category=None,
                next_retry_at=None,
                retry_reason=None,
                failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, a.raw)
            # Step is genuinely done now — grader passed and all summary
            # posthooks resolved. Notify. Grader branch above notifies
            # early when there are no summary posthooks; this branch
            # covers the common path where a summary is required.
            try:
                from general_beckman import _send_step_progress
                fresh = await get_task(a.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", a.raw or {})
            except Exception:
                pass
        else:
            await update_task(
                a.source_task_id, context=_json.dumps(ctx),
            )
        return

    logger.warning("posthook verdict: unknown kind", kind=a.kind)


async def _spawn_workflow_advance_if_mission(source: dict, raw: object) -> None:
    """After a posthook-mediated source completion (grade/summary), spawn a
    workflow_advance mechanical task so the workflow engine advances and,
    when all steps are terminal, fires `_maybe_complete_mission` which
    delivers the final artifact to Telegram.

    Without this, the final step passes grading but the mission never
    transitions out of 'active' — user sees step pings, never the result.
    """
    from src.infra.db import add_task

    mission_id = source.get("mission_id")
    if mission_id is None:
        return

    # Only workflow-driven tasks need advance. Direct /task enqueues do not.
    ctx_raw = source.get("context") or "{}"
    try:
        sctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
    except Exception:
        sctx = {}
    if not (isinstance(sctx, dict) and sctx.get("is_workflow_step")):
        return

    previous_result = raw if isinstance(raw, dict) else {}
    await add_task(
        title=f"Workflow advance: mission #{mission_id}",
        description="",
        agent_type="mechanical",
        mission_id=mission_id,
        depends_on=[],
        context=_mechanical_context(
            "workflow_advance",
            mission_id=mission_id,
            completed_task_id=source.get("id"),
            previous_result=previous_result,
        ),
    )


async def _summary_kinds_for_source(source: dict, source_ctx: dict) -> list:
    """Return summary:<name> kinds for large output artifacts on this source.

    Reads the stored artifact values from the blackboard; enqueues one
    summary kind per artifact whose stored text exceeds 3000 chars.
    """
    from src.workflows.engine.artifacts import ArtifactStore

    mission_id = source.get("mission_id")
    if mission_id is None:
        return []
    output_names = list(source_ctx.get("output_artifacts") or [])
    if not output_names:
        return []
    store = ArtifactStore()
    kinds: list = []
    for name in output_names:
        val = await store.retrieve(mission_id, name)
        if val and isinstance(val, str) and len(val) > 3000:
            kinds.append(f"summary:{name}")
    return kinds


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
