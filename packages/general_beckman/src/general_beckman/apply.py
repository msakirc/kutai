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

    # Bump max_worker_attempts when retry policy uses a category-specific
    # higher cap (currently only "no_model": 10). Without this, sweep.py
    # section 8 force-DLQs any pending task with worker_attempts >=
    # task.max_worker_attempts (default 6), beating the retry layer's
    # internal 10-attempt budget. Production 2026-05-02: 10 tasks DLQ'd
    # with "Worker attempts exceeded: 6/6" despite category=no_model
    # because sweep didn't know about the higher internal cap. Persist
    # the cap on the row so sweep respects it.
    if category == "no_model":
        from general_beckman.retry import _NO_MODEL_MAX_ATTEMPTS
        if max_attempts < _NO_MODEL_MAX_ATTEMPTS:
            max_attempts = _NO_MODEL_MAX_ATTEMPTS

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
    # Post-hook DLQ cascade: a grader/summarizer task exhausting its
    # retries means the source task can no longer advance through the
    # verdict path. Synthesise the terminal outcome instead of leaving
    # the source stuck in 'ungraded' forever.
    if task.get("agent_type") in ("grader", "artifact_summarizer"):
        try:
            await _posthook_dlq_cascade(task, error)
        except Exception as exc:
            logger.warning("posthook DLQ cascade failed",
                           task_id=task["id"], error=str(exc))
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
        max_attempts = int(source.get("max_worker_attempts") or 6)
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
