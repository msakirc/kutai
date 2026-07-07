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

import asyncio
import json
from datetime import timedelta
from typing import Iterable

from yazbunu import get_logger
from dabidabi.times import to_db, utc_now

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


# ──────────────────────────────────────────────────────────────────────────
# SP3b FIX 2 — per-source verdict serialization.
#
# A single source task can have MULTIPLE post-hook children completing
# concurrently: the chain cursor (emit→reflect→grade) plus independent
# blockers (grounding / verify_artifacts / test_run / ...). Each completion
# fires its resume via ``asyncio.create_task`` (continuations.fire_for_task),
# so several verdict appliers run as separate coroutines and do unserialized
# read-modify-write on the SAME ``tasks.context`` (``_pending_posthooks`` /
# ``_posthook_queue``). Lost updates → premature completion OR a permanent
# 'ungraded' stall.
#
# KutAI runs a SINGLE orchestrator process, so one asyncio.Lock per source_id
# serializes every read-modify-write of that source's context. The dict is
# unbounded in principle, but post-hook traffic is low and stale locks are
# cheap (one empty Lock object) — we drop the entry on best-effort cleanup
# once no applier holds it.
#
# Re-entrancy: ``asyncio.Lock`` is NOT reentrant, but the verdict/chain call
# graph is deeply nested within ONE coroutine chain (e.g.
# _apply_request_posthook → _advance_posthook_chain → _enqueue_posthook_llm_
# child → _apply_posthook_verdict auto-fail). A plain ``async with lock`` at
# every entry would self-deadlock. We track the source_ids the CURRENT task
# already holds via a contextvar set and make ``_source_verdict_guard`` a
# no-op re-acquire when already held — a lightweight reentrant lock scoped to
# the running coroutine chain. Concurrent appliers (separate create_task
# coroutines) get fresh contextvar copies, so they correctly block.
# ──────────────────────────────────────────────────────────────────────────
import contextvars as _contextvars
from contextlib import asynccontextmanager as _asynccontextmanager

_source_verdict_locks: dict[int, asyncio.Lock] = {}
_held_source_locks: _contextvars.ContextVar[frozenset[int]] = _contextvars.ContextVar(
    "_held_source_locks", default=frozenset()
)


def _source_verdict_lock(source_task_id: int) -> asyncio.Lock:
    """Return (creating on first use) the serialization lock for a source."""
    sid = int(source_task_id)
    lock = _source_verdict_locks.get(sid)
    if lock is None:
        lock = asyncio.Lock()
        _source_verdict_locks[sid] = lock
    return lock


@_asynccontextmanager
async def _source_verdict_guard(source_task_id: int):
    """Serialize read-modify-write of a source's context across concurrent
    verdict appliers. Reentrant within a single coroutine chain (see module
    note): if THIS task already holds the source lock, the inner ``async with``
    is a no-op pass-through rather than a self-deadlock."""
    sid = int(source_task_id)
    held = _held_source_locks.get()
    if sid in held:
        # Already held by an enclosing frame in this coroutine chain.
        yield
        return
    lock = _source_verdict_lock(sid)
    token = _held_source_locks.set(held | {sid})
    try:
        async with lock:
            yield
    finally:
        _held_source_locks.reset(token)
        # NOTE: intentionally NO eviction of _source_verdict_locks[sid] here.
        # CPython 3.10: Lock.release() clears _locked and *then* schedules the
        # next waiter.  In the window between those two steps lock.locked()
        # returns False while a waiter (applier B) is still queued on the OLD
        # Lock object.  Popping the dict entry in that window lets a new
        # applier C create a SECOND distinct Lock for the same source; B (old)
        # and C (new) then run their read-modify-write concurrently → lost
        # update → stranded 'ungraded' or double-completion.  This is the
        # exact race the lock was added to prevent.  Leaking one empty
        # asyncio.Lock per source-task-id is the intended cheap trade-off
        # (the module comment already concedes stale locks are cheap).


async def _record_and_resolve_confidence(
    task_id: int, correct: bool, source: str,
    reviewer_verdict_id: int | None = None,
) -> None:
    """Z10 T4B — record + immediately resolve a confidence claim.

    Skips silently if the source task has no confidence signal (record
    returns None) or already resolved. Safe in mechanical/skipped paths.

    Sibling loop: Z9 reinforce (mr_roboto/executors/record_verdict.py::
    _reinforce_winning_model). Kept deliberately separate — Z9 fires only
    on confirmed hypothesis verdicts and adjusts model SELECTION score;
    Z10 fires on every post-hook verdict and adjusts the LLM PROMPT via a
    per-bucket reliability rollup. Different signals, different surfaces,
    different aggregation cadence. See docs/handoff/
    2026-05-18-z0-and-backlog-handoff.md §2c.
    """
    from dabidabi import (
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
    from dabidabi import update_task
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
                from yalayut.recipes import pin_recipes_from_artifact
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
    from dabidabi import add_subtasks_atomically
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
    from dabidabi import add_task, update_task
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
        from dabidabi import get_task as _get
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
    from dabidabi import add_task, get_db, update_task
    # A MECHANICAL post-hook task that returned needs_review (e.g.
    # find_similar_missions: prior missions matched) has already done its job:
    # mr_roboto enqueued the founder Continue/Branch/Abort notice and rewrite
    # Rule 0c' synthesised the SOURCE's PostHookVerdict. The post-hook task
    # itself must now go TERMINAL — left non-terminal it was swept back,
    # re-run, and DLQ'd at the worker_attempts cap (bug 2026-05-26: #166396
    # find_similar DLQ'd 6× with its own needs_review string as the "error").
    # And a mechanical gate must never spawn a reviewer AGENT.
    _ctx = _parse_ctx(task)
    if task.get("agent_type") == "mechanical" and _ctx.get("posthook_kind"):
        await update_task(a.task_id, status="completed")
        logger.info(
            "mechanical post-hook needs_review completed (founder notice "
            "already surfaced; source verdict synthesised)",
            task_id=a.task_id, kind=_ctx.get("posthook_kind"),
        )
        return
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


# Unambiguous availability markers in a failure's error text. An availability
# failure (no model candidates / rate limit / daily quota / all models failed)
# must ride the backoff ladder and wait for capacity to return — it must NEVER
# take decide_retry's quality fast-path (immediate retry, no backoff) and burn
# worker_attempts to DLQ inside the quota-reset window. We sniff the text as a
# last line of defence because the upstream error_category is sometimes lost
# (mission_79 2026-05-30 task #225600: ModelCallFailed(error_category=
# "availability") raised in coulson, but _apply_failed received a stale
# "quality" category from the prior grader-FAIL row → fast-DLQ'd in seconds).
# Kept narrow: bare "unavailable" is excluded (it appears in "grader verdict
# unavailable", a genuine quality failure).
_AVAILABILITY_MARKERS: tuple[str, ...] = (
    "no model candidates",
    "no models available",
    "no models matched",
    "no candidates available",
    "all models failed",
    "rate limit",
    "rate_limit",
    "too many requests",
    "daily",
    "quota",
)


def _classify_availability_text(error: str | None) -> str | None:
    """Return "availability" if the error text unambiguously signals an
    availability/capacity failure, else None. Pure; safe on None/empty."""
    if not error or not isinstance(error, str):
        return None
    e = error.lower()
    if any(m in e for m in _AVAILABILITY_MARKERS):
        return "availability"
    return None


def _grade_verdict_is_availability(verdict) -> bool:
    """True when a grade FAIL verdict is an AUTO-FAIL caused because the grade
    CHILD couldn't get a model (cloud daily-exhausted / no candidates / load
    fail) — NOT a genuine quality rejection of the source artifact.

    posthook_continuations._grade_resume_err / _grade_resume build the auto-fail
    shape ``{"passed": False, "raw": "auto-fail: grader call failed (...)"}`` —
    the reason lives ONLY under the ``raw`` key behind an ``auto-fail:`` prefix.
    We require BOTH that shape AND an availability marker so a grader's free-text
    verdict (insight/strategy/...) that merely mentions "daily" / "quota" /
    "rate limit" about the artifact's CONTENT is never misread as availability —
    a raw sniff of the whole verdict text would false-positive (the genuine
    quality path must stay immediate-retry, not back off). Pure; safe on any
    shape. See project_grader_verdict_autofail_20260530 (PART 2)."""
    raw = getattr(verdict, "raw", None)
    if not isinstance(raw, dict):
        return False
    msg = raw.get("raw")
    if not _is_meaningful_text(msg) or "auto-fail" not in msg.lower():
        return False
    return _classify_availability_text(msg) == "availability"


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
    # Text sniff overrides a stale/wrong category: an availability failure
    # must never wear a "quality" (or any non-availability) label, or
    # decide_retry fast-DLQs it instead of backing off until capacity
    # returns (mission_79 #225600: "No model candidates available" carried
    # error_category="quality" → DLQ in seconds against daily-exhausted
    # gemini). The error text is the authoritative signal when present.
    sniffed = _classify_availability_text(a.error)
    if sniffed and category != "availability":
        logger.info(
            "availability sniff override",
            task_id=task.get("id"),
            stale_category=category,
            error=(a.error or "")[:120],
        )
        category = sniffed
    await _retry_or_dlq(task, category=category, error=a.error)


async def _apply_mission_advance(task: dict, a: MissionAdvance) -> None:
    from dabidabi import add_task
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
    from dabidabi import get_task as _get_task, update_task
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

    # T7 (Phase 3) — degenerate repeat detector. A quality-rejected worker
    # that emits byte-identical output across re-dispatches is not converging
    # (the proven symptom: 48 identical attempts). Hash THIS attempt's output
    # and compare to the immediately-prior _rejection_ledger entry's out_hash
    # (stamped in Phase 1). If they match AND are non-None, escalate to the DLQ
    # instead of re-dispatching a 49th identical attempt.
    #   - hashing parity: we hash task["result"] (the raw result field) with the
    #     SAME _output_hash the ledger appliers used (source.get("result") in
    #     apply.py) — apples-to-apples for the dominant grade/review/verify retry
    #     path that funnels here. The refetched task carries this attempt's result.
    #   - LIMIT (F6): exact-hash only; semantic near-duplicates won't match
    #     (out of scope for Phase 3).
    #   - guards: only quality rejections (availability/transient store no judged
    #     output and append nothing to the ledger); empty ledger or a None prior
    #     out_hash → no detection, normal retry; never fires on the first attempt
    #     (no prior entry exists yet).
    if category == "quality":
        ledger = ctx.get("_rejection_ledger") or []
        if ledger:
            prior_hash = ledger[-1].get("out_hash")
            try:
                from src.workflows.engine.hooks import _output_hash
                current_hash = _output_hash(task.get("result"))
            except Exception:  # pragma: no cover - defensive
                current_hash = None
            if prior_hash and current_hash and prior_hash == current_hash:
                logger.warning(
                    "degenerate repeat — identical output, escalating to DLQ",
                    task_id=task.get("id"),
                    attempts=attempts,
                    out_hash=current_hash,
                )
                await _dlq_write(
                    task,
                    error=(
                        "degenerate repeat: identical output across attempts, "
                        "not converging"
                    ),
                    category="quality",
                    attempts=attempts,
                )
                return

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

    # Rejection ledger (T1) — the "completed with empty result" quality
    # failure flows here (result_router emits Failed with no judged output
    # and no _schema_error). Record it so the retry prompt shows the prior
    # empty attempt. Availability/infra retries are NOT quality and append
    # nothing (spec C2). _retry_or_dlq is the single persist funnel for
    # Failed/Exhausted, so the append survives via the update_task below.
    if category == "quality" and "empty result" in (error or ""):
        from src.workflows.engine.hooks import append_rejection
        append_rejection(ctx, attempts, "empty result", None)

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
        from kara_kutu import upsert_mission_lesson
        from dabidabi import get_db
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
                "mission_id": source.get("mission_id"),
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
    from dabidabi import add_task, update_task
    from src.infra.dead_letter import quarantine_task, is_unresolved_dlq
    # Dedupe the user-facing alert: a fail-looping task is re-DLQ'd on each
    # retry/re-pend cycle. quarantine_task collapses to one row, but the notify
    # (+demand signal) must fire only on the FIRST quarantine — check BEFORE the
    # (idempotent) quarantine call creates/refreshes the row.
    try:
        _already_dlq = await is_unresolved_dlq(task["id"])
    except Exception:
        _already_dlq = False
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
    # SP3: the LLM posthook CHILDREN (reviewer/summarizer raw_dispatch) carry
    # source_task_id/mission_id in cont_state ONLY — their DLQ is handled by the
    # posthook.*.resume_err continuation (auto-fail verdict drives the source's
    # normal retry/DLQ), not by this cascade. Mechanical posthook tasks still
    # carry source_task_id+posthook_kind in their context and cascade here.
    _ctx_for_cascade = _parse_ctx(task)
    if (
        _ctx_for_cascade.get("source_task_id") is not None
        and _ctx_for_cascade.get("posthook_kind")
    ):
        try:
            await _posthook_dlq_cascade(task, error)
        except Exception as exc:
            logger.warning("posthook DLQ cascade failed",
                           task_id=task["id"], error=str(exc))
        # FIX 2.1 — the cascade may have flipped the SOURCE terminal
        # (failed for blocker kinds, completed for drained warning kinds)
        # inside the CHILD's DLQ write; fire the source's continuation now.
        await _fire_source_continuation(_ctx_for_cascade.get("source_task_id"))
    # Notify + demand signal fire ONCE per quarantine. A fail-looping task is
    # re-DLQ'd on each retry/re-pend cycle; quarantine_task collapses to one row,
    # but re-emitting the alert floods Telegram (and, when mission-scoped, spawns
    # a redundant critic gate per duplicate). Skip both on a re-DLQ — the user was
    # already alerted and the demand recorded on the first quarantine.
    if _already_dlq:
        return
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
            # Internal admin status ping, not outward-facing agent comms. Bypass
            # the SP6 critic gate \u2014 a stalled/vetoing critic must never silence
            # the user's visibility into task failures (mission_id != None alone
            # is not "outward-facing").
            critic_gate=False,
        ),
        depends_on=[],
    )
    # Yalayut demand signal: a DLQ'd task is unmet demand -- its intent could
    # not be satisfied. Record it (reactive `dlq` signal) WITHOUT importing
    # yalayut into this core-loop file; route through a mechanical task.
    _title = (task.get("title") or "").strip()
    if _title:
        await add_task(
            title=f"Demand signal: DLQ #{task['id']}",
            description="",
            agent_type="mechanical",
            mission_id=task.get("mission_id"),
            context=_mechanical_context(
                "yalayut_demand",
                source_step_pattern=f"dlq:{_title[:40]}",
                intent_keywords=[w for w in _title.split() if len(w) > 2][:12],
                signal_type="dlq",
                confidence=0.3,
            ),
            depends_on=[],
        )



_NULLISH_STRINGS = {"", "none", "null", "nil", "n/a", "na", "-"}


def _record_failed_model(ctx: dict) -> None:
    """Append the source's ``generating_model`` to ``ctx['failed_models']`` so
    the retry-escalation MODEL-EXCLUSION arm engages on the next worker pick.

    ``src.core.retry.get_model_constraints`` (read by fatih_hoca's
    requirements_builder at ``worker_attempts >= 3``) has TWO arms: a
    difficulty bump (keyed on attempt count alone, so it fires for any path
    that retries) and a model EXCLUSION (keyed on ``failed_models``). Before
    2026-06-04 only the grade-FAIL branch populated ``failed_models``, so every
    OTHER quality re-pend (verify_artifacts / code_review / test_run / semgrep /
    pattern_lint / shape & blocker checks / prior_art coverage) could re-draw
    the exact model that just produced the bad output — only the difficulty bump
    nudged it. Recording the failed model here closes that gap UNIVERSALLY:
    every quality re-pend funnels through ``_stamp_retry_feedback`` (its sole
    chokepoint), so this one call makes all of them escalate symmetrically.

    Idempotent (dedup) and a no-op when ``generating_model`` is unknown.
    """
    gen = ctx.get("generating_model") or ""
    if not gen:
        return
    failed = list(ctx.get("failed_models") or [])
    if gen not in failed:
        failed.append(gen)
        ctx["failed_models"] = failed


def _stamp_retry_feedback(
    ctx: dict,
    next_attempt: int,
    *,
    reason=None,
    prev_output=None,
) -> bool:
    """Prepare ``ctx`` for the next quality-retry attempt and report whether the
    attempt is a degenerate repeat. Called by EVERY quality re-pend branch
    (grade + all mechanical checks), so it is the single place that guarantees
    the per-attempt invariants hold for all of them:

    1. Tag freshly-written ``_schema_error``/``_prev_output`` with the attempt
       number they were written FOR. Readers gate on this so stale feedback from
       earlier failure modes (e.g. schema reject 2 attempts ago, then
       availability bounce that re-queues without rewriting) doesn't leak into
       the next prompt as ``"your last output failed: <unrelated>"``.
    2. Record the failing model in ``failed_models`` so the retry escalation's
       model-exclusion arm engages — not just the difficulty bump. This is a
       QUALITY-only chokepoint (availability/infra retries ride decide_retry,
       never this), so excluding the model is always correct here.
    3. Append the rejection to the conversation rejection ledger (T1/GAP-1) so
       EVERY quality re-pend — not just the 4 appliers that historically had a
       scattered ``_ledger_reject`` — feeds the "Prior attempts (do not repeat)"
       render in coulson/context.py. ``reason`` is the precise per-applier
       attribution (e.g. ``"grade: …"``); when omitted it falls back to the
       freshly-written ``_schema_error`` feedback text. ``prev_output`` is the
       raw produced output (``source.get("result")``) — hashed the SAME way the
       appliers and ``_retry_or_dlq`` hash it, so the ledger's ``out_hash`` is
       apples-to-apples within an applier-driven loop.
    4. Degenerate-repeat detection (T7/GAP-2): hash ``prev_output`` and compare
       to the IMMEDIATELY-PRIOR ledger entry's ``out_hash`` BEFORE appending the
       new entry (compare-then-append → a single attempt never self-matches).
       Return ``True`` when the hashes are non-None and equal — the caller must
       DLQ instead of re-pending a non-converging attempt. The verdict appliers
       re-pend DIRECTLY (never via ``_retry_or_dlq``), so this is the only place
       the detector can fire on the dominant grade/review/verify path.

       LIMIT (F6): exact-hash only; semantic near-duplicates won't match.
       Hash-parity caveat: the degenerate hooks path hashes the unwrapped
       envelope, not the raw result — if a loop mixes both the detector
       under-detects (never false-DLQs), which is the safe direction.
    """
    _record_failed_model(ctx)
    if "_schema_error" in ctx or "_prev_output" in ctx:
        ctx["_schema_error_for_attempt"] = int(next_attempt)

    escalate = False
    try:
        from src.workflows.engine.hooks import append_rejection, _output_hash

        cur = _output_hash(prev_output)
        ledger = ctx.get("_rejection_ledger") or []
        prior = ledger[-1].get("out_hash") if ledger else None
        escalate = bool(cur) and bool(prior) and cur == prior
        _reason = reason if reason is not None else (
            ctx.get("_schema_error") or "quality rejection"
        )
        append_rejection(ctx, next_attempt, _reason, cur)
    except Exception as _e:  # pragma: no cover - defensive
        logger.debug("rejection-ledger append skipped", error=str(_e))
    return escalate


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
        # Auto-fail verdicts (grade child couldn't get a model, or grader
        # incapable after retries) carry their message ONLY under ``raw`` —
        # _grade_resume_err / _grade_resume build {"passed": False, "raw":
        # "auto-fail: grader call failed (...)"}. Without reading ``raw`` the
        # real reason ("No model candidates available") was dropped and every
        # such DLQ showed the opaque "grader verdict unavailable" sentinel
        # (mission_79 #225586, 2026-05-30 — an availability failure hiding as
        # a grade DLQ). Read it as the last meaningful-text candidate.
        raw_val = candidate.get("raw")
        if _is_meaningful_text(raw_val) and not _is_title_echo(raw_val, source_title):
            return raw_val.strip()[:140]
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
    from dabidabi import get_task, update_task

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

    # SP3: the grader-specific DLQ cascade is gone. A failed grade CHILD
    # (agent_type 'reviewer', mission-less, source_task_id in cont_state) now
    # fires posthook.grade.resume_err, which applies an auto-fail verdict that
    # drives the source's normal retry/DLQ. The same holds for the summary
    # child (agent_type 'summarizer'): its DLQ is handled by
    # posthook.summary.resume_err, not here. Only mechanical posthook tasks —
    # which carry posthook_kind in their own context — still cascade below.
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

    if posthook_kind == "domain_layer_check":
        # Z3 T4C — domain_layer_check mechanical task DLQ'd (e.g. workspace
        # permission denied, unexpected semgrep crash).  A mechanical DLQ here
        # means the layer-filter wrapper itself crashed — NOT that semgrep found
        # violations (those are surfaced via verdict, not DLQ).  Soft-drop the
        # kind and let the source advance, mirroring pattern_lint's DLQ policy.
        source_ctx["_pending_posthooks"] = [
            k for k in (source_ctx.get("_pending_posthooks") or [])
            if k != "domain_layer_check"
        ]
        source_ctx["_domain_layer_check_dlq_reason"] = error[:300]
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
            "domain_layer_check DLQ soft-dropped (semgrep crash — not a findings violation)",
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
    # Z5 T4b — mobile_smoke joins this group: a DLQ'd Maestro flow gate
    # means the build step can't advance, so cascade source to failed.
    if posthook_kind in (
        "security_review", "accessibility_review", "contract_review",
        "performance_review", "adr_drift_check", "integration_replay",
        "integration_review", "adr_drift_judge", "visual_review", "mobile_smoke",
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

    if posthook_kind == "capture_hint":
        # Yalayut Phase 4 — capture_hint is warning-only telemetry; a DLQ of
        # the capture task itself must never cascade the source to failed.
        # Soft-drop the pending kind and advance the source if none remain.
        source_ctx["_pending_posthooks"] = [
            k for k in (source_ctx.get("_pending_posthooks") or [])
            if k != "capture_hint"
        ]
        new_pending = source_ctx["_pending_posthooks"]
        if not new_pending:
            await update_task(
                source_id, status="completed",
                context=_json.dumps(source_ctx),
                error=None, error_category=None,
                next_retry_at=None, retry_reason=None, failed_in_phase=None,
            )
            try:
                await _spawn_workflow_advance_if_mission(source, {})
            except Exception:
                pass
        else:
            await update_task(source_id, context=_json.dumps(source_ctx))
        logger.warning("capture_hint DLQ soft-dropped (warning-only kind)",
                       source_id=source_id, posthook_task_id=task["id"])
        return



# SP3: the LLM evaluation post-hooks (grade / code_review / summary:*) are now
# enqueued as raw_dispatch reviewer/summarizer CHILDREN with kind='overhead'
# directly by _enqueue_posthook_llm_child — they no longer flow through
# _posthook_agent_and_payload, so the old _OVERHEAD_POSTHOOK_AGENTS frozenset
# (grader/artifact_summarizer) is gone. _posthook_kind now only serves the
# mechanical post-hook branches that still reach add_task() below; mechanical
# post-hooks (verify_artifacts/test_run/pattern_lint/...) route via
# agent_type='mechanical' (runner='mechanical') on the main_work kind.
def _posthook_kind(agent_type: str) -> str:
    """Task ``kind`` for a spawned (mechanical) post-hook of the given agent_type."""
    return "main_work"


#: SP3b Task 6 — the RESULT-REWRITING post-hooks. Unlike every other (~40)
#: kind, these REWRITE the source's result in place rather than gating it, and
#: they must run BEFORE the terminal grade gate. They are NOT added to
#: ``_pending_posthooks`` (they gate nothing); the chain's ``_posthook_queue``
#: cursor sequences them ahead of grade instead.
_REWRITE_POSTHOOK_KINDS: frozenset[str] = frozenset(
    {"constrained_emit", "self_reflect"}
)


async def _fire_source_continuation(source_task_id) -> None:
    """FIX 2.1 chokepoint — fire the SOURCE's own continuation when a
    verdict-completion path just flipped it terminal.

    A source completed/failed by a verdict apply, a chain drain, or the
    post-hook DLQ cascade reaches terminal during ANOTHER task's
    on_task_finished, so its own continuation (e.g. the swap-chain
    image_done/image_err resumes) never fired until the reconcile TTL.
    Best-effort: a fire hiccup must never break the verdict path; the
    periodic reconcile remains the safety net.
    """
    if source_task_id is None:
        return
    try:
        from general_beckman.continuations import fire_if_terminal
        await fire_if_terminal(int(source_task_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("source continuation fire failed",
                       source_id=source_task_id, error=str(exc))


async def _apply_request_posthook(task: dict, a: RequestPostHook) -> None:
    """Park the source in `ungraded`, enqueue a post-hook task row.

    SP3b FIX 2 — the entire park-then-spawn region reads + rewrites the source
    ``context`` (``_pending_posthooks`` / ``_posthook_queue``). Serialize it
    against concurrent verdict appliers on the same source.
    """
    async with _source_verdict_guard(a.source_task_id):
        await _apply_request_posthook_locked(task, a)
    # FIX 2.1 — the locked body can complete the source synchronously (chain
    # drain via the direct _advance_posthook_chain_locked call, or the
    # trivial-source auto-fail short-circuit inside the child enqueue).
    await _fire_source_continuation(a.source_task_id)


async def _apply_request_posthook_locked(task: dict, a: RequestPostHook) -> None:
    import json as _json
    from dabidabi import add_task, get_task, update_task

    source = await get_task(a.source_task_id)
    if source is None:
        logger.warning("posthook: source missing", source_id=a.source_task_id)
        return

    # Use a.source_ctx (built by rewrite.py with result scalars merged in, e.g.
    # "draft" from incident/draft_update) rather than the DB-read ctx.  The DB
    # ctx is intentionally not updated with result scalars — it stores the task's
    # original input context.  a.source_ctx is the enriched view that posthook
    # handlers need to read fields like "draft", "incident_id", "status_kind".
    # ctx above is still used for _pending_posthooks tracking + update_task only.
    posthook_ctx = a.source_ctx if a.source_ctx else None

    # SP3b Task 6 — ordered rewrite→grade chain. When rewrite.py partitions the
    # constrained_emit→self_reflect→grade trio it emits a SINGLE RequestPostHook
    # for the chain HEAD, carrying the ordered ``_posthook_queue`` in source_ctx.
    # Sequence it as a cursor walk: stash the queue + park ONLY the gating kinds
    # (grade) in _pending_posthooks, then advance the cursor (spawns the head
    # with skip-drain). The rewriters never enter _pending_posthooks — they
    # rewrite, they don't gate, so grade-PASS still drains pending to empty.
    chain_queue = list((posthook_ctx or {}).get("_posthook_queue") or [])
    if chain_queue:
        ctx = _parse_ctx(source)
        pending = list(ctx.get("_pending_posthooks") or [])
        # Only the GATING members of the chain (grade + anything that isn't a
        # pure rewriter) go in pending — the source stays ungraded until those
        # resolve. Rewriters are sequenced by the queue cursor, not pending.
        for k in chain_queue:
            if k in _REWRITE_POSTHOOK_KINDS:
                continue
            if k not in pending:
                pending.append(k)
        ctx["_pending_posthooks"] = pending
        # Persist the ordered cursor (full source ctx survives the DB round-trip
        # as JSON) so resume continuations can advance after a restart.
        ctx["_posthook_queue"] = chain_queue
        # Carry the enriched scalars (draft/etc.) the rewriters' children read.
        for _k, _v in (posthook_ctx or {}).items():
            if _k in ("_posthook_queue", "_pending_posthooks"):
                continue
            ctx.setdefault(_k, _v)
        await update_task(
            a.source_task_id,
            status="ungraded",
            context=_json.dumps(ctx),
        )
        # We already hold the source guard — call the unlocked body directly.
        await _advance_posthook_chain_locked(a.source_task_id)
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

    if posthook_ctx is None:
        posthook_ctx = ctx

    # SP3: the LLM post-hook kinds (grade / code_review / summary:* and the SP3b
    # rewriters constrained_emit / self_reflect) spawn the raw_dispatch
    # reviewer/summarizer/emit/reflect CHILD directly with a durable
    # continuation — no cap-counted grader/code_reviewer/artifact_summarizer
    # agent task. Mechanical kinds (verify_artifacts/grounding/imports_check/
    # test_run/pattern_lint/...) keep the agent-task enqueue path below intact.
    if (
        a.kind == "grade"
        or a.kind == "code_review"
        or a.kind == "critic_gate"          # SP6 — admitted critic child
        or a.kind.startswith("summary:")
        or a.kind in _REWRITE_POSTHOOK_KINDS
    ):
        await _enqueue_posthook_llm_child(a.kind, source, posthook_ctx)
        return

    agent_type, payload = _posthook_agent_and_payload(a, source, posthook_ctx)

    # Z3 T2C: For integration_review, run the mechanical AST pre-check here
    # (async context) and inject the result into the payload before enqueue.
    if a.kind == "integration_review":
        payload = await _enrich_integration_review_payload(payload)

    await add_task(
        title=_posthook_title(a, source),
        description="",
        agent_type=agent_type,
        kind=_posthook_kind(agent_type),
        mission_id=source.get("mission_id"),
        depends_on=[],
        context=payload,
    )


async def _advance_posthook_chain(source_task_id: int) -> None:
    """Spawn the HEAD of the source's ordered ``_posthook_queue`` cursor.

    SP3b Task 6 — the constrained_emit→self_reflect→grade trio is sequenced,
    not fanned out: this function pops the queue head, persists the shortened
    queue, and enqueues that kind's child. Each rewriter's resume continuation
    calls back here after applying its verdict, spawning the NEXT kind; grade is
    the terminal entry and never chains further.

    Skip-drain (handoff #2): ``_enqueue_posthook_llm_child`` returns ``False``
    when a rewriter is a no-op (draft already conforms / empty / unconstrainable)
    — no child is spawned. In that case we MUST advance to the next kind in the
    same call so the source never stalls in 'ungraded' with a dead cursor. Loop
    until a child is actually spawned or the queue drains.

    SP3b FIX 2 — the cursor pop is a read-modify-write of the source context;
    serialize it against concurrent verdict appliers (reentrant within this
    coroutine chain, so callers that already hold the guard re-enter freely).
    """
    async with _source_verdict_guard(source_task_id):
        await _advance_posthook_chain_locked(source_task_id)
    # FIX 2.1 — a drained cursor completes the source synchronously
    # (_complete_source_if_no_pending); fire its continuation now.
    await _fire_source_continuation(source_task_id)


async def _advance_posthook_chain_locked(source_task_id: int) -> None:
    import json as _json
    from dabidabi import get_task, update_task

    while True:
        source = await get_task(source_task_id)
        if source is None:
            logger.warning("posthook chain: source missing",
                           source_id=source_task_id)
            return
        # SP3b FIX 3 — status re-check. An independent blocker (e.g. grounding /
        # test_run) may have failed concurrently and re-pended the source
        # ('pending' for retry, or 'failed'/'completed' via its own gate). If the
        # source is no longer 'ungraded', BAIL: spawning the next chain child
        # (e.g. a grade) would grade a stale draft against a task that already
        # moved on. The chain is abandoned safely — the blocker now owns the
        # source's fate, and a restart reconcile only re-drives genuinely-stuck
        # 'ungraded' rows.
        if source.get("status") != "ungraded":
            logger.debug(
                "posthook chain: source no longer ungraded, abandoning cursor",
                source_id=source_task_id, status=source.get("status"),
            )
            return
        ctx = _parse_ctx(source)
        queue = list(ctx.get("_posthook_queue") or [])
        if not queue:
            # Cursor drained. In the normal trio the terminal entry is grade,
            # which owns source completion via its gate verdict — nothing to do
            # here. But a chain that has rewriters with NO grade (e.g. a
            # requires_grading=False step that still opted into emit/reflect)
            # would otherwise strand the source in 'ungraded' forever. If no
            # gating post-hook remains pending, complete the source now.
            await _complete_source_if_no_pending(source_task_id, ctx)
            return
        head = queue[0]
        remaining = queue[1:]
        # Persist the shortened cursor BEFORE spawning the child so a restart
        # mid-spawn doesn't replay the head, and so the child's resume reads the
        # already-advanced queue.
        ctx["_posthook_queue"] = remaining
        await update_task(source_task_id, context=_json.dumps(ctx))
        spawned = await _enqueue_posthook_llm_child(head, source, ctx)
        if spawned:
            return
        # Skipped rewriter (no child) — drain to the next kind immediately.
        logger.debug("posthook chain: kind skipped, advancing",
                     source_id=source_task_id, kind=head)


async def reconcile_stranded_posthook_chains() -> int:
    """SP3b FIX 4 — re-drive post-hook chains stranded by a mid-advance crash.

    ``_advance_posthook_chain`` persists the shortened ``_posthook_queue`` in its
    OWN commit BEFORE ``_enqueue_posthook_llm_child`` writes the child's
    continuation row. A crash in that window leaves the source 'ungraded' with a
    non-empty queue, NO child task, and NO pending continuation — no event will
    ever re-drive it, so the source (and its mission step) hangs forever.

    This additive startup/periodic sweep finds those rows and re-calls
    ``_advance_posthook_chain`` (which re-reads the queue, re-persists, and
    spawns the head). It runs ALONGSIDE ``continuations.reconcile_continuations``
    — that pass recovers in-flight children whose continuation row DOES exist;
    this pass recovers the gap where it does NOT.

    Double-spawn guard (load-bearing): a source is re-driven ONLY when there is
    NO pending post-hook continuation referencing it. If a child IS in flight
    (continuation row pending), the queue is already advanced past its head and
    that child's resume will continue the chain — re-driving here would spawn a
    SECOND head and grade a stale draft twice. The per-source verdict guard
    additionally serializes this against any concurrent live advance.

    Returns the number of chains re-driven (for logging / tests).
    """
    import json as _json
    from dabidabi import get_db

    db = await get_db()

    # 1. Candidate sources: ungraded with a non-empty _posthook_queue. The
    #    JSON LIKE is a cheap pre-filter; we re-parse to confirm below.
    cur = await db.execute(
        "SELECT id, context FROM tasks "
        "WHERE status='ungraded' AND context LIKE '%_posthook_queue%'"
    )
    rows = await cur.fetchall()
    if not rows:
        return 0

    # 2. Source ids that already have a pending post-hook child in flight. Such
    #    chains are NOT stranded — the child's resume owns the next advance.
    ccur = await db.execute(
        "SELECT state_json FROM continuations "
        "WHERE status='pending' AND resume_name LIKE 'posthook.%'"
    )
    in_flight: set[int] = set()
    for (sj,) in await ccur.fetchall():
        try:
            st = _json.loads(sj) if isinstance(sj, str) else (sj or {})
        except (ValueError, TypeError):
            continue
        sid = st.get("source_task_id")
        if sid is not None:
            try:
                in_flight.add(int(sid))
            except (ValueError, TypeError):
                pass

    redriven = 0
    for (tid, raw_ctx) in rows:
        try:
            ctx = _json.loads(raw_ctx) if isinstance(raw_ctx, str) else (raw_ctx or {})
        except (ValueError, TypeError):
            continue
        if not isinstance(ctx, dict):
            continue
        queue = ctx.get("_posthook_queue") or []
        if not queue:
            continue  # LIKE false-positive (key present but empty/elsewhere)
        sid = int(tid)
        if sid in in_flight:
            continue  # a child IS in flight — do NOT double-spawn
        logger.warning(
            "reconcile: re-driving stranded post-hook chain",
            source_id=sid, queue=list(queue),
        )
        try:
            # Public guard-wrapped entry: re-reads + re-checks status under the
            # per-source lock, so a concurrent live advance can't race it.
            await _advance_posthook_chain(sid)
            redriven += 1
        except Exception as exc:  # noqa: BLE001 — one bad row must not abort the sweep
            logger.warning(
                "reconcile: stranded-chain re-drive failed",
                source_id=sid, error=str(exc),
            )
    if redriven:
        logger.info("reconcile: re-drove stranded post-hook chains",
                    count=redriven)
    return redriven


async def _complete_source_if_no_pending(source_task_id: int, ctx: dict) -> None:
    """Complete an ungraded source whose post-hook chain drained with no gating
    kind left pending.

    Only fires when ``_pending_posthooks`` is empty — i.e. there is no grade or
    mechanical blocker still in flight. Mirrors the grade-PASS completion path
    (clear failure metadata, fire workflow_advance, surface step progress) so a
    grade-less rewrite chain doesn't strand the source. Idempotent: no-ops once
    the source is no longer 'ungraded'.
    """
    import json as _json
    from dabidabi import get_task, update_task

    fresh = await get_task(source_task_id)
    if fresh is None or fresh.get("status") != "ungraded":
        return
    pending = list(ctx.get("_pending_posthooks") or [])
    if pending:
        return  # a gating post-hook (grade / mechanical) still owns completion
    await update_task(
        source_task_id, status="completed",
        context=_json.dumps(ctx),
        error=None, error_category=None,
        next_retry_at=None, retry_reason=None, failed_in_phase=None,
    )
    try:
        await _spawn_workflow_advance_if_mission(fresh, {})
    except Exception:  # noqa: BLE001
        pass
    try:
        from general_beckman import _send_step_progress
        again = await get_task(source_task_id)
        if again:
            await _send_step_progress(again, "completed", {})
    except Exception:  # noqa: BLE001
        pass


# Module-level name so tests can ``patch.object(apply, "enqueue", ...)`` and so
# _enqueue_posthook_llm_child resolves the (patchable) module global rather than
# a fresh per-call local. Lazily bound on first use to avoid the
# general_beckman.__init__ <-> apply import cycle at module-load time.
enqueue = None  # noqa: F811 — populated lazily by _enqueue_posthook_llm_child


# Checks that FULLY validate a produced artifact's shape/structure — passing one
# is a deterministic completeness authority, so the confab-prone LLM grade is
# skipped (the grade gate runs it inline; see kind=="grade" below). The *_shape
# suffix is the convention, but authority is a REGISTRY, not a naming rule:
# verify_adr_register is a full register validator that lacks the suffix and
# would otherwise leave step 4.14 (register.md) exposed to the same confab-grade
# → degenerate-repeat loop this short-circuit closes (task 567449). NARROW checks
# (e.g. verify_contains_product_name — a single-substring presence check) prove
# nothing about completeness and must NEVER be listed here.
_GRADE_AUTHORITATIVE_NON_SHAPE_CHECKS: frozenset[str] = frozenset({
    "verify_adr_register",
})


def _is_grade_authoritative_check(kind: str) -> bool:
    """True when passing *kind* deterministically proves artifact completeness,
    so the LLM grade may be skipped."""
    return (
        (kind.startswith("verify_") and kind.endswith("_shape"))
        or kind in _GRADE_AUTHORITATIVE_NON_SHAPE_CHECKS
    )


async def _enqueue_posthook_llm_child(kind: str, source: dict, source_ctx: dict,
                                      *, exclusions=None, attempt: int = 0) -> bool:
    """Enqueue the raw_dispatch reviewer/summarizer child with a durable
    continuation (CPS). mission_id rides in cont_state ONLY (never on the child
    row — SP3 child-spec hygiene). A trivial/degenerate source short-circuits to
    a verdict applied directly (no child).

    Returns ``True`` when a child task was actually enqueued, ``False`` when the
    spawn was a no-op (grade/code_review auto-fail short-circuit, or an SP3b
    rewriter that skipped because the draft already conforms / is empty /
    unconstrainable). The Task 6 cursor walk (``_advance_posthook_chain``) uses
    this to drain past a skipped rewriter without stalling the source."""
    global enqueue
    if enqueue is None:  # lazy bind: avoid __init__<->apply cycle at load time
        from general_beckman import enqueue as _enqueue
        enqueue = _enqueue
    source_id = source.get("id")
    mission_id = source.get("mission_id")
    gen_model = source_ctx.get("generating_model") or ""

    if kind == "grade":
        # ── Single source of truth: grade the CANONICAL produces artifact ──
        # For a step that declares a single ``produces`` path the artifact is
        # what ``materialize_produces`` wrote to disk (unconditionally), NOT
        # ``source['result']``. Schema'd steps strip ``write_file``, so the
        # agent NARRATES ("Wrote X.md containing …") instead of emitting the
        # body — leaving ``tasks.result`` with no ``##`` headers. Both readers
        # below (the deterministic schema gate at ``_draft`` and the LLM grader
        # via ``build_grading_spec`` → ``source.get('result')``) would then
        # false-reject a correct on-disk artifact and loop to a degenerate-repeat
        # DLQ (task 567373 [0.1] product_charter). Resolve the disk canonical
        # ONCE and override the in-memory source for the whole grade chain so it
        # judges the same artifact ``verify_charter_shape`` + downstream consumers
        # already read. ``None`` → non-produces step, keep ``source['result']``.
        try:
            from src.workflows.engine.hooks import resolve_produces_artifact
            _canon = resolve_produces_artifact(source, source_ctx)
            if isinstance(_canon, str) and _canon.strip():
                source = {**source, "result": _canon}  # local copy, never mutate caller
        except Exception:  # noqa: BLE001 — never let canonical resolution break grade
            pass
        # Fix #1 — deterministic artifact-schema gate BEFORE the LLM grade.
        # ``grade`` is the terminal chain entry, so any constrained_emit /
        # self_reflect rewrite has already landed on ``source.result``. Shape
        # and field-completeness are mechanical facts: a missing required field
        # or an object-where-an-array-is-required is FAIL with a precise reason,
        # unlike the prose grader's bare COMPLETE:NO that left capable producers
        # retrying blind to DLQ (#289735 single-object-vs-array, #289737 field
        # completeness). Route the failure through the SAME grade-FAIL retry /
        # escalation path (its message rides under ``error`` where
        # _grader_verdict_text reads it) and skip the wasted LLM grade. The
        # grader then judges semantics only, on shape-valid artifacts.
        _art_schema = source_ctx.get("artifact_schema")
        _gate_inputs = None  # upstream input artifacts for empty_ok exemption
        if isinstance(_art_schema, dict) and _art_schema:
            _draft = source.get("result")
            if isinstance(_draft, str) and _draft.strip():
                try:
                    from mr_roboto.schema_gate import schema_gate as _schema_gate
                    # Load upstream input artifacts ONLY when the schema declares
                    # an empty_ok_when_input_empty exemption (no-op otherwise).
                    # Anchors the exemption to a DIFFERENT task's artifact so a
                    # lazy producer can't self-grant an empty-scope pass.
                    try:
                        from src.workflows.engine.hooks import (
                            collect_empty_exemption_inputs as _cee,
                        )
                        _mid = source.get("mission_id")
                        if _mid is not None:
                            _gate_inputs = _cee(_art_schema, _mid)
                    except Exception:  # noqa: BLE001 — loader must never break grade
                        _gate_inputs = None
                    # A .md produces step's object/array schema must DEFER to
                    # verify_*_shape — the field-NAME substring fallback is
                    # meaningless on markdown prose (false pass AND false reject).
                    # Mirrors the producer gate (hooks.py produces_markdown).
                    _sg_produces = source_ctx.get("produces") or []
                    _sg_md = bool(_sg_produces) and all(
                        isinstance(p, str) and p.endswith(".md") for p in _sg_produces
                    )
                    _sg = _schema_gate(
                        output_value=_draft, schema=_art_schema, inputs=_gate_inputs,
                        produces_markdown=_sg_md,
                    )
                except Exception:  # noqa: BLE001 — never let the gate crash grade
                    _sg = {"passed": True, "error": ""}
                if not _sg.get("passed"):
                    await _apply_posthook_verdict(
                        {"id": source_id},
                        PostHookVerdict(
                            source_task_id=source_id, kind="grade", passed=False,
                            raw={"passed": False,
                                 "error": f"schema gate: {_sg.get('error')}"}))
                    return False  # verdict applied directly — no LLM grade spawned
        from src.core.grading import build_grading_spec, GradeResult
        excl = list(exclusions) if exclusions is not None else \
            list({m for m in [gen_model, *(source_ctx.get("grade_excluded_models") or [])] if m})
        built = build_grading_spec(source, excl)
        on_complete, on_error = "posthook.grade.resume", "posthook.grade.resume_err"
        if isinstance(built, GradeResult):
            await _apply_posthook_verdict(
                {"id": source_id},
                PostHookVerdict(source_task_id=source_id, kind="grade", passed=False,
                                raw={"passed": False, "raw": built.raw}))
            return False  # verdict applied directly — no child spawned
        # Blessed empty-scope short-circuit — placed AFTER build_grading_spec
        # so its deterministic trivial/degeneracy auto-fail (dogru_mu_samet)
        # has already run (a garbage artifact can't slip through this path).
        # When EVERY empty_ok_when_input_empty exemption is granted (upstream
        # scope genuinely empty — anchored to a DIFFERENT task's artifact, so a
        # lazy producer cannot self-grant), completeness is a proven
        # deterministic fact. The scope-blind LLM grader only ever false-rejects
        # such an artifact as "incomplete" (task #525016: empty compliance_overlay
        # for jurisdictions=[] → RELEVANT:NO/FAIL → DLQ). Skip the LLM grade —
        # schema gate + done_when + degeneracy floor already cover it. Non-empty
        # scope → not granted → graded normally (docs genuinely required).
        try:
            from src.workflows.engine.schema_dialect import (
                is_empty_scope_artifact as _is_empty_scope,
            )
            if _is_empty_scope(_art_schema, _gate_inputs):
                await _apply_posthook_verdict(
                    {"id": source_id},
                    PostHookVerdict(
                        source_task_id=source_id, kind="grade", passed=True,
                        raw={"passed": True,
                             "raw": "empty-scope exempted (schema gate + "
                                    "done_when + degeneracy floor pass); "
                                    "LLM grade skipped"}))
                return False  # blessed empty-scope — no LLM grade
        except Exception:  # noqa: BLE001 — never let the skip break grade
            pass
        # ── A passing shape verifier PROVES completeness only for STRUCTURED ──
        # ── artifacts; the LLM grade keeps TOPICALITY (and adequacy for prose) ──
        # The grader's COMPLETE axis is SEMANTIC ADEQUACY (grading.yaml: "adequate
        # depth, no stubs or hand-waving; NOT field presence"), NOT structural
        # presence. A verify_*_shape / verify_adr_register check proves STRUCTURE,
        # which ≈ substantive completeness ONLY when the returned structured value
        # IS the whole artifact — a pure .json config/decision (design_tokens, ADR,
        # taste_emphasis, surfaces). For a FREE-FORM authored doc (ANY .md
        # produces — charter, reverse_pitch, user_flow, premortem, register, …)
        # "adequate depth" is a real axis the verifier cannot see, so the LLM grade
        # must stay FULLY authoritative there (skipping/overriding COMPLETE would
        # drop legitimate stub/hand-waving detection on prose). Gate on the
        # codebase's authoritative structured-artifact predicate
        # (coulson._write_tools_redundant: structured-only schema AND no .md
        # produces) — the same seam that decides whether the result IS the clean
        # artifact for auto-strip / materialize.
        #
        # When eligible, we do NOT skip the grade: we run the shape check inline,
        # tag the continuation shape_verify_passed=True, and let the grade run. The
        # resume handler (posthook_continuations._grade_resume) then overrides a
        # COMPLETE-only FAIL to PASS — killing the confab that DLQ'd task 567449
        # [5.0a] design_tokens (a shape-valid .json FAILed COMPLETE:NO, then DLQ'd
        # as a "degenerate repeat" when the producer re-emitted the SAME correct
        # artifact) — while a RELEVANT:NO / COHERENT:NO FAIL stays terminal,
        # preserving topicality. Placed AFTER build_grading_spec so the
        # dogru_mu_samet degeneracy floor has already auto-failed garbage. On a
        # shape FAIL (a real earlier-attempt defect) the tag stays False and the
        # grade is fully authoritative. The verifier still runs again as its own
        # post-hook AFTER grade.
        # A registry-listed non-shape validator (verify_adr_register) is by
        # definition a FULL-ARTIFACT deterministic completeness proof (register.md
        # is a mechanical index with no adequacy/depth axis), so it is
        # override-eligible even though it authors a .md file — the .md exclusion
        # above targets prose whose depth the verifier can't see, which a
        # mechanical index is not. Prose *_shape checks are never registry members,
        # so this cannot re-admit them.
        _shape_verify_passed = False
        try:
            from coulson import _write_tools_redundant as _artifact_is_structured_only
            _registry_authoritative = any(
                isinstance(c, dict)
                and str(c.get("kind", "")) in _GRADE_AUTHORITATIVE_NON_SHAPE_CHECKS
                for c in (source_ctx.get("checks") or [])
            )
            _override_eligible = _registry_authoritative or (
                isinstance(_art_schema, dict) and bool(_art_schema)
                and _artifact_is_structured_only(_art_schema, source_ctx.get("produces"))
            )
        except Exception:  # noqa: BLE001 — predicate must never break grade
            _override_eligible = False
        try:
            _shape_check = next(
                (c for c in (source_ctx.get("checks") or [])
                 if isinstance(c, dict)
                 and _is_grade_authoritative_check(str(c.get("kind", "")))),
                None,
            ) if _override_eligible else None
            if _shape_check and source.get("mission_id") is not None:
                import mr_roboto as _mr
                _vpayload = dict(_shape_check.get("payload") or {})
                _vpayload.setdefault("action", _shape_check.get("kind"))
                _vaction = await _mr.run({
                    "id": source_id,
                    "mission_id": source.get("mission_id"),
                    "payload": _vpayload,
                })
                _shape_verify_passed = getattr(_vaction, "status", None) == "completed"
        except Exception:  # noqa: BLE001 — never let the probe break grade
            _shape_verify_passed = False
        cont_state = {"source_task_id": source_id, "kind": "grade",
                      "attempt": attempt, "exclusions": excl, "mission_id": mission_id,
                      "shape_verify_passed": _shape_verify_passed}

    elif kind == "code_review":
        from src.core.code_review import build_code_review_spec, CodeReviewResult
        excl = list(exclusions) if exclusions is not None else \
            list({m for m in [gen_model, *(source_ctx.get("review_excluded_models") or [])] if m})
        built = build_code_review_spec(source, excl)
        on_complete, on_error = "posthook.code_review.resume", "posthook.code_review.resume_err"
        if isinstance(built, CodeReviewResult):
            await _apply_posthook_verdict(
                {"id": source_id},
                PostHookVerdict(source_task_id=source_id, kind="code_review", passed=False,
                                raw={"passed": False, "issues": [], "raw": built.raw}))
            return False  # verdict applied directly — no child spawned
        cont_state = {"source_task_id": source_id, "kind": "code_review",
                      "attempt": attempt, "exclusions": excl, "mission_id": mission_id}

    elif kind.startswith("summary:"):
        artifact_name = kind.split(":", 1)[1]
        from src.workflows.engine.hooks import build_summary_spec
        from src.workflows.engine.artifacts import ArtifactStore
        text = ""
        if mission_id is not None:
            try:
                val = await ArtifactStore().retrieve(mission_id, artifact_name)
                if isinstance(val, str):
                    text = val
            except Exception:  # noqa: BLE001
                pass
        built = build_summary_spec(text, artifact_name)
        on_complete, on_error = "posthook.summary.resume", "posthook.summary.resume_err"
        cont_state = {"source_task_id": source_id, "kind": kind, "attempt": attempt,
                      "exclusions": [], "mission_id": mission_id,
                      "artifact_name": artifact_name}

    elif kind == "constrained_emit":
        # SP3b: re-emit the draft as schema-conforming JSON. The child is a
        # raw_dispatch OVERHEAD emit task (runs on husam); the resume handler
        # REWRITES the source result with the emitted JSON (Task 4 verdict).
        from src.core.reflection_posthook import (
            schema_response_format, should_skip_emit, build_emit_messages,
        )
        draft = source.get("result")
        if not isinstance(draft, str) or not draft.strip():
            return False  # nothing to emit — leave source untouched
        artifact_schema = source_ctx.get("artifact_schema")
        if not isinstance(artifact_schema, dict):
            return False  # no constrainable schema — emit is a no-op
        step_id = source_ctx.get("workflow_step_id") or "artifact"
        response_format = schema_response_format(artifact_schema, step_id)
        if response_format is None:
            return False  # unconstrainable (markdown/string) — validator covers it
        if should_skip_emit(draft, artifact_schema):
            # Draft already PASSES full schema validation (== the schema gate
            # would pass) — re-emitting would only risk tail-compression. Skip
            # the child. A draft that FAILS validation (missing nested fields /
            # wrong container) now falls through and gets re-emitted BEFORE the
            # gate, instead of flowing on to DLQ on a blind retry.
            return False
        messages = build_emit_messages(draft, response_format)
        spec = {
            "title": f"emit:task#{source_id}",
            "description": "Constrained re-emit of step output",
            "agent_type": "constrained_emit",
            "kind": "overhead",
            "priority": 1,
            "context": {"llm_call": {
                "raw_dispatch": True,
                "call_category": "overhead",
                "task": "structured_emit",
                "agent_type": "constrained_emit",
                "difficulty": 3,
                "messages": messages,
                "failures": [],
                "estimated_input_tokens": max(1000, len(messages[1]["content"]) // 4),
                # Output budget tracks the FULL draft (re-emit is ~1:1 reshape),
                # no 30000-char pre-cap. The 16000 ceiling is a model max-output
                # guard, not a content truncation of an input.
                "estimated_output_tokens": min(
                    16000, max(2000, len(draft) // 3),
                ),
                "prefer_speed": True,
                "response_format": response_format,
            }},
        }
        on_complete = "posthook.constrained_emit.resume"
        on_error = "posthook.constrained_emit.resume_err"
        cont_state = {"source_task_id": source_id, "kind": "constrained_emit",
                      "attempt": attempt, "exclusions": [], "mission_id": mission_id}
        built = spec

    elif kind == "self_reflect":
        # SP3b: role-specific self-review of the draft. The child is a
        # raw_dispatch OVERHEAD reviewer task (runs on husam); the resume
        # handler REWRITES the source result only when verdict=="fix" with a
        # non-degenerate corrected_result (warning severity — never fails the
        # source).
        from src.core.reflection_posthook import (
            build_reflect_messages, build_reflection_prompt,
        )
        draft = source.get("result")
        if not isinstance(draft, str) or not draft.strip():
            return False  # nothing to reflect on — leave source untouched
        agent_name = (
            source_ctx.get("agent_type") or source.get("agent_type") or "executor"
        )
        checklist = build_reflection_prompt(
            agent_name, iteration=1,
            stack=source_ctx.get("stack"), layer=source_ctx.get("layer"),
        )
        messages = build_reflect_messages(source, draft, checklist=checklist)
        # Estimate from the ACTUAL (untruncated) prompt so selection fits the
        # context window — else the call-level cap re-truncates the draft.
        _reflect_chars = sum(len(m.get("content") or "") for m in messages)
        spec = {
            "title": f"reflect:task#{source_id}",
            "description": "Self-reflection review of step output",
            "agent_type": "self_reflect",
            "kind": "overhead",
            "priority": 1,
            "context": {"llm_call": {
                "raw_dispatch": True,
                "call_category": "overhead",
                "task": "reviewer",
                "agent_type": "self_reflect",
                "difficulty": 6,
                "messages": messages,
                "failures": [],
                "estimated_input_tokens": max(800, _reflect_chars // 4),
                "estimated_output_tokens": 500,
                "prefer_speed": True,
            }},
        }
        on_complete = "posthook.self_reflect.resume"
        on_error = "posthook.self_reflect.resume_err"
        cont_state = {"source_task_id": source_id, "kind": "self_reflect",
                      "attempt": attempt, "exclusions": [], "mission_id": mission_id}
        built = spec

    elif kind == "critic_gate":
        # SP6 — admitted critic child. action_name + payload come from the
        # source step context (set by the workflow step that declared
        # post_hooks:["critic_gate"]).
        from mr_roboto.critic_gate import _build_critic_spec, _redact_payload, _hash_payload
        action_name = str(
            source_ctx.get("critic_action_name")
            or source_ctx.get("step_id")
            or "unknown"
        )
        raw_payload = source_ctx.get("critic_target_payload")
        redacted = _redact_payload(raw_payload)
        built = _build_critic_spec(action_name, redacted)
        on_complete = "posthook.critic.resume"
        on_error = "posthook.critic.resume_err"
        cont_state = {
            "source_task_id": source_id,
            "kind": "critic_gate",
            "action_name": action_name,
            "payload_hash": _hash_payload(redacted),
            "mission_id": source.get("mission_id"),
        }

    else:
        raise ValueError(f"_enqueue_posthook_llm_child: unsupported kind {kind!r}")

    # SP3b CRITICAL FIX: post-hook children MUST ride the oneshot lane. The lane
    # system only has oneshot/ongoing (lanes.py); the pump (next_task → pick_ready
    # _top_k(lane=LANE_ONESHOT)) only selects lane=='oneshot'. add_task persists an
    # unknown lane verbatim, so lane="overhead" produced rows the pump NEVER
    # dispatched — orphaning every emit/reflect/grade/code_review child and
    # stranding the source 'ungraded' forever. These are OVERHEAD-category LLM
    # calls (call_category lives in context.llm_call), which is orthogonal to the
    # admission lane — they belong on oneshot per SP3 design intent.
    await enqueue(built, parent_id=source_id, on_complete=on_complete,
                  on_error=on_error, cont_state=cont_state, lane="oneshot")
    return True  # child task enqueued


# Parameterized mechanical CHECK verbs — converted from standalone `.verify`
# workflow steps. Unlike `post_hooks` (a pure list[str] of registry kinds whose
# payload is standard/derived), a check carries its OWN payload (which file to
# check, min/max bounds), declared on the producer in a SEPARATE `checks` field:
#     "checks": [{"kind": "verify_adr_shape", "payload": {...}}]
# This keeps the two pots unmixed (no str|dict union on post_hooks). The check's
# kind is registered as a normal PostHookSpec; _CHECK_KINDS is derived from the
# registry by verb-name convention so converting a verb needs only a registry
# row + the JSON `checks` entry — no handler code.
# verify_* kinds that are NOT `checks`-pot members (they predate it / route
# through other paths): verify_artifacts (own handler, produces-derived) and
# verify_falsification_present (Z1 blocker path, post_hooks-wired on 3.1-3.7).
_NON_CHECK_VERIFY_KINDS: frozenset[str] = frozenset({
    "verify_artifacts",
    "verify_falsification_present",
})


def _derive_check_kinds() -> frozenset[str]:
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    return frozenset(
        k for k in POST_HOOK_REGISTRY
        if k.startswith("verify_") and k not in _NON_CHECK_VERIFY_KINDS
    )


_CHECK_KINDS: frozenset[str] = _derive_check_kinds()


def _find_check_payload(source_ctx: dict, kind: str) -> dict | None:
    """Return the declared payload for *kind* from the producer's `checks`."""
    for entry in (source_ctx.get("checks") or []):
        if isinstance(entry, dict) and entry.get("kind") == kind:
            pl = entry.get("payload")
            if isinstance(pl, dict):
                return pl
    return None


def _posthook_agent_and_payload(
    a: RequestPostHook, source: dict, source_ctx: dict,
) -> tuple[str, dict]:
    if a.kind == "verify_review_verdict":
        # Reviewer-failure routing check. Unlike the file-bounds checks below,
        # this verifier reads the SOURCE step's produced verdict (the reviewer's
        # own {status, issues[]} artifact), which the declared checks[].payload
        # does NOT carry. Build the base payload as usual, then inject the
        # reviewer's result as `review_result` by parsing source.result with the
        # SAME unwrap+json.loads the materializer / verify_falsification_present
        # path uses. On any parse failure leave review_result=None so mr_roboto
        # classifies it `malformed` (→ retry the reviewer, not route to a
        # producer). A dict result is used directly.
        payload = _find_check_payload(source_ctx, a.kind) or {}
        source_result = source.get("result")
        parsed: object | None = None
        if isinstance(source_result, (dict, list)):
            parsed = source_result
        elif isinstance(source_result, str) and source_result.strip():
            from coulson.grounding import unwrap_fenced_artifact
            candidate = unwrap_fenced_artifact(source_result) or source_result
            try:
                parsed = json.loads(candidate)
            except (ValueError, TypeError):
                parsed = None
        payload = {**payload, "action": a.kind, "review_result": parsed}
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": a.kind,
            "executor": "mechanical",
            "payload": payload,
        })

    if a.kind in _CHECK_KINDS:
        # Parameterized check: use the producer's declared `checks[].payload`
        # VERBATIM (it already names the exact file + bounds — no derivation
        # from `produces`, which is over-broad and uses different path strings).
        payload = _find_check_payload(source_ctx, a.kind) or {}
        payload = {**payload, "action": a.kind}  # ensure action present
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": a.kind,
            "executor": "mechanical",
            "payload": payload,
        })
    # SP3: grade / summary:* / code_review are handled before this function is
    # reached — _apply_request_posthook returns early for those kinds and
    # spawns the raw_dispatch reviewer/summarizer child via
    # _enqueue_posthook_llm_child. Only mechanical post-hook kinds fall through
    # to the branches below.
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
    if a.kind == "domain_layer_check":
        # Z3 T4C — layer-filtered semgrep using forbidden_in_domain.yml.
        # The mr_roboto verb run_semgrep_layer_filtered filters target_files
        # to only domain-layer files before invoking semgrep. Soft-skips when
        # semgrep is absent; real findings (ERROR severity) cascade the source.
        produces = list(source_ctx.get("produces") or [])
        workspace_path = source_ctx.get("workspace_path") or ""
        mission_id = source.get("mission_id")
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "domain_layer_check",
            "executor": "mechanical",
            "payload": {
                "action": "run_semgrep_layer_filtered",
                "rule_pack_path": "forbidden_in_domain.yml",
                "required_layer": "domain",
                "target_files": produces,
                "workspace_path": workspace_path,
                "mission_id": mission_id,
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
    if a.kind == "mobile_smoke":
        # Z5 T4b — run a recipe-driven Maestro mobile-QA flow against the
        # running app. flow_paths come from step context (the mobile
        # recipes ship a smoke-flow .yaml and the step declares them via
        # `maestro_flows`). Falls back to scanning produces for flow YAMLs
        # so a step that emits a *.flow.yaml needs no extra wiring.
        produces = list(source_ctx.get("produces") or [])
        flow_paths = list(source_ctx.get("maestro_flows") or [])
        if not flow_paths:
            flow_paths = [
                p for p in produces
                if isinstance(p, str)
                and (p.endswith(".flow.yaml") or p.endswith(".flow.yml")
                     or "maestro" in p.lower())
            ]
        workspace_path = source_ctx.get("workspace_path") or ""
        # Z5 P2 (2026-05-18 sweep) — auto-discover instantiated recipe
        # flows under the mission workspace. Mobile recipes (mobile_auth /
        # mobile_nav / mobile_persistence / mobile_offline_sync /
        # mobile_push / mobile_deep_links) each ship a flows/<name>.flow.yaml
        # under templates.smoke_flow; once a recipe is instantiated into the
        # workspace its flow lives at <workspace>/recipes/<name>/v1/flows/...
        # Without this fallback step 14.8 always sees flow_paths=[] (its
        # produces are .store/*.json only) and maestro_run soft-passes every
        # mission.
        if not flow_paths and workspace_path:
            try:
                import os as _os
                discovered: list[str] = []
                recipes_root = _os.path.join(workspace_path, "recipes")
                if _os.path.isdir(recipes_root):
                    for root, _dirs, files in _os.walk(recipes_root):
                        # Only enter flows/ directories to keep the walk cheap.
                        if _os.path.basename(root) != "flows":
                            continue
                        for fn in files:
                            if fn.endswith(".flow.yaml") or fn.endswith(".flow.yml"):
                                # Store paths relative to workspace_path so
                                # maestro_run (which joins workspace_path)
                                # resolves them correctly.
                                full = _os.path.join(root, fn)
                                rel = _os.path.relpath(full, workspace_path)
                                discovered.append(rel.replace("\\", "/"))
                if discovered:
                    discovered.sort()  # deterministic order
                    flow_paths = discovered
            except Exception as _disc_exc:
                logger.debug(
                    "mobile_smoke flow auto-discovery skipped: %s",
                    _disc_exc,
                )
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "mobile_smoke",
            "executor": "mechanical",
            "payload": {
                "action": "maestro",
                "flow_paths": flow_paths,
                "workspace_path": workspace_path,
                "extra_args": list(source_ctx.get("maestro_extra_args") or []),
                "timeout_s": float(source_ctx.get("maestro_timeout_s", 600.0)),
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
        candidates_path = source_ctx.get("candidates_path")
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "prior_art_min_coverage",
            "executor": "mechanical",
            "payload": {
                "action": "prior_art_min_coverage",
                "report_path": report_path,
                "report": source_ctx.get("report"),
                "candidates_path": candidates_path,
            },
        })
    if a.kind == "verify_falsification_present":
        # Z1 T2 (P4) — falsification triple check on phase-3 commitments.
        # Resolve artifacts from source.result. Most phase-3 steps emit a
        # single output artifact (functional_requirements, etc.); we wrap
        # the parsed result under its declared output_artifacts[0] name.
        source_result = source.get("result") or ""
        parsed: object = {}
        parse_error: str | None = None
        if isinstance(source_result, str) and source_result.strip():
            # Cloud LLMs frequently bury the artifact under chat narration
            # and/or a ```json fence (mission-81 3.2/3.7 DLQ'd empty=True
            # despite emitting a complete, valid artifact). Unwrap with the
            # same helper the materializer uses before parsing; fall back to
            # the raw string when there's no fence.
            from coulson.grounding import unwrap_fenced_artifact
            candidate = unwrap_fenced_artifact(source_result) or source_result
            try:
                parsed = json.loads(candidate)
            except (ValueError, TypeError) as exc:
                # Do NOT silently swallow into {} → empty=True (a misleading
                # "wiring bug" signal). Surface the parse error so the verdict
                # re-pends the PRODUCER with "re-emit valid JSON" feedback
                # (mission-90 567413: a single corrupt JSON seam DLQ'd a
                # near-valid 12k array at wa=1). _parse_error rides the payload.
                parsed = {}
                parse_error = str(exc)
        elif isinstance(source_result, (list, dict)):
            parsed = source_result
        output_names = list(source_ctx.get("output_artifacts") or [])
        artifacts: dict = {}
        if isinstance(parsed, dict):
            # Result is already a {name: value} mapping.
            artifacts = parsed
        elif output_names and parsed:
            artifacts = {output_names[0]: parsed}
        payload = {
            "action": "verify_falsification_present",
            "artifacts": artifacts,
        }
        if parse_error:
            payload["parse_error"] = parse_error
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "verify_falsification_present",
            "executor": "mechanical",
            "payload": payload,
        })
    # critic_gate is now an admitted posthook LLM child (SP6) — handled by the LLM-child route above.
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
    if a.kind == "visual_review":
        # Z4 T3B — vision-model diff against tunneled preview URL.
        # URL resolved from .preview/last_preview_url.txt (same pattern as
        # accessibility_review) with source_ctx fallback.
        # "pending:" marker (hosting deferred) suppresses URL — verb soft-skips.
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
        step_id = str(source.get("step_id") or source.get("id") or "")
        mission_id = source.get("mission_id") or 0
        routes = list(source_ctx.get("routes") or []) or None
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "visual_review",
            "executor": "mechanical",
            "payload": {
                "action": "visual_review",
                "workspace_path": workspace_path,
                "step_id": step_id,
                "mission_id": mission_id,
                "produces": produces,
                "routes": routes,
                "baseline_dir": None,
            },
        })
    if a.kind == "inject_lessons":
        # Z2 cross-mission lessons. The expander prepends this kind to the
        # first phase-0 task; mr_roboto's inject_lessons executor reads top
        # mission_lessons rows and stamps them into the next task's context
        # under `lessons_top_n`. Advisory — never blocks. (Sweep handoff
        # 2026-05-18, Z2 P1.)
        mission_id = source.get("mission_id") or 0
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "inject_lessons",
            "executor": "mechanical",
            "payload": {
                "action": "inject_lessons",
                "mission_id": int(mission_id) if mission_id else 0,
                "source_task_id": a.source_task_id,
            },
        })
    if a.kind == "capture_hint":
        # Yalayut Phase 4 — internal-hint auto-capture. Mechanical post-hook:
        # mr_roboto's capture_hint executor calls yalayut.capture_hint with
        # the source task + its outcome. Advisory — never fails the source.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "capture_hint",
            "executor": "mechanical",
            "payload": {
                "action": "capture_hint",
                "source_task": {
                    "id": a.source_task_id,
                    "title": source_ctx.get("title") or source.get("title", ""),
                    "description": source_ctx.get("description")
                    or source.get("description", ""),
                    "agent_type": source.get("agent_type", ""),
                },
                "outcome": {
                    "status": source.get("status", "completed"),
                    "iterations": int(source_ctx.get("iterations") or 0),
                    "result": source.get("result", ""),
                },
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
    if a.kind in ("copy_compliance_review", "brand_voice_lint",
                  "briefing_compose", "audit_completeness_check"):
        # Z7 T1.0 — humanish-layers posthook handlers.
        # Each runs as a mechanical task whose executor calls the handler
        # module in posthook_handlers/<kind>.py.  The handler receives the
        # full source task and result and returns {status: ok|fail, ...}.
        produces = list(source_ctx.get("produces") or [])
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": a.kind,
            "executor": "mechanical",
            "payload": {
                "action": a.kind,
                "produces": produces,
                "workspace_path": source_ctx.get("workspace_path") or "",
                "jurisdiction": source_ctx.get("jurisdiction") or "",
                "channel": source_ctx.get("channel") or "",
                "artifact_metadata": source_ctx.get("artifact_metadata") or {},
                "copy_path": source_ctx.get("copy_path") or "",
                "privacy_policy_path": source_ctx.get("privacy_policy_path") or "",
            },
        })
    if a.kind in ("demo_artifact_check", "demo_accessibility_check"):
        # Z7 T3B — demo pipeline posthook handlers (A3 + A3.r1).
        # Route to general_beckman.posthook_handlers.<kind> via mr_roboto dispatcher.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": a.kind,
            "executor": "mechanical",
            "payload": {
                "action": a.kind,
                "workspace_path": source_ctx.get("workspace_path") or "",
                "demo_cuts": source_ctx.get("demo_cuts") or {},
                "demo_vtt_path": source_ctx.get("demo_vtt_path") or "",
                "demo_cut_targets": source_ctx.get("demo_cut_targets") or {},
                "demo_accessibility_manifest_path": (
                    source_ctx.get("demo_accessibility_manifest_path") or ""
                ),
            },
        })
    if a.kind == "launch_readiness_gate":
        # Z7 T3A — A2.r1: pre-T-0 readiness gate for launch playbook.
        # Fires before publish_synchronized; runs 7 hard checks.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": a.kind,
            "executor": "mechanical",
            "payload": {
                "action": a.kind,
                "product_id": source_ctx.get("product_id") or "",
                "launch_id": source_ctx.get("launch_id") or 0,
                "channels": source_ctx.get("channels") or [],
            },
        })
    if a.kind == "incident_update_review":
        # Z7 T3D — B3: founder-review gate for incident status update drafts.
        # Fires after incident/draft_update; emits a founder_action with the draft.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": a.kind,
            "executor": "mechanical",
            "payload": {
                "action": a.kind,
                "incident_id": source_ctx.get("incident_id"),
                "product_id": source_ctx.get("product_id") or "",
                "draft": source_ctx.get("draft") or "",
                "status_kind": source_ctx.get("status_kind") or "investigating",
            },
        })
    if a.kind == "documentation_gap_detect":
        # Z7 T4 A8 — documentation_gap_detect: semantic-search question against
        # per-language support_docs collection; writes docs_gap_log row when no match.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": a.kind,
            "executor": "mechanical",
            "payload": {
                "action": a.kind,
                "question": source_ctx.get("question") or "",
                "product_id": source_ctx.get("product_id") or "",
            },
        })
    if a.kind == "outreach_deliverability_check":
        # Z7 T6 A7 — outreach_deliverability_check: read-only scan of bounce+complaint
        # rates; emits founder_action if thresholds exceeded.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": a.kind,
            "executor": "mechanical",
            "payload": {
                "action": a.kind,
                "product_id": source_ctx.get("product_id") or "",
                "list_id": source_ctx.get("list_id") or "",
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
    from dabidabi import update_task

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
                from dabidabi import get_task
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
                ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
            if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
                await _dlq_write(
                    source,
                    error="degenerate repeat: identical output across attempts, not converging",
                    category="quality", attempts=attempts,
                )
                return
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
        ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
    if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
        await _dlq_write(
            source,
            error="degenerate repeat: identical output across attempts, not converging",
            category="quality", attempts=attempts,
        )
        return
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
    from dabidabi import update_task

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
                from dabidabi import get_task
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
                ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
            if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
                await _dlq_write(
                    source,
                    error="degenerate repeat: identical output across attempts, not converging",
                    category="quality", attempts=attempts,
                )
                return
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
        ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
    if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
        await _dlq_write(
            source,
            error="degenerate repeat: identical output across attempts, not converging",
            category="quality", attempts=attempts,
        )
        return
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
    from dabidabi import update_task

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
                from dabidabi import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", verdict.raw or {})
            except Exception:
                pass
        else:
            # Other post-hooks still pending — persist the dropped
            # code_review so it is not re-run when the next verdict
            # arrives. Source stays 'ungraded' (no status change).
            await update_task(
                verdict.source_task_id,
                context=_json.dumps(ctx),
            )
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
                ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
            if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
                await _dlq_write(
                    source,
                    error="degenerate repeat: identical output across attempts, not converging",
                    category="quality", attempts=attempts,
                )
                return
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
        ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
    if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
        await _dlq_write(
            source,
            error="degenerate repeat: identical output across attempts, not converging",
            category="quality", attempts=attempts,
        )
        return
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
    from dabidabi import update_task

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
                from dabidabi import get_task
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
                ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
            if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
                await _dlq_write(
                    source,
                    error="degenerate repeat: identical output across attempts, not converging",
                    category="quality", attempts=attempts,
                )
                return
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
        ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
    if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
        await _dlq_write(
            source,
            error="degenerate repeat: identical output across attempts, not converging",
            category="quality", attempts=attempts,
        )
        return
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
    from dabidabi import update_task

    raw = verdict.raw or {}
    findings = raw.get("findings") or []
    skipped = bool(raw.get("skipped"))
    skipped_platform = bool(raw.get("skipped_platform"))

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
        if skipped_platform:
            logger.warning(
                f"{kind}: semgrep gate DID NOT RUN — platform skip "
                f"(Windows, no Docker fallback). Gate is NOT enforced.",
                source_id=verdict.source_task_id,
                kind=kind,
            )
            ctx[f"_{kind}_platform_skip"] = True
        else:
            logger.warning(
                f"{kind}: semgrep not installed — gate skipped",
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
            from dabidabi import get_task
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
    from dabidabi import update_task

    raw = verdict.raw or {}
    findings = raw.get("findings") or []
    skipped = bool(raw.get("skipped"))
    skipped_platform = bool(raw.get("skipped_platform"))

    # Soft-skip: semgrep not installed → advance, but surface platform skips
    # as WARNING so reviewers know the gate did not enforce anything.
    if skipped:
        if skipped_platform:
            logger.warning(
                f"{kind}: semgrep gate DID NOT RUN — platform skip "
                f"(Windows, no Docker fallback). Gate is NOT enforced. "
                f"Install semgrep via WSL or run Docker to enable this gate.",
                source_id=verdict.source_task_id,
                kind=kind,
            )
            # Stamp the platform-skip flag into context so mission audit logs
            # and downstream consumers can distinguish a real pass from a skip.
            ctx[f"_{kind}_platform_skip"] = True
        else:
            logger.warning(
                f"{kind}: semgrep not installed — gate skipped "
                f"(non-Windows; install semgrep to enable enforcement)",
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
                from dabidabi import get_task
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
                from dabidabi import get_task
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
                ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
            if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
                await _dlq_write(
                    source,
                    error="degenerate repeat: identical output across attempts, not converging",
                    category="quality", attempts=attempts,
                )
                return
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
        ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
    if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
        await _dlq_write(
            source,
            error="degenerate repeat: identical output across attempts, not converging",
            category="quality", attempts=attempts,
        )
        return
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


async def _apply_domain_layer_check_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Z3 T4C — domain_layer_check: layer-filtered semgrep with forbidden_in_domain.yml.

    Routes by registry default_severity (blocker). Real findings (ERROR severity
    per rule pack) trigger retry-with-feedback; exhausted budget DLQs the source.
    Soft-skips when semgrep is absent or no domain-layer files matched
    (skipped=True in raw). A semgrep crash is handled upstream in
    _posthook_dlq_cascade (soft-drop); this handler only sees valid verdicts.
    """
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY.get("domain_layer_check")
    default_severity = (spec.default_severity if spec else "blocker")
    step_threshold = ctx.get("semgrep_blocker_threshold") or default_severity
    if default_severity == "blocker":
        await _apply_semgrep_blocker_verdict(
            kind="domain_layer_check",
            findings_ctx_key="_domain_layer_check_findings",
            dlq_reason_ctx_key="_domain_layer_check_dlq_reason",
            blocker_threshold=step_threshold,
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )
    else:
        await _apply_semgrep_warning_verdict(
            kind="domain_layer_check",
            findings_ctx_key="_domain_layer_check_findings",
            dlq_reason_ctx_key="_domain_layer_check_dlq_reason",
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
    from dabidabi import update_task

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
                    from dabidabi import get_task
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
                    ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
                if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
                    await _dlq_write(
                        source,
                        error="degenerate repeat: identical output across attempts, not converging",
                        category="quality", attempts=attempts,
                    )
                    return
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
            ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
        if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
            await _dlq_write(
                source,
                error="degenerate repeat: identical output across attempts, not converging",
                category="quality", attempts=attempts,
            )
            return
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
            from dabidabi import get_task
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
    from dabidabi import update_task

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
                from dabidabi import get_task
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
                ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
            if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
                await _dlq_write(
                    source,
                    error="degenerate repeat: identical output across attempts, not converging",
                    category="quality", attempts=attempts,
                )
                return
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
        ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
    if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
        await _dlq_write(
            source,
            error="degenerate repeat: identical output across attempts, not converging",
            category="quality", attempts=attempts,
        )
        return
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

# Z1 blockers whose verdict judges LLM PRODUCER output (fabrication / thin
# grounding / malformed artifact), NOT a deterministic on-disk artifact. These
# must NOT single-shot DLQ — a retry on the same (or an escalated) model can
# emit a correct artifact — so the verdict dispatcher routes them through the
# retry-with-feedback rail (_apply_simple_blocker_verdict) instead of
# _apply_z1_mechanical_verdict. The remaining _Z1_BLOCKER_KINDS
# (compliance_template_present / compliance_blocker_check — file-presence +
# on-disk overlay checks; critic_gate — a veto) stay single-shot, since
# re-running the same artifact through the same model is pointless there.
# See project_quality_failure_escalation_20260604.
#
# verify_falsification_present judges the producer's requirement-bundle artifact
# straight out of tasks.result (produces=None) — a localized JSON glitch or a
# missing falsification field is fixable on a re-pend with feedback. Routing it
# single-shot DLQ'd a near-valid 12k array at wa=1 with a misleading
# `empty=True` (mission-90 567413). It belongs on the producer-re-pend rail.
_PRODUCER_QUALITY_Z1_BLOCKERS: frozenset[str] = frozenset({
    "prior_art_min_coverage",
    "verify_falsification_present",
})


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
    from dabidabi import update_task

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
                from dabidabi import get_task
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
                from dabidabi import get_task
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
    if a.kind == "mobile_smoke":
        return f"Mobile smoke flow for #{a.source_task_id}"
    if a.kind == "pattern_lint":
        return f"Pattern lint for #{a.source_task_id}"
    if a.kind == "design_system_check":
        return f"Design system check for #{a.source_task_id}"
    if a.kind == "domain_layer_check":
        return f"Domain layer check for #{a.source_task_id}"
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


def _adapt_shape_findings(verdict: PostHookVerdict) -> None:
    """Map a shape verb's problem lists into the ``findings`` simple_blocker reads.

    Shape verbs return ``{ok, question_problems|problems|missing|placeholders}``
    rather than the ``{findings:[...]}`` _apply_simple_blocker_verdict builds its
    retry feedback from. Without this map the producer's retry feedback summary
    is empty and the model re-emits the same bad shape. Mutates ``verdict.raw``
    in place; no-op when ``findings`` is already present.
    """
    raw = dict(verdict.raw or {})
    if raw.get("findings"):
        object.__setattr__(verdict, "raw", raw)
        return
    findings: list[dict] = []
    for p in (raw.get("question_problems") or []):
        if isinstance(p, dict):
            miss = p.get("missing_fields") or p.get("issues") or []
            findings.append({"why": f"{p.get('header', '?')}: {miss}"})
        else:
            findings.append({"why": str(p)})
    for p in (raw.get("problems") or []):
        findings.append(
            {"why": (p.get("why") or str(p)) if isinstance(p, dict) else str(p)}
        )
    for m in (raw.get("missing") or [])[:8]:
        findings.append({"why": f"missing: {m}"})
    for ph in (raw.get("placeholders") or [])[:5]:
        findings.append({"why": f"placeholder text: {ph}"})
    if not findings and raw.get("error"):
        findings.append({"why": str(raw["error"])})
    raw["findings"] = findings
    object.__setattr__(verdict, "raw", raw)


async def _load_mission_workflow(mission_id: int) -> dict | None:
    """Return a plain workflow dict ({"steps": [...]}) for a mission, or None.

    Resolves the workflow name, then loads + parses the JSON via the engine
    loader. Used by the reviewer-failure router (build_producer_index needs the
    steps' input/output_artifacts).

    Name resolution order:
    1. `missions.context.workflow_name` — the reliable source, seeded at mission
       expansion. Primary because the checkpoint table is written only on full-
       phase completion (and seeds name="" on first write), so it is empty /
       useless for in-flight missions whose reviewer fired before any phase
       completed (Class C: 2026-06-21 handoff).
    2. `workflow_checkpoints.workflow_name` — fallback, for missions that lack a
       context name but do have a checkpoint row."""
    try:
        from dabidabi import get_mission, get_workflow_checkpoint
        from src.workflows.engine.loader import load_workflow

        name = ""
        mission = await get_mission(int(mission_id))
        if mission:
            raw_ctx = mission.get("context")
            if isinstance(raw_ctx, str) and raw_ctx:
                import json as _json
                try:
                    name = str((_json.loads(raw_ctx) or {}).get("workflow_name") or "")
                except (ValueError, TypeError):
                    name = ""
            elif isinstance(raw_ctx, dict):
                name = str(raw_ctx.get("workflow_name") or "")

        if not name:
            checkpoint = await get_workflow_checkpoint(int(mission_id))
            if checkpoint and checkpoint.get("workflow_name"):
                name = str(checkpoint["workflow_name"])

        if not name:
            return None
        wf = load_workflow(name)
        return {"steps": list(wf.steps)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("review verdict: could not load mission workflow",
                       mission_id=mission_id, error=str(exc))
        return None


def _verdict_verify_opt_out() -> bool:
    """KUTAI_VERDICT_VERIFY=off disables Tier-2 refutation (Tier-1 grounding
    still runs in the mechanical verifier)."""
    import os
    return (os.environ.get("KUTAI_VERDICT_VERIFY") or "").strip().lower() in {
        "off", "0", "false", "no",
    }


async def _spawn_verdict_refuter(
    *, source: dict, mission_id, reviewer_id: str,
    kept_issues: list[dict], candidates: list[dict],
) -> bool:
    """Spawn the admitted Tier-2 refuter child for the surviving candidates.

    Resolves each candidate's artifact from disk, builds the batched refuter
    spec, and enqueues it as an OVERHEAD oneshot child parented to the reviewer
    with a durable ``posthook.verdict_verify.*`` continuation (so the reviewer
    stays parked until the refuter resumes). Returns True on enqueue, False on
    any failure (the caller then routes without Tier 2 — fail-safe)."""
    global enqueue
    if enqueue is None:  # lazy bind: avoid __init__<->apply cycle
        from general_beckman import enqueue as _enqueue
        enqueue = _enqueue
    try:
        from mr_roboto.verify_review_verdict import _resolve_artifact_content
        from mr_roboto.verdict_refuter import build_refuter_spec
        enriched: list[dict] = []
        for c in candidates:
            ta = c.get("target_artifact")
            content = await _resolve_artifact_content(mission_id, ta)
            enriched.append({
                "target_artifact": ta,
                "problem": c.get("problem"),
                "content": content or "",
            })
        spec = build_refuter_spec(enriched)
        cont_state = {
            "source_task_id": source["id"],
            "kind": "verdict_verify",
            "mission_id": mission_id,
            "reviewer_id": reviewer_id,
            "kept_issues": kept_issues,
            "candidates": [
                {"target_artifact": c.get("target_artifact"), "problem": c.get("problem")}
                for c in candidates
            ],
        }
        await enqueue(
            spec, parent_id=source["id"],
            on_complete="posthook.verdict_verify.resume",
            on_error="posthook.verdict_verify.resume_err",
            cont_state=cont_state, lane="oneshot",
        )
        return True
    except Exception as exc:  # noqa: BLE001 — never let a spawn failure halt routing
        logger.warning(
            "verdict refuter spawn failed — routing without Tier 2",
            source_id=source.get("id"), error=str(exc),
        )
        return False


async def _route_review_fail(
    *, source: dict, ctx: dict, pending: list[str], mission_id,
    reviewer_id: str, issues: list[dict], source_task_id,
) -> None:
    """Route a reviewer FAIL (issues already Tier-1/Tier-2 filtered) to the
    at-fault producer(s). Extracted verbatim from the original
    _apply_review_verdict fail tail so both the inline path and the refuter
    resume share one routing implementation."""
    import json as _json
    from dabidabi import update_task

    wf = await _load_mission_workflow(mission_id) if mission_id is not None else None
    if wf is None:
        # Can't localise without the workflow graph — fall back to a reviewer DLQ
        # so the mission doesn't silently swallow the rejection.
        logger.warning(
            "review verdict FAIL but no workflow — DLQ reviewer",
            source_id=source_task_id, mission_id=mission_id,
        )
        new_pending = [k for k in pending if k != "verify_review_verdict"]
        ctx["_pending_posthooks"] = new_pending
        await update_task(source_task_id, context=_json.dumps(ctx))
        await _retry_or_dlq(
            source, category="quality",
            error="reviewer rejected artifact but workflow graph unavailable",
        )
        return

    review_result = {"status": "fail", "issues": issues or []}
    try:
        from general_beckman.review_routing import route_review_failure
        # Hand the reviewer's context down so escalation can persist the
        # _review_halt payload (for restart/nudge re-rendering) in the same
        # parking write — no extra DB read.
        _rev_ctx = source.get("context")
        if isinstance(_rev_ctx, str):
            try:
                _rev_ctx = _json.loads(_rev_ctx) if _rev_ctx else {}
            except (ValueError, TypeError):
                _rev_ctx = {}
        if not isinstance(_rev_ctx, dict):
            _rev_ctx = {}
        outcome = await route_review_failure(
            mission_id=int(mission_id),
            reviewer_id=str(reviewer_id),
            review_result=review_result,
            workflow=wf,
            reviewer_task_id=source["id"],
            reviewer_ctx=_rev_ctx,
        )
        logger.info(
            "review verdict FAIL routed to producers",
            source_id=source_task_id, mission_id=mission_id,
            reviewer_id=reviewer_id, outcome=outcome,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("review routing raised — DLQ reviewer",
                       source_id=source_task_id, error=str(exc))
        new_pending = [k for k in pending if k != "verify_review_verdict"]
        ctx["_pending_posthooks"] = new_pending
        await update_task(source_task_id, context=_json.dumps(ctx))
        await _retry_or_dlq(
            source, category="quality",
            error=f"review routing failed: {str(exc)[:200]}",
        )
        return

    # Drain the reviewer's verify_review_verdict kind regardless of outcome.
    new_pending = [k for k in pending if k != "verify_review_verdict"]
    ctx["_pending_posthooks"] = new_pending

    routed = list(outcome.get("routed") or [])
    escalated = bool(outcome.get("escalated"))

    # Close the review loop: when at least one producer was actually re-pended
    # (routed non-empty AND nothing escalated), RE-PEND THE REVIEWER ITSELF back
    # to `pending` so it re-reviews the FIXED artifacts (a COMPLETED reviewer
    # never re-runs when its producer re-completes). Bounded by the producers'
    # worker_attempts cap (primary) + the reviewer's own bump (backstop).
    if routed and not escalated:
        attempts = int(source.get("worker_attempts") or 0) + 1
        max_attempts = int(source.get("max_worker_attempts") or 15)
        prev_output = source.get("result") or ""
        if isinstance(prev_output, str) and prev_output.strip():
            ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
        ctx["_schema_error"] = (
            "Previous review REJECTED upstream artifact(s); the "
            f"producer(s) {routed} were re-pended to fix it. Re-review the "
            "corrected artifacts fresh."
        )
        _stamp_retry_feedback(ctx, attempts, prev_output=source.get("result"))
        await update_task(
            source_task_id, status="pending",
            worker_attempts=attempts, max_worker_attempts=max_attempts,
            context=_json.dumps(ctx),
            result=None,
            error="reviewer rejected artifact — re-pended to re-review after producer fix",
            error_category="quality", next_retry_at=None,
            retry_reason=None, failed_in_phase=None,
        )
        logger.info(
            "review verdict FAIL — re-pended reviewer to re-review after producer fix",
            source_id=source_task_id, mission_id=mission_id,
            reviewer_id=reviewer_id, routed=routed, worker_attempts=attempts,
        )
        return

    # Escalated / nothing localisable: route_review_failure has escalated to the
    # founder-halt AND already PARKED the reviewer in ``waiting_human``. Persist
    # the drained _pending_posthooks; the reviewer keeps its parked status and
    # the founder-halt card owns resumption.
    await update_task(source_task_id, context=_json.dumps(ctx))
    return


async def _complete_review_pass(
    *, source: dict, ctx: dict, pending: list[str], source_task_id,
    raw: dict, mission_id, reviewer_id: str,
) -> None:
    """Complete a reviewer whose (possibly Tier-2 filtered) verdict is a pass:
    drain the verify kind, complete-if-empty, advance the mission."""
    import json as _json
    from dabidabi import update_task

    new_pending = [k for k in pending if k != "verify_review_verdict"]
    ctx["_pending_posthooks"] = new_pending
    if not new_pending:
        await update_task(
            source_task_id, status="completed",
            context=_json.dumps(ctx),
            error=None, error_category=None, next_retry_at=None,
            retry_reason=None, failed_in_phase=None,
        )
        await _spawn_workflow_advance_if_mission(source, raw)
        try:
            from general_beckman import _send_step_progress
            from dabidabi import get_task
            fresh = await get_task(source_task_id)
            if fresh:
                await _send_step_progress(fresh, "completed", raw)
        except Exception:
            pass
    else:
        await update_task(source_task_id, context=_json.dumps(ctx))
    logger.info(
        "review verdict PASS — reviewer completed",
        source_id=source_task_id, mission_id=mission_id, reviewer_id=reviewer_id,
    )


async def _finish_review_after_refuter(
    *, source_task_id, reviewer_id: str, mission_id, final_issues: list[dict],
) -> None:
    """Re-derive + apply the verdict after the Tier-2 refuter resolved.

    Loads the parked reviewer, and routes the surviving issues to producers
    (still blocking) or completes the reviewer as pass (the refuter dropped
    every blocking finding — the mission is NOT halted on confabulation)."""
    from dabidabi import get_task
    from mr_roboto.verify_review_verdict import _is_blocking_issue

    source = await get_task(source_task_id)
    if source is None:
        logger.warning("verdict-verify finish: reviewer task missing",
                       source_id=source_task_id)
        return
    ctx = _parse_ctx(source)
    pending = list(ctx.get("_pending_posthooks") or [])
    final_issues = final_issues or []
    blocking = any(_is_blocking_issue(i) for i in final_issues if isinstance(i, dict))
    if blocking:
        await _route_review_fail(
            source=source, ctx=ctx, pending=pending, mission_id=mission_id,
            reviewer_id=str(reviewer_id), issues=final_issues,
            source_task_id=source_task_id,
        )
    else:
        logger.info(
            "verdict-verify: all blocking findings dropped by refuter — passing",
            source_id=source_task_id, mission_id=mission_id,
        )
        await _complete_review_pass(
            source=source, ctx=ctx, pending=pending, source_task_id=source_task_id,
            raw={}, mission_id=mission_id, reviewer_id=str(reviewer_id),
        )


async def _apply_review_verdict(
    *, source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Apply a verify_review_verdict outcome from a reviewer step.

    PASS    -> the reviewer accepted the artifact. Drain the
               verify_review_verdict kind and (when nothing else is pending)
               COMPLETE the reviewer + spawn a workflow_advance so the mission
               flows past the satisfied review. Without this branch a clean
               PASS fell through to the malformed handler and got re-pended /
               DLQ'd — breaking the happy path of every wired reviewer.
    FAIL    -> route to the at-fault producer(s) via review_routing.
               * producer(s) re-pended (routed, not escalated) -> RE-PEND the
                 reviewer itself back to `pending` too, so it re-reviews the
                 fixed artifacts after the producers re-complete (the reviewer
                 depends_on its producers; advance.py never re-runs a COMPLETED
                 reviewer, so completing it here would let the mission flow past
                 an unsatisfied review). Loop is bounded by the producers'
                 worker_attempts cap (primary) + the reviewer's own bump
                 (backstop). Mission is NOT advanced.
               * escalated / nothing localisable -> route_review_failure has
                 PARKED the reviewer in waiting_human (safety park — an
                 escalated review must NOT advance unreviewed) and sent the
                 founder-halt card. Do NOT complete/advance; the card
                 (Regenerate producer / Accept anyway) owns resumption.
    MALFORMED (and any non-fail that reached here) -> the reviewer task itself
               failed to produce a parseable verdict: normal retry/DLQ on the
               reviewer task (drain the pending kind first).
    """
    import json as _json
    from dabidabi import update_task

    raw = verdict.raw or {}
    verdict_class = str(raw.get("verdict_class") or "").lower()
    # The reviewer's OWN step id (not a phase fallback) — the router resolves
    # the producer set from this step's input_artifacts.
    reviewer_id = str(ctx.get("workflow_step_id") or ctx.get("step_id") or "")
    mission_id = source.get("mission_id")

    if verdict_class == "fail":
        issues = raw.get("issues") or []
        tier2 = raw.get("tier2_candidates") or []
        # Tier-2 verdict verification (2026-06-26): before halting on findings a
        # single reviewer flagged but that Tier-1 grounding (mr_roboto) could not
        # refute deterministically, spawn ONE admitted adversarial refuter to
        # drop the findings it cannot support. The reviewer stays parked
        # (ungraded) on the in-flight refuter continuation; its resume routes the
        # survivors / completes the reviewer. KUTAI_VERDICT_VERIFY=off disables
        # Tier 2 (Tier 1 still ran in the mechanical verifier).
        if tier2 and mission_id is not None and not _verdict_verify_opt_out():
            spawned = await _spawn_verdict_refuter(
                source=source, mission_id=mission_id, reviewer_id=reviewer_id,
                kept_issues=issues, candidates=tier2,
            )
            if spawned:
                logger.info(
                    "review verdict FAIL — deferred to Tier-2 refuter",
                    source_id=verdict.source_task_id, mission_id=mission_id,
                    candidates=len(tier2),
                )
                return
            # spawn failed → fall through and route now (fail-safe).
        await _route_review_fail(
            source=source, ctx=ctx, pending=pending, mission_id=mission_id,
            reviewer_id=reviewer_id, issues=issues,
            source_task_id=verdict.source_task_id,
        )
        return

    if verdict_class == "pass":
        # Happy path: the reviewer accepted the artifact. Drain the
        # verify_review_verdict kind; when nothing else is pending on the
        # reviewer, COMPLETE it and advance the mission. Shared with the Tier-2
        # refuter resume (which completes a reviewer whose blocking findings were
        # all dropped) via _complete_review_pass.
        await _complete_review_pass(
            source=source, ctx=ctx, pending=pending,
            source_task_id=verdict.source_task_id, raw=raw,
            mission_id=mission_id, reviewer_id=reviewer_id,
        )
        return

    # Malformed (or any non-fail/non-pass verdict that reached here): the
    # reviewer task itself failed to emit a parseable verdict. Drain the kind
    # and route the REVIEWER task through normal retry/DLQ.
    new_pending = [k for k in pending if k != "verify_review_verdict"]
    ctx["_pending_posthooks"] = new_pending
    await update_task(verdict.source_task_id, context=_json.dumps(ctx))
    error_str = str(raw.get("error") or "reviewer produced no parseable verdict")[:300]
    await _retry_or_dlq(source, category="quality", error=error_str)


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
    from dabidabi import update_task

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
                from dabidabi import get_task
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
                ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
            if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
                await _dlq_write(
                    source,
                    error="degenerate repeat: identical output across attempts, not converging",
                    category="quality", attempts=attempts,
                )
                return
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
        ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
    if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
        await _dlq_write(
            source,
            error="degenerate repeat: identical output across attempts, not converging",
            category="quality", attempts=attempts,
        )
        return
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


# Z5 T4b — mobile_smoke verdict. The Maestro verb result has no `findings`
# list (it carries {flows_run, passed, failed, exit, error}), so the generic
# _apply_simple_blocker_verdict's findings-based summary would be empty.
# This handler mirrors _apply_test_run_verdict: pass drops the kind from
# pending; fail retries the source with the Maestro flow failure detail and
# DLQs on attempts-exhausted (bonus path honoured). A soft-skipped run
# (Maestro CLI absent) arrives as passed=True via rewrite.py Rule 0b reading
# the `ok` key — so it drains pending without blocking.
async def _apply_mobile_smoke_verdict(
    source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict,
) -> None:
    """Apply a mobile_smoke (Maestro flow) post-hook verdict to the source."""
    import json as _json
    from dabidabi import update_task

    raw = verdict.raw or {}

    if verdict.passed:
        new_pending = [k for k in pending if k != "mobile_smoke"]
        ctx["_pending_posthooks"] = new_pending
        if raw.get("skipped"):
            logger.debug(
                "mobile_smoke: soft-skipped",
                source_id=verdict.source_task_id,
                reason=raw.get("reason") or "",
            )
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
                from dabidabi import get_task
                fresh = await get_task(verdict.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", raw)
            except Exception:
                pass
        else:
            await update_task(verdict.source_task_id, context=_json.dumps(ctx))
        return

    # Fail path: at least one Maestro flow failed (or the verb itself failed).
    failed_n = int(raw.get("failed") or 0)
    flows_n = int(raw.get("flows_run") or 0)
    error_detail = raw.get("error") or ""
    stdout_tail = (raw.get("stdout_tail") or "")[:400]
    if error_detail:
        error_str = f"mobile_smoke: {error_detail}"
    elif failed_n:
        error_str = f"mobile_smoke: {failed_n}/{flows_n} Maestro flow(s) failed"
    else:
        error_str = "mobile_smoke: Maestro gate failed"
    if stdout_tail:
        error_str = (error_str + f" output={stdout_tail!r}")[:500]
    error_str = error_str[:500]

    feedback = (
        "The Maestro mobile smoke flow failed (sign in → onboard → core "
        "action → sign out). Fix the app behaviour or the flow YAML so the "
        f"Maestro run goes green. Details: {error_str}"
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
                ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
            if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
                await _dlq_write(
                    source,
                    error="degenerate repeat: identical output across attempts, not converging",
                    category="quality", attempts=attempts,
                )
                return
            await update_task(
                verdict.source_task_id, status="pending",
                worker_attempts=attempts, max_worker_attempts=max_attempts,
                error=error_str, error_category="quality",
                next_retry_at=None, context=_json.dumps(ctx),
            )
            return
        await _dlq_write(
            source, error=error_str or "mobile_smoke gate exhausted",
            category="quality", attempts=attempts,
        )
        return

    ctx["_schema_error"] = feedback
    prev_output = source.get("result") or ""
    if isinstance(prev_output, str) and prev_output.strip():
        ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
    if _stamp_retry_feedback(ctx, attempts, reason=error_str, prev_output=source.get("result")):
        await _dlq_write(
            source,
            error="degenerate repeat: identical output across attempts, not converging",
            category="quality", attempts=attempts,
        )
        return
    await update_task(
        verdict.source_task_id, status="pending",
        worker_attempts=attempts, error=error_str, error_category="quality",
        next_retry_at=None, context=_json.dumps(ctx),
    )


async def _maybe_spawn_adr_drift_judge(
    *, source: dict, ctx: dict, pending: list[str], verdict: PostHookVerdict
) -> bool:
    """Z3 R3 — spawn adr_drift_judge LLM task for judgment_only ADRs.

    Reads ``verdict.raw.judgment_only_adr_ids`` (populated by check_adr_drift
    when an ADR's falsification_signal is v1-string / null / unknown-shape).
    Enqueues a single ``adr_drift_judge`` task carrying the ADR ids + paths
    + produced_files.

    MUST be called BEFORE ``_apply_simple_blocker_verdict`` for the
    adr_drift_check verdict: it appends ``adr_drift_judge`` to both *ctx*'s
    ``_pending_posthooks`` and the in-memory *pending* list IN PLACE, so the
    subsequent simple_blocker pass sees a non-empty pending list and does NOT
    prematurely complete the source / fire workflow_advance. The judge's own
    verdict (kind=``adr_drift_judge``, also in the simple_blocker tuple) then
    drains the last pending entry and completes the source properly.

    Verdict precedence: mechanical-fail > judge-fail > judge-pass — enforced
    by the caller gating on ``verdict.passed`` (judge only spawns on a
    mechanical pass).

    Returns True when a judge was spawned (ctx/pending mutated), else False.
    """
    raw = verdict.raw or {}
    judgment_ids: list[str] = list(raw.get("judgment_only_adr_ids") or [])
    if not judgment_ids:
        return False

    import json as _json
    from dabidabi import add_task

    workspace_path = ctx.get("workspace_path") or ""
    produced = list(ctx.get("produces") or [])

    # Resolve adr_paths from workspace .adr/ dir.
    adr_paths: dict[str, str] = {}
    if workspace_path:
        import os as _os
        adr_dir = _os.path.join(workspace_path, ".adr")
        for adr_id in judgment_ids:
            candidate = _os.path.join(adr_dir, f"{adr_id}.json")
            if _os.path.isfile(candidate):
                adr_paths[adr_id] = candidate

    judge_ctx = {
        "source_task_id": source.get("id"),
        "posthook_kind": "adr_drift_judge",
        "adr_ids": judgment_ids,
        "adr_paths": adr_paths,
        "produced_files": produced,
        "workspace_path": workspace_path,
    }

    # Mutate ctx + pending IN PLACE so the caller's simple_blocker pass keeps
    # the source ungraded until the judge verdict lands.
    if "adr_drift_judge" not in pending:
        pending.append("adr_drift_judge")
    ctx["_pending_posthooks"] = list(pending)

    await add_task(
        title=f"ADR drift judge for {len(judgment_ids)} judgment_only ADR(s)",
        description=(
            "Judge whether the produced files drift from the listed ADRs. "
            "Read each ADR + relevant files; emit per-ADR verdict."
        ),
        agent_type="adr_drift_judge",
        context=_json.dumps(judge_ctx),
        mission_id=source.get("mission_id"),
    )

    logger.info(
        "adr_drift_judge spawned",
        source_id=source.get("id"),
        adr_count=len(judgment_ids),
    )
    return True


async def _maybe_spawn_integration_bisect(
    *, source: dict, ctx: dict, verdict: PostHookVerdict
) -> bool:
    """Z3 R4 — spawn an advisory integration_bisect mechanical task.

    Called when an ``integration_replay`` verdict failed. Reads the replayed
    commit list from ``verdict.raw.commits_replayed`` and enqueues a
    fire-and-forget mechanical ``integration_bisect`` task. The bisect verb's
    dispatch wrapper (mr_roboto.__init__) upserts a ``mission_lessons`` row
    when it isolates a breaking pair — that lesson is the entire point.

    NOT a post-hook: the task carries no ``source_task_id``/``posthook_kind``
    so it never produces a PostHookVerdict and never gates the source. The
    source has already been retried/DLQ'd by ``_apply_simple_blocker_verdict``.

    Returns True when a bisect task was enqueued, else False (too few commits).
    """
    raw = verdict.raw or {}
    commits = list(raw.get("commits_replayed") or [])
    # Bisect needs at least 2 commits to narrow a pair.
    if len(commits) < 2:
        return False

    import json as _json
    from dabidabi import add_task

    workspace_path = ctx.get("workspace_path") or ""
    if not workspace_path:
        return False

    suite_glob = ctx.get("integration_suite_glob") or "tests/integration/**"
    stack = str(ctx.get("tech_stack_detected") or ctx.get("stack") or "unknown")

    # Deliberately enqueued with mission_id=None: the bisect is advisory, not
    # a workflow step. A mission-scoped task would make rewrite.py emit a
    # spurious MissionAdvance on completion. The real mission_id is carried in
    # the payload so the mission_lessons row is still attributed correctly.
    await add_task(
        title=f"Integration bisect for #{source.get('id')} ({len(commits)} commits)",
        description=(
            "Binary-search the replayed commit list for the regression. "
            "Advisory — emits a mission_lessons row, does not gate."
        ),
        agent_type="mechanical",
        mission_id=None,
        context={
            # No source_task_id / posthook_kind — deliberately NOT a post-hook.
            "executor": "mechanical",
            "payload": {
                "action": "integration_bisect",
                "commits": commits,
                "suite_glob": suite_glob,
                "workspace_path": workspace_path,
                "mission_id": source.get("mission_id"),
                "stack": stack,
                "source_task_id": source.get("id"),
            },
        },
    )
    logger.info(
        "integration_bisect spawned (advisory)",
        source_id=source.get("id"), commit_count=len(commits),
    )
    return True


async def _apply_posthook_verdict(task: dict, a: PostHookVerdict) -> None:
    """Apply a post-hook verdict back to the source task.

    SP3b FIX 2 — every per-kind verdict handler below does a read-modify-write
    of the source ``context`` (drains ``_pending_posthooks`` / rewrites
    ``result``). Concurrent appliers (the chain grade child + an independent
    blocker like grounding, each fired in its own ``asyncio.create_task``) would
    otherwise clobber each other's pending-list updates → premature completion
    or a permanent 'ungraded' stall. Serialize on the source. The guard is
    reentrant within a coroutine chain, so the auto-fail short-circuit inside
    ``_enqueue_posthook_llm_child`` (which already holds the guard) re-enters
    freely without self-deadlock.
    """
    async with _source_verdict_guard(a.source_task_id):
        await _apply_posthook_verdict_locked(task, a)
    # FIX 2.1 — verdict appliers flip the source terminal (grade-PASS
    # completion, attempt-cap DLQ, blocker fails). Fire the source's own
    # continuation now instead of leaving it for the reconcile TTL.
    await _fire_source_continuation(a.source_task_id)


async def _apply_posthook_verdict_locked(task: dict, a: PostHookVerdict) -> None:
    import json as _json
    from dabidabi import get_task, update_task, add_task
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

    # SP3b — result-REWRITE verdict path. Unlike every other (~40) kind, which
    # gate (pass/fail → retry/DLQ) or surface (founder_action), the reflection
    # and constrained_emit post-hooks REWRITE the source's result in place
    # (reflection's corrected_result; emit's schema-conforming JSON). This
    # branch runs BEFORE the per-kind gate dispatch; the default action="gate"
    # leaves all existing kinds untouched. No second idempotency guard is added
    # here — the SP1/SP3 claim-then-fire CAS already fires this once per child
    # terminal event, and update_task to the same value is naturally idempotent.
    # The rewrite only mutates `result`; the ordered chain (Task 6) advances to
    # the next pending post-hook by leaving the source ungraded.
    # Truthy guard (not ``is not None``): an empty-string rewrite must never
    # silently clear the source result — fall through to the gate path so the
    # original draft survives.
    if getattr(a, "action", "gate") == "rewrite" and bool(a.new_result):
        await update_task(int(a.source_task_id), result=a.new_result)
        logger.debug(
            "posthook verdict: rewrote source result",
            source_id=a.source_task_id, kind=a.kind,
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

        # Availability masquerade (PART 2). When the grade CHILD itself
        # couldn't get a model, the auto-fail verdict is an AVAILABILITY
        # failure, not a quality rejection — the source artifact was never
        # judged. Hardcoding error_category="quality" below burned the
        # quality-sized worker-attempt cap and fast-DLQ'd against an
        # exhausted pool (mission_79 #225586/#225597, 2026-05-30). Founder
        # principle: can't get capacity → WAIT, not DLQ. Route through the
        # shared availability machinery (_retry_or_dlq → decide_retry, whose
        # effective_max_attempts floors the cap at the 15-step ladder so it
        # rides the backoff to a quota reset instead of DLQ'ing at max=6).
        # We do NOT add the generator to grade_excluded_models/failed_models
        # (no model misbehaved — it was capacity) and clear pending posthooks
        # so the re-run re-attaches a fresh grade. Guarded on the auto-fail
        # SHAPE in _grade_verdict_is_availability so a grader insight that
        # merely mentions "daily"/"quota" stays on the immediate quality path.
        if _grade_verdict_is_availability(a):
            ctx["_pending_posthooks"] = []
            await update_task(a.source_task_id, context=_json.dumps(ctx))
            logger.info(
                "grade auto-fail is availability — backing off, not quality-DLQ",
                source_id=a.source_task_id, error=error_str[:120],
            )
            await _retry_or_dlq(source, category="availability", error=error_str)
            return

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
                    ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
                if _stamp_retry_feedback(ctx, attempts, reason=f"grade: {error_str}", prev_output=source.get("result")):
                    await _dlq_write(
                        source,
                        error="degenerate repeat: identical output across attempts, not converging",
                        category="quality", attempts=attempts,
                    )
                    return
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
            ctx["_prev_output"] = prev_output[:6000]  # fallback only — artifact-backed continuation reads full draft (T3)
        if _stamp_retry_feedback(ctx, attempts, reason=f"grade: {error_str}", prev_output=source.get("result")):
            await _dlq_write(
                source,
                error="degenerate repeat: identical output across attempts, not converging",
                category="quality", attempts=attempts,
            )
            return
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

    # Z3 T2C + T3 + T4B + T5 + Z4 T3B — integration_review/security/accessibility/
    # contract/performance/adr_drift_check/integration_replay/adr_drift_judge/
    # visual_review share the simple blocker-or-pass pattern: the producing
    # verb (mechanical OR config-only LLM reviewer) emits {verdict, findings}.
    if a.kind in (
        "integration_review",
        "security_review", "accessibility_review", "contract_review",
        "performance_review", "adr_drift_check", "integration_replay",
        "adr_drift_judge", "visual_review",
    ):
        # Z3 R3 — ADR drift gray-zone path: when the mechanical check passed
        # but some ADRs were judgment_only, spawn an LLM judge. This MUST run
        # BEFORE simple_blocker so the judge kind is in `pending` — otherwise
        # simple_blocker completes the source + fires workflow_advance and the
        # judge verdict can no longer gate anything. Best-effort: a spawn
        # failure leaves pending untouched and the source completes normally.
        if a.kind == "adr_drift_check" and a.passed:
            try:
                await _maybe_spawn_adr_drift_judge(
                    source=source, ctx=ctx, pending=pending, verdict=a,
                )
            except Exception as _exc:
                logger.debug(
                    "adr_drift_judge spawn skipped",
                    source_id=source.get("id"), error=str(_exc),
                )
        await _apply_simple_blocker_verdict(
            kind=a.kind, source=source, ctx=ctx, pending=pending, verdict=a,
            feedback_prefix=f"{a.kind} gate",
        )
        # Z3 R4 — integration_replay fail → spawn an advisory integration_bisect
        # mechanical task to narrow the breaking commit pair. Fire-and-forget:
        # it is NOT a post-hook (carries no source_task_id/posthook_kind) so it
        # never gates the source — its only job is to emit a mission_lessons
        # row. Needs >= 2 replayed commits to bisect anything.
        if a.kind == "integration_replay" and not a.passed:
            try:
                await _maybe_spawn_integration_bisect(source=source, ctx=ctx, verdict=a)
            except Exception as _exc:
                logger.debug(
                    "integration_bisect spawn skipped",
                    source_id=source.get("id"), error=str(_exc),
                )
        # Z4 T4A — fire founder-loop visual-review notification (best-effort).
        if a.kind == "visual_review":
            try:
                from mr_roboto._visual_review_notify import enqueue_visual_review_notice
                # PostHookVerdict carries payload on .raw, not .result. (Was
                # silently {} → captured_paths always [] → Telegram album never
                # sent. Sweep handoff 2026-05-18, Z4 P1.)
                _vr_result = a.raw or {}
                if isinstance(_vr_result, dict):
                    await enqueue_visual_review_notice(
                        mission_id=int(source.get("mission_id") or 0),
                        step_id=str(
                            ctx.get("workflow_step_id")
                            or ctx.get("step_id")
                            or source.get("id")
                            or ""
                        ),
                        verdict=_vr_result.get("verdict", "pass"),
                        findings=list(_vr_result.get("findings") or []),
                        captured_paths=list(_vr_result.get("captured_paths") or []),
                        workspace_path=ctx.get("workspace_path") or None,
                    )
            except Exception as _exc:
                logger.debug(
                    "visual_review_notify skipped",
                    source_id=source.get("id"), error=str(_exc),
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

    if a.kind == "mobile_smoke":
        await _apply_mobile_smoke_verdict(
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

    if a.kind == "domain_layer_check":
        await _apply_domain_layer_check_verdict(
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

    if a.kind == "verify_review_verdict":
        # Reviewer-failure routing. The reviewer step ran a mr_roboto
        # verify_review_verdict check over its own *_review_result. A FAIL
        # verdict means the reviewer correctly REJECTED an upstream artifact:
        # route the failure to the at-fault PRODUCER(s) (re-pend their existing
        # rows with feedback), NOT back to the reviewer itself — falling
        # through to the default blocker rail would wrongly re-pend the
        # reviewer. A MALFORMED verdict means the reviewer task itself produced
        # no parseable verdict → it is a genuine reviewer failure → normal
        # retry/DLQ on the reviewer task. (Verdict payload rides verdict.raw;
        # see rewrite.py Rule 0c which carries verdict_class/issues there.)
        await _apply_review_verdict(source=source, ctx=ctx, pending=pending, verdict=a)
        return

    if a.kind in _CHECK_KINDS:
        # Parameterized mechanical check (converted standalone .verify step).
        # Adapt the verb's problem lists into `findings`, then share the
        # existing re-pend-with-feedback rail. Blocker semantics: a failed
        # check re-pends the PRODUCER (not the validator) with the problems.
        _adapt_shape_findings(a)
        await _apply_simple_blocker_verdict(
            kind=a.kind, source=source, ctx=ctx, pending=pending, verdict=a,
            feedback_prefix=f"{a.kind.replace('_', ' ')} gate",
        )
        return

    if a.kind in _PRODUCER_QUALITY_Z1_BLOCKERS:
        # A Z1 blocker whose verdict judges LLM PRODUCER output (e.g.
        # prior_art_min_coverage caught the synthesizer fabricating Habitica/
        # Streaks instead of grounding in the fetched candidates, #289710
        # 2026-06-04). Unlike the other Z1 blockers — which are deterministic
        # against on-disk artifacts, so re-running the SAME model re-emits the
        # SAME artifact and single-shot DLQ is correct — a stronger/escalated
        # model WOULD ground here. Route it through the retry-with-escalation
        # rail (_apply_simple_blocker_verdict + the _stamp_retry_feedback model
        # exclusion) instead of _apply_z1_mechanical_verdict's single-shot DLQ,
        # so worker_attempts climb to 3 and get_model_constraints excludes the
        # fabricating model + bumps difficulty. See the escalation audit
        # (project_quality_failure_escalation_20260604).
        _adapt_shape_findings(a)
        await _apply_simple_blocker_verdict(
            kind=a.kind, source=source, ctx=ctx, pending=pending, verdict=a,
            feedback_prefix=f"{a.kind.replace('_', ' ')} gate",
        )
        return

    if a.kind in _Z1_MECHANICAL_KINDS:
        await _apply_z1_mechanical_verdict(
            source=source, ctx=ctx, pending=pending, verdict=a,
        )
        return

    # Z7 T1.0 — humanish-layers posthooks.
    # copy_compliance_review is a blocker (privacy mismatch = blocker);
    # brand_voice_lint is a blocker (per registry default_severity).
    # briefing_compose + audit_completeness_check are warnings (advisory).
    if a.kind == "copy_compliance_review":
        await _apply_simple_blocker_verdict(
            kind=a.kind, source=source, ctx=ctx, pending=pending, verdict=a,
            feedback_prefix="copy_compliance_review gate",
        )
        return

    if a.kind == "brand_voice_lint":
        await _apply_simple_blocker_verdict(
            kind=a.kind, source=source, ctx=ctx, pending=pending, verdict=a,
            feedback_prefix="brand_voice_lint gate",
        )
        return

    if a.kind in ("briefing_compose", "audit_completeness_check"):
        # Warning severity: soft-drop pending kind and advance source.
        import json as _json
        from dabidabi import update_task as _update_task
        new_pending = [k for k in pending if k != a.kind]
        ctx["_pending_posthooks"] = new_pending
        if not a.passed:
            ctx[f"_{a.kind}_warning"] = (
                (a.raw or {}).get("error") or (a.raw or {}).get("summary") or "needs_review"
            )[:300]
        if not new_pending:
            await _update_task(
                a.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None, error_category=None,
                next_retry_at=None, retry_reason=None, failed_in_phase=None,
            )
            try:
                await _spawn_workflow_advance_if_mission(source, a.raw or {})
            except Exception:
                pass
        else:
            await _update_task(a.source_task_id, context=_json.dumps(ctx))
        return

    # Z7 T3B — demo pipeline posthook verdicts (A3 + A3.r1).
    # Both are blockers (per registry default_severity).
    if a.kind in ("demo_artifact_check", "demo_accessibility_check"):
        await _apply_simple_blocker_verdict(
            kind=a.kind, source=source, ctx=ctx, pending=pending, verdict=a,
            feedback_prefix=f"{a.kind} gate",
        )
        return

    # Z7 T3A — A2.r1: launch_readiness_gate posthook verdict.
    # Blocker: all 7 checks must pass before T-0 publish fires.
    # ready / ready_with_warnings → passes gate (warnings surfaced but not blocking).
    # blocked → keeps pending, T-0 frozen until founder acts.
    if a.kind == "launch_readiness_gate":
        await _apply_simple_blocker_verdict(
            kind=a.kind, source=source, ctx=ctx, pending=pending, verdict=a,
            feedback_prefix="launch_readiness_gate",
        )
        return

    # Z7 T3D — B3: incident_update_review posthook verdict.
    # Blocker: founder must review the draft before publish_status is called.
    # Pass: drop kind from pending, advance source task.
    # Fail: keep pending, re-surface to founder.
    if a.kind == "incident_update_review":
        await _apply_simple_blocker_verdict(
            kind=a.kind, source=source, ctx=ctx, pending=pending, verdict=a,
            feedback_prefix="incident_update_review gate",
        )
        return

    # Z7 T6 A7 — outreach_deliverability_check posthook verdict.
    # Warning severity: advisory — pauses campaign via DB flag + founder_action,
    # but never blocks the source outreach/send task from completing.
    if a.kind == "outreach_deliverability_check":
        import json as _json
        from dabidabi import update_task as _update_task
        new_pending = [k for k in pending if k != a.kind]
        ctx["_pending_posthooks"] = new_pending
        if not a.passed:
            ctx["_outreach_deliverability_warning"] = (
                (a.raw or {}).get("issue") or (a.raw or {}).get("error") or "needs_review"
            )[:300]
        if not new_pending:
            await _update_task(
                a.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None, error_category=None,
                next_retry_at=None, retry_reason=None, failed_in_phase=None,
            )
            try:
                await _spawn_workflow_advance_if_mission(source, a.raw or {})
            except Exception:
                pass
        else:
            await _update_task(a.source_task_id, context=_json.dumps(ctx))
        return

    # Z7 T4 A8 — documentation_gap_detect posthook verdict.
    # Warning severity: advisory — logs docs gaps but doesn't block escalation.
    # Always soft-drops the kind from pending and advances the source task.
    if a.kind == "documentation_gap_detect":
        import json as _json
        from dabidabi import update_task as _update_task
        new_pending = [k for k in pending if k != a.kind]
        ctx["_pending_posthooks"] = new_pending
        if not new_pending:
            await _update_task(a.source_task_id, status="completed",
                               context=_json.dumps(ctx))
        else:
            await _update_task(a.source_task_id, context=_json.dumps(ctx))
        return

    # Yalayut Phase 4 — capture_hint posthook verdict.
    # Warning severity: pure telemetry (replaces the old skills.py
    # auto-capture). A capture miss must NEVER block or DLQ the source —
    # always soft-drop the kind. The mechanical executor no-ops on
    # <2-iteration / failed tasks and returns ok=True even on internal
    # failure, so the verdict is effectively always passed. Without this
    # branch the kind falls through to the "unknown kind" warning at the
    # end of this function and the source is stranded in 'ungraded'
    # forever (capture_hint auto-wires on every task via triggers=["*"]).
    if a.kind == "capture_hint":
        new_pending = [k for k in pending if k != "capture_hint"]
        ctx["_pending_posthooks"] = new_pending
        if not new_pending:
            await update_task(
                a.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None, error_category=None,
                next_retry_at=None, retry_reason=None, failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, a.raw or {})
            # capture_hint may be the last hook to resolve (after grade),
            # so it owns the step-completion notification in that case.
            try:
                from general_beckman import _send_step_progress
                fresh = await get_task(a.source_task_id)
                if fresh:
                    await _send_step_progress(fresh, "completed", a.raw or {})
            except Exception:
                pass
        else:
            await update_task(a.source_task_id, context=_json.dumps(ctx))
        return

    # Z2 cross-mission lessons. Advisory — never blocks. Same soft-resolve
    # shape as capture_hint above. (Sweep handoff 2026-05-18, Z2 P1.)
    if a.kind == "inject_lessons":
        new_pending = [k for k in pending if k != "inject_lessons"]
        ctx["_pending_posthooks"] = new_pending
        if not new_pending:
            await update_task(
                a.source_task_id, status="completed",
                context=_json.dumps(ctx),
                error=None, error_category=None,
                next_retry_at=None, retry_reason=None, failed_in_phase=None,
            )
            await _spawn_workflow_advance_if_mission(source, a.raw or {})
        else:
            await update_task(a.source_task_id, context=_json.dumps(ctx))
        return

    if a.kind == "grade" and a.passed:
        # Remove "grade" from pending; spawn summary tasks for large artifacts.
        pending = [k for k in pending if k != "grade"]
        new_summary_kinds = await _summary_kinds_for_source(source, ctx)
        for kind in new_summary_kinds:
            pending.append(kind)
            await _enqueue_posthook_llm_child(kind, source, ctx)
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
    from dabidabi import add_task

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
