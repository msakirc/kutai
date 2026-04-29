"""General Beckman — the task master.

Public API (everything else is internal):
  - next_task() -> Task | None
  - on_task_finished(task_id, result) -> None
  - enqueue(spec) -> int
"""
from __future__ import annotations

from general_beckman.types import Task, AgentResult

__all__ = ["next_task", "on_task_finished", "enqueue", "on_model_swap", "Task", "AgentResult"]


# Admission-fingerprint cache. When the input state (in_flight + loaded model
# + cloud rpd + candidate ids) hasn't changed since last tick AND last tick
# admitted nothing, this tick's result is deterministic — skip the per-
# candidate Fatih Hoca scan. Single saturated task #4885 was rejected 601x in
# a 42-min window before this guard, each rejection costing a full 13-model
# filter+score pass (2026-04-27 log analysis). Reset by passing
# ``force=True`` or by any state delta.
_last_admission_fp: tuple | None = None
_last_admission_admitted: bool = True


def _admission_fingerprint(snap, candidates) -> tuple:
    """Cheap stable hash of admission-relevant state.

    Captures in_flight (auth source for local saturation), loaded model
    name, idle bucket (10s granularity — matches pressure ladder steps),
    cloud rpd-remaining per provider, and candidate task IDs.
    """
    if snap is None:
        # No snapshot → can't trust state; bypass cache.
        return ("nosnap", tuple(c.get("id") for c in candidates))
    in_flight = tuple(sorted(
        (c.task_id, c.model, c.provider, c.is_local)
        for c in (snap.in_flight_calls or [])
    ))
    local = snap.local
    loaded = local.model_name if local else None
    idle_bucket = int((local.idle_seconds if local else 0) // 10)
    is_swapping = bool(local.is_swapping) if local else False
    cloud_rpd = tuple(sorted(
        (p, st.limits.rpd.remaining if st.limits and st.limits.rpd else None)
        for p, st in (snap.cloud or {}).items()
    ))
    cand_ids = tuple(sorted(c.get("id", 0) for c in candidates))
    return (in_flight, loaded, idle_bucket, is_swapping, cloud_rpd, cand_ids)


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

    top_k = int(os.environ.get("BECKMAN_TOP_K", "5"))

    await fire_due()

    # One fresh snapshot per tick. Admission gates on snapshot.pressure_for(pick.model);
    # the in-flight list in the snapshot — pushed by the dispatcher — is the
    # authoritative "what's running right now" signal. No artificial
    # concurrency cap: local is naturally serial via llama-server --parallel 1
    # (surfaced as -1.0 local_pressure), cloud is bounded by per-pool
    # pressure thresholds.
    #
    # refresh_snapshot() is awaited (not cached read) because ticks fire back-
    # to-back in the orchestrator pump and the background 2s refresh loop is
    # too coarse to catch in-flight pushes that just landed on the sidecar.
    try:
        import nerd_herd
        snap = await nerd_herd.refresh_snapshot()
    except Exception:
        snap = None

    try:
        import fatih_hoca
    except Exception:
        fatih_hoca = None  # type: ignore

    from src.infra.logging_config import get_logger
    _log = get_logger("beckman.admission")

    candidates = await _queue.pick_ready_top_k(k=top_k)

    # Skip the entire scan if state hasn't changed since last tick and last
    # tick admitted nothing — result is deterministic. Mechanical tasks are
    # the one exception (no Hoca call, no pressure gate); admit them even on
    # a cache hit so blackboard writes / git commits don't stall behind LLM
    # saturation.
    global _last_admission_fp, _last_admission_admitted
    fp = _admission_fingerprint(snap, candidates)
    if (
        not _last_admission_admitted
        and _last_admission_fp == fp
        and not any((c.get("agent_type") or "") == "mechanical" for c in candidates)
    ):
        return None

    for task in candidates:
        # Defensive guard: a pending row past worker_attempts cap should
        # never have reached this point (sweep section 8 is supposed to
        # catch them) but if a fast retry bumped it between sweep ticks,
        # force DLQ here instead of admitting + immediately failing again.
        # Mechanical tasks have no attempt cap — skip the check for them.
        # (Handoff item A.)
        attempts = int(task.get("worker_attempts") or 0)
        max_att = int(task.get("max_worker_attempts") or 6)
        if (
            task.get("agent_type") != "mechanical"
            and max_att > 0
            and attempts >= max_att
        ):
            try:
                from general_beckman.apply import _dlq_write
                fresh = dict(task)
                fresh["failed_in_phase"] = "worker"
                await _dlq_write(
                    fresh,
                    error=f"Worker attempts exceeded at admission: "
                    f"{attempts}/{max_att}",
                    category=task.get("error_category") or "worker",
                    attempts=attempts,
                )
                _log.warning(
                    f"admission: task #{task['id']} REJECT past cap "
                    f"({attempts}/{max_att}) — forced to DLQ"
                )
            except Exception as e:
                _log.warning(f"admission: cap-guard DLQ write failed #{task['id']}: {e}")
            continue

        agent_type = task.get("agent_type") or ""
        difficulty = task.get("difficulty", 5)

        # Mechanical tasks have no LLM, no Hoca pick, no pressure gate.
        # Per design: mechanical is unbounded — local-only backpressure
        # applies only to LLM tasks. Claim and return directly.
        if agent_type == "mechanical":
            if not await _claim_task(task["id"]):
                _log.debug(f"admission: task #{task['id']} (mechanical) claim race lost")
                continue
            _log.info(f"admission: task #{task['id']} ADMIT mechanical")
            task["status"] = "processing"
            _last_admission_fp = fp
            _last_admission_admitted = True
            return task

        pick = None
        select_err = None
        if fatih_hoca is not None:
            try:
                pick = fatih_hoca.select(
                    task=agent_type,
                    agent_type=agent_type,
                    difficulty=difficulty,
                )
            except Exception as e:
                select_err = repr(e)
                pick = None
        if pick is None:
            _log.debug(
                f"admission: task #{task['id']} agent={agent_type} d={difficulty} "
                f"select=None err={select_err}"
            )
            continue

        if snap is not None:
            try:
                from fatih_hoca.estimates import estimate_for
                from general_beckman.btable_cache import get_btable

                class _TaskShim:
                    def __init__(self, t):
                        self.agent_type = t.get("agent_type", "")
                        ctx = t.get("context") or {}
                        if isinstance(ctx, str):
                            import json as _json
                            try:
                                ctx = _json.loads(ctx)
                            except Exception:
                                ctx = {}
                        self.context = ctx

                shim = _TaskShim(task)
                estimates = estimate_for(
                    shim,
                    btable=get_btable(),
                    model_is_thinking=getattr(pick.model, "is_thinking", False),
                )
                prov_key = getattr(pick.model, "provider", "")
                prov_state = snap.cloud.get(prov_key) if hasattr(snap, "cloud") else None
                consecutive_failures = getattr(prov_state, "consecutive_failures", 0)
                breakdown = snap.pressure_for(
                    pick.model,
                    task_difficulty=difficulty,
                    est_per_call_tokens=estimates.per_call_tokens,
                    est_per_task_tokens=estimates.total_tokens,
                    est_iterations=estimates.iterations,
                    est_call_cost=getattr(
                        pick.model, "estimated_cost",
                        lambda *_: 0.0,
                    )(estimates.in_tokens, estimates.out_tokens),
                    cap_needed=5.0,
                    consecutive_failures=consecutive_failures,
                )
                pressure = breakdown.scalar
            except Exception as e:
                _log.debug(f"admission: task #{task['id']} pressure_for raised {e!r}; fail-open")
                pressure = 1.0  # fail-open: admit rather than starve
        else:
            pressure = 1.0

        urgency = compute_urgency(task)
        thr = threshold(urgency)
        if pressure < thr:
            _log.info(
                f"admission: task #{task['id']} REJECT model={pick.model.name} "
                f"pressure={pressure:.3f} urgency={urgency:.3f} threshold={thr:.3f}"
            )
            continue

        if not await _claim_task(task["id"]):
            _log.debug(f"admission: task #{task['id']} claim race lost")
            continue

        # Reserve the in-flight slot at admission. The peer registry
        # (src.core.in_flight) represents three GPU-slot lifecycle states:
        # (1) admitted-not-yet-calling, (2) mid-call, (3) between calls
        # in the ReAct iteration gap. Without this reserve, state (1) has
        # no writer — the dispatcher's begin_call only fires AFTER agent
        # pre-work (RAG + chain-context + file-tree scan, often 10-20s),
        # during which the next Beckman tick reads an empty in-flight list
        # and admits a second local task onto a lane the first will
        # re-occupy. Dispatcher's begin_call UPSERTS the same task-{id}
        # key later, so mid-task model changes remain accurate. Release
        # fires via orchestrator._dispatch finally → release_task.
        #
        # Fail-open: if the peer registry is unreachable, still admit.
        # Nerd_herd being down shouldn't halt the system.
        try:
            from src.core.in_flight import reserve_task
            await reserve_task(task["id"], pick)
        except Exception as e:
            _log.warning(f"admission: reserve_task failed #{task['id']}: {e}")

        _log.info(
            f"admission: task #{task['id']} ADMIT model={pick.model.name} "
            f"pressure={pressure:.3f} urgency={urgency:.3f} threshold={thr:.3f}"
        )
        task["preselected_pick"] = pick
        task["status"] = "processing"
        _last_admission_fp = fp
        _last_admission_admitted = True
        return task

    _last_admission_fp = fp
    _last_admission_admitted = False
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

    # Persist generating_model + accumulate failed_models. The retry-
    # recovery layers (model exclusion at attempts >= 3, difficulty
    # bump in src/core/retry.py::get_model_constraints) gate on
    # ctx.failed_models. Agent emits ``result.generating_model`` /
    # ``result.model`` but no code path was writing it back to ctx —
    # so failed_models stayed [] across 5+ retries and R1/R2 had
    # nothing to act on. Live signal: mission 57 task 4441 hit DLQ at
    # attempts=5 with empty failed_models. Fix: always record the
    # current run's model in ctx.generating_model; on quality-failure
    # status, append it to failed_models so the next retry's selector
    # can see it. Idempotent (no duplicates).
    _model = (
        (result or {}).get("generating_model")
        or (result or {}).get("model")
        or ""
    )
    _status = (result or {}).get("status") or "completed"
    if _model:
        _ctx_changed = False
        if task_ctx.get("generating_model") != _model:
            task_ctx["generating_model"] = _model
            _ctx_changed = True
        # Quality-class failures (worker schema-fail, exhausted, timeout,
        # disguised failure) all flow through status="failed" here.
        # needs_clarification is NOT a model failure — agent worked, just
        # needs human input.
        if _status == "failed":
            _failed = list(task_ctx.get("failed_models") or [])
            if _model not in _failed:
                _failed.append(_model)
                task_ctx["failed_models"] = _failed
                _ctx_changed = True
                log.info(
                    "tracked failed_model",
                    task_id=task_id,
                    model=_model,
                    failed_count=len(_failed),
                )
        if _ctx_changed:
            try:
                from src.infra.db import update_task as _ut
                import json as _json
                await _ut(task_id, context=_json.dumps(task_ctx))
                # Refresh local task dict so downstream route_result
                # / apply_actions see the persisted state.
                task["context"] = _json.dumps(task_ctx)
            except Exception as e:
                log.warning("failed_model persist failed", task_id=task_id, error=str(e))

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
        if not _bookkeeping:
            status = (result or {}).get("status", "completed")
            if status in ("completed", "failed", "needs_clarification"):
                if task.get("mission_id"):
                    await _send_step_progress(task, status, result)
                else:
                    await _send_standalone_completion(task, status, result)
    except Exception as e:
        log.debug("progress ping failed", task_id=task_id, error=str(e))

    try:
        from general_beckman.queue_profile_push import invalidate_completed_id_cache
        invalidate_completed_id_cache(task_id)
    except Exception:
        pass

    try:
        from general_beckman.queue_profile_push import build_and_push
        await build_and_push()
    except Exception as e:
        log.debug("queue_profile push failed", task_id=task_id, error=str(e))


async def _send_standalone_completion(task: dict, status: str, result: dict) -> None:
    """Deliver completion for a mission-less task back to the user.

    Standalone tasks (created from generic messages) carry chat_id in
    context. Without this path, the user sees 'Task #N queued' and then
    radio silence even after the task finishes.
    """
    if task.get("agent_type") in ("mechanical", "grader", "artifact_summarizer"):
        return
    import json as _json
    ctx_raw = task.get("context") or "{}"
    try:
        ctx = _json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
    except Exception:
        ctx = {}
    chat_id = ctx.get("chat_id")
    if not chat_id:
        return
    from src.app.telegram_bot import get_telegram
    tg = get_telegram()
    if tg is None:
        return
    title = (task.get("title") or "").strip() or f"task #{task['id']}"
    icon = {
        "completed": "✅",
        "failed": "❌",
        "needs_clarification": "❓",
    }.get(status, "ℹ️")
    body_parts = [f"{icon} Task #{task['id']} — {title[:80]}"]
    if status == "completed":
        out = (result or {}).get("result") or ""
        if isinstance(out, str) and out.strip():
            excerpt = out.strip()
            if len(excerpt) > 3500:
                excerpt = excerpt[:3500] + "\n...(truncated)"
            body_parts.append(excerpt)
    elif status == "failed":
        err = (result or {}).get("error") or "error"
        body_parts.append(f"Error: {str(err)[:500]}")
    await tg.send_notification("\n\n".join(body_parts))


async def _send_step_progress(task: dict, status: str, result: dict) -> None:
    """Send a one-line Telegram progress update when a mission step finishes.

    Fires from on_task_finished, which runs BEFORE the grader verdict.
    A worker that finished is only "done" from the workflow's POV once
    grading passes. Check the live DB status to avoid premature ticks
    on steps that are queued for re-grade or retry.
    """
    if task.get("agent_type") in ("mechanical", "grader", "artifact_summarizer"):
        return
    # Always compare the raw agent-reported status against the live DB
    # status before pinging. The rewrite layer can flip actions between
    # the two (e.g. RequestClarification → CompleteWithReusedAnswer when
    # clarification_history exists). If the rewrite resolved it, the DB
    # row is already "completed" and the needs_clarification ping would
    # wrongly re-alarm the user.
    if status in ("completed", "needs_clarification", "failed"):
        from src.infra.db import get_task as _get_task
        live = await _get_task(task["id"])
        live_status = (live or {}).get("status", "")
        # Silent when DB already shows the step done from a different
        # path. For "completed" the prior gate rule holds (skip if not
        # yet completed — grader still running). For
        # "needs_clarification"/"failed" we silence when the rewrite
        # short-circuited to completed.
        if status == "completed" and live_status != "completed":
            return
        if status in ("needs_clarification", "failed") and live_status == "completed":
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

