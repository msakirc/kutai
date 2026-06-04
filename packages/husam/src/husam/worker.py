"""husam.run — the non-agentic single-call worker.

SP3b Task 2: the dispatcher is now a dumb pipe. Its only public primitive is
``LLMDispatcher.execute(pick, messages, ...)`` (one attempt: load + call +
meter; returns a raw ``CallResult``/``CallError``). ALL selection, result
mapping, and error-shaping that used to live in ``LLMDispatcher.dispatch`` /
``_do_dispatch`` / ``_result_to_dict`` now lives here.

Flow: orchestrator pump -> husam.run(task) -> (select via fatih_hoca, or honour
the Beckman-preselected pick) -> get_dispatcher().execute(...) -> map CallResult
to the legacy response dict (or raise ModelCallFailed/RuntimeError on failure).

Purity: husam imports NOTHING from coulson. All cross-package imports are LAZY
(inside ``run``) to keep import-time dependencies minimal and avoid cycles.
"""
from __future__ import annotations

from typing import Any

# Dispatcher instance call counters are bumped here (they live on the dispatcher
# singleton so get_stats() stays accurate). We bump the dispatcher's attributes
# directly (they were incremented in _do_dispatch historically).


async def _remaining_budget(mission_id: int | None) -> float | None:
    """Return max(0, ceiling - spent) for the given mission, or None if uncapped.

    Ported VERBATIM from ``llm_dispatcher._remaining_budget``. Returns None when:
      * mission_id is None (standalone call, no mission context)
      * mission row not found
      * cost_ceiling_usd is NULL (no ceiling set)
    """
    if mission_id is None:
        return None
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT cost_ceiling_usd, spent_usd FROM missions WHERE id = ?",
        (mission_id,),
    )
    row = await cur.fetchone()
    if not row or row[0] is None:
        return None
    ceiling = float(row[0])
    spent = float(row[1] or 0.0)
    return max(0.0, ceiling - spent)


async def run(task: dict) -> dict:
    """Select + execute a raw_dispatch task; return the legacy response dict.

    ``task`` is the spec the orchestrator pump hands us:
        {"context": {"llm_call": {...}},
         "kind": "main_work" | "overhead",
         "preselected_pick": <in-memory Pick or None>}

    Re-hydrates ``context.llm_call`` into selection kwargs, honours the
    Beckman-attached ``preselected_pick`` (iteration 0, no failures), and runs
    the dumb dispatcher primitive once. Retries do NOT happen here — they live
    in coulson.react (MAIN_WORK transport retry) and Beckman lifecycle's
    availability-retry path (OVERHEAD). husam just surfaces the failure.

    Raises:
        ModelCallFailed: MAIN_WORK selection/call failures (and pool-empty).
        RuntimeError:    OVERHEAD selection/call failures.
    """
    # LAZY imports — keep husam's import-time graph minimal and coulson-free.
    import fatih_hoca
    import hallederiz_kadir
    from fatih_hoca import SelectionFailure
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    from src.core.router import ModelCallFailed
    from .mapping import result_to_dict

    # ── Spec unpacking (ported from LLMDispatcher.dispatch) ──────────────────
    _ctx = task.get("context")
    llm_call = _ctx.get("llm_call", {}) if isinstance(_ctx, dict) else {}
    if not isinstance(llm_call, dict):
        llm_call = {}

    cat_str = llm_call.get("call_category") or task.get("kind") or "main_work"
    try:
        category = CallCategory(cat_str)
    except ValueError:
        category = CallCategory.MAIN_WORK

    # Recover the in-memory Pick object that Beckman attached at admission.
    # The orchestrator passes it through task["preselected_pick"] so the
    # serialisation round-trip (Pick -> DB -> spec) is avoided. When the Pick
    # is absent (fallback / test path), we call fatih_hoca.select as before.
    # Admission already gated this in Beckman (pool_pressure + fatih_hoca.select
    # + in_flight.reserve_task); we must not repeat those gates.
    preselected_pick = task.get("preselected_pick")

    task_name: str = llm_call.get("task") or ""
    agent_type: str = llm_call.get("agent_type") or ""
    difficulty: int = int(llm_call.get("difficulty") or 5)
    messages: list[dict] = llm_call.get("messages") or []
    tools = llm_call.get("tools")
    failures = llm_call.get("failures") or []
    # mission_id rides on the spec (top-level) when present; dispatch() historically
    # passed mission_id=None (it never extracted it), so default None preserves
    # that. We additionally honour an explicit top-level mission_id so the
    # budget-pause path works once Beckman threads it through.
    mission_id = task.get("mission_id")
    if mission_id is not None:
        try:
            mission_id = int(mission_id)
        except (TypeError, ValueError):
            mission_id = None

    # Selection-hint kwargs (mirror dispatch()'s rehydration). None values are
    # forwarded as-is, matching the previous _do_dispatch signature defaults.
    prefer_speed = llm_call.get("prefer_speed")
    prefer_local = llm_call.get("prefer_local")
    needs_json_mode = llm_call.get("needs_json_mode")
    needs_thinking_in = llm_call.get("needs_thinking")
    needs_function_calling_in = llm_call.get("needs_function_calling")
    min_context_in = llm_call.get("min_context") or 0
    response_format_in = llm_call.get("response_format")
    estimated_input_tokens = int(llm_call.get("estimated_input_tokens") or 0)
    estimated_output_tokens = int(llm_call.get("estimated_output_tokens") or 0)
    urgency_in = llm_call.get("urgency")
    if urgency_in is None:
        urgency_in = 0.5

    # ── _do_dispatch body (ported faithfully) ────────────────────────────────
    dispatcher = get_dispatcher()
    dispatcher._total_calls += 1
    is_overhead = category == CallCategory.OVERHEAD
    if is_overhead:
        dispatcher._overhead_calls += 1

    messages = messages or []
    failures = failures or []

    # needs_thinking: default True for MAIN_WORK, always False for OVERHEAD.
    needs_thinking = needs_thinking_in if needs_thinking_in is not None else (not is_overhead)
    if is_overhead:
        needs_thinking = False

    # Pass tools hint to selector for function calling requirement.
    needs_function_calling = (
        needs_function_calling_in if needs_function_calling_in is not None else bool(tools)
    )
    if tools:
        needs_function_calling = True

    # min_context is consumed by _ensure_local_model (inside execute), not by
    # fatih_hoca.select.
    _min_context_kw = int(min_context_in or 0)

    # response_format is forwarded to hallederiz_kadir.call (via execute).
    _response_format_kw = response_format_in
    # If caller supplies response_format, the selected model MUST support JSON
    # mode (production triage 2026-05-01): set needs_json_mode so the selector's
    # eligibility filter excludes models that lack it. Mirrors _do_dispatch's
    # ``kwargs.setdefault("needs_json_mode", True)`` — only fills the default
    # when the caller didn't set it (None == unset on the rehydrated spec).
    if _response_format_kw is not None and needs_json_mode is None:
        needs_json_mode = True

    # Selection hints forwarded to fatih_hoca.select(). These mirror exactly
    # what _do_dispatch forwarded via **kwargs after its pops: prefer_speed,
    # prefer_local, needs_json_mode, estimated_input_tokens,
    # estimated_output_tokens (urgency is passed explicitly below). Values are
    # forwarded as-is (incl. None) — the selector's bool/int params treat None
    # as falsy / unset, identical to the pre-SP3b behaviour.
    select_extra: dict[str, Any] = {
        "prefer_speed": prefer_speed,
        "prefer_local": prefer_local,
        "needs_json_mode": needs_json_mode,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
    }

    # Telemetry plumbing. task_obj / iteration_n are runtime objects; the spec
    # round-trip can't carry them, so they default to None/0 here (matching the
    # prior behaviour for serialised raw_dispatch tasks).
    _task_obj_kw = task.get("task_obj")
    _iteration_n_kw = int(task.get("iteration_n") or 0)

    if preselected_pick is not None and not failures:
        # Iteration 0: reuse Beckman's admission-time Hoca query.
        pick = preselected_pick
    else:
        # Mid-task urgency: admission urgency (urgency_in) + finish-bias, with
        # an extra bump while adapting around failures. Single source of truth
        # = fatih_hoca.mid_task_urgency (shared with the coulson ReAct loop).
        from fatih_hoca.urgency import mid_task_urgency
        _urgency = mid_task_urgency(urgency_in, has_failures=bool(failures))
        remaining = await _remaining_budget(mission_id)
        pick = fatih_hoca.select(
            task=task_name,
            agent_type=agent_type,
            difficulty=difficulty,
            needs_thinking=needs_thinking,
            needs_function_calling=needs_function_calling,
            failures=failures,
            call_category=category.value,
            remaining_budget_usd=remaining,
            urgency=_urgency,
            **select_extra,
        )

    # Handle SelectionFailure before the None / Pick path.
    if isinstance(pick, SelectionFailure):
        if pick.reason == "budget" and mission_id is not None:
            from general_beckman.lifecycle_events import emit_pause
            await emit_pause(
                mission_id,
                reason="no_model_fits_budget",
                triggered_by="auto:budget",
            )
        task_desc = task_name or agent_type or category.value
        if is_overhead:
            raise RuntimeError(
                f"OVERHEAD call failed — selection failed: {pick.reason}: {pick.detail}. "
                f"Task: {task_desc}"
            )
        raise ModelCallFailed(
            call_id=task_desc,
            last_error=f"selection failed: {pick.reason}: {pick.detail}",
            error_category="budget" if pick.reason == "budget" else "availability",
        )

    if pick is None:
        task_desc = task_name or agent_type or category.value
        # Forensics: pool drained mid-task (see _do_dispatch rationale).
        try:
            from src.infra.admission_forensics import record_admission_violation
            _t_id_forensic = (
                _task_obj_kw.get("id") if isinstance(_task_obj_kw, dict) else None
            )
            _t_agent_forensic = (
                _task_obj_kw.get("agent_type") if isinstance(_task_obj_kw, dict) else None
            )
            await record_admission_violation(
                site="dispatcher_pool_empty",
                phase=category.value,
                task_id=_t_id_forensic,
                call_category=category.value,
                agent_type=_t_agent_forensic or "",
                difficulty=difficulty,
                reason="no_candidates",
                error_category="availability",
                error_message=f"No model candidates after {len(failures)} failure(s)",
                extra={
                    "failures_count": len(failures),
                    "failure_models": [getattr(f, "model", "") for f in failures[:10]],
                    "is_overhead": is_overhead,
                    "iteration_n": _iteration_n_kw,
                },
            )
        except Exception:
            pass
        if is_overhead:
            raise RuntimeError(
                f"OVERHEAD call failed: no model candidates available. "
                f"Task: {task_desc}"
            )
        # Pool empty mid-task — surface as availability so orchestrator routes
        # through on_task_finished's normal availability-retry path.
        raise ModelCallFailed(
            call_id=task_desc,
            last_error="No model candidates available",
            error_category="availability",
        )

    # Pool pressure is enforced INSIDE the selector (single source of truth).
    model = pick.model

    result = await dispatcher.execute(
        pick=pick,
        messages=messages,
        category=category,
        task=task_name,
        agent_type=agent_type,
        difficulty=difficulty,
        tools=tools,
        needs_thinking=needs_thinking,
        min_context=_min_context_kw,
        response_format=_response_format_kw,
        task_obj=_task_obj_kw,
        iteration_n=_iteration_n_kw,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
    )

    if isinstance(result, hallederiz_kadir.CallResult):
        return result_to_dict(result, model)

    # CallError path — primitive already recorded the pick failure. husam does
    # not retry internally (single retry surface lives in coulson.react /
    # Beckman lifecycle). Same exception shape for OVERHEAD + MAIN_WORK so the
    # orchestrator's ModelCallFailed handler catches both.
    task_desc = task_name or agent_type or category.value
    raise ModelCallFailed(
        call_id=task_desc,
        last_error=(
            f"OVERHEAD call failed: {result.message}. Task: {task_desc}"
            if is_overhead
            else result.message
        ),
        error_category=result.category,
    )
