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


async def _run_image(task: dict, image_call: dict) -> dict:
    """Image-modality lane. Single-shot. Mirrors dispatcher.execute()'s telemetry
    envelope (begin_call -> call -> _record_pick success/failure -> end_call finally
    -> record_call_tokens -> record_call_cost) but calls paintress, not the LLM
    dispatcher primitive. Husam is permitted to reach the dispatcher; here it only
    borrows the _record_pick delegator."""
    import time as _time
    import paintress
    import fatih_hoca
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import SelectionFailure
    from src.core.in_flight import begin_call, end_call
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    from src.core.router import ModelCallFailed
    from src.core import heartbeat as _hb

    pick = task.get("preselected_pick")
    if pick is None or not isinstance(getattr(pick, "model", None), ImageModelInfo):
        ctx = task.get("context") if isinstance(task.get("context"), dict) else {}
        pick = fatih_hoca.select(
            needs_image=True,
            quality_tier=image_call.get("quality_tier", "fast"),
            failures=list((ctx or {}).get("failed_models") or []),
        )
    if pick is None or isinstance(pick, SelectionFailure):
        raise ModelCallFailed(call_id="image",
                              last_error="no eligible image provider",
                              error_category="availability")
    model = pick.model

    task_id = None
    try:
        from src.core.heartbeat import current_task_id
        task_id = current_task_id.get()
    except Exception:
        pass

    agent_type = image_call.get("agent_type", "image")
    difficulty = int(image_call.get("difficulty", 5) or 5)
    disp = get_dispatcher()

    call_id = await begin_call(
        category=CallCategory.IMAGE.value,
        model_name=getattr(model, "name", ""),
        provider=getattr(model, "provider", ""),
        is_local=bool(getattr(model, "is_local", False)),
        task_id=task_id,
        est_tokens=0,
    )

    started = _time.time()
    result = None
    try:
        s = paintress.ImageSpec(
            prompt=image_call.get("prompt", ""),
            out_dir=image_call.get("out_dir") or ".",
            negative_prompt=image_call.get("negative_prompt"),
            width=int(image_call.get("width", 1024)),
            height=int(image_call.get("height", 1024)),
            seed=image_call.get("seed"),
            quality_tier=image_call.get("quality_tier", "fast"),
            filename_hint=image_call.get("filename_hint"),
        )
        try:
            async with _hb.keepalive():
                if getattr(pick.model, "is_local", False):
                    import asyncio
                    # 1. Free VRAM by unloading any current local LLM. shutdown()
                    #    is internally guarded (no-op if nothing loaded); DaLLaMa
                    #    lazy-reloads on the next LLM task's ensure_model.
                    try:
                        from src.models.local_model_manager import get_local_manager
                        await get_local_manager().shutdown()
                    except Exception as _e:
                        from src.infra.logging_config import get_logger
                        get_logger("husam.image").warning(
                            "local_image: dallama shutdown failed: %s", _e)
                    # 2. Poll free VRAM until the image model fits (or ~30s).
                    try:
                        import nerd_herd
                        deadline = _time.time() + 30.0
                        needed = int(getattr(pick.model, "vram_mb", 0) or 0)
                        while _time.time() < deadline:
                            # Live in-process singleton snapshot: vram_available_mb
                            # is a live GPU read, so it reflects VRAM freed by the
                            # shutdown() above. NOT module-level refresh_snapshot()
                            # (async coroutine in this context) nor snapshot()
                            # (client cache).
                            snap = nerd_herd._get_singleton().snapshot()
                            if int(getattr(snap, "vram_available_mb", 0) or 0) >= needed:
                                break
                            await asyncio.sleep(0.5)
                    except Exception as _e:
                        from src.infra.logging_config import get_logger
                        get_logger("husam.image").warning(
                            "local_image: vram poll failed: %s", _e)
                    # 3. Start clair_obscur (idempotent; returns base_url).
                    try:
                        import clair_obscur
                        co_base = await clair_obscur.start()
                        try:
                            pick.model.endpoint = co_base
                        except Exception:
                            pass  # paintress local_server falls back to clair_obscur.base_url()
                    except Exception as _e:
                        raise ModelCallFailed(
                            call_id=getattr(pick.model, "name", "image"),
                            last_error=f"clair_obscur_start_failed:{_e}",
                            error_category="availability",
                        )
                    # 4. Record exactly one swap (charge against hoca's budget).
                    try:
                        import nerd_herd
                        nerd_herd.record_swap(getattr(pick.model, "name", ""))
                    except Exception:
                        pass
                result = await paintress.generate(pick, s)
        except Exception as exc:
            # Preserve an already-categorized ModelCallFailed (e.g. the
            # clair_obscur.start handover failure raises error_category=
            # "availability" so Beckman rides the transient backoff ladder and
            # degrades local→cloud). Re-wrapping as "raw_exception" (NOT in
            # TRANSIENT_CATEGORIES) would defeat that graceful degrade. Only
            # genuinely-uncategorized exceptions become "raw_exception".
            err_cat = getattr(exc, "error_category", None) if isinstance(exc, ModelCallFailed) else None
            if err_cat:
                await disp._record_pick(pick=pick, task="image", category=CallCategory.IMAGE,
                                        success=False, error_category=err_cat,
                                        agent_type=agent_type, difficulty=difficulty)
                raise
            await disp._record_pick(pick=pick, task="image", category=CallCategory.IMAGE,
                                    success=False, error_category="raw_exception",
                                    agent_type=agent_type, difficulty=difficulty)
            raise ModelCallFailed(call_id=getattr(model, "name", "image"),
                                  last_error=f"{exc.__class__.__name__}:{exc}",
                                  error_category="raw_exception")
        if result.error is None:
            await disp._record_pick(pick=pick, task="image", category=CallCategory.IMAGE,
                                    success=True, error_category="",
                                    agent_type=agent_type, difficulty=difficulty)
        else:
            # FIX 5: unknown_provider is a permanent misconfig (no provider by
            # that name), but "fatal" is NOT a category beckman's decide_retry /
            # TRANSIENT_CATEGORIES recognizes — it would fall through to the
            # availability-backoff branch and get retried 3×-with-backoff before
            # DLQ anyway, just under a misleading name. Beckman has no dedicated
            # terminal/non-retryable category (verified in retry.py), so we use
            # "availability" (a recognized TRANSIENT_CATEGORIES member) for both
            # arms. The retry ladder DLQs it predictably; the category is honest.
            err_cat = "availability"
            await disp._record_pick(pick=pick, task="image", category=CallCategory.IMAGE,
                                    success=False, error_category=err_cat,
                                    agent_type=agent_type, difficulty=difficulty)
            raise ModelCallFailed(call_id=getattr(model, "name", "image"),
                                  last_error=result.error, error_category=err_cat)
    finally:
        await end_call(call_id)

    duration_ms = int((_time.time() - started) * 1000)
    try:
        from src.infra.db import record_call_tokens, record_call_cost
        await record_call_tokens(
            task_id=task_id, agent_type=agent_type,
            workflow_step_id=image_call.get("workflow_step_id"),
            workflow_phase=image_call.get("workflow_phase"),
            call_category=CallCategory.IMAGE.value,
            model=getattr(model, "name", ""), provider=getattr(model, "provider", ""),
            is_streaming=False, prompt_tokens=0, completion_tokens=0,
            reasoning_tokens=0, total_tokens=0, duration_ms=duration_ms,
            iteration_n=int(image_call.get("iteration_n", 0) or 0), success=True,
        )
        if result.cost > 0.0 and task_id is not None:
            await record_call_cost(task_id, float(result.cost))
    except Exception:
        pass  # telemetry best-effort

    return {
        "content": result.path, "path": result.path, "provider": result.provider,
        "model": result.model,
        # FIX 7: emit BOTH "cost" (legacy readers) and "cost_usd". Beckman's
        # on_task_finished reads result.get("cost_usd") for mission spend
        # tracking; without it image cost never accrues to the mission (moot
        # today since providers are free=0, but correct when a paid one lands).
        "cost": result.cost, "cost_usd": result.cost,
        "latency": result.latency,
        "seed_used": result.seed_used,
        "is_local": getattr(model, "is_local", False),
        "ran_on": result.provider,
    }


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
    # ── Image modality branch (FIRST — before any text rehydration) ──────────
    # An image task carries ``context.image_call`` (NOT llm_call). If we fell
    # through to the text path below it would KeyError on the missing llm_call /
    # dispatch a text model. So we branch at the very top of run().
    import json as _json
    _ctx = task.get("context")
    if isinstance(_ctx, str):
        try:
            _ctx = _json.loads(_ctx)
        except Exception:
            _ctx = {}
    _image_call = _ctx.get("image_call") if isinstance(_ctx, dict) else None
    if isinstance(_image_call, dict) and _image_call.get("raw_dispatch"):
        return await _run_image({**task, "context": _ctx}, _image_call)

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
