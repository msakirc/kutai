"""Per-iter Hoca selection + result mapping for the ReAct loop.

Phase C.2b — coulson owns the per-iter Pick and the inner
transport-retry loop, calling ``dispatcher.execute(pick, messages, ...)``
directly instead of routing through the legacy
``dispatcher.request → beckman.enqueue`` per-iter sub-tasking. Single
retry surface inside coulson; dispatcher becomes a one-attempt primitive.
"""
from __future__ import annotations

from typing import Any

import fatih_hoca
from fatih_hoca.types import Failure, Pick
from fatih_hoca.urgency import mid_task_urgency

from yazbunu import get_logger

logger = get_logger("coulson.dispatch")


def pick_for_iter(
    *,
    reqs: Any,
    task: dict,
    failures: list[Failure],
    iteration: int,
    remaining_budget: float,
    diag_out: dict | None = None,
) -> Pick | None:
    """Select the model for the current ReAct iteration.

    RC-A (mission 74): reuse the model Beckman reserved for this task
    across EVERY no-failure iteration — not just iteration 0 — as long as
    it is still servable *right now*. Re-selecting fresh each turn re-races
    the live pool (the GPU is often busy with a sibling task seconds after
    admission, and a cloud model that fit at admission may have hit its TPM
    / daily cap since), which is the ``no_candidates`` mechanism: admission
    keeps passing while the worker keeps failing into the same wall → DLQ.

    The held pick is ``task["_held_pick"]`` (the model actually running
    after any earlier re-select) or, on the first iteration, the admission
    ``task["preselected_pick"]``. Re-select only when:
      (a) ``failures`` are present — failure-adaptation excludes flaky
          models / escalates; or
      (b) the held model is no longer servable (unloaded / swapped-out /
          rate-limited / daily-exhausted), per ``fatih_hoca.is_servable``.

    Every fresh selection is stamped onto ``task["_held_pick"]`` so the
    next no-failure iter reuses the live model, not the stale preselect.

    Mid-task urgency is derived via ``fatih_hoca.mid_task_urgency`` from the
    task's admission urgency (``_admission_urgency``, stamped by Beckman) plus
    a finish-bias, with an extra bump while failures are being adapted around
    (user design 2026-05-03: "mid task urgency of the task can be a little
    higher than pre dispatch urgency to help react loops finish"). A started
    task is thus never judged stricter than it was at admission.
    """
    if not failures:
        held = task.get("_held_pick") or task.get("preselected_pick")
        if held is not None and fatih_hoca.is_servable(model=held.model, reqs=reqs):
            return held

    urgency = mid_task_urgency(
        task.get("_admission_urgency"), has_failures=bool(failures),
    )

    pick = fatih_hoca.select(
        task=reqs.effective_task or reqs.primary_capability,
        agent_type=reqs.agent_type,
        difficulty=reqs.difficulty,
        needs_thinking=reqs.needs_thinking,
        needs_function_calling=reqs.needs_function_calling,
        needs_vision=reqs.needs_vision,
        local_only=reqs.local_only,
        prefer_speed=reqs.prefer_speed,
        prefer_quality=reqs.prefer_quality,
        prefer_local=reqs.prefer_local,
        estimated_input_tokens=reqs.estimated_input_tokens,
        estimated_output_tokens=reqs.estimated_output_tokens,
        priority=reqs.priority,
        exclude_models=list(reqs.exclude_models or []),
        remaining_budget=remaining_budget,
        failures=failures,
        call_category="main_work",
        urgency=urgency,
        diag_out=diag_out,
    )
    # Track the live model so the next no-failure iter reuses THIS pick
    # (the one actually running) rather than re-racing the pool again.
    if pick is not None:
        task["_held_pick"] = pick
    return pick


# Failure categories where the chosen model is unavailable *right now* but a
# DIFFERENT model would serve. ``retryable=False`` on these means "don't retry
# the same model", not "abandon the task" — the transport loop must re-select
# (the failed model is excluded via the appended Failure), falling back to
# cloud when local can't load. Live 2026-06-16: llama-server couldn't bind its
# port, every local load returned category="loading", and the loop terminated
# instead of re-selecting → tasks pinned to dead local while cloud sat idle.
_RESELECTABLE_CATEGORIES = ("loading", "circuit_breaker")


def _transport_should_terminate(result: Any, transport_attempt: int,
                                max_attempts: int) -> bool:
    """Decide whether a CallError ends the task or triggers a re-select.

    Terminate when attempts are spent, or when the error is non-retryable AND
    not re-selectable (e.g. a malformed request — a different model won't
    help). A ``loading`` / ``circuit_breaker`` failure with attempts remaining
    re-selects a different model instead of terminating.
    """
    if transport_attempt >= max_attempts:
        return True
    reselectable = getattr(result, "category", None) in _RESELECTABLE_CATEGORIES
    if not getattr(result, "retryable", False) and not reselectable:
        return True
    return False


def result_to_response_dict(result: Any, model: Any) -> dict:
    """Map a ``hallederiz_kadir.CallResult`` to the legacy response dict.

    Same shape that ``LLMDispatcher._result_to_dict`` produced when the
    react loop went through ``dispatcher.request``. Kept here so coulson
    is the only owner of the ReAct call shape.
    """
    return {
        "content": result.content,
        "model": result.model,
        "model_name": result.model_name,
        "cost": result.cost,
        "usage": result.usage,
        "tool_calls": result.tool_calls,
        "latency": result.latency,
        "thinking": result.thinking,
        "is_local": result.is_local,
        "ran_on": "local" if result.is_local else result.provider,
        "provider": result.provider,
        "task": result.task,
        "finish_reason": getattr(result, "finish_reason", None),
        "capability_score": 0.0,
        "difficulty": 5,
    }


def _summarize_diag(diag: dict | None) -> str:
    """Render the selector's empty-pool diag into a compact one-line
    snapshot_summary (capped downstream at 1000 chars by the recorder).

    WS-1 (handoff 2026-05-25): the pre-fix forensics wrote a blank
    snapshot_summary, so the DB could not say WHICH filter emptied the
    pool. This names the stage, the per-reason histogram, and — the key
    signal — the FC-capable models that were rejected and why.
    """
    if not diag:
        return ""
    stage = diag.get("empty_stage")
    parts = [f"stage={stage}", f"eligible={diag.get('eligible_count')}"]
    fr = diag.get("filter_reasons") or {}
    if fr:
        hist = ", ".join(
            f"{c}×{r}"
            for r, c in sorted(fr.items(), key=lambda kv: -kv[1])
        )
        parts.append(f"reasons=[{hist}]")
    fc = diag.get("fc_capable_rejected") or {}
    if fc:
        fcs = ", ".join(f"{k}:{v}" for k, v in list(fc.items())[:8])
        parts.append(f"fc_capable_rejected=[{fcs}]")
    if stage == "pressure":
        parts.append(f"threshold={diag.get('pressure_threshold')}")
        sc = diag.get("pressure_scalars") or {}
        if sc:
            scs = ", ".join(f"{k}={v}" for k, v in list(sc.items())[:6])
            parts.append(f"scalars=[{scs}]")
    return " ".join(parts)


async def record_pool_empty_forensics(
    *,
    task: dict,
    failures: list[Failure],
    difficulty: int,
    iteration_n: int,
    diag: dict | None = None,
) -> None:
    """Pool drained mid-task — capture context for offline tuning.

    Ported from ``LLMDispatcher._do_dispatch``'s pool-empty branch. The
    pressure model failed to predict that retry would find no candidates
    after the initial pick admitted; record what was on the table.

    ``diag`` is the selector's per-call empty-pool diagnostic (populated
    via ``pick_for_iter(diag_out=...)``); it names the filter that emptied
    the pool so the DB row is actionable instead of blank.
    """
    try:
        from src.infra.admission_forensics import record_admission_violation
        t_id = task.get("id") if isinstance(task, dict) else None
        t_agent = task.get("agent_type") if isinstance(task, dict) else None
        extra = {
            "failures_count": len(failures),
            "failure_models": [getattr(f, "model", "") for f in failures[:10]],
            "is_overhead": False,
            "iteration_n": iteration_n,
        }
        if diag:
            extra["diag"] = {
                k: diag.get(k)
                for k in (
                    "empty_stage", "eligible_count", "filter_reasons",
                    "fc_capable_rejected", "pressure_threshold",
                    "pressure_scalars",
                )
                if k in diag
            }
        await record_admission_violation(
            site="coulson_pool_empty",
            phase="main_work",
            task_id=t_id,
            call_category="main_work",
            agent_type=t_agent or "",
            difficulty=difficulty,
            reason="no_candidates",
            error_category="availability",
            error_message=f"No model candidates after {len(failures)} failure(s)",
            snapshot_summary=_summarize_diag(diag),
            extra=extra,
        )
    except Exception:
        pass
