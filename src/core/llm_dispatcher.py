# llm_dispatcher.py
"""
Centralized LLM Dispatcher — thin ask-load-call-retry loop.

All LLM calls in KutAI go through this dispatcher. It categorizes each
request and applies appropriate routing rules:

  MAIN_WORK  — Agent execution (ReAct iterations, single-shot, shopping,
               sub-agents). The actual task work the user cares about.
               CAN trigger model swaps.

  OVERHEAD   — Classifiers, graders, self-reflection, subtask classification.
               System housekeeping. CANNOT trigger model swaps — uses
               whatever is loaded or falls back to cloud.

Routing is fully delegated to fatih_hoca.select(). This module owns only:
  - ask (fatih_hoca.select)
  - load (ensure_local_model)
  - call (hallederiz_kadir.call)
  - retry (recurse with accumulated failures, max 5)
"""

from __future__ import annotations

import asyncio
import copy
from enum import Enum
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("core.llm_dispatcher")


# ─── In-flight registry ──────────────────────────────────────────────────────
# The in-flight registry lives in src.core.in_flight as a peer module so
# Beckman can call reserve_task() at admission time without importing
# dispatcher. Dispatcher uses begin_call / end_call from there; orchestrator
# uses release_task. All writers funnel through the peer module → nerd_herd.
from src.core.in_flight import (
    begin_call as _begin_call,
    end_call as _end_call,
    in_flight_snapshot,  # re-exported for back-compat with callers that imported here
    release_task,  # re-exported
)


# ─── Context-size resolver ───────────────────────────────────────────────────

def _resolve_load_ctx(*, need_ctx: int, min_context: int, est_in: int, est_out: int) -> int:
    """Pick's need_ctx wins; fall back to min_context, then the legacy
    token heuristic. Kept as a thin fallback for cloud picks / tests that
    don't carry need_ctx."""
    if need_ctx and need_ctx > 0:
        return need_ctx
    if min_context and min_context > 0:
        return min_context
    if est_in or est_out:
        return int((est_in + est_out) * 1.3) + 512
    return 0


# ─── Call Categories ─────────────────────────────────────────────────────────

class CallCategory(Enum):
    """Two categories — simple and clear."""
    MAIN_WORK = "main_work"   # Agent execution, can trigger swaps
    OVERHEAD = "overhead"     # Classifier/grader/reflection, no swaps
    IMAGE = "image"     # Image generation via husam → paintress (single-shot)


# ─── LLM Dispatcher ─────────────────────────────────────────────────────────

# ─── ModelCallFailed re-export for convenience ───────────────────────────────
# Some callers imported ModelCallFailed from here historically.  Keep the
# re-export so those imports continue to work.
from src.core.router import ModelCallFailed  # noqa: E402


# Note: the mission-budget helper (`_remaining_budget`) and the
# select->execute->map orchestration (`dispatch`, `_do_dispatch`,
# `_result_to_dict`) moved to `husam` (SP3b Task 2). The dispatcher is now a
# dumb pipe — its only call-execution surface is `execute()`.


class LLMDispatcher:
    """Centralized LLM call coordinator.

    The ONLY component that should trigger model swaps, acquire GPU slots,
    and route between local and cloud.

    Usage:
        dispatcher = get_dispatcher()
        result = await dispatcher.execute(pick=pick, messages=messages, ...)

    Note: the legacy ``request()`` shim (the blocking
    ``beckman.enqueue(await_inline=True)`` bridge) was deleted in SP5 once
    its last callers — shopping, ``single_shot.run``,
    ``reflection.self_reflect``, ``constrained_emit.maybe_apply`` — were
    retired. All LLM work now flows through ``beckman.enqueue()``.
    """

    def __init__(self):
        self._total_calls = 0
        self._overhead_calls = 0

    async def execute(
        self,
        *,
        pick: Any,
        messages: list[dict],
        category: CallCategory,
        task: str,
        agent_type: str,
        difficulty: int,
        tools: list[dict] | None,
        needs_thinking: bool,
        min_context: int,
        response_format: Any,
        task_obj: Any,
        iteration_n: int,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> Any:
        """One attempt against `pick`. No selection. No retry.

        Phase C primitive — the public entry that runtime/coulson will call
        per-iter once the per-iter Hoca selection lands in coulson.react
        (C.2b). Today called by `husam.run`.

        Returns ``hallederiz_kadir.CallResult`` on success or
        ``hallederiz_kadir.CallError`` on a load/transport failure. The
        caller decides whether to raise, retry on the same pick, or
        re-select via Hoca.

        Behavior preserved from the original inline body:
          * in_flight begin_call before any await (Beckman tick visibility)
          * swap intent recorded BEFORE ensure_model (budget reflects
            attempted swaps, not just successful ones)
          * heartbeat keepalive around swap + call (300s watchdog)
          * record_pick fires for success and for hallederiz CallError;
            skipped for local-load failures (which raise CallError(loading)
            without ever attempting the call)
          * record_pick fires for raw-exception path before re-raising
          * end_call always fires in finally
        """
        import hallederiz_kadir

        model = pick.model

        # Register in-flight BEFORE swap + any other awaits.
        # Rationale: swap can take 10-30s. Without early registration, the
        # next Beckman tick (~3s cadence) reads an empty in-flight list
        # and admits a second local task onto a GPU that's mid-swap.
        _active_task_id = None
        try:
            from src.core.heartbeat import current_task_id as _ctid
            _active_task_id = _ctid.get()
        except Exception:
            pass

        # est_tokens for begin_call: Beckman's reserve_task already set the
        # projection at admission. begin_call's max(prior_est, passed)
        # preserves the Beckman value regardless of what we pass here.
        _call_id = await _begin_call(
            category=category.value,
            model_name=model.name,
            provider=model.provider,
            is_local=model.is_local,
            task_id=_active_task_id,
            est_tokens=0,
        )

        result: Any = None
        try:
            # Local model load
            if model.is_local and getattr(model, "location", "") != "ollama":
                is_thinking = model.thinking_model and needs_thinking
                _min_ctx = _resolve_load_ctx(
                    need_ctx=int(getattr(pick, "need_ctx", 0) or 0),
                    min_context=min_context,
                    est_in=estimated_input_tokens,
                    est_out=estimated_output_tokens,
                )
                # Record swap intent BEFORE ensure_model so the budget
                # reflects ATTEMPTED swaps, not just successful ones
                # (production triage 2026-05-01: 96% force-kill rate
                # despite swap-budget gate — undercount was the cause).
                _swap_intended = (model.is_local and not model.is_loaded)
                if _swap_intended:
                    import nerd_herd as _nerd_herd
                    _nerd_herd.record_swap(model.name)
                from src.core import heartbeat as _hb_ka
                async with _hb_ka.keepalive():
                    ok, _swap_happened = await self._ensure_local_model(
                        model, needs_thinking=is_thinking,
                        load_timeout=pick.estimated_load_seconds or 0.0,
                        estimated_context=_min_ctx,
                    )
                # Feed the process-level local-inference liveness gate: 5
                # consecutive cross-model load failures → selector lays off ALL
                # local (routes to cloud) instead of admitting every task
                # against a dead llama-server. Any success resets it.
                try:
                    import nerd_herd as _nh_live
                    _nh_live.record_local_load(bool(ok))
                except Exception:
                    pass
                if not ok:
                    # Loading failure — return non-retryable CallError.
                    # No record_pick (matches prior raise-before-record
                    # contract: loading failures don't pollute pick_log).
                    return hallederiz_kadir.CallError(
                        category="loading",
                        message=f"Failed to load local model {model.name}",
                        retryable=False,
                    )

            _messages = self._prepare_messages(messages, model)
            # Per-call hard cap: cost-runaway protection for cloud only.
            # Local has no wall-clock cap — stream-inactivity watchdog
            # inside hallederiz_kadir handles hung-call detection.
            timeout = 600.0 if not model.is_local else 0.0

            try:
                from src.core import heartbeat as _hb
                _hb.bump()
            except Exception:
                pass

            try:
                from src.core import heartbeat as _hb_ka
                async with _hb_ka.keepalive():
                    result = await hallederiz_kadir.call(
                        model=model,
                        messages=_messages,
                        tools=tools,
                        timeout=timeout,
                        task=task or category.value,
                        needs_thinking=needs_thinking,
                        estimated_input_tokens=estimated_input_tokens,
                        estimated_output_tokens=estimated_output_tokens,
                        response_format=response_format,
                        task_obj=task_obj,
                        iteration_n=iteration_n,
                        call_category=category.value,
                    )
            except Exception:
                logger.exception(
                    "hallederiz_kadir.call raised raw exception | "
                    f"model={model.name} task={task or category.value}"
                )
                await self._record_pick(
                    pick=pick, task=task, category=category,
                    success=False, error_category="raw_exception",
                    agent_type=agent_type, difficulty=difficulty,
                )
                raise
        finally:
            await _end_call(_call_id)

        if isinstance(result, hallederiz_kadir.CallResult):
            await self._record_pick(
                pick=pick, task=task, category=category,
                success=True, error_category="",
                agent_type=agent_type, difficulty=difficulty,
            )
        else:
            # hallederiz CallError
            logger.debug(
                f"{category.value} candidate failed | model={model.name} "
                f"category={result.category} error={result.message[:80]}"
            )
            await self._record_pick(
                pick=pick, task=task, category=category,
                success=False, error_category=result.category,
                agent_type=agent_type, difficulty=difficulty,
            )

        return result

    async def _record_pick(
        self,
        *,
        pick: Any,
        task: str,
        category: CallCategory,
        success: bool,
        error_category: str = "",
        agent_type: str = "",
        difficulty: int | None = None,
    ) -> None:
        """Fire-and-forget pick_log write. Delegates to telemetry.pick_recorder.

        Kept as a thin instance method (not inlined at the execute() call
        sites) so tests that monkeypatch ``LLMDispatcher._record_pick`` and
        callers asserting ``hasattr(LLMDispatcher, "_record_pick")`` keep
        working. The 52-LOC body moved to src/telemetry/pick_recorder.py
        (Modularization Finish Plan Phase 4) — pick telemetry is not part of
        the dispatcher's load→call loop.
        """
        from src.telemetry.pick_recorder import record_pick
        await record_pick(
            pick=pick,
            task=task,
            category=category,
            success=success,
            error_category=error_category,
            agent_type=agent_type,
            difficulty=difficulty,
        )

    async def _ensure_local_model(
        self,
        model: "ModelInfo",
        needs_thinking: bool = False,
        needs_vision: bool = False,
        agent_type: str = "",
        task: str = "",
        estimated_context: int = 0,
        load_timeout: float = 0.0,
    ) -> tuple[bool, bool]:
        """Ensure the local model is loaded with correct vision/thinking state.

        Returns (ok, swap_happened):
          ok           — True if model is ready, False if load failed.
          swap_happened — True if a swap actually occurred (model changed).
        """
        from src.models.local_model_manager import get_local_manager

        manager = get_local_manager()
        needs_vision_load = needs_vision and model.has_vision
        needs_thinking_reload = (
            model.thinking_model
            and needs_thinking
            and not manager._thinking_enabled
        )
        # Source of truth = DaLLaMa, not registry. ModelInfo.is_loaded is
        # set by mark_loaded/mark_unloaded which only fire on swap paths;
        # IdleUnloader stops llama-server without touching registry, so
        # the flag goes stale True. Reading manager.current_model +
        # manager.is_loaded asks DaLLaMa directly: is THIS model
        # actually serving on the port right now? If not, force a
        # reload (DaLLaMa relaunches llama-server). Production triage
        # 2026-05-07: 5 minutes of OpenAIException Connection error
        # on Qwen3.5-9B because is_loaded was stale True after idle
        # unload and dispatcher kept skipping ensure_model.
        this_actually_loaded = (
            manager.is_loaded and manager.current_model == model.name
        )
        # llama-server fixes n_ctx at load and cannot grow it at runtime. If the
        # already-loaded instance was started with a smaller context than this
        # task needs, reusing it silently overflows the prompt (intake #73:
        # Qwen3.5-9B loaded at ctx 5786 under transient RAM pressure, then reused
        # for a ~14k-token analyst prompt -> context_overflow). Force a reload so
        # ensure_model re-sizes the window via the min_context floor. Without this
        # the ctx-aware guard inside ensure_model is unreachable on the
        # already-loaded path because we short-circuit before calling it.
        loaded_ctx_insufficient = (
            this_actually_loaded
            and estimated_context > 0
            and manager.loaded_context_length < estimated_context
        )
        needs_reload = (
            not this_actually_loaded
            or needs_thinking_reload
            or (needs_vision_load and not manager._vision_enabled)
            or loaded_ctx_insufficient
        )
        if not needs_reload:
            manager.keep_alive()
            return True, False

        before = manager.current_model
        reason = f"{agent_type}:{task}" if agent_type or task else "request"
        success = await manager.ensure_model(
            model.name,
            reason=reason,
            enable_thinking=needs_thinking,
            enable_vision=needs_vision_load,
            min_context=estimated_context,
            load_timeout=load_timeout,
        )
        after = manager.current_model
        swap_happened = success and (before != after)
        return success, swap_happened

    def _prepare_messages(
        self,
        messages: list[dict],
        model: "ModelInfo",
    ) -> list[dict]:
        """Prepare messages for the given model.

        - Redacts secrets for cloud models
        - Adapts trailing assistant prefill for thinking models
        """
        _messages = messages

        # Redact secrets from messages sent to cloud models
        if not model.is_local:
            try:
                from src.security.sensitivity import redact_secrets
                _messages = []
                for msg in messages:
                    _m = dict(msg)
                    if isinstance(_m.get("content"), str):
                        _m["content"] = redact_secrets(_m["content"])
                    _messages.append(_m)
            except Exception:
                _messages = messages

        # Thinking models reject assistant prefills — convert to user message
        if model.thinking_model and _messages and _messages[-1].get("role") == "assistant":
            last = _messages[-1]
            _messages = list(_messages[:-1])
            _messages.append({
                "role": "user",
                "content": (
                    "Your previous response (continue from here, "
                    "do NOT repeat this):\n\n" + last["content"]
                ),
            })

        return _messages

    # Loaded-model introspection (`get_loaded_litellm_name` etc.) moved to
    # src/models/introspection.py (Modularization Finish Plan Phase 4). The
    # speed/thinking/name variants were dead (zero callers) and were deleted.

    def get_stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "overhead_calls": self._overhead_calls,
            "overhead_pct": (
                f"{self._overhead_calls / self._total_calls * 100:.1f}%"
                if self._total_calls > 0 else "0%"
            ),
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_dispatcher: LLMDispatcher | None = None


def get_dispatcher() -> LLMDispatcher:
    """Get or create the global LLM dispatcher singleton."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = LLMDispatcher()
    return _dispatcher
