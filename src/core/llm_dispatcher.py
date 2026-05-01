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


# ─── Call Categories ─────────────────────────────────────────────────────────

class CallCategory(Enum):
    """Two categories — simple and clear."""
    MAIN_WORK = "main_work"   # Agent execution, can trigger swaps
    OVERHEAD = "overhead"     # Classifier/grader/reflection, no swaps


# ─── LLM Dispatcher ─────────────────────────────────────────────────────────

class LLMDispatcher:
    """Centralized LLM call coordinator.

    The ONLY component that should trigger model swaps, acquire GPU slots,
    and route between local and cloud.

    Usage:
        dispatcher = get_dispatcher()
        result = await dispatcher.request(
            category=CallCategory.MAIN_WORK,
            task="coder",
            agent_type="coder",
            difficulty=6,
            messages=messages,
            tools=tools,
        )
    """

    def __init__(self):
        self._total_calls = 0
        self._overhead_calls = 0

    async def request(
        self,
        category: CallCategory,
        task: str = "",
        agent_type: str = "",
        difficulty: int = 5,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        failures: list | None = None,
        preselected_pick: Any = None,
        **kwargs,
    ) -> dict:
        """Route an LLM call through the dispatcher.

        Args:
            category: MAIN_WORK or OVERHEAD
            task: Task profile key (e.g. "coder", "reviewer", "router")
            agent_type: Agent type string for selection hints
            difficulty: 1-10 scale
            messages: Chat messages
            tools: Optional tool definitions
            failures: Accumulated Failure objects from previous attempts
            **kwargs: Additional selection hints forwarded to fatih_hoca.select()

        Returns:
            dict with response content, model info, cost, etc.

        Raises:
            ModelCallFailed: When all MAIN_WORK candidates exhausted
            RuntimeError: When all OVERHEAD candidates exhausted
        """
        import fatih_hoca
        import hallederiz_kadir
        from src.core.router import ModelCallFailed
        from fatih_hoca.types import Failure

        self._total_calls += 1
        is_overhead = category == CallCategory.OVERHEAD
        if is_overhead:
            self._overhead_calls += 1

        messages = messages or []
        failures = failures or []
        max_recursion = 5

        needs_thinking = kwargs.pop("needs_thinking", not is_overhead)
        if is_overhead:
            needs_thinking = False

        # Pass tools hint to selector for function calling requirement
        needs_function_calling = kwargs.pop("needs_function_calling", bool(tools))
        if tools:
            needs_function_calling = True

        # min_context is consumed by _ensure_local_model (below), not by
        # fatih_hoca.select. Pop so it doesn't leak into the selector's
        # unknown-kwarg rejection.
        _min_context_kw = int(kwargs.pop("min_context", 0) or 0)

        # response_format is forwarded to hallederiz_kadir.call below; pop
        # here so it doesn't leak into the selector's kwarg rejection.
        # Re-injected into the call kwargs at the dispatch site.
        _response_format_kw = kwargs.pop("response_format", None)

        # Telemetry plumbing. task_obj is the agent/workflow Task dict (id,
        # agent_type, context.workflow_step_id/phase). iteration_n is the
        # ReAct loop counter. Both feed model_call_tokens for the B-table
        # rollup; without them rows have NULL keys and the rollup discards
        # them. None is acceptable for non-task callers (shopping pipeline,
        # ad-hoc graders) — those rows will be filtered out at rollup.
        _task_obj_kw = kwargs.pop("task_obj", None)
        _iteration_n_kw = int(kwargs.pop("iteration_n", 0) or 0)

        if preselected_pick is not None and not failures:
            # Iteration 0: reuse Beckman's admission-time Hoca query.
            pick = preselected_pick
        else:
            pick = fatih_hoca.select(
                task=task,
                agent_type=agent_type,
                difficulty=difficulty,
                needs_thinking=needs_thinking,
                needs_function_calling=needs_function_calling,
                failures=failures,
                call_category=category.value,
                **kwargs,
            )

        if pick is None:
            task_desc = task or agent_type or category.value
            if is_overhead:
                raise RuntimeError(
                    f"OVERHEAD call failed: no model candidates available. "
                    f"Task: {task_desc}"
                )
            raise ModelCallFailed(
                call_id=task_desc,
                last_error="No model candidates available",
                error_category="no_model",
            )

        # Pool pressure is enforced INSIDE selector (single source of truth).
        # Selector returns None when no candidate clears the urgency-derived
        # threshold; dispatcher just trusts that result. No second-guess
        # gate here.
        model = pick.model

        # Register in-flight BEFORE swap + any other awaits.
        # Rationale: swap can take 10-30s. Without early registration, the
        # next Beckman tick (~3s cadence) reads an empty in-flight list
        # and admits a second local task onto a GPU that's mid-swap.
        # Registering here also means retry recursion updates the list to
        # whatever new model Hoca picks, so mid-task model switches stay
        # accurate. task_id flows via the heartbeat contextvar; the
        # finally-block at the end of the call removes the entry along
        # every exit path (load failure, call failure, success).
        _active_task_id = None
        try:
            from src.core.heartbeat import current_task_id as _ctid
            _active_task_id = _ctid.get()
        except Exception:
            pass

        _call_id = await _begin_call(
            category=category.value,
            model_name=model.name,
            provider=model.provider,
            is_local=model.is_local,
            task_id=_active_task_id,
        )
        try:
            # Load local model if needed. Previously tried to pass a
            # prompt-derived required_ctx here (commit adb3d7c) — on an 8 GB
            # GPU that bumped context_length high enough to OOM the compute
            # buffer at warmup, regressing a working setup. Revert to letting
            # local_model_manager.ensure_model compute ctx from LIVE VRAM.
            # Prompts that exceed the allocated ctx get truncated by
            # llama-server — less catastrophic than every load failing. If
            # truncation becomes a real problem, address it with smarter
            # message-history pruning at the agent layer (drop oldest
            # tool results) rather than forcing VRAM blowups.
            if model.is_local and getattr(model, "location", "") != "ollama":
                is_thinking = model.thinking_model and needs_thinking
                # Caller (agent) passes min_context=reqs.effective_context_needed.
                # Plumbed to ensure_local_model so calculate_dynamic_context's
                # floor bumps ctx up to task need — otherwise ctx collapses to
                # the 4096 safe-minimum on tight VRAM and litellm rejects any
                # larger prompt with "exceeds the available context size".
                _min_ctx = _min_context_kw
                if _min_ctx <= 0:
                    # Legacy/standalone callers without a ModelRequirements
                    # object — derive the same formula reqs.effective_context_needed
                    # uses, so behaviour matches whether caller plumbs min_context
                    # explicitly or not.
                    _in = int(kwargs.get("estimated_input_tokens", 0) or 0)
                    _out = int(kwargs.get("estimated_output_tokens", 0) or 0)
                    if _in or _out:
                        _min_ctx = int((_in + _out) * 1.3) + 512
                # Record swap intent BEFORE ensure_model so the budget
                # reflects ATTEMPTED swaps, not just successful ones. Old
                # path recorded only after ok=True + swap_happened=True;
                # a load timeout / circuit-break still cost a process
                # kill on the prior model but didn't decrement the budget,
                # letting the next swap through immediately. Production
                # triage 2026-05-01: 96% force-kill rate (2058/2133 stops)
                # despite swap-budget gate — undercount was the cause.
                #
                # Pre-determine swap intent: not-loaded → swap will happen.
                _swap_intended = (model.is_local and not model.is_loaded)
                if _swap_intended:
                    import nerd_herd as _nerd_herd
                    _nerd_herd.record_swap(model.name)
                # Local model swap can take 30+s and waits behind any
                # in-flight call on the previous model — no per-step bump
                # inside ensure_model. Background-pump heartbeats so the
                # 300s watchdog doesn't kill the task during a slow swap.
                from src.core import heartbeat as _hb_ka
                async with _hb_ka.keepalive():
                    ok, swap_happened = await self._ensure_local_model(
                        model, needs_thinking=is_thinking,
                        load_timeout=pick.estimated_load_seconds or 0.0,
                        estimated_context=_min_ctx,
                    )
                if not ok:
                    task_desc = task or agent_type or category.value
                    if is_overhead:
                        raise RuntimeError(
                            f"OVERHEAD call failed: local model load failed. "
                            f"Task: {task_desc}"
                        )
                    raise ModelCallFailed(
                        call_id=task_desc,
                        last_error=f"Failed to load local model {model.name}",
                        error_category="loading",
                    )

            _messages = self._prepare_messages(messages, model)
            # Per-call hard cap: cost-runaway protection for cloud only. Local
            # has no wall-clock cap — stream-inactivity watchdog inside
            # hallederiz_kadir handles hung-call detection. The earlier
            # min_time-as-timeout coupling was wrong: min_time is a scoring
            # estimate, not a kill threshold.
            timeout = 600.0 if not model.is_local else 0.0  # 0 = no outer cap

            # Heartbeat the orchestrator's no-progress watchdog at call entry.
            # task_id flows via contextvar, no plumbing needed.
            try:
                from src.core import heartbeat as _hb
                _hb.bump()
            except Exception:
                pass

            try:
                # Cloud non-streaming calls block awaiting litellm.acompletion
                # — no per-token bump like local streaming gives. With timeout
                # of 600s a slow cloud call would starve the 300s watchdog.
                # Background-pump heartbeats around every cloud call. (Local
                # streaming already bumps every 5s from inside the accumulator,
                # so the keepalive is mostly a no-op there but cheap to leave
                # uniform.)
                from src.core import heartbeat as _hb_ka
                async with _hb_ka.keepalive():
                    result = await hallederiz_kadir.call(
                        model=model,
                        messages=_messages,
                        tools=tools,
                        timeout=timeout,
                        task=task or category.value,
                        needs_thinking=needs_thinking,
                        estimated_input_tokens=kwargs.get("estimated_input_tokens", 0),
                        estimated_output_tokens=kwargs.get("estimated_output_tokens", 0),
                        response_format=_response_format_kw,
                        task_obj=_task_obj_kw,
                        iteration_n=_iteration_n_kw,
                        call_category=category.value,
                    )
            except Exception as exc:
                # hallederiz_kadir wraps known failures as CallError. A raw
                # exception here is a bug in hallederiz — but we still want
                # the pick recorded so telemetry reflects the failure.
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
            return self._result_to_dict(result, model)

        # CallError path
        last_error = result.message
        last_category = result.category
        logger.debug(
            f"{category.value} candidate failed | model={model.name} "
            f"category={result.category} error={result.message[:80]}"
        )
        await self._record_pick(
            pick=pick, task=task, category=category,
            success=False, error_category=last_category,
            agent_type=agent_type, difficulty=difficulty,
        )

        if not result.retryable or len(failures) >= max_recursion:
            task_desc = task or agent_type or category.value
            if is_overhead:
                raise RuntimeError(
                    f"OVERHEAD call failed: {last_error}. Task: {task_desc}"
                )
            raise ModelCallFailed(
                call_id=task_desc,
                last_error=last_error,
                error_category=last_category,
            )

        # Build failure record.
        new_failure = Failure(
            model=model.litellm_name,
            reason=last_category,
            latency=None,
        )

        return await self.request(
            category=category,
            task=task,
            agent_type=agent_type,
            difficulty=difficulty,
            messages=messages,
            tools=tools,
            failures=failures + [new_failure],
            preselected_pick=None,
            needs_thinking=needs_thinking,
            needs_function_calling=needs_function_calling,
            min_context=_min_context_kw,
            task_obj=_task_obj_kw,
            iteration_n=_iteration_n_kw,
            response_format=_response_format_kw,
            **kwargs,
        )


    @staticmethod
    def _estimate_prompt_tokens(messages: list) -> int:
        """Rough prompt-token count from message content. 1 token ≈ 4 chars.

        Used to size llama-server's KV cache. Overestimating wastes a
        little VRAM; underestimating truncates the prompt. Skews
        slightly high — char-to-token ratio is ~3.5 for English code,
        ~4 for prose.
        """
        total_chars = 0
        for m in messages or []:
            content = m.get("content") if isinstance(m, dict) else ""
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text", "")
                        if isinstance(text, str):
                            total_chars += len(text)
        # 1 token ≈ 3 chars for mixed prose+JSON+code (skews high
        # vs the classic 4 ratio — truncation is costlier than over-
        # allocating a few hundred MB of KV).
        return max(0, total_chars // 3)

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
        """Fire-and-forget pick_log write. Never propagates errors."""
        try:
            import os
            from src.infra import pick_log as _pick_log_mod

            db_path = os.getenv("DB_PATH") or "kutai.db"
            model = getattr(pick, "model", None)
            picked_model = getattr(model, "name", "") if model is not None else ""
            picked_score = float(getattr(pick, "composite", 0.0) or 0.0)
            task_name = task or category.value
            cat_value = category.value if isinstance(category, CallCategory) else str(category)

            await _pick_log_mod.write_pick_log_row(
                db_path=db_path,
                task_name=task_name,
                picked_model=picked_model,
                picked_score=picked_score,
                category=cat_value,
                success=success,
                error_category=error_category,
                provider=("local" if getattr(model, "is_local", False) else (getattr(model, "provider", "local") or "local")),
                agent_type=agent_type,
                difficulty=difficulty,
            )
        except Exception as e:  # noqa: BLE001 — telemetry must never break dispatch
            logger.debug("pick_log record failed: %s", e)

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
        needs_reload = (
            not model.is_loaded
            or needs_thinking_reload
            or (needs_vision_load and not manager._vision_enabled)
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

    def _result_to_dict(
        self,
        result: "CallResult",
        model: "ModelInfo",
    ) -> dict:
        """Convert a hallederiz_kadir CallResult to the legacy response dict format."""
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
            "capability_score": 0.0,
            "difficulty": 5,
        }

    def _get_loaded_model_name(self) -> str | None:
        """Get the currently loaded local model's name."""
        try:
            from src.models.local_model_manager import get_local_manager
            manager = get_local_manager()
            return manager.current_model
        except Exception:
            return None

    def _get_loaded_litellm_name(self) -> str | None:
        """Get the currently loaded local model's litellm_name."""
        try:
            from src.models.local_model_manager import get_local_manager
            from src.models.model_registry import get_registry
            manager = get_local_manager()
            if not manager.current_model:
                return None
            registry = get_registry()
            info = registry.get(manager.current_model)
            return info.litellm_name if info else None
        except Exception:
            return None

    def get_loaded_model_speed(self) -> float:
        """Get the currently loaded model's measured tok/s. Returns 0 if unknown."""
        try:
            from src.models.local_model_manager import get_local_manager
            manager = get_local_manager()
            if manager.runtime_state and manager.runtime_state.measured_tps > 0:
                return manager.runtime_state.measured_tps
            if manager.current_model:
                from src.models.model_registry import get_registry
                info = get_registry().get(manager.current_model)
                if info:
                    return info.tokens_per_second
        except Exception:
            pass
        return 0.0

    def is_loaded_model_thinking(self) -> bool:
        """Check if the currently loaded model has thinking enabled."""
        try:
            from src.models.local_model_manager import get_local_manager
            manager = get_local_manager()
            if manager.runtime_state:
                return manager.runtime_state.thinking_enabled
        except Exception:
            pass
        return False

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
