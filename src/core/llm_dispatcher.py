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

import kuleden_donen_var

from src.infra.logging_config import get_logger

logger = get_logger("core.llm_dispatcher")


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

        model = pick.model

        # Load local model if needed
        if model.is_local and getattr(model, "location", "") != "ollama":
            is_thinking = model.thinking_model and needs_thinking
            ok, swap_happened = await self._ensure_local_model(
                model, needs_thinking=is_thinking,
                load_timeout=pick.estimated_load_seconds or 0.0,
            )
            if swap_happened:
                import nerd_herd as _nerd_herd
                _nerd_herd.record_swap(model.name)
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

        _kdv_handle = None
        if not model.is_local:
            _kdv_handle = kuleden_donen_var.begin_call(model.provider, model.name)
        try:
            try:
                result = await hallederiz_kadir.call(
                    model=model,
                    messages=_messages,
                    tools=tools,
                    timeout=timeout,
                    task=task or category.value,
                    needs_thinking=needs_thinking,
                    estimated_output_tokens=kwargs.get("estimated_output_tokens", 0),
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
                )
                raise
        finally:
            if _kdv_handle is not None:
                kuleden_donen_var.end_call(_kdv_handle)

        if isinstance(result, hallederiz_kadir.CallResult):
            await self._record_pick(
                pick=pick, task=task, category=category,
                success=True, error_category="",
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
            **kwargs,
        )


    async def _record_pick(
        self,
        *,
        pick: Any,
        task: str,
        category: CallCategory,
        success: bool,
        error_category: str = "",
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
