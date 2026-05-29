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

def _request_kwargs_to_spec(category: "CallCategory", **kwargs) -> dict:
    """Map dispatcher.request() kwargs into a Beckman task spec dict.

    The call payload (messages, tools, flags, etc.) is stuffed into
    ``spec["context"]["llm_call"]``.  The sentinel ``raw_dispatch=True``
    tells the orchestrator pump to route this task directly to
    ``husam.run()`` instead of an agent class.
    """
    import uuid as _uuid
    import time as _time

    task_name: str = kwargs.get("task", "") or ""
    agent_type: str = kwargs.get("agent_type", "") or ""
    difficulty: int = int(kwargs.get("difficulty", 5) or 5)
    priority: int = int(kwargs.get("priority", 5) or 5)
    mission_id = kwargs.get("mission_id")
    parent_task_id = kwargs.get("parent_task_id")

    # Map CallCategory → kind string (matches the tasks.kind column).
    kind = category.value  # "main_work" or "overhead"

    # Unique suffix prevents add_task dedup from silently dropping concurrent
    # calls that have identical (title, description, agent_type, mission_id,
    # parent_task_id).  Uses a short random token + epoch ms.
    _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
    title = f"llm_call:{task_name or kind}:{_suffix}"
    description = f"LLM {category.value} call"

    # Build the llm_call payload that dispatch() will read back.
    llm_call: dict = {
        "raw_dispatch": True,          # sentinel for orchestrator pump
        "call_category": category.value,
        "task": task_name,
        "agent_type": agent_type,
        "difficulty": difficulty,
        "messages": kwargs.get("messages") or [],
        "tools": kwargs.get("tools"),
        "failures": kwargs.get("failures") or [],
        "preselected_pick": None,      # not serialisable; re-selected in dispatch
        "prefer_speed": kwargs.get("prefer_speed"),
        "prefer_local": kwargs.get("prefer_local"),
        "needs_json_mode": kwargs.get("needs_json_mode"),
        "needs_thinking": kwargs.get("needs_thinking"),
        "needs_function_calling": kwargs.get("needs_function_calling"),
        "min_context": kwargs.get("min_context"),
        "response_format": kwargs.get("response_format"),
        "estimated_input_tokens": kwargs.get("estimated_input_tokens"),
        "estimated_output_tokens": kwargs.get("estimated_output_tokens"),
        "urgency": kwargs.get("urgency"),
        # task_obj / iteration_n are runtime objects; don't attempt to serialise
    }
    # Strip None values to keep context compact.
    llm_call = {k: v for k, v in llm_call.items() if v is not None or k in (
        "raw_dispatch", "task", "agent_type", "difficulty", "messages",
        "failures", "call_category",
    )}
    # raw_dispatch must always be present.
    llm_call["raw_dispatch"] = True

    spec: dict = {
        "title": title,
        "description": description,
        "agent_type": agent_type or kind,
        "kind": kind,
        # Phase D — single-call dispatcher.request → Beckman.enqueue path
        # is always a direct lane (no ReAct loop, no tools). Orchestrator
        # pump dispatches by task.runner.
        "runner": "direct",
        "priority": priority,
        "context": {"llm_call": llm_call},
    }
    if mission_id is not None:
        spec["mission_id"] = mission_id
    if parent_task_id is not None:
        spec["parent_task_id"] = parent_task_id

    return spec


def _task_result_to_request_response(result: "TaskResult") -> dict:
    """Map a TaskResult back to the legacy response dict expected by all callers.

    Callers expect at minimum ``{"content": str, ...}``.  husam.run stores
    its full husam.mapping.result_to_dict() output under TaskResult.result.  On_task_finished
    serialises that into a JSON string before passing it to TaskResult, so we
    handle both dict and JSON-string shapes.
    """
    import json as _json

    raw = result.result
    if isinstance(raw, str):
        try:
            payload = _json.loads(raw)
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            payload = {}
    elif isinstance(raw, dict):
        payload = raw
    else:
        payload = {}
    # Ensure the mandatory "content" key is always present.
    if "content" not in payload:
        payload = dict(payload)
        payload["content"] = ""
    return payload


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

    # ─── Public alias ─────────────────────────────────────────────────────

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
        """DEPRECATION ALIAS — routes through Beckman.

        All callers will migrate to beckman.enqueue() over time. Until
        then this preserves the public API by:
          1. Converting kwargs → Beckman task spec
          2. Calling beckman.enqueue(spec, await_inline=True)
          3. Mapping TaskResult back to the legacy response dict

        Raises ModelCallFailed / RuntimeError on failure, identical to the
        previous direct-dispatch behaviour.
        """
        import general_beckman

        spec = _request_kwargs_to_spec(
            category,
            task=task,
            agent_type=agent_type,
            difficulty=difficulty,
            messages=messages,
            tools=tools,
            failures=failures,
            preselected_pick=preselected_pick,
            **kwargs,
        )

        # Keep the PARENT task's heartbeat fresh while we wait on the child
        # raw_dispatch task. dispatcher.request now blocks awaiting an
        # inline-waiter future for an entire Beckman admission + dispatch
        # cycle (queue → reserve → select → call → grade) — easily 60+s
        # under cloud cooldowns or local swaps. The parent's no-progress
        # watchdog only sees the dispatched task's heartbeat, and that
        # task's contextvar id is the PARENT's, not the child's. Without
        # this wrapper the parent goes 300s without a bump while the child
        # is making real progress, the watchdog kills the runner, and the
        # mission step lands as "task wedged" even though work was happening.
        # Production 2026-05-04: 11+ wedged ❌ pings within 5 minutes after
        # the dispatcher.request → enqueue alias landed.
        from src.core import heartbeat as _hb
        async with _hb.keepalive():
            result = await general_beckman.enqueue(spec, await_inline=True)

        if result.status == "failed":
            err = result.error or "LLM call failed"
            is_overhead = category == CallCategory.OVERHEAD
            if is_overhead:
                raise RuntimeError(
                    f"OVERHEAD call failed: {err}. Task: {task or category.value}"
                )
            raise ModelCallFailed(
                call_id=task or category.value,
                last_error=err,
                error_category="dispatch",
            )

        return _task_result_to_request_response(result)

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
                _min_ctx = min_context
                if _min_ctx <= 0 and (estimated_input_tokens or estimated_output_tokens):
                    _min_ctx = int((estimated_input_tokens + estimated_output_tokens) * 1.3) + 512
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
        """Fire-and-forget pick_log write. Never propagates errors."""
        try:
            import os
            from src.infra import pick_log as _pick_log_mod

            db_path = os.getenv("DB_PATH") or "kutai.db"
            model = getattr(pick, "model", None)
            picked_model = getattr(model, "name", "") if model is not None else ""
            # Read score from Pick.score (populated by selector). The
            # legacy `composite` attribute never existed on Pick — every
            # row was getting picked_score=0.0 silently. Now persists
            # ScoredModel.score from the post-utilization rank step.
            picked_score = float(getattr(pick, "score", 0.0) or 0.0)
            # Top-5 candidate summary from the same select() invocation.
            # Persists into model_pick_log.snapshot_summary so offline
            # analysis can see runner-up scores alongside the winner —
            # diagnoses "did we have a clear winner or a near-tie?"
            snapshot_summary = str(getattr(pick, "top_summary", "") or "")
            task_name = task or category.value
            cat_value = category.value if isinstance(category, CallCategory) else str(category)

            # Resolve the active task id from the heartbeat ContextVar —
            # same pattern used at request() entry (lines 663-664). Defensive:
            # overhead calls and tests that don't set the ContextVar get None.
            _active_task_id: int | None = None
            try:
                from src.core.heartbeat import current_task_id as _ctid2
                _active_task_id = _ctid2.get()
            except Exception:
                pass

            await _pick_log_mod.write_pick_log_row(
                db_path=db_path,
                task_name=task_name,
                picked_model=picked_model,
                picked_score=picked_score,
                category=cat_value,
                success=success,
                error_category=error_category,
                snapshot_summary=snapshot_summary,
                provider=("local" if getattr(model, "is_local", False) else (getattr(model, "provider", "local") or "local")),
                agent_type=agent_type,
                difficulty=difficulty,
                task_id=_active_task_id,
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
