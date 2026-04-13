"""Main call function — the talking layer's entry point.

Receives a ModelInfo + messages from the dispatcher and handles:
  - completion_kwargs building (sampling, api_base, api_key, tools)
  - local vs cloud routing (DaLLaMa infer / KDV pre/post)
  - litellm.acompletion with streaming
  - response parsing and quality check
  - metrics, audit, logging
"""
from __future__ import annotations
import asyncio
import time
from typing import Callable
import litellm
from .types import CallResult, CallError
from .response import parse_response
from .retry import execute_with_retry, classify_error

litellm.suppress_debug_info = True
litellm.return_response_headers = True
litellm.request_timeout = 120


# ─── Lazy singleton accessors ──────────────────────────────────────────────

def _get_logger():
    try:
        from src.infra.logging_config import get_logger
        return get_logger("talking_layer")
    except Exception:
        import logging
        return logging.getLogger("talking_layer")


def _get_dallama():
    try:
        from src.models.local_model_manager import get_local_manager
        return get_local_manager()
    except Exception:
        return None


def _kdv_pre_call(litellm_name: str, provider: str, estimated_tokens: int):
    """Returns: (allowed, wait_seconds, daily_exhausted)"""
    try:
        from src.core.router import get_kdv
        kdv = get_kdv()
        pre = kdv.pre_call(litellm_name, provider, estimated_tokens)
        return (pre.allowed, pre.wait_seconds, pre.daily_exhausted)
    except Exception:
        return (True, 0.0, False)


def _kdv_post_call(litellm_name, provider, headers, token_count):
    try:
        from src.core.router import get_kdv
        get_kdv().post_call(litellm_name, provider, headers=headers, token_count=token_count)
    except Exception:
        pass


def _kdv_record_failure(litellm_name, provider, reason):
    try:
        from src.core.router import get_kdv
        get_kdv().record_failure(litellm_name, provider, reason)
    except Exception:
        pass


def _get_sampling_params(task, sampling_overrides=None):
    try:
        from src.models.model_profiles import get_sampling_params
        return get_sampling_params(task, sampling_overrides=sampling_overrides)
    except Exception:
        return None


def _record_metrics(model_name, cost, latency_ms, tokens):
    try:
        from src.infra.metrics import track_model_call_metrics
        track_model_call_metrics(model=model_name, cost=cost, latency_ms=latency_ms, tokens=tokens)
    except Exception:
        pass


async def _record_audit(agent_type, model_litellm, task, cost, latency):
    try:
        from src.infra.audit import audit, ACTOR_AGENT, ACTION_MODEL_CALL
        await audit(actor=f"{ACTOR_AGENT}:{agent_type or 'unknown'}", action=ACTION_MODEL_CALL,
                    target=model_litellm, details=f"task={task} cost=${cost:.4f} latency={latency:.1f}s")
    except Exception:
        pass


def _check_quality(content):
    """Returns: (is_ok, reason)"""
    if not content or len(content) < 20:
        return (True, "")
    try:
        from dogru_mu_samet import assess
        result = assess(content)
        if result.is_degenerate:
            return (False, result.summary)
    except Exception:
        pass
    return (True, "")


# ─── Streaming ─────────────────────────────────────────────────────────────

async def _stream_with_accumulator(completion_kwargs, partial_content_ref):
    """Call litellm with streaming, accumulate content for partial recovery."""
    completion_kwargs["stream"] = True
    chunks = []
    accumulated = ""
    role = "assistant"
    finish_reason = None

    async for chunk in await litellm.acompletion(**completion_kwargs):
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta:
            if delta.content:
                accumulated += delta.content
                partial_content_ref[0] = accumulated
            if delta.role:
                role = delta.role
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        chunks.append(chunk)

    last = chunks[-1] if chunks else None
    resp = litellm.ModelResponse(
        id=getattr(last, "id", "0"),
        choices=[{"message": {"role": role, "content": accumulated}, "finish_reason": finish_reason or "stop"}],
        usage=getattr(last, "usage", None),
    )
    return resp


# ─── Main Call Function ────────────────────────────────────────────────────

async def call(
    model,
    messages: list[dict],
    tools: list[dict] | None,
    timeout: float,
    task: str,
    needs_thinking: bool,
    estimated_output_tokens: int = 1000,
) -> CallResult | CallError:
    """Execute an LLM call against a single model.

    This is the talking layer's main entry point. The dispatcher calls this
    once per candidate model.

    Args:
        model: ModelInfo object from router scoring
        messages: chat messages (already redacted/adapted by dispatcher)
        tools: tool definitions or None
        timeout: seconds (computed by dispatcher)
        task: task label for sampling profile and logging
        needs_thinking: whether to enable thinking for this call
        estimated_output_tokens: for max_tokens calculation
    """
    logger = _get_logger()
    is_thinking = model.thinking_model and needs_thinking
    is_local = model.is_local
    is_ollama = model.location == "ollama"

    # ── Sampling ──
    if is_thinking:
        sampling_params = None
    else:
        sampling_params = _get_sampling_params(
            task, sampling_overrides=getattr(model, "sampling_overrides", None))

    # ── Max tokens ──
    _max_tokens = min(estimated_output_tokens * 2, model.max_tokens)

    # ── HTTP timeout ──
    http_timeout = max(10.0, float(timeout) - 5.0)

    # ── Build completion kwargs ──
    completion_kwargs = dict(
        model=model.litellm_name, messages=messages,
        max_tokens=_max_tokens, timeout=http_timeout,
    )

    if sampling_params:
        for k, v in sampling_params.items():
            completion_kwargs[k] = v
    elif model.thinking_model and not is_thinking:
        completion_kwargs["temperature"] = 0.3

    if model.api_base:
        completion_kwargs["api_base"] = model.api_base
    if is_local and not is_ollama:
        completion_kwargs["api_key"] = "sk-no-key"
        completion_kwargs["num_retries"] = 0

    # ── Tools ──
    use_tools = None
    if tools and model.supports_function_calling:
        use_tools = tools
        completion_kwargs["tools"] = tools
        completion_kwargs["tool_choice"] = "auto"
    elif tools and not model.supports_function_calling and model.supports_json_mode:
        completion_kwargs["response_format"] = {"type": "json_object"}

    # ── Rate limiting (cloud only) ──
    if not is_local:
        estimated_tokens = estimated_output_tokens * 3
        allowed, wait_secs, daily_exhausted = _kdv_pre_call(
            model.litellm_name, model.provider, estimated_tokens)
        if daily_exhausted:
            return CallError(category="daily_exhausted",
                           message=f"Daily limit exhausted for {model.name}", retryable=False)
        if not allowed:
            if wait_secs > 0:
                await asyncio.sleep(wait_secs)
            else:
                return CallError(category="rate_limited",
                               message=f"Rate limited for {model.name}", retryable=True)

    # ── GPU semaphore (local only) ──
    local_manager = None
    _inf_gen = None
    if is_local and not is_ollama:
        local_manager = _get_dallama()
        if local_manager:
            gpu_timeout = min(timeout, 120.0)
            granted = await local_manager.acquire_inference_slot(
                priority=getattr(model, "priority", 5),
                task_id=task, agent_type=task, timeout=gpu_timeout)
            if not granted:
                return CallError(category="gpu_busy",
                               message=f"GPU queue timeout for {model.name}", retryable=True)
            _inf_gen = local_manager.mark_inference_start()
            if local_manager.current_model != model.name:
                actual = local_manager.current_model
                local_manager.mark_inference_end(_inf_gen)
                local_manager.release_inference_slot()
                return CallError(category="loading",
                               message=f"Model swapped during GPU wait: {model.name} → {actual}",
                               retryable=True)

    # ── Streaming decision ──
    use_stream = is_local and not is_ollama and not use_tools

    # ── Execute with retry ──
    max_retries = 2 if is_local else 3
    partial_content_ref = [""]
    call_start = time.time()

    async def _health_check_fn():
        if local_manager:
            return await local_manager._health_check()
        return True

    def _swap_check_fn():
        if local_manager:
            return local_manager.swap_started_at > 0
        return False

    async def _do_call():
        if use_stream:
            return await _stream_with_accumulator(completion_kwargs, partial_content_ref)
        else:
            return await litellm.acompletion(**completion_kwargs)

    try:
        raw_result = await execute_with_retry(
            call_fn=_do_call, max_retries=max_retries, timeout=timeout,
            is_local=is_local, model_name=model.name,
            health_check=_health_check_fn if local_manager else None,
            is_swap_in_progress=_swap_check_fn if local_manager else None,
            partial_content_ref=partial_content_ref,
        )
    finally:
        if local_manager and _inf_gen is not None:
            local_manager.mark_inference_end(_inf_gen)
            local_manager.release_inference_slot()

    # ── Handle retry failure ──
    if isinstance(raw_result, CallError):
        if not is_local:
            _kdv_record_failure(model.litellm_name, model.provider, raw_result.category)
        return raw_result

    # ── Parse response ──
    call_latency = time.time() - call_start
    parsed = parse_response(raw_result, model_name=model.name,
                           is_local=is_local, is_thinking=is_thinking)

    # ── Cloud: record usage via KDV ──
    if not is_local and raw_result.usage:
        total_tokens = ((raw_result.usage.prompt_tokens or 0)
                       + (raw_result.usage.completion_tokens or 0))
        hidden = getattr(raw_result, "_hidden_params", None)
        headers = {}
        if hidden:
            headers = dict(hidden.get("additional_headers") or hidden.get("headers") or {})
        _kdv_post_call(model.litellm_name, model.provider, headers=headers, token_count=total_tokens)

    # ── Local: update measured speed ──
    if is_local and raw_result.usage:
        output_tokens = raw_result.usage.completion_tokens or 0
        if output_tokens > 0 and call_latency > 0:
            tok_per_sec = output_tokens / call_latency
            logger.info("llm performance", model_name=model.name,
                       output_tokens=output_tokens, latency=f"{call_latency:.1f}s",
                       speed=f"{tok_per_sec:.1f} tok/s")
            try:
                from src.models.model_registry import get_registry
                get_registry().update_measured_speed(model.name, tok_per_sec)
            except Exception:
                pass
            if local_manager and local_manager.runtime_state:
                try:
                    local_manager.runtime_state.measured_tps = tok_per_sec
                except Exception:
                    pass

    # ── Quality check ──
    content = parsed["content"]
    quality_ok, quality_reason = _check_quality(content)
    if not quality_ok:
        logger.warning("quality check failed", model_name=model.name, reason=quality_reason)
        return CallError(category="quality_failure",
                        message=f"Quality check failed: {quality_reason}",
                        retryable=True, partial_content=content)

    # ── Metrics ──
    total_tokens = (parsed["usage"].get("prompt_tokens", 0)
                   + parsed["usage"].get("completion_tokens", 0))
    _record_metrics(model.name, parsed["cost"], call_latency * 1000, total_tokens)

    # ── Audit ──
    try:
        await _record_audit(task, model.litellm_name, task, parsed["cost"], call_latency)
    except Exception:
        pass

    return CallResult(
        content=content, tool_calls=parsed["tool_calls"], thinking=parsed["thinking"],
        usage=parsed["usage"], cost=parsed["cost"], latency=call_latency,
        model=model.litellm_name, model_name=model.name,
        is_local=is_local, provider=model.provider if not is_local else "local", task=task,
    )
