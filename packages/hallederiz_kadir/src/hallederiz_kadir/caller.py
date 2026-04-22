"""Main call function — HaLLederiz Kadir's entry point.

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
# No global request_timeout — each call passes its own timeout via
# completion_kwargs["timeout"], computed from the dispatcher's budget.
# A global cap here silently truncates long generations (e.g. 300s coder
# calls capped to 120s).


# ─── Lazy singleton accessors ──────────────────────────────────────────────

def _get_logger():
    try:
        from src.infra.logging_config import get_logger
        return get_logger("hallederiz_kadir")
    except Exception:
        import logging
        return logging.getLogger("hallederiz_kadir")


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
    """Call litellm with streaming, accumulate content for partial recovery.

    Two abort triggers:
      1. Degenerate repetition (dogru_mu_samet streaming callback).
      2. Stream-inactivity watchdog: no chunk in N seconds = hung.

    Replaces the old wall-clock per-call timeout. Healthy slow generation
    keeps producing chunks (or thinking-content deltas) — the watchdog
    only fires on real silence.
    """
    import asyncio as _asyncio
    from dogru_mu_samet.streaming import make_stream_callback
    # Size capping is llama-server's job (ctx-size); we only watch for
    # repetition / low-entropy degeneration during streaming.
    _should_abort = make_stream_callback(max_size=200_000, check_interval=2000)

    # First-chunk wait covers prefill (10k-token prompts on 9B ≈ 50s).
    # Inter-chunk wait covers between-token silence (real stream that's
    # alive emits something — content or reasoning_content — every few
    # hundred ms; sustained 20s silence means the server is wedged).
    INITIAL_CHUNK_TIMEOUT = 180.0
    INTER_CHUNK_TIMEOUT = 20.0

    completion_kwargs["stream"] = True
    chunks = []
    accumulated = ""
    accumulated_reasoning = ""
    role = "assistant"
    finish_reason = None
    # Tool call accumulation: {index: {"id": ..., "name": ..., "arguments": ...}}
    tool_call_parts: dict[int, dict] = {}

    stream = await litellm.acompletion(**completion_kwargs)
    stream_iter = stream.__aiter__()
    chunk_timeout = INITIAL_CHUNK_TIMEOUT
    # Heartbeat throttling — bumping a contextvar-backed dict on every
    # token delta is wasteful. Bump at most every 5 seconds.
    HB_INTERVAL = 5.0
    import time as _time
    _last_hb = _time.monotonic()
    while True:
        try:
            chunk = await _asyncio.wait_for(stream_iter.__anext__(), timeout=chunk_timeout)
        except StopAsyncIteration:
            break
        except _asyncio.TimeoutError:
            _get_logger().warning(
                f"stream-inactivity watchdog: no chunk in {chunk_timeout:.0f}s — "
                f"aborting (accumulated {len(accumulated)} chars)"
            )
            raise _asyncio.TimeoutError(
                f"Stream silent for {chunk_timeout:.0f}s"
            )
        # After the first chunk, switch to the tighter inter-chunk window.
        chunk_timeout = INTER_CHUNK_TIMEOUT
        # Heartbeat the orchestrator's no-progress watchdog. Throttled
        # so we don't churn the dict on every token delta — chunks at
        # 50+ tok/s is fine; one bump every 5s is enough.
        _now = _time.monotonic()
        if _now - _last_hb >= HB_INTERVAL:
            _last_hb = _now
            try:
                from src.core import heartbeat as _hb
                _hb.bump()
            except Exception:
                pass
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta:
            if delta.content:
                accumulated += delta.content
                partial_content_ref[0] = accumulated
            # Capture reasoning_content from thinking models (DeepSeek, etc.)
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                accumulated_reasoning += rc
            # Accumulate streamed tool_calls deltas
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index if hasattr(tc_delta, "index") else 0
                    if idx not in tool_call_parts:
                        tool_call_parts[idx] = {"id": "", "name": "", "arguments": ""}
                    part = tool_call_parts[idx]
                    if hasattr(tc_delta, "id") and tc_delta.id:
                        part["id"] = tc_delta.id
                    fn = getattr(tc_delta, "function", None)
                    if fn:
                        if hasattr(fn, "name") and fn.name:
                            part["name"] = fn.name
                        if hasattr(fn, "arguments") and fn.arguments:
                            part["arguments"] += fn.arguments
            if delta.role:
                role = delta.role
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        chunks.append(chunk)

        if _should_abort(accumulated):
            _get_logger().warning(
                "stream aborted: degenerate content detected "
                f"at {len(accumulated)} chars"
            )
            finish_reason = "length"
            break

    # If content is empty but reasoning was captured, rescue it
    if not accumulated.strip() and accumulated_reasoning.strip():
        import re
        accumulated = re.sub(r"</?think>", "", accumulated_reasoning).strip()

    # Reassemble tool calls into the format parse_response expects
    assembled_tool_calls = None
    if tool_call_parts:
        assembled_tool_calls = []
        for idx in sorted(tool_call_parts):
            p = tool_call_parts[idx]
            assembled_tool_calls.append({
                "id": p["id"] or f"call_{idx}",
                "type": "function",
                "function": {"name": p["name"], "arguments": p["arguments"]},
            })

    last = chunks[-1] if chunks else None
    msg_dict: dict = {"role": role, "content": accumulated}
    if assembled_tool_calls:
        msg_dict["tool_calls"] = assembled_tool_calls
    resp = litellm.ModelResponse(
        id=getattr(last, "id", "0"),
        choices=[{"message": msg_dict, "finish_reason": finish_reason or "stop"}],
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

    This is HaLLederiz Kadir's main entry point. The dispatcher calls this
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
    # Local models: don't cap — llama-server enforces ctx-size naturally,
    # and the post-execute hook summarizes long artifacts for downstream.
    # Cloud models: cap to avoid runaway cost.
    if is_local:
        _max_tokens = None  # omit from request → server uses full context
    else:
        _max_tokens = min(estimated_output_tokens * 2, model.max_tokens)

    # ── Per-request reasoning override ──
    # If a thinking-capable model is loaded with --reasoning on but this
    # call wants thinking OFF (OVERHEAD / grader / classifier), llama-server
    # would otherwise burn thousands of invisible reasoning tokens before
    # emitting content. Sending reasoning_budget=0 + chat_template_kwargs
    # tells the server to skip thinking for this request only — no swap.
    _suppress_reasoning = (
        is_local and model.thinking_model and not is_thinking
    )

    # ── HTTP timeout ──
    http_timeout = max(10.0, float(timeout) - 5.0)

    # ── Build completion kwargs ──
    completion_kwargs = dict(
        model=model.litellm_name, messages=messages,
        timeout=http_timeout,
    )
    if _max_tokens is not None:
        completion_kwargs["max_tokens"] = _max_tokens

    if _suppress_reasoning:
        # llama.cpp v8668+: reasoning_budget caps thinking tokens. 0 disables.
        # chat_template_kwargs is a legacy hint — deprecated but harmless,
        # kept for Ollama-served GGUFs that still honor it.
        extra = dict(completion_kwargs.get("extra_body") or {})
        extra["reasoning_budget"] = 0
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        completion_kwargs["extra_body"] = extra

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

    # ── Local model manager reference (for health checks, speed tracking) ──
    local_manager = None
    if is_local and not is_ollama:
        local_manager = _get_dallama()

    # ── Streaming decision ──
    # Always stream local models — including tool calls.  Non-streaming +
    # thinking model + tools caused llama-server to silently drop requests
    # (2026-04-15 incident: 190s "all slots idle" while request sat
    # unprocessed).  The accumulator reassembles tool_call deltas.
    use_stream = is_local and not is_ollama

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

    # Mark this call in-flight so DaLLaMa's idle-unload watchdog does not
    # yank the model out from under a long-running inference. Without this,
    # calls exceeding idle_timeout_seconds got cancelled mid-generation
    # (observed 2026-04-22: 97s writer call on Qwen3.5-35B unloaded at
    # 90s, connection_error returned to dispatcher).
    inflight_gen = None
    if local_manager is not None:
        try:
            inflight_gen = local_manager.begin_inference()
        except Exception:
            inflight_gen = None
    try:
        raw_result = await execute_with_retry(
            call_fn=_do_call, max_retries=max_retries, timeout=timeout,
            is_local=is_local, model_name=model.name,
            health_check=_health_check_fn if local_manager else None,
            is_swap_in_progress=_swap_check_fn if local_manager else None,
            partial_content_ref=partial_content_ref,
        )
    finally:
        if inflight_gen is not None and local_manager is not None:
            try:
                local_manager.end_inference(inflight_gen)
            except Exception:
                pass

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
