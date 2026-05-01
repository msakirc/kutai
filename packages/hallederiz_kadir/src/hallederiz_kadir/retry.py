"""Retry loop and error classification for LLM calls."""
from __future__ import annotations
import asyncio
import time
from typing import Callable, Awaitable
from .types import CallError


def _extract_status_code(exc) -> int | None:
    """Pull HTTP status off a LiteLLM exception. Tries the common shapes:
    e.status_code, e.response.status_code, e.response.status. Returns
    None when no numeric status is available (raw asyncio errors,
    network errors before the response landed, etc.)."""
    sc = getattr(exc, "status_code", None)
    if isinstance(sc, int):
        return sc
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None) or getattr(resp, "status", None)
        if isinstance(sc, int):
            return sc
    return None


def _classify_by_status(status_code: int | None, error_lc: str) -> str | None:
    """Map an HTTP status code to a retry category. Returns None when the
    code carries no actionable signal (200s, 1xx, or unknown 4xx/5xx that
    text matching can disambiguate further). Status codes outrank text
    matches because providers wrap consistent semantics behind status
    while phrasing varies (Gemini "is not found for api version" vs
    OpenRouter "No endpoints found" vs OpenAI "model_not_found" — all
    HTTP 404)."""
    if status_code is None:
        return None
    if status_code in (401, 403):
        return "auth_failure"
    if status_code == 404:
        return "model_not_found"
    if status_code == 408:
        return "timeout"
    if status_code == 413 or status_code == 422:
        # 413 Payload Too Large; 422 Unprocessable Entity is what some
        # providers return for context-overflow before the model sees the
        # request. Disambiguate via text — "context" hint stays useful
        # since 422 also covers schema/validation failures.
        if "context" in error_lc or "exceeds" in error_lc:
            return "context_overflow"
        return None
    if status_code == 429:
        return "rate_limited"
    if status_code in (500, 502, 503, 504):
        return "server_error"
    return None


def classify_error(error: str, status_code: int | None = None) -> str:
    """Classify a provider error into a retry category.

    Order of precedence:
    1. HTTP status code (when provided) — providers consistently wrap
       semantics behind status (404=not_found, 429=rate_limit, 401/403=
       auth, 5xx=server_error). Text phrasing varies by provider and
       version, status does not.
    2. Text fallback — covers locally-raised errors with no HTTP context
       (asyncio.TimeoutError, GPU-queue messages, llama.cpp loading state,
       circuit-breaker, etc.) and disambiguates buckets that share a
       status (e.g. daily_exhausted vs rpm rate_limited both 429).
    """
    e = error.lower()

    # 1. Status-code-driven categories (when caller provided status).
    cat = _classify_by_status(status_code, e)
    if cat is not None:
        # Refine status-derived rate_limited into daily_exhausted when
        # body parser already wrote a Daily marker into the error msg.
        # daily_exhausted retries differently in the dispatcher (model
        # added to failures, not the call retried with backoff).
        if cat == "rate_limited" and "daily limit exhausted" in e:
            return "daily_exhausted"
        return cat

    # 2. Text fallback for non-HTTP errors and additional disambiguation.
    if "gpu queue timeout" in e or "gpu access denied" in e:
        return "gpu_busy"
    if any(k in e for k in ("rate limit", "rate_limit", "429",
                             "too many requests", "tokens per minute",
                             "resource_exhausted")):
        return "rate_limited"
    if "daily limit exhausted" in e:
        return "daily_exhausted"
    if "loading model" in e:
        return "loading"
    if "circuit breaker" in e or "failed to load local model" in e:
        return "circuit_breaker"
    if "no models available" in e or "no models matched" in e:
        return "no_model"
    if any(k in e for k in (
        "api key", "authentication", "unauthorized", "billing",
        "key limit exceeded",  # OpenRouter: credit/spend cap on the key
        "insufficient_quota", "insufficient credits", "credit balance",
    )):
        return "auth_failure"
    # Provider-specific not-found tags. Loose substring "404" + "not found"
    # was previously matched here as fallback but it false-positives on
    # error bodies that ECHO the request payload back — Groq's
    # json_validate_failed includes the failed_generation field which can
    # carry arbitrary user code (FastAPI handlers raise HTTPException 404
    # not_found), and Anthropic / Gemini bodies sometimes embed conversation
    # context. Production triage 2026-05-01: groq/openai/gpt-oss-20b's
    # constrained_emit error contained generated FastAPI router code with
    # `404`/`"not found"` literals; classifier flagged the model as
    # model_not_found and mark_dead'd it for the rest of the session. Only
    # match unambiguous provider tags now; rely on status_code 404 from
    # _classify_by_status (above) for the structured-not-found case.
    if (
        "is not found for api version" in e        # Gemini retired ids
        or "model_not_found" in e                  # OpenAI native code
        or "no endpoints found" in e               # OpenRouter routing
    ):
        return "model_not_found"
    if any(k in e for k in ("timeout", "timed out")):
        return "timeout"
    if any(k in e for k in ("connection", "network", "dns", "refused")):
        return "connection_error"
    if any(k in e for k in ("500", "internal server error")):
        return "server_error"
    if (
        "exceeds the available context size" in e
        or "context_length_exceeded" in e
        # Groq phrasing for context overflow on small-context models like
        # allam-2-7b. Production triage 2026-05-01: "Please reduce the
        # length of the messages or completion." was classifying as
        # "unknown" → retried generically → same model picked again →
        # same overflow → DLQ.
        or "please reduce the length of the messages" in e
        or "maximum context length" in e
        or "request too large for model" in e  # tpm-style overflow
    ):
        return "context_overflow"
    # Groq's "model does not support JSON output" — non-retryable on
    # the same model. Categorize so dispatcher records it in failures
    # and selector reselects a json-capable peer. Pre-this fix it
    # classified as "unknown" → generic retry → same incapable model.
    if "does not support json output" in e:
        return "json_unsupported"
    return "unknown"


async def execute_with_retry(
    call_fn: Callable[[], Awaitable],
    max_retries: int,
    timeout: float,
    is_local: bool,
    model_name: str,
    health_check: Callable[[], Awaitable[bool]] | None = None,
    is_swap_in_progress: Callable[[], bool] | None = None,
    partial_content_ref: list | None = None,
) -> dict | CallError:
    """Execute an LLM call with retry, timeout, and error handling.

    Args:
        call_fn: async callable that makes the litellm call and returns parsed response
        max_retries: max attempts (2 for local, 3 for cloud)
        timeout: seconds for asyncio.wait_for
        is_local: whether this is a local model
        model_name: for error messages
        health_check: async fn to check if local server is alive
        is_swap_in_progress: fn returning True if model swap is happening
        partial_content_ref: single-element list updated by streaming with partial content

    Returns:
        Raw litellm response on success, CallError on failure.
    """
    last_error: str | None = None
    # Captured headers from the most recent exception, when LiteLLM surfaces
    # them on the exception object (RateLimitError, BadRequestError, etc.
    # all carry .response.headers when the call reached the provider).
    # Forwarded to KDV.record_attempt so 4xx/5xx responses still update
    # x-ratelimit-* counters.
    last_headers: dict[str, str] | None = None
    # Captured HTTP status off the most recent exception. Drives error
    # classification with priority over text matching — providers wrap
    # consistent HTTP semantics (404=not_found, 429=rate_limit, 401/403=
    # auth, 5xx=server_error) but vary their human-readable phrasing.
    last_status_code: int | None = None

    for attempt in range(max_retries):
        try:
            # timeout==0 → no outer wall-clock cap. Used for local models
            # where the stream-inactivity watchdog inside the call_fn is
            # the sole hung-detection mechanism. Cloud calls keep the
            # outer cap as cost-runaway protection.
            if timeout and timeout > 0:
                result = await asyncio.wait_for(call_fn(), timeout=timeout)
            else:
                result = await call_fn()
            return result

        except asyncio.TimeoutError:
            last_error = f"Timeout on {model_name}"
            if is_local and health_check:
                alive = await health_check()
                if not alive:
                    break
            continue

        except Exception as e:
            error_str = str(e).lower()
            last_error = str(e)
            # Try common LiteLLM exception shapes: e.response.headers,
            # e.headers, e.response.headers when response is a Response-like
            # object. Fall back to None on any AttributeError. Headers from
            # 4xx/5xx are still authoritative rate-limit signals.
            try:
                resp = getattr(e, "response", None)
                hdrs = getattr(resp, "headers", None) if resp is not None else None
                if hdrs is None:
                    hdrs = getattr(e, "headers", None)
                if hdrs is not None:
                    last_headers = dict(hdrs)
            except Exception:
                last_headers = None
            last_status_code = _extract_status_code(e)

            # Status-code-driven branches first (when present). HTTP semantics
            # are stable across providers; text phrasing is not.
            if last_status_code in (401, 403):
                break  # auth — never retryable
            if last_status_code == 404:
                break  # provider says id doesn't exist; same id won't resurrect
            if last_status_code == 429:
                if attempt < max_retries - 1:
                    await asyncio.sleep((attempt + 1) * 5)
                    continue
                break

            # Text-fallback branches for errors that arrived without a status
            # code (network errors before response, asyncio cancellations,
            # local llama-server loading state, GPU queue messages).

            # Auth/billing
            if any(kw in error_str for kw in (
                "api key", "authentication", "unauthorized",
                "billing", "credit", "quota",
            )):
                break

            # 404 / model not found (text fallback when status missing).
            # Loose "404"+"not found" check removed — false-positives on
            # request-echo bodies (Groq json_validate_failed payload).
            if (
                "is not found for api version" in error_str
                or "model_not_found" in error_str
                or "no endpoints found" in error_str
            ):
                break

            # Rate limit (text fallback)
            is_rate_limit = any(kw in error_str for kw in (
                "rate limit", "rate_limit", "429",
                "too many requests", "tokens per minute",
                "resource_exhausted",
            ))
            if is_rate_limit:
                if attempt < max_retries - 1:
                    await asyncio.sleep((attempt + 1) * 5)
                    continue
                break

            # Local model loading — wait for it
            if "loading model" in error_str and is_local and health_check:
                load_wait = 0
                while load_wait < 30:
                    await asyncio.sleep(5)
                    load_wait += 5
                    if await health_check():
                        break
                if await health_check():
                    continue
                break

            # Local 5xx during swap — wait briefly. Status code path catches
            # 500/502/503/504; text fallback for raw Connection errors that
            # sometimes carry "500" in the body without a status attribute.
            is_server_error = (
                last_status_code in (500, 502, 503, 504)
                or "500" in error_str or "internal server error" in error_str
            )
            if is_server_error and is_local and is_swap_in_progress:
                if is_swap_in_progress():
                    swap_wait = 0
                    while is_swap_in_progress() and swap_wait < 10:
                        await asyncio.sleep(2)
                        swap_wait += 2
                    break  # model changed, let dispatcher re-select

            # Generic retry with small backoff
            if attempt < 1:
                await asyncio.sleep(2)
                continue
            break

    # All retries exhausted
    category = classify_error(last_error or "Unknown", last_status_code)
    partial = None
    if partial_content_ref and partial_content_ref[0]:
        partial = partial_content_ref[0]

    return CallError(
        category=category,
        message=last_error or "Unknown error",
        retryable=category in ("timeout", "rate_limited", "loading",
                               "server_error", "gpu_busy", "connection_error",
                               "context_overflow", "json_unsupported"),
        partial_content=partial,
        headers=last_headers,
        status_code=last_status_code,
    )
