"""Retry loop and error classification for LLM calls."""
from __future__ import annotations
import asyncio
import time
from typing import Callable, Awaitable
from .types import CallError


def classify_error(error: str) -> str:
    """Classify an error string into a category for retry/routing decisions."""
    e = error.lower()
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
    if any(k in e for k in ("api key", "authentication", "unauthorized", "billing")):
        return "auth_failure"
    if any(k in e for k in ("timeout", "timed out")):
        return "timeout"
    if any(k in e for k in ("connection", "network", "dns", "refused")):
        return "connection_error"
    if any(k in e for k in ("500", "internal server error")):
        return "server_error"
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

    for attempt in range(max_retries):
        try:
            result = await asyncio.wait_for(call_fn(), timeout=timeout)
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

            # Auth/billing — not retryable
            if any(kw in error_str for kw in (
                "api key", "authentication", "unauthorized",
                "billing", "credit", "quota",
            )):
                break

            # Rate limit — backoff then retry
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

            # Local 500 during swap — wait briefly
            is_server_error = "500" in error_str or "internal server error" in error_str
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
    category = classify_error(last_error or "Unknown")
    partial = None
    if partial_content_ref and partial_content_ref[0]:
        partial = partial_content_ref[0]

    return CallError(
        category=category,
        message=last_error or "Unknown error",
        retryable=category in ("timeout", "rate_limited", "loading",
                               "server_error", "gpu_busy", "connection_error"),
        partial_content=partial,
    )
