"""Tests for error classification and retry logic."""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import AsyncMock
from hallederiz_kadir.retry import classify_error, execute_with_retry
from hallederiz_kadir.types import CallError


def test_classify_timeout():
    assert classify_error("Timeout on qwen3-30b") == "timeout"
    assert classify_error("Connection timed out") == "timeout"

def test_classify_rate_limit():
    assert classify_error("rate limit exceeded") == "rate_limited"
    assert classify_error("429 Too Many Requests") == "rate_limited"

def test_classify_auth():
    assert classify_error("Invalid API key") == "auth_failure"
    assert classify_error("billing quota exceeded") == "auth_failure"


def test_classify_openrouter_key_limit():
    """OpenRouter 'Key limit exceeded' (account-level credit cap, 403)
    must classify as auth_failure — not retryable, triggers provider-
    wide mark_dead in caller.py error path. Pre-fix this fell through
    to 'unknown' and dispatcher kept burning recursion attempts."""
    assert classify_error(
        "OpenrouterException - Key limit exceeded (total limit). "
        "Manage it using https://openrouter.ai/settings/keys"
    ) == "auth_failure"
    assert classify_error("insufficient_quota: please add credits") == "auth_failure"
    assert classify_error("credit balance is too low") == "auth_failure"


def test_classify_model_not_found():
    """404 NOT_FOUND from a provider (e.g. Gemini retiring a *-preview-MM-DD
    slug) classifies as model_not_found — caller marks dead, retry skips."""
    assert classify_error(
        "litellm.NotFoundError: GeminiException - 404 NOT_FOUND. "
        "models/gemini-2.5-flash-preview-05-20 is not found"
    ) == "model_not_found"
    assert classify_error(
        "models/gemini-2.5-flash-preview-05-20 is not found for API version v1beta"
    ) == "model_not_found"
    assert classify_error("model_not_found: foo") == "model_not_found"
    # Generic 404 without not-found keyword stays unknown — could be a
    # transient routing/proxy 404 unrelated to the model id.
    assert classify_error("upstream returned 404") == "unknown"

def test_classify_gpu_busy():
    assert classify_error("GPU queue timeout for qwen3-30b") == "gpu_busy"

def test_classify_daily_exhausted():
    assert classify_error("Daily limit exhausted for groq/llama") == "daily_exhausted"

def test_classify_loading():
    assert classify_error("loading model qwen3-30b") == "loading"

def test_classify_circuit_breaker():
    assert classify_error("circuit breaker active for model") == "circuit_breaker"
    assert classify_error("Failed to load local model xyz") == "circuit_breaker"

def test_classify_no_model():
    assert classify_error("No models available") == "no_model"

def test_classify_connection():
    assert classify_error("Connection refused") == "connection_error"

def test_classify_server_error():
    assert classify_error("500 Internal Server Error") == "server_error"

def test_classify_unknown():
    assert classify_error("something weird happened") == "unknown"


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def test_retry_success_first_attempt():
    call_fn = AsyncMock(return_value={"content": "hello"})
    result = run_async(execute_with_retry(
        call_fn=call_fn, max_retries=3, timeout=10.0,
        is_local=False, model_name="test",
    ))
    assert result == {"content": "hello"}
    assert call_fn.call_count == 1

def test_retry_timeout_returns_call_error():
    call_fn = AsyncMock(side_effect=asyncio.TimeoutError)
    result = run_async(execute_with_retry(
        call_fn=call_fn, max_retries=2, timeout=0.1,
        is_local=False, model_name="test",
    ))
    assert isinstance(result, CallError)
    assert result.category == "timeout"
    assert result.retryable is True

def test_retry_auth_error_no_retry():
    call_fn = AsyncMock(side_effect=Exception("Invalid API key"))
    result = run_async(execute_with_retry(
        call_fn=call_fn, max_retries=3, timeout=10.0,
        is_local=False, model_name="test",
    ))
    assert isinstance(result, CallError)
    assert result.category == "auth_failure"
    assert call_fn.call_count == 1  # no retry for auth errors

def test_retry_preserves_partial_content():
    call_fn = AsyncMock(side_effect=asyncio.TimeoutError)
    partial = ["some partial output"]
    result = run_async(execute_with_retry(
        call_fn=call_fn, max_retries=1, timeout=0.1,
        is_local=True, model_name="test",
        partial_content_ref=partial,
    ))
    assert isinstance(result, CallError)
    assert result.partial_content == "some partial output"
