"""Tests for error classification and retry logic."""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import AsyncMock
from hallederiz_kadir.retry import classify_error, execute_with_retry
from hallederiz_kadir.types import CallError


def test_classify_by_status_code_outranks_text():
    """HTTP status code drives classification when present. Provider
    text varies (Gemini 'is not found for api version' vs OpenRouter
    'No endpoints found' vs OpenAI 'model_not_found') but 404 is 404.
    User feedback 2026-05-01: 'we should also respect to http codes,
    text search is not enough'."""
    # 404 → model_not_found regardless of body text
    assert classify_error("any garbage body", status_code=404) == "model_not_found"
    assert classify_error("", status_code=404) == "model_not_found"
    # 401/403 → auth_failure even if text doesn't match auth keywords
    assert classify_error("internal error", status_code=401) == "auth_failure"
    assert classify_error("forbidden", status_code=403) == "auth_failure"
    # 429 → rate_limited
    assert classify_error("provider message", status_code=429) == "rate_limited"
    # 5xx → server_error
    assert classify_error("upstream broken", status_code=500) == "server_error"
    assert classify_error("bad gateway", status_code=502) == "server_error"
    assert classify_error("svc unavailable", status_code=503) == "server_error"
    assert classify_error("gateway timeout", status_code=504) == "server_error"
    # 408 → timeout
    assert classify_error("client request timeout", status_code=408) == "timeout"
    # Daily-exhausted refinement: 429 + body marker → daily_exhausted
    # (dispatcher handles these differently — adds to failures vs backoff).
    assert classify_error("Daily limit exhausted for X", status_code=429) == "daily_exhausted"
    # Status code missing → falls back to text matching
    assert classify_error("rate limit exceeded", status_code=None) == "rate_limited"
    # 2xx never reached classify_error (success path), but defensive: returns unknown
    assert classify_error("ok", status_code=200) == "unknown"
    # Unmapped 4xx falls back to text matching
    assert classify_error("rate limit exceeded", status_code=418) == "rate_limited"


def test_classify_groq_context_overflow():
    """Groq returns 'Please reduce the length of the messages...' for
    context overflow on small-context models (allam-2-7b @ 4K). Was
    classifying as 'unknown' → retried generically → same model picked
    again. Production triage 2026-05-01: ~4 tasks failed via this path."""
    body = (
        'litellm.BadRequestError: GroqException - {"error":{"message":'
        '"Please reduce the length of the messages or completion. '
        'Currently, the model has 4096 tokens of context and the '
        'request is 6243 tokens.","type":"invalid_request_error"}}'
    )
    assert classify_error(body) == "context_overflow"
    # Other phrasings
    assert classify_error("maximum context length is 8192") == "context_overflow"
    assert classify_error("Request too large for model") == "context_overflow"


def test_classify_groq_json_unsupported():
    """Groq returns 'This model does not support JSON output' when
    response_format=json_object hits a non-supporting model (compound,
    compound-mini). Caller's selection should already filter via
    needs_json_mode, but if it slips through the classifier must mark
    it retryable so dispatcher reselects to a json-capable peer."""
    body = (
        'litellm.BadRequestError: GroqException - {"error":{"message":'
        '"This model does not support JSON output","type":'
        '"invalid_request_error","param":"response_format"}}'
    )
    assert classify_error(body) == "json_unsupported"


def test_classify_timeout():
    assert classify_error("Timeout on qwen3-30b") == "timeout"
    assert classify_error("Connection timed out") == "timeout"

def test_classify_rate_limit():
    assert classify_error("rate limit exceeded") == "rate_limited"
    assert classify_error("429 Too Many Requests") == "rate_limited"

def test_classify_auth():
    assert classify_error("Invalid API key") == "auth_failure"
    # "billing quota exceeded" used to land on auth_failure because
    # "billing" matched first. As of 2026-05-02, "quota exceeded" is
    # an explicit rate_limited marker — quota exhaustion is transient,
    # not a credentials problem. Mass-mark-dead must NOT fire.
    assert classify_error("billing quota exceeded") == "rate_limited"
    # Pure billing-without-quota-marker still classifies as auth_failure.
    assert classify_error("billing details required") == "auth_failure"


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
    slug) classifies as model_not_found — caller marks dead, retry skips.

    Note: status code 404 is the structured signal (covered by
    test_classify_by_status_code_outranks_text). Text fallback only matches
    unambiguous provider tags — loose "404"+"not found" substring matching
    was removed because error bodies echo request payloads (Groq
    json_validate_failed contains failed_generation that can include
    arbitrary user code with HTTP 404 / 'not found' literals)."""
    # Status 404 → model_not_found regardless of text
    assert classify_error("provider rejected", status_code=404) == "model_not_found"
    # Provider-specific tags in text (no status code)
    assert classify_error(
        "models/gemini-2.5-flash-preview-05-20 is not found for API version v1beta"
    ) == "model_not_found"
    assert classify_error("model_not_found: foo") == "model_not_found"
    # OpenRouter retired-model signal: id in OR catalog but no provider
    # currently serves it. Caller marks dead so future selections skip it.
    assert classify_error(
        'OpenrouterException - {"error":{"message":"No endpoints found for x"}}'
    ) == "model_not_found"
    # Generic 404 in body without status code stays unknown —
    # could be a transient routing/proxy 404 unrelated to the model id.
    assert classify_error("upstream returned 404") == "unknown"
    # Crucial regression: error body echoing user code that contains
    # "404" + "not found" literals must NOT misclassify (production
    # 2026-05-01: Groq json_validate_failed payload included generated
    # FastAPI HTTPException 404 strings → model wrongly marked dead).
    groq_json_fail = (
        'litellm.BadRequestError: GroqException - {"error":{"message":'
        '"Failed to generate JSON. Please adjust your prompt. See '
        "'failed_generation' for more details.\","
        '"code":"json_validate_failed",'
        '"failed_generation":"raise HTTPException(404, \\"item not found\\")"}}'
    )
    assert classify_error(groq_json_fail) == "unknown"
    # Same body with status_code=400 (the actual HTTP status) → unknown
    assert classify_error(groq_json_fail, status_code=400) == "unknown"

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


def test_classify_gemini_quota_resource_exhausted_is_rate_limited():
    """Production 2026-05-02 task #7059: Gemini free-tier
    RESOURCE_EXHAUSTED comes back as litellm.BadRequestError (NOT 429
    in status_code). Body contains 'check your plan and billing
    details' AND 'Quota exceeded' AND 'RESOURCE_EXHAUSTED'. The
    'billing' substring used to short-circuit to auth_failure before
    the rate_limited branch saw the body markers, mass-marking all
    16 gemini ids dead.

    Quota markers must outrank billing/credit text in classification."""
    body = (
        'litellm.BadRequestError: Vertex_ai_betaException BadRequestError - '
        'b\'{"error":{"code":429,"message":"You exceeded your current quota, '
        'please check your plan and billing details. For more information '
        'on this error, head to: https://ai.google.dev/gemini-api/docs/'
        'rate-limits.\\n* Quota exceeded for metric: '
        'generativelanguage.googleapis.com/generate_content_free_tier_'
        'requests, limit: 20, model: gemini-2.5-flash","status":'
        '"RESOURCE_EXHAUSTED"}}\''
    )
    # Even without a 429 status_code, body markers must trigger rate_limited.
    assert classify_error(body) == "rate_limited"
    assert classify_error(body, status_code=400) == "rate_limited"


def test_classify_quota_exceeded_phrasings():
    """Gemini emits multiple quota phrasings depending on which limit
    tripped. All must classify as rate_limited, not auth_failure."""
    assert classify_error("Quota exceeded for metric X") == "rate_limited"
    assert classify_error("exceeded your current quota") == "rate_limited"
    assert classify_error("RESOURCE_EXHAUSTED") == "rate_limited"
    # auth-shaped errors WITHOUT quota markers still classify as auth.
    assert classify_error("invalid api key, check billing") == "auth_failure"
    assert classify_error("unauthorized") == "auth_failure"
