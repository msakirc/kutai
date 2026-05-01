"""Result and error types for HaLLederiz Kadir."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CallResult:
    """Successful LLM call result."""
    content: str
    tool_calls: list[dict] | None
    thinking: str | None
    usage: dict
    cost: float
    latency: float
    model: str
    model_name: str
    is_local: bool
    provider: str
    task: str


@dataclass
class CallError:
    """Failed LLM call with classification."""
    category: str
    message: str
    retryable: bool
    partial_content: str | None = None
    # Response headers from the failing call, when available. LiteLLM exposes
    # these on its exception types (RateLimitError.response.headers etc.).
    # Captured by execute_with_retry and forwarded to KDV.record_attempt so
    # x-ratelimit-* counters stay in sync with the provider's view even on
    # 4xx/5xx responses (which still consume request quota).
    headers: dict[str, str] | None = None
    # HTTP status code from the failing response, when the exception carried
    # one. Used by the caller's mark_dead path: status 404 means the provider
    # said "no such id" — same id won't resurrect, marking dead skips it
    # for the rest of the process.
    status_code: int | None = None
