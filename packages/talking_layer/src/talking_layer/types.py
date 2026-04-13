"""Result and error types for the talking layer."""
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
