"""Tests for CallResult and CallError dataclasses."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from talking_layer.types import CallResult, CallError


def test_call_result_construction():
    r = CallResult(content="hello", tool_calls=None, thinking=None,
                   usage={"prompt_tokens": 10, "completion_tokens": 5},
                   cost=0.0, latency=1.5, model="openai/qwen3-30b",
                   model_name="qwen3-30b", is_local=True, provider="local", task="executor")
    assert r.content == "hello"
    assert r.is_local is True
    assert r.cost == 0.0


def test_call_result_with_tool_calls():
    r = CallResult(content="", tool_calls=[{"id": "1", "name": "search", "arguments": {"q": "test"}}],
                   thinking=None, usage={"prompt_tokens": 100, "completion_tokens": 50},
                   cost=0.001, latency=2.3, model="groq/llama-8b", model_name="llama-8b",
                   is_local=False, provider="groq", task="executor")
    assert r.tool_calls[0]["name"] == "search"
    assert r.is_local is False


def test_call_error_construction():
    e = CallError(category="timeout", message="Timeout on qwen3-30b", retryable=True)
    assert e.category == "timeout"
    assert e.retryable is True
    assert e.partial_content is None


def test_call_error_with_partial_content():
    e = CallError(category="timeout", message="Timeout on qwen3-30b", retryable=True,
                  partial_content="The analysis shows...")
    assert e.partial_content == "The analysis shows..."
