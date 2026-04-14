"""Tests for response parsing — content, tool_calls, thinking, think-tags, cost."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock
from hallederiz_kadir.response import parse_response


def _make_response(content="hello", tool_calls=None, reasoning_content=None,
                   thinking=None, usage=None, finish_reason="stop"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.reasoning_content = reasoning_content
    msg.thinking = thinking
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = (usage or {}).get("prompt_tokens", 10)
    resp.usage.completion_tokens = (usage or {}).get("completion_tokens", 5)
    return resp


def test_parse_simple_content():
    resp = _make_response(content="Hello world")
    result = parse_response(resp, model_name="test", is_local=True, is_thinking=False)
    assert result["content"] == "Hello world"
    assert result["tool_calls"] is None
    assert result["thinking"] is None

def test_parse_tool_calls():
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "search"
    tc.function.arguments = '{"query": "test"}'
    resp = _make_response(content="", tool_calls=[tc])
    result = parse_response(resp, model_name="test", is_local=False, is_thinking=False)
    assert result["tool_calls"][0]["name"] == "search"
    assert result["tool_calls"][0]["arguments"] == {"query": "test"}

def test_parse_thinking_from_reasoning_content():
    resp = _make_response(content="answer", reasoning_content="let me think...")
    result = parse_response(resp, model_name="test", is_local=True, is_thinking=True)
    assert result["thinking"] == "let me think..."
    assert result["content"] == "answer"

def test_parse_thinking_from_think_tags():
    resp = _make_response(content="<think>reasoning</think>The answer is 42")
    result = parse_response(resp, model_name="test", is_local=True, is_thinking=True)
    assert result["thinking"] == "reasoning"

def test_strip_think_tags_when_not_requested():
    resp = _make_response(content="<think>internal</think>The answer is 42")
    result = parse_response(resp, model_name="test", is_local=True, is_thinking=False)
    assert "<think>" not in result["content"]
    assert "The answer is 42" in result["content"]
    assert result["thinking"] is None

def test_rescue_reasoning_content_when_content_empty():
    resp = _make_response(content="", reasoning_content="<think>actual answer here</think>")
    result = parse_response(resp, model_name="test", is_local=True, is_thinking=False)
    assert result["content"] == "actual answer here"

def test_strip_think_preserves_content_when_all_in_think():
    resp = _make_response(content="<think>The only content</think>")
    result = parse_response(resp, model_name="test", is_local=True, is_thinking=False)
    assert result["content"] == "The only content"

def test_strip_unclosed_think_tag():
    resp = _make_response(content="<think>reasoning that got cut off")
    result = parse_response(resp, model_name="test", is_local=True, is_thinking=False)
    assert "<think>" not in result["content"]

def test_parse_malformed_tool_call_arguments():
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "search"
    tc.function.arguments = "not json"
    resp = _make_response(content="", tool_calls=[tc])
    result = parse_response(resp, model_name="test", is_local=False, is_thinking=False)
    assert result["tool_calls"][0]["arguments"] == {}

def test_cost_zero_for_local():
    resp = _make_response(content="hello")
    result = parse_response(resp, model_name="test", is_local=True, is_thinking=False)
    assert result["cost"] == 0.0
