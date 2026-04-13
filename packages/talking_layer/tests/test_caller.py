"""Tests for the main call() function with mocked litellm and backends."""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import AsyncMock, MagicMock, patch
from talking_layer.caller import call
from talking_layer.types import CallResult, CallError


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_model_info(is_local=True, litellm_name="openai/qwen3-30b",
                     name="qwen3-30b", thinking_model=False,
                     supports_function_calling=True, supports_json_mode=False,
                     api_base=None, max_tokens=4096, location="local",
                     provider="llama_cpp", sampling_overrides=None,
                     is_free=False, tokens_per_second=0.0):
    m = MagicMock()
    m.is_local = is_local
    m.litellm_name = litellm_name
    m.name = name
    m.thinking_model = thinking_model
    m.supports_function_calling = supports_function_calling
    m.supports_json_mode = supports_json_mode
    m.api_base = api_base or ("http://localhost:8080" if is_local else None)
    m.max_tokens = max_tokens
    m.location = location
    m.provider = provider
    m.sampling_overrides = sampling_overrides
    m.is_free = is_free
    m.tokens_per_second = tokens_per_second
    m.has_vision = False
    return m


def _make_litellm_response(content="Hello", tool_calls=None,
                           prompt_tokens=10, completion_tokens=5):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.reasoning_content = None
    msg.thinking = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp._hidden_params = {}
    return resp


@patch("talking_layer.caller._get_dallama", return_value=None)
@patch("talking_layer.caller.litellm")
def test_call_local_success(mock_litellm, mock_dallama):
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=True)
    result = run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                           tools=None, timeout=60.0, task="executor",
                           needs_thinking=False, estimated_output_tokens=500))
    assert isinstance(result, CallResult)
    assert result.content == "Hello"
    assert result.is_local is True
    assert result.cost == 0.0


@patch("talking_layer.caller.litellm")
def test_call_cloud_success(mock_litellm):
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=False, litellm_name="groq/llama-8b",
                             name="llama-8b", location="cloud", provider="groq", api_base=None)
    with patch("talking_layer.caller._kdv_pre_call", return_value=(True, 0.0, False)), \
         patch("talking_layer.caller._kdv_post_call"), \
         patch("talking_layer.caller.litellm.completion_cost", return_value=0.001):
        result = run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                               tools=None, timeout=60.0, task="executor",
                               needs_thinking=False, estimated_output_tokens=500))
    assert isinstance(result, CallResult)
    assert result.is_local is False
    assert result.provider == "groq"


@patch("talking_layer.caller.litellm")
def test_call_timeout_returns_call_error(mock_litellm):
    mock_litellm.acompletion = AsyncMock(side_effect=asyncio.TimeoutError)
    model = _make_model_info(is_local=False, litellm_name="groq/llama-8b",
                             location="cloud", provider="groq", api_base=None)
    with patch("talking_layer.caller._kdv_pre_call", return_value=(True, 0.0, False)), \
         patch("talking_layer.caller._kdv_record_failure"):
        result = run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                               tools=None, timeout=1.0, task="executor",
                               needs_thinking=False, estimated_output_tokens=500))
    assert isinstance(result, CallError)
    assert result.category == "timeout"
    assert result.retryable is True


@patch("talking_layer.caller._get_dallama", return_value=None)
@patch("talking_layer.caller.litellm")
def test_call_sets_api_key_for_local(mock_litellm, mock_dallama):
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=True)
    run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                   tools=None, timeout=60.0, task="executor",
                   needs_thinking=False, estimated_output_tokens=500))
    kwargs = mock_litellm.acompletion.call_args[1]
    assert kwargs["api_key"] == "sk-no-key"


@patch("talking_layer.caller._get_dallama", return_value=None)
@patch("talking_layer.caller.litellm")
def test_call_tools_set_tool_choice(mock_litellm, mock_dallama):
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=True, supports_function_calling=True)
    tools = [{"type": "function", "function": {"name": "search"}}]
    run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                   tools=tools, timeout=60.0, task="executor",
                   needs_thinking=False, estimated_output_tokens=500))
    kwargs = mock_litellm.acompletion.call_args[1]
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"


@patch("talking_layer.caller._get_dallama", return_value=None)
@patch("talking_layer.caller.litellm")
def test_call_json_mode_fallback(mock_litellm, mock_dallama):
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=True, supports_function_calling=False, supports_json_mode=True)
    tools = [{"type": "function", "function": {"name": "search"}}]
    run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                   tools=tools, timeout=60.0, task="executor",
                   needs_thinking=False, estimated_output_tokens=500))
    kwargs = mock_litellm.acompletion.call_args[1]
    assert "tools" not in kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
