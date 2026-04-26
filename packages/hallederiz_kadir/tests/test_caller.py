"""Tests for the main call() function with mocked litellm and backends."""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import AsyncMock, MagicMock, patch
from hallederiz_kadir.caller import call
from hallederiz_kadir.types import CallResult, CallError


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


def _local_patches():
    """Common patches for local model tests: no DaLLaMa, mock streaming."""
    resp = _make_litellm_response()
    return (
        resp,
        patch("hallederiz_kadir.caller._get_dallama", return_value=None),
        patch("hallederiz_kadir.caller._stream_with_accumulator",
              new_callable=AsyncMock, return_value=resp),
    )


@patch("hallederiz_kadir.caller.litellm")
def test_call_local_success(mock_litellm):
    """Local model call succeeds — returns CallResult."""
    resp, p_dallama, p_stream = _local_patches()
    with p_dallama, p_stream:
        model = _make_model_info(is_local=True)
        result = run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                               tools=None, timeout=60.0, task="executor",
                               needs_thinking=False, estimated_output_tokens=500))
    assert isinstance(result, CallResult)
    assert result.content == "Hello"
    assert result.is_local is True
    assert result.cost == 0.0


@patch("hallederiz_kadir.caller.litellm")
def test_call_cloud_success(mock_litellm):
    """Cloud model call succeeds — uses KDV pre/post."""
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=False, litellm_name="groq/llama-8b",
                             name="llama-8b", location="cloud", provider="groq", api_base=None)
    with patch("hallederiz_kadir.caller._kdv_pre_call", return_value=(True, 0.0, False)), \
         patch("hallederiz_kadir.caller._kdv_post_call"), \
         patch("hallederiz_kadir.caller.litellm.completion_cost", return_value=0.001):
        result = run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                               tools=None, timeout=60.0, task="executor",
                               needs_thinking=False, estimated_output_tokens=500))
    assert isinstance(result, CallResult)
    assert result.is_local is False
    assert result.provider == "groq"


@patch("hallederiz_kadir.caller.litellm")
def test_call_timeout_returns_call_error(mock_litellm):
    """Timeout returns CallError with category='timeout'."""
    mock_litellm.acompletion = AsyncMock(side_effect=asyncio.TimeoutError)
    model = _make_model_info(is_local=False, litellm_name="groq/llama-8b",
                             location="cloud", provider="groq", api_base=None)
    with patch("hallederiz_kadir.caller._kdv_pre_call", return_value=(True, 0.0, False)), \
         patch("hallederiz_kadir.caller._kdv_record_failure"):
        result = run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                               tools=None, timeout=1.0, task="executor",
                               needs_thinking=False, estimated_output_tokens=500))
    assert isinstance(result, CallError)
    assert result.category == "timeout"
    assert result.retryable is True


@patch("hallederiz_kadir.caller.litellm")
def test_call_sets_api_key_for_local(mock_litellm):
    """Local models get api_key='sk-no-key' in completion_kwargs.

    For local models without tools, streaming is used. The kwargs are passed
    to _stream_with_accumulator which internally calls litellm.acompletion.
    We verify the kwargs via the stream mock.
    """
    resp = _make_litellm_response()
    stream_mock = AsyncMock(return_value=resp)
    with patch("hallederiz_kadir.caller._get_dallama", return_value=None), \
         patch("hallederiz_kadir.caller._stream_with_accumulator", stream_mock):
        model = _make_model_info(is_local=True)
        run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                       tools=None, timeout=60.0, task="executor",
                       needs_thinking=False, estimated_output_tokens=500))
    # _stream_with_accumulator receives completion_kwargs as first arg
    kwargs = stream_mock.call_args[0][0]  # first positional arg
    assert kwargs["api_key"] == "sk-no-key"


@patch("hallederiz_kadir.caller.litellm")
def test_call_tools_set_tool_choice(mock_litellm):
    """When tools provided and model supports FC, tool_choice='auto'.

    Local models always stream — tool calls go through _stream_with_accumulator.
    """
    resp = _make_litellm_response()
    stream_mock = AsyncMock(return_value=resp)
    with patch("hallederiz_kadir.caller._get_dallama", return_value=None), \
         patch("hallederiz_kadir.caller._stream_with_accumulator", stream_mock):
        model = _make_model_info(is_local=True, supports_function_calling=True)
        tools = [{"type": "function", "function": {"name": "search"}}]
        run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                       tools=tools, timeout=60.0, task="executor",
                       needs_thinking=False, estimated_output_tokens=500))
    kwargs = stream_mock.call_args[0][0]  # completion_kwargs passed to stream accumulator
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"


@patch("hallederiz_kadir.caller.litellm")
def test_call_json_mode_fallback(mock_litellm):
    """When tools given but model lacks FC, falls back to json_mode.

    No FC means use_tools=None, so local models use streaming path.
    """
    resp = _make_litellm_response()
    stream_mock = AsyncMock(return_value=resp)
    with patch("hallederiz_kadir.caller._get_dallama", return_value=None), \
         patch("hallederiz_kadir.caller._stream_with_accumulator", stream_mock):
        model = _make_model_info(is_local=True, supports_function_calling=False, supports_json_mode=True)
        tools = [{"type": "function", "function": {"name": "search"}}]
        run_async(call(model=model, messages=[{"role": "user", "content": "hello"}],
                       tools=tools, timeout=60.0, task="executor",
                       needs_thinking=False, estimated_output_tokens=500))
    kwargs = stream_mock.call_args[0][0]
    assert "tools" not in kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


# ── response_format (constrained decoding) ─────────────────────────────


_RF_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "test_artifact",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["a", "b"],
            "properties": {"a": {}, "b": {}},
        },
    },
}


def _make_model_info_with_jschema(**kwargs):
    js = kwargs.pop("supports_json_schema", False)
    m = _make_model_info(**kwargs)
    m.supports_json_schema = js
    return m


@patch("hallederiz_kadir.caller.litellm")
def test_response_format_passthrough_when_supported(mock_litellm):
    """Capable model + caller-provided json_schema -> kwargs include it as-is."""
    resp, p_dallama, p_stream = _local_patches()
    stream_mock = p_stream.kwargs["new"] if "new" in p_stream.kwargs else AsyncMock(return_value=resp)
    with p_dallama, patch(
        "hallederiz_kadir.caller._stream_with_accumulator",
        new_callable=AsyncMock, return_value=resp,
    ) as sm:
        model = _make_model_info_with_jschema(
            is_local=True, supports_function_calling=False,
            supports_json_mode=True, supports_json_schema=True,
        )
        run_async(call(
            model=model, messages=[{"role": "user", "content": "hi"}],
            tools=None, timeout=60.0, task="executor",
            needs_thinking=False, estimated_output_tokens=500,
            response_format=_RF_JSON_SCHEMA,
        ))
    kwargs = sm.call_args[0][0]
    assert kwargs["response_format"] == _RF_JSON_SCHEMA


@patch("hallederiz_kadir.caller.litellm")
def test_response_format_degrades_to_json_object_when_no_schema(mock_litellm):
    """json_schema requested but model only supports json_mode -> json_object fallback."""
    resp = _make_litellm_response()
    with patch("hallederiz_kadir.caller._get_dallama", return_value=None), \
         patch(
            "hallederiz_kadir.caller._stream_with_accumulator",
            new_callable=AsyncMock, return_value=resp,
         ) as sm:
        model = _make_model_info_with_jschema(
            is_local=True, supports_function_calling=False,
            supports_json_mode=True, supports_json_schema=False,
        )
        run_async(call(
            model=model, messages=[{"role": "user", "content": "hi"}],
            tools=None, timeout=60.0, task="executor",
            needs_thinking=False, estimated_output_tokens=500,
            response_format=_RF_JSON_SCHEMA,
        ))
    kwargs = sm.call_args[0][0]
    assert kwargs["response_format"] == {"type": "json_object"}


@patch("hallederiz_kadir.caller.litellm")
def test_response_format_dropped_when_no_json_at_all(mock_litellm):
    """Model lacks both json_schema AND json_mode -> response_format dropped entirely."""
    resp = _make_litellm_response()
    mock_litellm.acompletion = AsyncMock(return_value=resp)
    with patch(
        "hallederiz_kadir.caller._kdv_pre_call",
        return_value=(True, 0.0, False),
    ):
        model = _make_model_info_with_jschema(
            is_local=False, location="cloud", provider="anthropic",
            supports_function_calling=True,
            supports_json_mode=False, supports_json_schema=False,
        )
        run_async(call(
            model=model, messages=[{"role": "user", "content": "hi"}],
            tools=None, timeout=60.0, task="executor",
            needs_thinking=False, estimated_output_tokens=500,
            response_format=_RF_JSON_SCHEMA,
        ))
    kwargs = mock_litellm.acompletion.call_args.kwargs
    assert "response_format" not in kwargs


@patch("hallederiz_kadir.caller.litellm")
def test_response_format_ignored_with_function_calling_tools(mock_litellm):
    """When tools+FC active, response_format is suppressed (mutually exclusive)."""
    resp = _make_litellm_response()
    with patch("hallederiz_kadir.caller._get_dallama", return_value=None), \
         patch(
            "hallederiz_kadir.caller._stream_with_accumulator",
            new_callable=AsyncMock, return_value=resp,
         ) as sm:
        model = _make_model_info_with_jschema(
            is_local=True, supports_function_calling=True,
            supports_json_mode=True, supports_json_schema=True,
        )
        tools = [{"type": "function", "function": {"name": "search"}}]
        run_async(call(
            model=model, messages=[{"role": "user", "content": "hi"}],
            tools=tools, timeout=60.0, task="executor",
            needs_thinking=False, estimated_output_tokens=500,
            response_format=_RF_JSON_SCHEMA,
        ))
    kwargs = sm.call_args[0][0]
    # tools branch wins; json_schema not propagated (would conflict with tool_choice)
    assert kwargs.get("tools") == tools
    assert kwargs.get("response_format") != _RF_JSON_SCHEMA
