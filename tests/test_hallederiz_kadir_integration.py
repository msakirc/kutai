"""Integration test: dispatcher -> HaLLederiz Kadir -> mocked litellm.

Targets ``LLMDispatcher.execute()`` — one attempt against a pick, no
selection, no retry (the legacy ``request()`` shim and its candidate-
fallback loop were deleted in SP5; fallback/retry now lives in the
selector feedback + HaLLederiz Kadir).
"""
import asyncio, sys, os
from unittest.mock import AsyncMock, MagicMock, patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_litellm_response(content="The answer is 42"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.reasoning_content = None
    msg.thinking = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 50
    resp.usage.completion_tokens = 10
    resp._hidden_params = {}
    return resp


def _make_model_info(name="test-model", is_local=False):
    m = MagicMock()
    m.name = name
    m.litellm_name = f"openai/{name}"
    m.is_local = is_local
    m.location = "local" if is_local else "cloud"
    m.provider = "llama_cpp" if is_local else "test_provider"
    m.thinking_model = False
    m.has_vision = False
    m.supports_function_calling = True
    m.supports_json_mode = False
    m.api_base = "http://localhost:8080" if is_local else None
    m.max_tokens = 4096
    m.sampling_overrides = None
    m.is_free = True
    m.tokens_per_second = 0.0
    m.is_loaded = True
    m.capabilities = {}
    m.context_length = 32000
    return m


def _make_pick(model):
    pick = MagicMock()
    pick.model = model
    pick.score = 8.5
    pick.need_ctx = 0
    pick.estimated_load_seconds = 0.0
    return pick


@patch("hallederiz_kadir.caller.litellm")
@patch("hallederiz_kadir.caller._kdv_pre_call",
       return_value=(True, 0.0, False, "", None))
@patch("hallederiz_kadir.caller._kdv_post_call")
@patch("hallederiz_kadir.caller._record_metrics")
@patch("hallederiz_kadir.caller._record_audit", new_callable=AsyncMock)
def test_full_pipeline_cloud(mock_audit, mock_metrics, mock_kdv_post,
                              mock_kdv_pre, mock_litellm):
    """Full pipeline: dispatcher.execute(pick) -> HaLLederiz Kadir -> cloud model."""
    import hallederiz_kadir

    # Cloud calls stream since 2026-05-01; mock the stream accumulator with a
    # canned final response (same pattern as packages/hallederiz_kadir tests).
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    mock_litellm.completion_cost = MagicMock(return_value=0.001)

    model = _make_model_info(is_local=False)
    pick = _make_pick(model)

    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    dispatcher = LLMDispatcher()

    with patch.object(dispatcher, "_record_pick", new_callable=AsyncMock) as mock_pick_log, \
         patch("hallederiz_kadir.caller._stream_with_accumulator",
               new_callable=AsyncMock, return_value=_make_litellm_response()):
        result = run_async(dispatcher.execute(
            pick=pick,
            messages=[{"role": "user", "content": "What is 6*7?"}],
            category=CallCategory.MAIN_WORK,
            task="executor",
            agent_type="executor",
            difficulty=5,
            tools=None,
            needs_thinking=False,
            min_context=4096,
            response_format=None,
            task_obj=None,
            iteration_n=0,
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        ))

    assert isinstance(result, hallederiz_kadir.CallResult)
    assert result.content == "The answer is 42"
    # Success path records the pick exactly once.
    assert mock_pick_log.await_count == 1
    assert mock_pick_log.await_args.kwargs["success"] is True


@patch("hallederiz_kadir.caller.litellm")
@patch("hallederiz_kadir.caller._kdv_pre_call",
       return_value=(True, 0.0, False, "", None))
@patch("hallederiz_kadir.caller._kdv_post_call")
@patch("hallederiz_kadir.caller._record_metrics")
@patch("hallederiz_kadir.caller._record_audit", new_callable=AsyncMock)
def test_call_error_surfaces_to_caller(mock_audit, mock_metrics, mock_kdv_post,
                                        mock_kdv_pre, mock_litellm):
    """execute() does NOT retry: a HaLLederiz CallError is returned as-is
    (re-selection is the caller's job — selector feedback, not dispatcher)."""
    from hallederiz_kadir.types import CallError

    model = _make_model_info(name="model-a", is_local=False)
    pick = _make_pick(model)

    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    dispatcher = LLMDispatcher()
    err = CallError(category="timeout", message="Timeout on model-a", retryable=True)

    with patch.object(dispatcher, "_record_pick", new_callable=AsyncMock) as mock_pick_log, \
         patch("hallederiz_kadir.call", new=AsyncMock(return_value=err)):
        result = run_async(dispatcher.execute(
            pick=pick,
            messages=[{"role": "user", "content": "test"}],
            category=CallCategory.MAIN_WORK,
            task="executor",
            agent_type="executor",
            difficulty=5,
            tools=None,
            needs_thinking=False,
            min_context=4096,
            response_format=None,
            task_obj=None,
            iteration_n=0,
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        ))

    assert isinstance(result, CallError)
    assert result.category == "timeout"
    # Failure path records the pick with the error category.
    assert mock_pick_log.await_count == 1
    assert mock_pick_log.await_args.kwargs["success"] is False
    assert mock_pick_log.await_args.kwargs["error_category"] == "timeout"
