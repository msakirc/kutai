"""Integration test: dispatcher → HaLLederiz Kadir → mocked litellm."""
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


def _make_scored(model):
    scored = MagicMock()
    scored.model = model
    scored.score = 8.5
    scored.capability_score = 7.2
    scored.reasons = ["test"]
    return scored


def _make_reqs():
    from dataclasses import dataclass, field
    @dataclass
    class FakeReqs:
        task: str = "executor"
        primary_capability: str = "general"
        difficulty: int = 5
        estimated_output_tokens: int = 500
        estimated_input_tokens: int = 1000
        needs_thinking: bool = False
        needs_vision: bool = False
        needs_function_calling: bool = False
        local_only: bool = False
        prefer_speed: bool = False
        min_score: float = 0.0
        agent_type: str = "executor"
        effective_task: str = "executor"
        model_override: str | None = None
        priority: int = 5
        exclude_models: list = field(default_factory=list)
        effective_context_needed: int = 4096
    return FakeReqs()


@patch("hallederiz_kadir.caller.litellm")
@patch("hallederiz_kadir.caller._kdv_pre_call", return_value=(True, 0.0, False))
@patch("hallederiz_kadir.caller._kdv_post_call")
@patch("hallederiz_kadir.caller._record_metrics")
@patch("hallederiz_kadir.caller._record_audit", new_callable=AsyncMock)
def test_full_pipeline_cloud(mock_audit, mock_metrics, mock_kdv_post,
                              mock_kdv_pre, mock_litellm):
    """Full pipeline: dispatcher → HaLLederiz Kadir → cloud model."""
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    mock_litellm.completion_cost = MagicMock(return_value=0.001)

    model = _make_model_info(is_local=False)
    scored = _make_scored(model)

    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    dispatcher = LLMDispatcher()
    reqs = _make_reqs()

    with patch.object(dispatcher, "_select_candidates", return_value=[scored]), \
         patch.object(dispatcher, "_prepare_messages", return_value=[{"role": "user", "content": "test"}]):
        result = run_async(dispatcher.request(
            category=CallCategory.MAIN_WORK,
            reqs=reqs,
            messages=[{"role": "user", "content": "What is 6*7?"}],
        ))

    assert result["content"] == "The answer is 42"
    assert result["capability_score"] == 7.2


@patch("hallederiz_kadir.caller.litellm")
@patch("hallederiz_kadir.caller._kdv_pre_call", return_value=(True, 0.0, False))
@patch("hallederiz_kadir.caller._kdv_post_call")
@patch("hallederiz_kadir.caller._record_metrics")
@patch("hallederiz_kadir.caller._record_audit", new_callable=AsyncMock)
def test_full_pipeline_fallback_on_error(mock_audit, mock_metrics, mock_kdv_post,
                                          mock_kdv_pre, mock_litellm):
    """First candidate fails, second succeeds."""
    from hallederiz_kadir.types import CallError

    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    mock_litellm.completion_cost = MagicMock(return_value=0.001)

    model1 = _make_model_info(name="model-a", is_local=False)
    model2 = _make_model_info(name="model-b", is_local=False)
    scored1 = _make_scored(model1)
    scored2 = _make_scored(model2)

    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    dispatcher = LLMDispatcher()
    reqs = _make_reqs()

    # First call fails (timeout), second succeeds
    call_count = [0]
    original_call = None

    # Grab the real call before patching so we can delegate to it on success
    from hallederiz_kadir.caller import call as _real_call

    async def mock_talker_call(model, messages, tools, timeout, task, needs_thinking, estimated_output_tokens=1000):
        call_count[0] += 1
        if call_count[0] == 1:
            return CallError(category="timeout", message="Timeout on model-a", retryable=True)
        # Delegate to the real caller (litellm already mocked via decorators above)
        return await _real_call(model=model, messages=messages, tools=tools,
                                timeout=timeout, task=task, needs_thinking=needs_thinking,
                                estimated_output_tokens=estimated_output_tokens)

    with patch.object(dispatcher, "_select_candidates", return_value=[scored1, scored2]), \
         patch.object(dispatcher, "_prepare_messages", return_value=[{"role": "user", "content": "test"}]), \
         patch("hallederiz_kadir.call", side_effect=mock_talker_call):
        result = run_async(dispatcher.request(
            category=CallCategory.MAIN_WORK,
            reqs=reqs,
            messages=[{"role": "user", "content": "test"}],
        ))

    assert result["content"] == "The answer is 42"
    assert call_count[0] == 2  # first failed, second succeeded
