import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_dispatcher_calls_begin_end_for_cloud():
    from src.core.llm_dispatcher import LLMDispatcher, CallCategory
    import hallederiz_kadir

    fake_model = MagicMock(
        is_local=False,
        provider="anthropic",
        thinking_model=False,
        has_vision=False,
        litellm_name="anthropic/claude-sonnet-4-6",
    )
    fake_model.name = "claude-sonnet-4-6"
    fake_pick = MagicMock(
        model=fake_model,
        composite=0.7,
        estimated_load_seconds=0.0,
        min_time_seconds=60.0,
    )

    call_result = hallederiz_kadir.CallResult(
        content="hello",
        model="anthropic/claude-sonnet-4-6",
        model_name="claude-sonnet-4-6",
        cost=0.0,
        usage={},
        tool_calls=[],
        latency=0.1,
        thinking="",
        is_local=False,
        provider="anthropic",
        task="coder",
    )

    with patch("fatih_hoca.select", return_value=fake_pick), \
         patch("src.core.llm_dispatcher.kuleden_donen_var.begin_call") as mock_begin, \
         patch("src.core.llm_dispatcher.kuleden_donen_var.end_call") as mock_end, \
         patch("hallederiz_kadir.call", new=AsyncMock(return_value=call_result)), \
         patch("src.infra.pick_log.write_pick_log_row", new=AsyncMock()):
        mock_begin.return_value = MagicMock()
        d = LLMDispatcher()
        await d.request(
            category=CallCategory.MAIN_WORK,
            task="coder", difficulty=7, messages=[], tools=None,
        )
    mock_begin.assert_called_once_with("anthropic", "claude-sonnet-4-6")
    mock_end.assert_called_once()


@pytest.mark.asyncio
async def test_dispatcher_ends_call_even_on_exception():
    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    fake_model = MagicMock(
        is_local=False,
        provider="anthropic",
        thinking_model=False,
        has_vision=False,
        litellm_name="anthropic/claude-sonnet-4-6",
    )
    fake_model.name = "claude-sonnet-4-6"
    fake_pick = MagicMock(
        model=fake_model,
        composite=0.7,
        estimated_load_seconds=0.0,
        min_time_seconds=60.0,
    )

    async def raise_err(*a, **k):
        raise RuntimeError("boom")

    with patch("fatih_hoca.select", return_value=fake_pick), \
         patch("src.core.llm_dispatcher.kuleden_donen_var.begin_call") as mock_begin, \
         patch("src.core.llm_dispatcher.kuleden_donen_var.end_call") as mock_end, \
         patch("hallederiz_kadir.call", new=raise_err), \
         patch("src.infra.pick_log.write_pick_log_row", new=AsyncMock()):
        mock_begin.return_value = MagicMock()
        d = LLMDispatcher()
        with pytest.raises(Exception):
            await d.request(
                category=CallCategory.MAIN_WORK,
                task="coder", difficulty=7, messages=[], tools=None,
            )
    mock_end.assert_called()
