import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_dispatcher_records_swap_after_swap(monkeypatch):
    """After dispatcher triggers a local swap via ensure_local_model,
    nerd_herd.record_swap must be called with the loaded model name."""
    from src.core.llm_dispatcher import LLMDispatcher, CallCategory
    from hallederiz_kadir import CallResult

    # Patch fatih_hoca.select to return a local model that triggers swap.
    # NOTE: MagicMock's `name` kwarg sets the mock display name, not .name attr.
    fake_model = MagicMock(is_local=True, location="gguf",
                           provider="local", thinking_model=False)
    fake_model.name = "qwen3-8b"
    fake_pick = MagicMock(model=fake_model, min_time_seconds=30)
    fake_result = CallResult(
        content="ok", tool_calls=None, thinking=None,
        usage={}, cost=0.0, latency=0.1,
        model="qwen3-8b", model_name="qwen3-8b",
        is_local=True, provider="local", task="coder",
    )

    with patch("fatih_hoca.select", return_value=fake_pick), \
         patch("nerd_herd.record_swap") as mock_record, \
         patch("hallederiz_kadir.call", new=AsyncMock(return_value=fake_result)):
        d = LLMDispatcher()
        # Arrange: ensure_local_model returns True + reports swap_happened=True
        d._ensure_local_model = AsyncMock(return_value=(True, True))
        await d.request(
            category=CallCategory.MAIN_WORK,
            task="coder", difficulty=5, messages=[], tools=None,
        )
        mock_record.assert_called_once_with("qwen3-8b")
