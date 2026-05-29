import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_dispatcher_records_swap_after_swap(monkeypatch):
    """After husam.run triggers a local swap via ensure_local_model,
    nerd_herd.record_swap must be called with the loaded model name."""
    import src.core.llm_dispatcher as mod
    mod._dispatcher = None
    d = mod.get_dispatcher()

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

    task_spec = {
        "context": {
            "llm_call": {
                "raw_dispatch": True,
                "call_category": "main_work",
                "task": "coder",
                "agent_type": "",
                "difficulty": 5,
                "messages": [],
                "tools": None,
                "failures": [],
            }
        },
        "kind": "main_work",
        "preselected_pick": None,
    }

    with patch("fatih_hoca.select", return_value=fake_pick), \
         patch("nerd_herd.record_swap") as mock_record, \
         patch("hallederiz_kadir.call", new=AsyncMock(return_value=fake_result)):
        import husam
        # Arrange: ensure_local_model returns True + reports swap_happened=True
        d._ensure_local_model = AsyncMock(return_value=(True, True))
        await husam.run(task_spec)
        mock_record.assert_called_once_with("qwen3-8b")
