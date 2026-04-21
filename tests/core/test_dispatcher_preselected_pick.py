"""Dispatcher honours Task.preselected_pick on iteration 0, falls back to Hoca on retries."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import hallederiz_kadir
from hallederiz_kadir.types import CallResult


def _fake_model(name: str = "claude-sonnet-4-6"):
    return SimpleNamespace(
        name=name,
        litellm_name=name,
        is_local=False,
        is_loaded=False,
        is_free=False,
        provider="anthropic",
        thinking_model=False,
        has_vision=False,
        demoted=False,
        location="cloud",
    )


def _fake_pick(model, composite: float = 0.7):
    return SimpleNamespace(
        model=model,
        composite=composite,
        score=composite,
        min_time_seconds=8.0,
        estimated_load_seconds=0.0,
    )


def _fake_call_result(model_name: str = "claude-sonnet-4-6") -> CallResult:
    return CallResult(
        content="hi",
        tool_calls=None,
        thinking=None,
        usage={},
        cost=0.0,
        latency=0.1,
        model=model_name,
        model_name=model_name,
        is_local=False,
        provider="anthropic",
        task="coder",
    )


@pytest.mark.asyncio
async def test_preselected_pick_skips_first_hoca_call():
    from src.core.llm_dispatcher import CallCategory, LLMDispatcher

    model = _fake_model()
    preselected = _fake_pick(model, composite=0.7)

    with patch("fatih_hoca.select") as select_spy, \
         patch.object(hallederiz_kadir, "call",
                      new=AsyncMock(return_value=_fake_call_result())), \
         patch("src.infra.pick_log.write_pick_log_row", new=AsyncMock()):
        d = LLMDispatcher()
        await d.request(
            category=CallCategory.MAIN_WORK,
            task="coder",
            difficulty=7,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            preselected_pick=preselected,
        )
    assert select_spy.call_count == 0


@pytest.mark.asyncio
async def test_no_preselected_pick_calls_hoca_normally():
    from src.core.llm_dispatcher import CallCategory, LLMDispatcher

    model = _fake_model()
    pick = _fake_pick(model, composite=0.7)

    with patch("fatih_hoca.select", return_value=pick) as select_spy, \
         patch.object(hallederiz_kadir, "call",
                      new=AsyncMock(return_value=_fake_call_result())), \
         patch("src.infra.pick_log.write_pick_log_row", new=AsyncMock()):
        d = LLMDispatcher()
        await d.request(
            category=CallCategory.MAIN_WORK,
            task="coder",
            difficulty=7,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        )
    assert select_spy.call_count == 1
