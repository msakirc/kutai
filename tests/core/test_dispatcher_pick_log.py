"""Dispatcher writes model_pick_log rows post-iteration (success + failure)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import hallederiz_kadir
from hallederiz_kadir.types import CallError, CallResult


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


def _fake_pick(model, composite: float = 0.65):
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
async def test_dispatcher_writes_pick_log_on_success():
    from src.core.llm_dispatcher import CallCategory, LLMDispatcher

    model = _fake_model("claude-sonnet-4-6")
    pick = _fake_pick(model, composite=0.65)

    writes: list[dict] = []

    async def fake_write(**kw):
        writes.append(kw)

    with patch("fatih_hoca.select", return_value=pick), \
         patch.object(
             hallederiz_kadir, "call",
             new=AsyncMock(return_value=_fake_call_result()),
         ), \
         patch("src.infra.pick_log.write_pick_log_row", new=fake_write):
        d = LLMDispatcher()
        result = await d.request(
            category=CallCategory.MAIN_WORK,
            task="coder",
            difficulty=7,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        )

    assert result["content"] == "hi"
    assert len(writes) == 1
    assert writes[0]["picked_model"] == "claude-sonnet-4-6"
    assert writes[0]["success"] is True
    assert writes[0]["task_name"] == "coder"
    assert writes[0]["category"] == "main_work"


@pytest.mark.asyncio
async def test_dispatcher_writes_pick_log_on_failure():
    """Non-retryable error path must still write a pick_log row with success=False."""
    from src.core.llm_dispatcher import CallCategory, LLMDispatcher
    from src.core.router import ModelCallFailed

    model = _fake_model("claude-sonnet-4-6")
    pick = _fake_pick(model, composite=0.65)

    writes: list[dict] = []

    async def fake_write(**kw):
        writes.append(kw)

    err = CallError(category="auth", message="bad key", retryable=False)

    with patch("fatih_hoca.select", return_value=pick), \
         patch.object(hallederiz_kadir, "call", new=AsyncMock(return_value=err)), \
         patch("src.infra.pick_log.write_pick_log_row", new=fake_write):
        d = LLMDispatcher()
        with pytest.raises(ModelCallFailed):
            await d.request(
                category=CallCategory.MAIN_WORK,
                task="coder",
                difficulty=7,
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
            )

    assert len(writes) == 1
    assert writes[0]["success"] is False
    assert writes[0]["picked_model"] == "claude-sonnet-4-6"
    assert writes[0]["error_category"] == "auth"


@pytest.mark.asyncio
async def test_dispatcher_writes_pick_log_on_raw_exception():
    """hallederiz_kadir raising raw exception must still produce a pick_log row."""
    from src.core.llm_dispatcher import CallCategory, LLMDispatcher

    model = _fake_model("claude-sonnet-4-6")
    pick = _fake_pick(model, composite=0.65)

    writes: list[dict] = []

    async def fake_write(**kw):
        writes.append(kw)

    async def raise_boom(**_kw):
        raise RuntimeError("boom")

    with patch("fatih_hoca.select", return_value=pick), \
         patch.object(hallederiz_kadir, "call", new=raise_boom), \
         patch("src.infra.pick_log.write_pick_log_row", new=fake_write):
        d = LLMDispatcher()
        with pytest.raises(RuntimeError, match="boom"):
            await d.request(
                category=CallCategory.MAIN_WORK,
                task="coder",
                difficulty=7,
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
            )

    assert len(writes) == 1
    assert writes[0]["success"] is False
    assert writes[0]["error_category"] == "raw_exception"
    assert writes[0]["picked_model"] == "claude-sonnet-4-6"
