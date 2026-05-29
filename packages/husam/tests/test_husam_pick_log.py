"""husam.run writes model_pick_log rows post-execute (success + failure).

Migrated from tests/core/test_dispatcher_pick_log.py (SP3b Task 2): the
select → execute → map flow moved into husam.run. The pick_log write itself
still fires inside ``LLMDispatcher.execute`` (a surviving dumb-pipe helper);
these tests confirm husam → execute still produces the same rows.
"""
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


def _fake_pick(model, composite: float = 0.65,
               top_summary: str = "claude-sonnet-4-6=8.4, gpt-4o=7.2"):
    return SimpleNamespace(
        model=model,
        composite=composite,
        score=composite,
        top_summary=top_summary,
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


def _task(task="coder", difficulty=7, tools=None):
    return {
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "main_work",
            "task": task,
            "agent_type": "",
            "difficulty": difficulty,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
            "failures": [],
        }},
        "kind": "main_work",
        "preselected_pick": None,
    }


@pytest.mark.asyncio
async def test_husam_writes_pick_log_on_success():
    import husam

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
        result = await husam.run(_task())

    assert result["content"] == "hi"
    assert len(writes) == 1
    assert writes[0]["picked_model"] == "claude-sonnet-4-6"
    assert writes[0]["success"] is True
    assert writes[0]["task_name"] == "coder"
    assert writes[0]["category"] == "main_work"
    assert writes[0]["picked_score"] == pytest.approx(0.65)
    assert writes[0]["snapshot_summary"] == "claude-sonnet-4-6=8.4, gpt-4o=7.2"


@pytest.mark.asyncio
async def test_husam_writes_pick_log_on_failure():
    """Non-retryable error path must still write a pick_log row with success=False."""
    import husam
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
        with pytest.raises(ModelCallFailed):
            await husam.run(_task())

    assert len(writes) == 1
    assert writes[0]["success"] is False
    assert writes[0]["picked_model"] == "claude-sonnet-4-6"
    assert writes[0]["error_category"] == "auth"


@pytest.mark.asyncio
async def test_husam_writes_pick_log_on_raw_exception():
    """hallederiz_kadir raising raw exception must still produce a pick_log row."""
    import husam

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
        with pytest.raises(RuntimeError, match="boom"):
            await husam.run(_task())

    assert len(writes) == 1
    assert writes[0]["success"] is False
    assert writes[0]["error_category"] == "raw_exception"
    assert writes[0]["picked_model"] == "claude-sonnet-4-6"
