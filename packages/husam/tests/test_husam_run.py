"""husam.run owns select -> dispatcher.execute -> map.

Task 2 (SP3b): husam.run is the non-agentic worker the orchestrator pump
dispatches raw_dispatch specs to. It unpacks the spec, selects (or honours
the preselected pick), calls the dumb dispatcher primitive
(``LLMDispatcher.execute``), and maps the ``CallResult`` back to the legacy
response dict that ``dispatch`` used to return.

These tests patch ``husam.worker.get_dispatcher().execute`` to return a real
``hallederiz_kadir.CallResult`` so we exercise the husam mapping/selection
glue without touching DaLLaMa / hallederiz transport.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import hallederiz_kadir
from hallederiz_kadir.types import CallError, CallResult


# ─── doubles (mirroring tests/test_llm_dispatcher.py + tests/core) ────────────

def _fake_model(name: str = "claude-sonnet-4-6", is_local: bool = False):
    return SimpleNamespace(
        name=name,
        litellm_name=name,
        is_local=is_local,
        is_loaded=False,
        is_free=False,
        provider="anthropic",
        thinking_model=False,
        has_vision=False,
        demoted=False,
        location="cloud",
    )


def _fake_pick(model, composite: float = 0.65,
               top_summary: str = "claude-sonnet-4-6=8.4"):
    return SimpleNamespace(
        model=model,
        composite=composite,
        score=composite,
        top_summary=top_summary,
        min_time_seconds=8.0,
        estimated_load_seconds=0.0,
    )


def _fake_call_result(content: str = "hi",
                      model_name: str = "claude-sonnet-4-6") -> CallResult:
    return CallResult(
        content=content,
        tool_calls=None,
        thinking=None,
        usage={},
        cost=0.0,
        latency=0.1,
        model=model_name,
        model_name=model_name,
        is_local=False,
        provider="anthropic",
        task="reviewer",
    )


def _task(**llm_call_extra) -> dict:
    llm_call = {
        "raw_dispatch": True,
        "call_category": "overhead",
        "task": "reviewer",
        "agent_type": "reviewer",
        "difficulty": 3,
        "messages": [{"role": "user", "content": "hi"}],
    }
    llm_call.update(llm_call_extra)
    return {"context": {"llm_call": llm_call}}


# ─── Step 1: husam.run selects + executes + returns legacy dict ───────────────

@pytest.mark.asyncio
async def test_husam_run_selects_and_executes_returns_legacy_dict():
    import husam

    model = _fake_model("claude-sonnet-4-6")
    pick = _fake_pick(model)
    result_obj = _fake_call_result(content="reviewed-ok")

    fake_execute = AsyncMock(return_value=result_obj)

    # husam.run lazily imports get_dispatcher from src.core.llm_dispatcher,
    # so patch the source module (the lazy `from ... import get_dispatcher`
    # binds at call time and picks up the patched attribute).
    with patch("fatih_hoca.select", return_value=pick), \
         patch("src.core.llm_dispatcher.get_dispatcher") as mock_get:
        mock_get.return_value = SimpleNamespace(
            execute=fake_execute, _total_calls=0, _overhead_calls=0,
        )
        out = await husam.run(_task())

    assert out["content"] == "reviewed-ok"
    # legacy-dict shape preserved (result_to_dict keys)
    for key in ("content", "model", "model_name", "cost", "usage",
                "tool_calls", "latency", "thinking", "is_local",
                "ran_on", "provider", "task"):
        assert key in out, f"missing key: {key}"
    fake_execute.assert_awaited_once()
