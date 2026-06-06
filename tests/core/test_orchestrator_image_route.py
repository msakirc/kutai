"""Tests for image-lane routing in Orchestrator._dispatch (Task 12).

A task whose context.image_call.raw_dispatch is True must reach husam.run via
the SAME raw path that context.llm_call.raw_dispatch uses — there is no
dispatcher.dispatch(). husam was made image-aware in Task 11; this proves the
orchestrator hands image tasks to it.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

import husam
from src.core.orchestrator import Orchestrator


def _make_orch() -> Orchestrator:
    """Build an Orchestrator without starting any background loops."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.telegram = None
    orch.shutdown_event = None
    orch._shutting_down = False
    orch._running_futures = []
    orch.running = False
    return orch


@pytest.mark.asyncio
async def test_image_call_routes_to_husam(monkeypatch):
    """_dispatch must detect image_call.raw_dispatch and route the task through
    husam.run (the SAME raw path llm_call.raw_dispatch uses)."""
    orch = _make_orch()

    seen = {}

    async def _fake_husam_run(spec):
        seen["spec"] = spec
        return {
            "path": "/tmp/x.png",
            "provider": "pollinations",
            "cost": 0.0,
            "latency": 0.1,
            "seed_used": 1,
        }

    monkeypatch.setattr(husam, "run", _fake_husam_run)

    finished = {}

    async def _fake_finish(task_id, result=None):
        finished["task_id"] = task_id
        finished["result"] = result

    monkeypatch.setattr("general_beckman.on_task_finished", _fake_finish)

    task = {
        "id": 1,
        "kind": "image",
        "context": {"image_call": {"raw_dispatch": True, "prompt": "a dog"}},
        "preselected_pick": object(),
    }

    with patch(
        "src.core.orchestrator.inject_chain_context",
        new_callable=AsyncMock,
        return_value=task,
    ), patch(
        "src.core.orchestrator.release_task_locks", new_callable=AsyncMock
    ), patch(
        "src.workflows.engine.hooks.should_skip_workflow_step",
        new_callable=AsyncMock,
        return_value=(False, ""),
    ), patch(
        "src.core.orchestrator.get_agent",
        return_value=type("P", (), {"enable_self_reflection": False})(),
    ), patch(
        "src.core.metrics_push.push_metrics", new_callable=AsyncMock
    ):
        await orch._dispatch(task)

    # ── Non-negotiable: husam.run received the image context ──
    assert "spec" in seen, "husam.run was never invoked for image_call task"
    assert seen["spec"]["context"]["image_call"]["prompt"] == "a dog"

    # on_task_finished fired and the husam result round-tripped into it.
    assert finished.get("task_id") == 1
    raw = (finished["result"] or {}).get("result")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    assert payload["path"] == "/tmp/x.png"


@pytest.mark.asyncio
async def test_image_failure_surfaces_model_for_failed_models(monkeypatch):
    """On a ModelCallFailed from husam, the orchestrator's failure result MUST
    carry the failed model name (from mcf.call_id) so on_task_finished can append
    it to context.failed_models. Without this, retry-provider-exclusion is a
    production no-op. This drives the REAL except-ModelCallFailed path."""
    from src.core.router import ModelCallFailed

    orch = _make_orch()

    async def _failing_husam_run(spec):
        raise ModelCallFailed(call_id="pollinations/flux",
                              last_error="quality_failure:blank",
                              error_category="availability")

    monkeypatch.setattr(husam, "run", _failing_husam_run)

    finished = {}

    async def _fake_finish(task_id, result=None):
        finished["task_id"] = task_id
        finished["result"] = result

    monkeypatch.setattr("general_beckman.on_task_finished", _fake_finish)

    task = {
        "id": 7,
        "kind": "image",
        "context": {"image_call": {"raw_dispatch": True, "prompt": "a cat"}},
        "preselected_pick": object(),
    }

    with patch(
        "src.core.orchestrator.inject_chain_context",
        new_callable=AsyncMock,
        return_value=task,
    ), patch(
        "src.core.orchestrator.release_task_locks", new_callable=AsyncMock
    ), patch(
        "src.workflows.engine.hooks.should_skip_workflow_step",
        new_callable=AsyncMock,
        return_value=(False, ""),
    ), patch(
        "src.core.orchestrator.get_agent",
        return_value=type("P", (), {"enable_self_reflection": False})(),
    ), patch(
        "src.core.metrics_push.push_metrics", new_callable=AsyncMock
    ):
        await orch._dispatch(task)

    res = finished.get("result") or {}
    assert res.get("status") == "failed"
    # The fix: failed model name surfaced from mcf.call_id → drives failed_models.
    assert res.get("model") == "pollinations/flux", (
        "orchestrator dropped the failed model name; failed_models propagation "
        "is a no-op on the real path"
    )
