"""Test failed_models accumulation in on_task_finished (2026-04-27 fix).

Mission 57 task 4441 hit DLQ at worker_attempts=5 with ``failed_models=[]``
because no code path was persisting the generating model to ctx. R1/R2
(model exclusion + difficulty bump at attempts >= 3) gate on
ctx.failed_models — empty list = nothing to exclude = same model picked
every retry = same garbage produced = DLQ.

Fix lives in ``general_beckman.on_task_finished``: read result.model /
result.generating_model, write to ctx.generating_model always, append
to ctx.failed_models when status=='failed'. Idempotent.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from general_beckman import on_task_finished


def _make_db(initial_ctx: dict | None = None):
    """Return (fake_get_task, fake_update_task, captured_updates)."""
    initial_ctx = initial_ctx or {}
    state = {
        "task": {
            "id": 4441,
            "agent_type": "analyst",
            "context": json.dumps(initial_ctx),
            "worker_attempts": 0,
            "mission_id": 57,
        }
    }
    captured = []

    async def fake_get_task(tid):
        return dict(state["task"])

    async def fake_update_task(tid, **kw):
        captured.append(kw)
        if "context" in kw:
            state["task"]["context"] = kw["context"]

    return fake_get_task, fake_update_task, captured, state


@pytest.mark.asyncio
async def test_completed_run_persists_generating_model():
    fake_get, fake_update, captured, state = _make_db()
    with patch("src.infra.db.get_task", fake_get), \
         patch("src.infra.db.update_task", fake_update), \
         patch("src.workflows.engine.hooks.is_workflow_step", return_value=False), \
         patch("general_beckman.result_router.route_result", return_value=[]), \
         patch("general_beckman.rewrite.rewrite_actions", return_value=[]), \
         patch("general_beckman.apply.apply_actions", AsyncMock()):
        await on_task_finished(4441, {
            "status": "completed",
            "result": "ok",
            "model": "Qwen3.5-9B",
        })

    # Should have persisted generating_model
    persisted_ctx = json.loads(state["task"]["context"])
    assert persisted_ctx.get("generating_model") == "Qwen3.5-9B"
    # No failure → failed_models still untouched
    assert persisted_ctx.get("failed_models", []) == []


@pytest.mark.asyncio
async def test_failed_run_appends_to_failed_models():
    fake_get, fake_update, captured, state = _make_db()
    with patch("src.infra.db.get_task", fake_get), \
         patch("src.infra.db.update_task", fake_update), \
         patch("src.workflows.engine.hooks.is_workflow_step", return_value=False), \
         patch("general_beckman.result_router.route_result", return_value=[]), \
         patch("general_beckman.rewrite.rewrite_actions", return_value=[]), \
         patch("general_beckman.apply.apply_actions", AsyncMock()):
        await on_task_finished(4441, {
            "status": "failed",
            "error": "schema validation",
            "model": "Qwen3.5-9B",
        })

    persisted_ctx = json.loads(state["task"]["context"])
    assert persisted_ctx.get("generating_model") == "Qwen3.5-9B"
    assert persisted_ctx.get("failed_models") == ["Qwen3.5-9B"]


@pytest.mark.asyncio
async def test_repeated_failure_idempotent():
    """Same model failing twice — should appear only once in list."""
    fake_get, fake_update, captured, state = _make_db(
        initial_ctx={"failed_models": ["Qwen3.5-9B"]}
    )
    with patch("src.infra.db.get_task", fake_get), \
         patch("src.infra.db.update_task", fake_update), \
         patch("src.workflows.engine.hooks.is_workflow_step", return_value=False), \
         patch("general_beckman.result_router.route_result", return_value=[]), \
         patch("general_beckman.rewrite.rewrite_actions", return_value=[]), \
         patch("general_beckman.apply.apply_actions", AsyncMock()):
        await on_task_finished(4441, {
            "status": "failed",
            "error": "schema again",
            "model": "Qwen3.5-9B",
        })

    persisted_ctx = json.loads(state["task"]["context"])
    assert persisted_ctx.get("failed_models") == ["Qwen3.5-9B"]


@pytest.mark.asyncio
async def test_different_model_appended():
    fake_get, fake_update, captured, state = _make_db(
        initial_ctx={"failed_models": ["Qwen3.5-9B"]}
    )
    with patch("src.infra.db.get_task", fake_get), \
         patch("src.infra.db.update_task", fake_update), \
         patch("src.workflows.engine.hooks.is_workflow_step", return_value=False), \
         patch("general_beckman.result_router.route_result", return_value=[]), \
         patch("general_beckman.rewrite.rewrite_actions", return_value=[]), \
         patch("general_beckman.apply.apply_actions", AsyncMock()):
        await on_task_finished(4441, {
            "status": "failed",
            "error": "x",
            "model": "Apriel-1.6-15b",
        })

    persisted_ctx = json.loads(state["task"]["context"])
    assert persisted_ctx.get("failed_models") == ["Qwen3.5-9B", "Apriel-1.6-15b"]


@pytest.mark.asyncio
async def test_needs_clarification_does_not_track_as_failure():
    """needs_clarification means agent worked, just needs human input —
    not a model failure. Should NOT append to failed_models."""
    fake_get, fake_update, captured, state = _make_db()
    with patch("src.infra.db.get_task", fake_get), \
         patch("src.infra.db.update_task", fake_update), \
         patch("src.workflows.engine.hooks.is_workflow_step", return_value=False), \
         patch("general_beckman.result_router.route_result", return_value=[]), \
         patch("general_beckman.rewrite.rewrite_actions", return_value=[]), \
         patch("general_beckman.apply.apply_actions", AsyncMock()):
        await on_task_finished(4441, {
            "status": "needs_clarification",
            "question": "what?",
            "model": "Qwen3.5-9B",
        })

    persisted_ctx = json.loads(state["task"]["context"])
    # generating_model still tracked
    assert persisted_ctx.get("generating_model") == "Qwen3.5-9B"
    # but failed_models stays empty
    assert persisted_ctx.get("failed_models", []) == []


@pytest.mark.asyncio
async def test_no_model_in_result_skips_persist():
    """Result without model field shouldn't crash or persist garbage."""
    fake_get, fake_update, captured, state = _make_db()
    with patch("src.infra.db.get_task", fake_get), \
         patch("src.infra.db.update_task", fake_update), \
         patch("src.workflows.engine.hooks.is_workflow_step", return_value=False), \
         patch("general_beckman.result_router.route_result", return_value=[]), \
         patch("general_beckman.rewrite.rewrite_actions", return_value=[]), \
         patch("general_beckman.apply.apply_actions", AsyncMock()):
        await on_task_finished(4441, {
            "status": "failed",
            "error": "x",
            # No model / generating_model
        })

    persisted_ctx = json.loads(state["task"]["context"])
    assert "generating_model" not in persisted_ctx
    assert persisted_ctx.get("failed_models", []) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
