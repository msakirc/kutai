"""Tests for ``BaseAgent._maybe_constrained_emit`` post-execution pass.

Phase B of constrained decoding: after the ReAct loop or single-shot
returns, run a fix-up dispatch with response_format:json_schema so
required fields are guaranteed. Failure modes the gate must handle:

* Non-completed results pass through (failures, clarifies, subtasks).
* Non-workflow tasks pass through (no schema to constrain).
* Markdown / string schemas pass through (unconstrainable).
* Dispatch error -> keep draft, never regress.
* Empty / non-JSON emit -> keep draft, never regress.
* Constrainable + completion -> dispatch fired with json_schema RF.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.base import BaseAgent


class _FakeAgent(BaseAgent):
    name = "fake"


def _wf_task(schema: dict, step_id: str = "test_step") -> dict:
    return {
        "id": 999,
        "context": json.dumps({
            "is_workflow_step": True,
            "workflow_step_id": step_id,
            "artifact_schema": schema,
        }),
    }


_OBJECT_SCHEMA = {"db_client": {"type": "object", "required_fields": ["a", "b"]}}
_ARRAY_SCHEMA = {"items": {"type": "array", "min_items": 1, "item_fields": ["x"]}}
_MARKDOWN_SCHEMA = {"doc": {"type": "markdown"}}


@pytest.mark.asyncio
async def test_passes_through_non_completed_result():
    agent = _FakeAgent()
    result = {"status": "failed", "result": "boom", "error": "x"}
    out = await agent._maybe_constrained_emit(_wf_task(_OBJECT_SCHEMA), result)
    assert out is result


@pytest.mark.asyncio
async def test_passes_through_empty_draft():
    agent = _FakeAgent()
    result = {"status": "completed", "result": "   "}
    out = await agent._maybe_constrained_emit(_wf_task(_OBJECT_SCHEMA), result)
    assert out is result


@pytest.mark.asyncio
async def test_passes_through_non_workflow_task():
    agent = _FakeAgent()
    task = {"id": 1, "context": json.dumps({"artifact_schema": _OBJECT_SCHEMA})}
    result = {"status": "completed", "result": '{"a":1,"b":2}'}
    out = await agent._maybe_constrained_emit(task, result)
    assert out is result


@pytest.mark.asyncio
async def test_passes_through_markdown_schema():
    agent = _FakeAgent()
    result = {"status": "completed", "result": "# title\n\nbody"}
    out = await agent._maybe_constrained_emit(_wf_task(_MARKDOWN_SCHEMA), result)
    assert out is result


@pytest.mark.asyncio
async def test_passes_through_when_no_schema():
    agent = _FakeAgent()
    task = {
        "id": 1,
        "context": json.dumps({"is_workflow_step": True}),
    }
    result = {"status": "completed", "result": "x"}
    out = await agent._maybe_constrained_emit(task, result)
    assert out is result


@pytest.mark.asyncio
async def test_dispatch_failure_keeps_draft():
    agent = _FakeAgent()
    result = {"status": "completed", "result": "draft text"}
    fake_dispatcher = AsyncMock()
    fake_dispatcher.request.side_effect = RuntimeError("boom")
    with patch(
        "src.core.llm_dispatcher.get_dispatcher",
        return_value=fake_dispatcher,
    ):
        out = await agent._maybe_constrained_emit(
            _wf_task(_OBJECT_SCHEMA), result,
        )
    assert out["result"] == "draft text"
    assert "constrained_emit_applied" not in out


@pytest.mark.asyncio
async def test_empty_emit_keeps_draft():
    agent = _FakeAgent()
    result = {"status": "completed", "result": "draft text"}
    fake_dispatcher = AsyncMock()
    fake_dispatcher.request = AsyncMock(return_value={"content": "", "model": "m"})
    with patch(
        "src.core.llm_dispatcher.get_dispatcher",
        return_value=fake_dispatcher,
    ):
        out = await agent._maybe_constrained_emit(
            _wf_task(_OBJECT_SCHEMA), result,
        )
    assert out["result"] == "draft text"
    assert "constrained_emit_applied" not in out


@pytest.mark.asyncio
async def test_non_json_emit_keeps_draft():
    agent = _FakeAgent()
    result = {"status": "completed", "result": "draft text"}
    fake_dispatcher = AsyncMock()
    fake_dispatcher.request = AsyncMock(
        return_value={"content": "this is not json", "model": "m"},
    )
    with patch(
        "src.core.llm_dispatcher.get_dispatcher",
        return_value=fake_dispatcher,
    ):
        out = await agent._maybe_constrained_emit(
            _wf_task(_OBJECT_SCHEMA), result,
        )
    assert out["result"] == "draft text"


@pytest.mark.asyncio
async def test_valid_emit_replaces_draft():
    agent = _FakeAgent()
    result = {"status": "completed", "result": "draft text", "model": "draft_model"}
    valid_emit = '{"a": 1, "b": 2}'
    fake_dispatcher = AsyncMock()
    fake_dispatcher.request = AsyncMock(
        return_value={"content": valid_emit, "model": "emit_model"},
    )
    with patch(
        "src.core.llm_dispatcher.get_dispatcher",
        return_value=fake_dispatcher,
    ):
        out = await agent._maybe_constrained_emit(
            _wf_task(_OBJECT_SCHEMA), result,
        )
    assert out["result"] == valid_emit
    assert out["constrained_emit_applied"] is True
    # Original draft model should still be in metadata so we know who
    # produced the actual work — not overwritten by the emit model.
    assert out["model"] == "draft_model"


@pytest.mark.asyncio
async def test_dispatch_called_with_json_schema_response_format():
    agent = _FakeAgent()
    result = {"status": "completed", "result": "draft"}
    valid = '{"a":1,"b":2}'
    fake_dispatcher = AsyncMock()
    fake_dispatcher.request = AsyncMock(
        return_value={"content": valid, "model": "m"},
    )
    with patch(
        "src.core.llm_dispatcher.get_dispatcher",
        return_value=fake_dispatcher,
    ):
        await agent._maybe_constrained_emit(_wf_task(_OBJECT_SCHEMA), result)

    call_kwargs = fake_dispatcher.request.call_args.kwargs
    rf = call_kwargs.get("response_format")
    assert rf is not None
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["type"] == "object"
    assert sorted(rf["json_schema"]["schema"]["required"]) == ["a", "b"]


@pytest.mark.asyncio
async def test_skip_emit_when_draft_already_parses_with_all_keys():
    """Mission 57 task 4441 5.4b: draft was 30751 chars with full
    empty_states + error_states arrays. Constrained_emit then
    compressed to 12826 chars and lost tail content. Skip when draft
    already parses with all required keys present."""
    agent = _FakeAgent()
    schema = {
        "form_specs": {"type": "object", "required_fields": ["forms"]},
        "empty_error_state_specs": {
            "type": "object",
            "required_fields": ["empty_states", "error_states"],
        },
    }
    task = {
        "id": 4441,
        "context": json.dumps({
            "is_workflow_step": True,
            "workflow_step_id": "5.4b",
            "artifact_schema": schema,
        }),
    }
    draft = json.dumps({
        "form_specs": {"forms": [{"id": "x"}]},
        "empty_error_state_specs": {
            "empty_states": [{"id": "e1"}],
            "error_states": [{"id": "err1"}],
        },
    })
    result = {"status": "completed", "result": draft}
    fake_dispatcher = AsyncMock()
    fake_dispatcher.request = AsyncMock(
        return_value={"content": '{"replaced":"x"}', "model": "m"},
    )
    with patch(
        "src.core.llm_dispatcher.get_dispatcher",
        return_value=fake_dispatcher,
    ):
        out = await agent._maybe_constrained_emit(task, result)
    # Dispatcher MUST NOT have been called — skipped because draft is clean.
    assert fake_dispatcher.request.call_count == 0
    assert out["result"] == draft
    assert "constrained_emit_applied" not in out


@pytest.mark.asyncio
async def test_emit_still_fires_when_draft_missing_keys():
    """If the draft is JSON but missing a required artifact key, the
    emit pass should still fire to reshape."""
    agent = _FakeAgent()
    schema = {
        "form_specs": {"type": "object", "required_fields": ["forms"]},
        "empty_error_state_specs": {
            "type": "object", "required_fields": ["empty_states"],
        },
    }
    task = {
        "id": 1,
        "context": json.dumps({
            "is_workflow_step": True,
            "workflow_step_id": "5.4b",
            "artifact_schema": schema,
        }),
    }
    draft = json.dumps({"form_specs": {"forms": []}})  # missing empty_error_state_specs
    result = {"status": "completed", "result": draft}
    valid = '{"form_specs":{"forms":[]},"empty_error_state_specs":{"empty_states":[{"id":"e"}]}}'
    fake_dispatcher = AsyncMock()
    fake_dispatcher.request = AsyncMock(
        return_value={"content": valid, "model": "m"},
    )
    with patch(
        "src.core.llm_dispatcher.get_dispatcher",
        return_value=fake_dispatcher,
    ):
        out = await agent._maybe_constrained_emit(task, result)
    # Emit fired because key was missing.
    assert fake_dispatcher.request.call_count == 1
    assert out.get("constrained_emit_applied") is True


@pytest.mark.asyncio
async def test_emit_still_fires_when_draft_is_prose():
    """Non-JSON draft -> emit must fire to constrain shape."""
    agent = _FakeAgent()
    schema = {"x": {"type": "object", "required_fields": ["a"]}}
    task = {
        "id": 2,
        "context": json.dumps({
            "is_workflow_step": True,
            "workflow_step_id": "test",
            "artifact_schema": schema,
        }),
    }
    result = {"status": "completed", "result": "Just prose, not JSON."}
    fake_dispatcher = AsyncMock()
    fake_dispatcher.request = AsyncMock(
        return_value={"content": '{"a":"ok"}', "model": "m"},
    )
    with patch(
        "src.core.llm_dispatcher.get_dispatcher",
        return_value=fake_dispatcher,
    ):
        await agent._maybe_constrained_emit(task, result)
    assert fake_dispatcher.request.call_count == 1


@pytest.mark.asyncio
async def test_array_schema_translates_correctly_at_call():
    agent = _FakeAgent()
    result = {"status": "completed", "result": "draft"}
    valid = '[{"x":"v"}]'
    fake_dispatcher = AsyncMock()
    fake_dispatcher.request = AsyncMock(
        return_value={"content": valid, "model": "m"},
    )
    with patch(
        "src.core.llm_dispatcher.get_dispatcher",
        return_value=fake_dispatcher,
    ):
        await agent._maybe_constrained_emit(_wf_task(_ARRAY_SCHEMA), result)
    rf = fake_dispatcher.request.call_args.kwargs["response_format"]
    assert rf["json_schema"]["schema"]["type"] == "array"
    assert rf["json_schema"]["schema"]["minItems"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
