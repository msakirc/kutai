"""Test the recency reorder of structured-output instructions in
``BaseAgent._build_context`` (handoff item Q).

The schema instruction (``## Required Output Format``) and retry hint
(``## IMPORTANT: Previous Output Was Invalid``) must appear at the END
of the user prompt — small models attend more strongly to end-of-prompt
content. Previously these blocks were emitted mid-prompt where they got
buried under deps, skills, RAG, etc.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.base import BaseAgent


class _Agent(BaseAgent):
    name = "fake"


def _wf_task_with_schema(schema: dict, _prev: str = "", retry: int = 0) -> dict:
    ctx = {
        "is_workflow_step": True,
        "artifact_schema": schema,
        "mission_id": 1,
    }
    if retry > 0:
        ctx["_schema_error"] = "Missing required fields: ['x']"
    if _prev:
        ctx["_prev_output"] = _prev
    return {
        "id": 99,
        "title": "T",
        "description": "do thing",
        "mission_id": 1,
        "worker_attempts": retry,
        "context": json.dumps(ctx),
    }


@pytest.mark.asyncio
async def test_required_output_format_at_end():
    """Schema block is the last big section (or second-last when retry
    section follows it)."""
    schema = {"my_artifact": {"type": "object", "required_fields": ["a", "b"]}}
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value="content")
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(_wf_task_with_schema(schema))

    schema_idx = ctx.find("## Required Output Format")
    task_idx = ctx.find("## Task (PRIMARY")
    assert schema_idx > 0, "Required Output Format missing"
    assert task_idx >= 0, "Task block missing"
    assert schema_idx > task_idx, "Schema block must come AFTER task block"
    # Nothing after schema except possibly a retry block
    tail = ctx[schema_idx:]
    # Allowed sections after schema: only retry-related
    forbidden_after = [
        "## Task (PRIMARY",
        "## Results from Previous Steps",
        "## Project Memory",
    ]
    for f in forbidden_after:
        assert f not in tail, f"`{f}` appears after `## Required Output Format`"


@pytest.mark.asyncio
async def test_retry_block_at_very_end():
    """When retrying, retry hint is the LAST section."""
    schema = {"my_artifact": {"type": "object", "required_fields": ["a", "b"]}}
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value="content")
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(
            _wf_task_with_schema(schema, _prev='{"a":"v"}', retry=2),
        )

    retry_idx = ctx.find("## IMPORTANT: Previous Output Was Invalid")
    schema_idx = ctx.find("## Required Output Format")
    assert retry_idx > 0, "retry hint missing"
    assert schema_idx > 0, "schema block missing"
    assert retry_idx > schema_idx, "retry hint must come AFTER schema block"
    # Retry block should be near the very end — no other major sections after
    tail_after_retry = ctx[retry_idx:]
    # Only allowed: the previous-output dump itself (part of retry block)
    assert "## Task (PRIMARY" not in tail_after_retry
    assert "## Results from Previous Steps" not in tail_after_retry


@pytest.mark.asyncio
async def test_no_schema_no_tail_blocks():
    """When task has no artifact_schema, no tail block is emitted."""
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context({
            "id": 1, "title": "t", "description": "x",
            "context": json.dumps({}),
        })
    assert "## Required Output Format" not in ctx
    assert "## IMPORTANT: Previous Output" not in ctx


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
