"""Test that clarify-action results on ``triggers_clarification`` steps
skip the schema validation gate (2026-04-27 fix).

Mission 57 task 4376 (step 0.5 human_clarification_request) burned 5
retries because:
  - Agent emitted ``{"action":"clarify","question":"..."}`` correctly.
  - Clarify text landed in ``result.question``, NOT ``result.result``.
  - ``output_value = result.get("result", "")`` → empty.
  - Empty + producer task + artifact_schema present → schema validator
    failed with "empty output for schema clarification_request".
  - Failure status set BEFORE the triggers_clarification override block
    that would have routed to needs_clarification correctly.

Fix: detect clarify-action shape (``status==needs_clarification`` OR
non-empty ``question``/``clarification`` field) on a
triggers_clarification step and skip schema validation. The override
block also extracts clarify text from any of the three fields so it
fires regardless of which one the agent populated.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.workflows.engine.hooks import _post_execute_workflow_step_impl


def _clarify_step_task(question_field: str = "question") -> tuple[dict, dict]:
    ctx = {
        "is_workflow_step": True,
        "workflow_step_id": "0.5",
        "workflow_phase": "phase_0",
        "mission_id": 999,
        "output_artifacts": ["clarification_request"],
        "triggers_clarification": True,
        "artifact_schema": {
            "clarification_request": {"type": "array", "min_items": 3}
        },
    }
    task = {
        "id": 4376,
        "agent_type": "analyst",
        "context": json.dumps(ctx),
        "mission_id": 999,
        "worker_attempts": 0,
    }
    result = {
        "status": "completed",
        "result": "",
        question_field: (
            "1. Audience? A) Families B) Roommates\n"
            "2. Platform? A) iOS B) Android\n"
            "3. Scope? A) MVP B) Full"
        ),
    }
    return task, result


@pytest.mark.asyncio
async def test_clarify_in_question_field_skips_schema():
    task, result = _clarify_step_task(question_field="question")
    fake_db = AsyncMock()
    with patch("src.infra.db.update_task", AsyncMock()), \
         patch("src.infra.db.get_db", AsyncMock(return_value=fake_db)):
        await _post_execute_workflow_step_impl(task, result)

    assert result["status"] == "needs_clarification"
    assert "Schema validation" not in (result.get("error") or "")
    assert result.get("clarification") and "Audience" in result["clarification"]


@pytest.mark.asyncio
async def test_clarify_in_clarification_field_skips_schema():
    task, result = _clarify_step_task(question_field="clarification")
    fake_db = AsyncMock()
    with patch("src.infra.db.update_task", AsyncMock()), \
         patch("src.infra.db.get_db", AsyncMock(return_value=fake_db)):
        await _post_execute_workflow_step_impl(task, result)

    assert result["status"] == "needs_clarification"
    assert "Schema validation" not in (result.get("error") or "")


@pytest.mark.asyncio
async def test_status_already_needs_clarification_skips_schema():
    """Result with status=needs_clarification but empty fields still
    skips schema (same flow, different result shape)."""
    task, _ = _clarify_step_task()
    result = {
        "status": "needs_clarification",
        "result": "",
    }
    fake_db = AsyncMock()
    with patch("src.infra.db.update_task", AsyncMock()), \
         patch("src.infra.db.get_db", AsyncMock(return_value=fake_db)):
        await _post_execute_workflow_step_impl(task, result)

    # No schema-validation error
    assert "Schema validation" not in (result.get("error") or "")


@pytest.mark.asyncio
async def test_non_clarify_step_still_validates_schema():
    """Producer step WITHOUT triggers_clarification still gets validation."""
    ctx = {
        "is_workflow_step": True,
        "workflow_step_id": "0.2",
        "mission_id": 999,
        "output_artifacts": ["data"],
        "artifact_schema": {
            "data": {"type": "object", "required_fields": ["a", "b"]}
        },
    }
    task = {
        "id": 1, "agent_type": "analyst",
        "context": json.dumps(ctx),
        "mission_id": 999, "worker_attempts": 0,
    }
    result = {"status": "completed", "result": ""}

    fake_db = AsyncMock()
    with patch("src.infra.db.update_task", AsyncMock()), \
         patch("src.infra.db.get_db", AsyncMock(return_value=fake_db)):
        await _post_execute_workflow_step_impl(task, result)

    # Empty output on producer non-clarify step → schema fail.
    assert result["status"] == "failed"
    assert "empty output" in (result.get("error") or "")


@pytest.mark.asyncio
async def test_clarify_step_with_history_validates_normally():
    """Second run after human answered — clarification_history present
    means agent should now produce real artifact, not clarify again.
    Schema validation should run normally."""
    ctx = {
        "is_workflow_step": True,
        "workflow_step_id": "0.5",
        "workflow_phase": "phase_0",
        "mission_id": 999,
        "output_artifacts": ["clarification_request"],
        "triggers_clarification": True,
        "clarification_history": [{"question": "x", "answer": "y"}],
        "artifact_schema": {
            "clarification_request": {"type": "array", "min_items": 3}
        },
    }
    task = {
        "id": 4376, "agent_type": "analyst",
        "context": json.dumps(ctx),
        "mission_id": 999, "worker_attempts": 0,
    }
    # Status NOT needs_clarification, no question field — second run path.
    result = {"status": "completed", "result": ""}
    fake_db = AsyncMock()
    with patch("src.infra.db.update_task", AsyncMock()), \
         patch("src.infra.db.get_db", AsyncMock(return_value=fake_db)):
        await _post_execute_workflow_step_impl(task, result)
    # Empty output — schema validator should fire (no clarify-action signal).
    assert result["status"] == "failed"
    assert "empty output" in (result.get("error") or "")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
