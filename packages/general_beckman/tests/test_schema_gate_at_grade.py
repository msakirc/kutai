"""Fix #1 — deterministic artifact-schema gate at the grade boundary.

Before spending an LLM grade, ``_enqueue_posthook_llm_child("grade", ...)``
checks the (post-rewrite) source result against the step's ``artifact_schema``.
A shape/completeness failure is a mechanical fact: route it straight through
the existing grade-FAIL retry/escalation path with the validator's precise
message — and DO NOT spawn the LLM grade. Shape-valid artifacts proceed to the
semantic grader as before.

Regression anchors: #289735 (single object where a JSON array of >=5 is
required) and #289737 (field completeness).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest


_ARRAY_SCHEMA = {
    "user_stories": {
        "type": "array",
        "min_items": 5,
        "item_fields": ["story_id", "epic", "title", "story",
                        "acceptance_criteria", "priority"],
    }
}


def _story(i):
    return {
        "story_id": f"US-00{i}", "epic": "core", "title": f"t{i}",
        "story": "As a user, I want X, so that Y",
        "acceptance_criteria": "Given/When/Then", "priority": "High",
    }


@pytest.mark.asyncio
async def test_grade_schema_fail_routes_to_grade_fail_verdict_no_llm(monkeypatch):
    import general_beckman.apply as apply_mod

    captured: list = []

    async def fake_apply(child_task, verdict):
        captured.append(verdict)

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", fake_apply)

    draft = json.dumps(_story(1))  # ONE object — schema demands an array of >=5
    source = {
        "id": 42, "mission_id": 7, "result": draft,
        "title": "user stories", "description": "write user stories",
    }
    source_ctx = {"artifact_schema": _ARRAY_SCHEMA}

    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=901)) as enq:
        spawned = await apply_mod._enqueue_posthook_llm_child(
            "grade", source, source_ctx,
        )

    # No LLM grade child spawned; a grade-FAIL verdict applied directly.
    assert spawned is False
    enq.assert_not_awaited()
    assert len(captured) == 1
    v = captured[0]
    assert v.kind == "grade"
    assert v.passed is False
    # The validator's precise reason rides where _grader_verdict_text reads it.
    assert v.raw.get("error")
    assert "list items" in v.raw["error"] or "array" in v.raw["error"].lower()


@pytest.mark.asyncio
async def test_grade_schema_pass_proceeds_to_llm_grade(monkeypatch):
    import general_beckman.apply as apply_mod

    async def fake_apply(child_task, verdict):  # pragma: no cover - must not fire
        raise AssertionError("schema-valid artifact must not short-circuit grade")

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", fake_apply)

    draft = json.dumps([_story(i) for i in range(1, 6)])  # valid array of 5
    source = {
        "id": 43, "mission_id": 7, "result": draft,
        "title": "user stories", "description": "write user stories",
    }
    source_ctx = {"artifact_schema": _ARRAY_SCHEMA}

    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=902)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", source, source_ctx)

    # Shape-valid → the LLM grade child IS enqueued (semantic judgment proceeds).
    enq.assert_awaited_once()


@pytest.mark.asyncio
async def test_grade_no_schema_proceeds_to_llm_grade(monkeypatch):
    """No artifact_schema on the step → gate is vacuous, grade proceeds."""
    import general_beckman.apply as apply_mod

    source = {
        "id": 44, "mission_id": 7, "result": "some prose answer",
        "title": "t", "description": "d",
    }
    source_ctx = {}  # no artifact_schema

    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=903)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", source, source_ctx)

    enq.assert_awaited_once()
