"""Test the missing-input-artifact NOTE in BaseAgent._build_context.

Ported from the dead-code ``hooks.pre_execute_workflow_step`` as part
of handoff item D. The NOTE prevents agents from searching the
filesystem for artifacts whose upstream phase was skipped or DLQ'd.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.base import BaseAgent


class _Agent(BaseAgent):
    name = "fake"


def _wf_task(input_artifacts: list[str], mission_id: int = 1) -> dict:
    return {
        "id": 1,
        "title": "t",
        "description": "do thing",
        "mission_id": mission_id,
        "context": json.dumps({
            "is_workflow_step": True,
            "input_artifacts": input_artifacts,
            "mission_id": mission_id,
        }),
    }


@pytest.mark.asyncio
async def test_missing_artifacts_emit_note():
    fake_store = AsyncMock()
    # All three names missing — both forms (bare and _summary).
    fake_store.retrieve = AsyncMock(return_value=None)

    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(
            _wf_task(["spec_summary", "api_design", "open_risks"]),
        )

    assert "## Missing Input Artifacts" in ctx
    assert "spec_summary" in ctx
    assert "api_design" in ctx
    assert "open_risks" in ctx
    assert "Do NOT call read_file" in ctx


@pytest.mark.asyncio
async def test_present_artifacts_no_note():
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value="content here")

    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(
            _wf_task(["spec_summary"]),
        )

    assert "## Missing Input Artifacts" not in ctx


@pytest.mark.asyncio
async def test_summary_fallback_satisfies_check():
    """When the bare name is missing but ``<name>_summary`` exists, the
    artifact is considered present."""
    fake_store = AsyncMock()

    async def _retrieve(mid, name):
        return "summary content" if name == "feature_design_summary" else None

    fake_store.retrieve = AsyncMock(side_effect=_retrieve)

    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(
            _wf_task(["feature_design"]),
        )
    assert "## Missing Input Artifacts" not in ctx


@pytest.mark.asyncio
async def test_non_workflow_task_skipped():
    """Non-workflow tasks shouldn't trigger artifact-store calls."""
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)

    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        await _Agent()._build_context({
            "id": 2, "title": "t", "description": "x",
            "context": json.dumps({"input_artifacts": ["x"]}),
        })

    fake_store.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_no_input_artifacts_skipped():
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)

    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        await _Agent()._build_context({
            "id": 3, "title": "t", "description": "x",
            "context": json.dumps({"is_workflow_step": True}),
        })

    fake_store.retrieve.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
