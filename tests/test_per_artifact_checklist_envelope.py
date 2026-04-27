"""Test per-artifact checklist correctly handles envelope-wrapped
previous output (2026-04-27 fix).

Mission 57 task 4441 burned 5 retries because the per-artifact checklist
in ``BaseAgent._build_context`` falsely marked every required field as
missing ``[ ]``. Cause: ``_prev_output`` was stored as the agent's raw
envelope ``{"action":"final_answer","result":"<artifact-json>"}``;
``json.loads`` on it gave ``{action, result}`` keys, the parser walked
that wrong dict looking for ``form_specs`` and never found it.

Fix: ``_unwrap_envelope`` runs before parsing for the checklist, peeling
the envelope so the parser sees the artifact JSON directly. Also done at
storage time in ``apply.py`` so future retries store the bare form.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.base import BaseAgent


class _Agent(BaseAgent):
    name = "test"


_INNER = {
    "form_specs": {"forms": [{"id": "x"}]},
    "empty_error_state_specs": {
        "empty_states": [{"id": "e1"}],
        "error_states": [{"id": "err1"}],
    },
}
_SCHEMA = {
    "form_specs": {"type": "object", "required_fields": ["forms"]},
    "empty_error_state_specs": {
        "type": "object",
        "required_fields": ["empty_states", "error_states"],
    },
}


def _task(prev_output: str) -> dict:
    return {
        "id": 4441,
        "title": "5.4b",
        "description": "do",
        "worker_attempts": 3,
        "context": json.dumps({
            "is_workflow_step": True,
            "workflow_step_id": "5.4b",
            "artifact_schema": _SCHEMA,
            "_schema_error": "Grader rejected: stylistic",
            "_prev_output": prev_output,
        }),
    }


@pytest.mark.asyncio
async def test_envelope_wrapped_prev_unwraps_for_checklist():
    """The mission-57-task-4441 case."""
    envelope = json.dumps({
        "action": "final_answer",
        "result": json.dumps(_INNER),
    })
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(_task(envelope))

    # All populated fields should be marked [x], not [ ]
    pi = ctx.find("Per-artifact checklist")
    assert pi >= 0, "checklist block missing"
    block = ctx[pi:pi + 800]
    assert "[x] forms" in block
    assert "[x] empty_states" in block
    assert "[x] error_states" in block
    assert "[ ] forms" not in block
    assert "[ ] empty_states" not in block


@pytest.mark.asyncio
async def test_bare_prev_still_works():
    """Bare JSON (no envelope) works as before."""
    bare = json.dumps(_INNER)
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(_task(bare))

    pi = ctx.find("Per-artifact checklist")
    block = ctx[pi:pi + 800]
    assert "[x] forms" in block
    assert "[x] empty_states" in block
    assert "[x] error_states" in block


@pytest.mark.asyncio
async def test_missing_fields_still_flagged():
    """When the previous output legitimately misses fields, checklist
    still marks them [ ]."""
    partial = {
        "form_specs": {"forms": [{"id": "x"}]},
        # empty_error_state_specs MISSING
    }
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(_task(json.dumps(partial)))

    pi = ctx.find("Per-artifact checklist")
    block = ctx[pi:pi + 800]
    assert "[x] forms" in block
    # empty_error_state_specs not present in prev → both children [ ]
    assert "[ ] empty_states" in block
    assert "[ ] error_states" in block


@pytest.mark.asyncio
async def test_non_json_prev_no_crash():
    """Prose-shaped _prev_output shouldn't crash the checklist."""
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(_task("just some prose"))

    # Falls through gracefully — checklist still rendered with all [ ]
    pi = ctx.find("Per-artifact checklist")
    assert pi >= 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
