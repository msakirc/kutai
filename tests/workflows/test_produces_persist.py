"""intake #73 — the engine must persist a schema'd agent step's output to its
DECLARED `produces` path (subdir + extension), not `<name>.md` at the mission
root. Schema'd steps have write_file auto-stripped, so the engine is the only
thing that can land the file; previously it wrote `intake_todo_draft.md` at the
root while the step declared `.intake/intake_todo_draft.json` → grounding DLQ.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from unittest.mock import AsyncMock, patch


def _run(tmp_path, produces, output_obj):
    import src.tools.workspace as ws
    from src.workflows.engine import hooks

    mid = 999
    ctx = {
        "is_workflow_step": True,
        "workflow_step_id": "0.0a.draft",
        "mission_id": mid,
        "output_artifacts": ["intake_todo_draft"],
        "produces": produces,
        "artifact_schema": {"intake_todo_draft": {"type": "json", "required_keys": ["items"]}},
    }
    task = {"id": 1, "mission_id": mid, "context": json.dumps(ctx)}
    result = {"result": json.dumps(output_obj)}

    with patch.object(ws, "WORKSPACE_DIR", str(tmp_path)), \
         patch.object(hooks, "get_artifact_store", return_value=AsyncMock()), \
         patch.object(hooks, "_check_phase_completion", new=AsyncMock()):
        asyncio.run(hooks.post_execute_workflow_step(task, result))
    return mid


def test_persists_to_declared_json_produces_path(tmp_path):
    obj = {"_schema_version": "1",
           "items": [{"n": i, "category": "Audience", "question": f"q{i}"} for i in range(1, 15)]}
    mid = _run(tmp_path, [f"mission_999/.intake/intake_todo_draft.json"], obj)

    fp = tmp_path / f"mission_{mid}" / ".intake" / "intake_todo_draft.json"
    assert fp.is_file(), f"declared produces file not written: {fp}"
    data = json.loads(fp.read_text(encoding="utf-8"))
    assert "items" in data and len(data["items"]) == 14


def test_does_not_clobber_existing_produces_file(tmp_path):
    # An agent-written (richer) file must be preserved, not overwritten.
    d = tmp_path / "mission_999" / ".intake"
    d.mkdir(parents=True)
    (d / "intake_todo_draft.json").write_text('{"items": ["AGENT_WROTE_THIS"]}', encoding="utf-8")
    _run(tmp_path, ["mission_999/.intake/intake_todo_draft.json"],
         {"_schema_version": "1", "items": [{"n": 1}]})
    kept = (d / "intake_todo_draft.json").read_text(encoding="utf-8")
    assert "AGENT_WROTE_THIS" in kept
