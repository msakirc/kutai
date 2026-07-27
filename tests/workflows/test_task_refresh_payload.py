"""refresh_workflow_step_payload must live-reload a MECHANICAL step's payload.

Bug (m90 567458): a mechanical step's `payload` freezes in task.context at
expansion time. Mechanical tasks dispatch via mr_roboto.run(payload) and
BYPASS coulson's _refresh_workflow_step_config (the LLM-worker refresh that
live-reloads checks/instruction). So a workflow-JSON payload edit — e.g.
wiring `html_paths` onto 5.30c annotate_html_oids — never reaches an
already-expanded row, and the executor keeps starving on the stale payload
('annotate_html_oids: must supply html_text or html_paths').

This refresh mirrors refresh_workflow_agent_type: dispatch-time, reads the
live step, substitutes {mission_id}, syncs payload onto the frozen row.
"""
import asyncio
import json
from unittest.mock import patch


def _run(task, ctx, live_step):
    from src.workflows.engine.task_refresh import refresh_workflow_step_payload

    captured = {}

    class FakeCursor:
        async def fetchone(self):
            return (json.dumps({"workflow_name": "i2p_v3"}),)

        async def close(self):
            pass

    class FakeDB:
        async def execute(self, *a, **k):
            return FakeCursor()

    async def fake_get_db():
        return FakeDB()

    class FakeWF:
        def get_step(self, _sid):
            return live_step

    def fake_load_workflow(_name):
        return FakeWF()

    async def fake_update_task(task_id, **kwargs):
        captured["task_id"] = task_id
        captured.update(kwargs)

    with patch("src.infra.db.get_db", fake_get_db), \
         patch("general_beckman.update_task", fake_update_task), \
         patch("src.workflows.engine.loader.load_workflow", fake_load_workflow):
        result = asyncio.run(refresh_workflow_step_payload(task, ctx))

    return result, captured


_STALE = {"action": "annotate_html_oids"}


def test_mechanical_payload_synced_from_live_json():
    # Row expanded before html_paths was wired → payload lacks it.
    task = {"id": 567458, "mission_id": 90}
    ctx = {"is_workflow_step": True, "workflow_step_id": "5.30c",
           "payload": dict(_STALE)}
    live_step = {
        "agent": "mechanical",
        "executor": "annotate_html_oids",
        "payload": {"action": "annotate_html_oids",
                    "html_paths": ["mission_{mission_id}/.web/"]},
    }
    result, captured = _run(task, ctx, live_step)
    # {mission_id} substituted, html_paths now present in-memory.
    assert ctx["payload"]["html_paths"] == ["mission_90/.web/"]
    assert result["html_paths"] == ["mission_90/.web/"]
    # Persisted so the re-pended row dispatches with the live payload.
    assert json.loads(captured["context"])["payload"]["html_paths"] == \
        ["mission_90/.web/"]


def test_payload_unchanged_no_spurious_write():
    task = {"id": 1, "mission_id": 90}
    live_payload = {"action": "annotate_html_oids",
                    "html_paths": ["mission_90/.web/"]}
    ctx = {"is_workflow_step": True, "workflow_step_id": "5.30c",
           "payload": dict(live_payload)}
    live_step = {"agent": "mechanical", "executor": "annotate_html_oids",
                 "payload": {"action": "annotate_html_oids",
                             "html_paths": ["mission_{mission_id}/.web/"]}}
    _result, captured = _run(task, ctx, live_step)
    assert captured.get("context") is None  # no diff → no write


def test_non_workflow_step_left_alone():
    # An ad-hoc / runtime-injected mechanical task (no workflow_step_id) must
    # not be clobbered by a JSON lookup.
    task = {"id": 2, "mission_id": 90}
    ctx = {"payload": {"action": "regen_bundle", "axis": "tone"}}
    live_step = {"agent": "mechanical", "payload": {"action": "other"}}
    result, captured = _run(task, ctx, live_step)
    assert ctx["payload"] == {"action": "regen_bundle", "axis": "tone"}
    assert captured.get("context") is None
