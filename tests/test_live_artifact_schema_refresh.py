"""post_execute_workflow_step must validate against the LIVE workflow schema.

mission #81 #291858 (2026-06-04): step 1.14 produced a valid (decorated)
go_no_go_decision recommendation, but the workflow_advance retry re-validated
it against the producer task's FROZEN artifact_schema — captured at expander
time, before recommendation gained equals_lenient. So a /dlq retry DLQ'd a
now-valid artifact forever. _live_artifact_schema re-reads the step schema from
the live workflow JSON so schema edits reach in-flight missions on retry.
"""
import asyncio

import src.infra.db as db
from src.workflows.engine.hooks import _live_artifact_schema
from src.workflows.engine.schema_dialect import validate_value


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _FakeCursor:
    async def fetchone(self):
        return None  # no mission row → helper defaults to i2p_v3

    async def close(self):
        pass


class _FakeDB:
    async def execute(self, *a, **k):
        return _FakeCursor()


def test_live_schema_pulls_equals_lenient(monkeypatch):
    async def fake_get_db():
        return _FakeDB()

    monkeypatch.setattr(db, "get_db", fake_get_db)

    schema = _run(_live_artifact_schema(mission_id=999, step_id="1.14"))

    assert schema is not None
    rule = schema["go_no_go_decision"]["fields"]["recommendation"]
    assert rule.get("equals_lenient") is True
    # The exact artifact that DLQ'd #291858 now validates against the live schema.
    obj = {
        "scores": {"market_attractiveness": 5},
        "weighted_score": 5.2,
        "recommendation": "Conditional (needs_clarification)",
    }
    assert validate_value(schema["go_no_go_decision"], obj, "go_no_go_decision") is None


def test_live_schema_none_for_unknown_step(monkeypatch):
    async def fake_get_db():
        return _FakeDB()

    monkeypatch.setattr(db, "get_db", fake_get_db)

    # Template-expanded / unknown ids are not in wf.steps → None (caller keeps
    # the frozen prefixed snapshot rather than clobbering it).
    assert _run(_live_artifact_schema(999, "F7.5_feature_3")) is None
    assert _run(_live_artifact_schema(999, "")) is None
