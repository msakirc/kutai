"""GraderAgent wraps grade_task and returns a posthook_verdict payload."""
import json
import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_grader_returns_posthook_verdict_shape(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task
    await init_db()
    source_id = await add_task(
        title="source", description="", agent_type="writer",
        context=json.dumps({"output_artifacts": ["out"]}),
    )

    from src.agents.grader import GraderAgent
    agent = GraderAgent()
    grader_task = {
        "id": 500,
        "agent_type": "grader",
        "context": json.dumps({
            "source_task_id": source_id,
            "generating_model": "qwen-7b",
        }),
    }

    fake_verdict = {
        "passed": True,
        "score": 0.9,
        "grader_model": "claude-sonnet",
        "cost": 0.002,
    }
    with patch("src.core.grading.grade_task", AsyncMock(return_value=fake_verdict)):
        result = await agent.execute(grader_task)

    assert result["status"] == "completed"
    assert "posthook_verdict" in result
    pv = result["posthook_verdict"]
    assert pv["kind"] == "grade"
    assert pv["source_task_id"] == source_id
    assert pv["passed"] is True
    assert pv["raw"] == fake_verdict


@pytest.mark.asyncio
async def test_grader_missing_source_returns_failed(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db
    await init_db()

    from src.agents.grader import GraderAgent
    agent = GraderAgent()
    task = {
        "id": 600,
        "agent_type": "grader",
        "context": json.dumps({"source_task_id": 99999}),
    }
    result = await agent.execute(task)
    assert result["status"] == "failed"
    assert "missing" in result["error"].lower() or "not found" in result["error"].lower()


def test_base_agent_does_not_self_transition_to_ungraded():
    """BaseAgent's ReAct loop must return status='completed', never status='ungraded'."""
    import inspect
    from src.agents import base as _base_mod

    src = inspect.getsource(_base_mod)
    # No more transition_task(..., "ungraded", ...) call.
    assert '"ungraded"' not in src, (
        "base.py still references 'ungraded' — agent must not self-transition"
    )
