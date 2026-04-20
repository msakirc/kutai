"""Unit tests for workflow_engine.advance error paths."""
import pytest

from workflow_engine import advance, AdvanceResult
from src.infra import db as _db_mod


async def _fresh_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "advance.db")
    monkeypatch.setattr(_db_mod, "DB_PATH", test_db)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    await _db_mod.init_db()


@pytest.mark.asyncio
async def test_completed_task_id_not_found_returns_failed(tmp_path, monkeypatch):
    """When the completed_task_id doesn't exist, advance reports failed."""
    await _fresh_db(tmp_path, monkeypatch)
    result = await advance(mission_id=99, completed_task_id=99999,
                           previous_result={"status": "completed"})
    assert isinstance(result, AdvanceResult)
    assert result.status == "failed"
    assert "99999" in result.error


@pytest.mark.asyncio
async def test_non_workflow_step_is_noop(tmp_path, monkeypatch):
    """A completed task that isn't a workflow step is a clean no-op."""
    await _fresh_db(tmp_path, monkeypatch)
    tid = await _db_mod.add_task(
        title="t", description="", agent_type="coder", mission_id=1,
        context={},  # no workflow_step / is_workflow_step marker
    )
    result = await advance(mission_id=1, completed_task_id=tid,
                           previous_result={"status": "completed"})
    assert result.status == "completed"
    assert result.next_subtasks == []
    assert result.artifacts == {}


@pytest.mark.asyncio
async def test_post_hook_needs_clarification_propagates(tmp_path, monkeypatch):
    """If post_execute_workflow_step flips the result to needs_clarification,
    advance surfaces that status (question becomes the error body)."""
    await _fresh_db(tmp_path, monkeypatch)
    tid = await _db_mod.add_task(
        title="t", description="", agent_type="coder", mission_id=1,
        context={"is_workflow_step": True},
    )

    async def fake_post_hook(task, result):
        result["status"] = "needs_clarification"
        result["question"] = "Which branch?"

    async def fake_extract(*a, **k):
        return {}

    monkeypatch.setattr("src.workflows.engine.hooks.post_execute_workflow_step",
                        fake_post_hook)
    monkeypatch.setattr("src.workflows.engine.pipeline_artifacts.extract_pipeline_artifacts",
                        fake_extract)

    result = await advance(mission_id=1, completed_task_id=tid,
                           previous_result={"status": "completed"})
    assert result.status == "needs_clarification"
    assert "Which branch?" in result.error


@pytest.mark.asyncio
async def test_post_hook_raises_returns_failed(tmp_path, monkeypatch):
    """If post_execute_workflow_step raises, advance catches and reports failed."""
    await _fresh_db(tmp_path, monkeypatch)
    tid = await _db_mod.add_task(
        title="t", description="", agent_type="coder", mission_id=1,
        context={"is_workflow_step": True},
    )

    async def exploding_post_hook(task, result):
        raise RuntimeError("hook exploded")

    async def fake_extract(*a, **k):
        return {}

    monkeypatch.setattr("src.workflows.engine.hooks.post_execute_workflow_step",
                        exploding_post_hook)
    monkeypatch.setattr("src.workflows.engine.pipeline_artifacts.extract_pipeline_artifacts",
                        fake_extract)

    result = await advance(mission_id=1, completed_task_id=tid,
                           previous_result={"status": "completed"})
    assert result.status == "failed"
    assert "hook exploded" in result.error
