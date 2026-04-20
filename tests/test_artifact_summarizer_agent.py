"""ArtifactSummarizerAgent wraps _llm_summarize and returns a posthook_verdict."""
import json
import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_artifact_summarizer_success(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db
    await init_db()

    from src.workflows.engine.artifacts import ArtifactStore
    await ArtifactStore().store(1, "big_out", "x" * 5000)

    from src.agents.artifact_summarizer import ArtifactSummarizerAgent
    agent = ArtifactSummarizerAgent()
    task = {
        "id": 700,
        "mission_id": 1,
        "agent_type": "artifact_summarizer",
        "context": json.dumps({
            "source_task_id": 100,
            "artifact_name": "big_out",
        }),
    }
    fake_summary = "short summary of content" * 5
    with patch(
        "src.workflows.engine.hooks._llm_summarize",
        AsyncMock(return_value=fake_summary),
    ):
        result = await agent.execute(task)

    assert result["status"] == "completed"
    pv = result["posthook_verdict"]
    assert pv["kind"] == "summary:big_out"
    assert pv["passed"] is True
    assert pv["raw"]["summary"] == fake_summary


@pytest.mark.asyncio
async def test_artifact_summarizer_degenerate_output_marked_failed(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db
    await init_db()
    from src.workflows.engine.artifacts import ArtifactStore
    await ArtifactStore().store(1, "big_out", "x" * 5000)

    from src.agents.artifact_summarizer import ArtifactSummarizerAgent
    agent = ArtifactSummarizerAgent()
    task = {
        "id": 701,
        "mission_id": 1,
        "agent_type": "artifact_summarizer",
        "context": json.dumps({
            "source_task_id": 100,
            "artifact_name": "big_out",
        }),
    }
    # _llm_summarize returning None indicates degenerate/empty output.
    with patch(
        "src.workflows.engine.hooks._llm_summarize",
        AsyncMock(return_value=None),
    ):
        result = await agent.execute(task)

    pv = result["posthook_verdict"]
    assert pv["passed"] is False
