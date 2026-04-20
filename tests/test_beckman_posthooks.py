"""Post-hook pipeline tests: actions, policy, apply, migrations."""
import pytest
from general_beckman.result_router import (
    Action, RequestPostHook, PostHookVerdict,
)


def test_request_posthook_is_action():
    a = RequestPostHook(source_task_id=1, kind="grade", source_ctx={})
    assert isinstance(a, RequestPostHook)
    # Action is a Union; isinstance check works via dataclass identity.
    assert a.source_task_id == 1
    assert a.kind == "grade"


def test_posthook_verdict_is_action():
    v = PostHookVerdict(
        source_task_id=2, kind="grade", passed=True, raw={"score": 0.9},
    )
    assert v.passed is True
    assert v.raw == {"score": 0.9}


from general_beckman.posthooks import determine_posthooks


def test_mechanical_task_needs_no_posthooks():
    task = {"agent_type": "mechanical"}
    assert determine_posthooks(task, {}, {}) == []


def test_shopping_pipeline_task_needs_no_posthooks():
    task = {"agent_type": "shopping_pipeline"}
    assert determine_posthooks(task, {}, {}) == []


def test_grader_task_needs_no_posthooks():
    task = {"agent_type": "grader"}
    assert determine_posthooks(task, {}, {}) == []


def test_artifact_summarizer_task_needs_no_posthooks():
    task = {"agent_type": "artifact_summarizer"}
    assert determine_posthooks(task, {}, {}) == []


def test_llm_agent_task_needs_grade_by_default():
    task = {"agent_type": "writer"}
    assert determine_posthooks(task, {}, {}) == ["grade"]


def test_requires_grading_false_opts_out():
    task = {"agent_type": "writer"}
    ctx = {"requires_grading": False}
    assert determine_posthooks(task, ctx, {}) == []


import json
import pytest
from general_beckman.result_router import RequestPostHook
from general_beckman.apply import _apply_one


@pytest.mark.asyncio
async def test_apply_request_posthook_grade_enqueues_grader_and_parks_source(tmp_path, monkeypatch):
    # Set up a throwaway DB.
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task
    await init_db()
    source_id = await add_task(
        title="source work",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({"generating_model": "qwen-7b"}),
    )

    source_task = await get_task(source_id)
    action = RequestPostHook(
        source_task_id=source_id,
        kind="grade",
        source_ctx=json.loads(source_task["context"]),
    )
    await _apply_one(source_task, action)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "ungraded"
    ctx = json.loads(refreshed["context"])
    assert ctx["_pending_posthooks"] == ["grade"]

    # Grader task exists.
    from src.infra.db import get_db
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, agent_type, mission_id, context FROM tasks "
        "WHERE agent_type = 'grader'"
    )
    rows = list(await cursor.fetchall())
    assert len(rows) == 1
    grader_ctx = json.loads(rows[0]["context"])
    assert grader_ctx["source_task_id"] == source_id
    assert grader_ctx["generating_model"] == "qwen-7b"


@pytest.mark.asyncio
async def test_grade_verdict_pass_small_output_completes_source(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task
    await init_db()
    source_id = await add_task(
        title="source",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({
            "_pending_posthooks": ["grade"],
            "output_artifacts": ["short_out"],
        }),
    )
    await update_task(source_id, status="ungraded", result="short result (<3KB)")

    from general_beckman.result_router import PostHookVerdict
    verdict = PostHookVerdict(
        source_task_id=source_id, kind="grade", passed=True,
        raw={"score": 0.9},
    )
    from general_beckman.apply import _apply_one
    grade_task_row = {"id": 999, "mission_id": 1, "agent_type": "grader"}
    await _apply_one(grade_task_row, verdict)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "completed"
    ctx = json.loads(refreshed["context"])
    assert ctx["_pending_posthooks"] == []


@pytest.mark.asyncio
async def test_grade_pass_spawns_summary_for_large_artifact(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task, get_db
    await init_db()

    source_id = await add_task(
        title="source",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({
            "_pending_posthooks": ["grade"],
            "output_artifacts": ["big_out"],
        }),
    )
    await update_task(source_id, status="ungraded", result="..." * 2000)

    from src.workflows.engine.artifacts import ArtifactStore
    store = ArtifactStore()
    await store.store(1, "big_out", "x" * 5000)

    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_one
    verdict = PostHookVerdict(
        source_task_id=source_id, kind="grade", passed=True, raw={},
    )
    grade_row = {"id": 999, "mission_id": 1, "agent_type": "grader"}
    await _apply_one(grade_row, verdict)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "ungraded"
    ctx = json.loads(refreshed["context"])
    assert ctx["_pending_posthooks"] == ["summary:big_out"]

    db = await get_db()
    cursor = await db.execute(
        "SELECT id, context FROM tasks WHERE agent_type='artifact_summarizer'"
    )
    rows = list(await cursor.fetchall())
    assert len(rows) == 1
    sum_ctx = json.loads(rows[0]["context"])
    assert sum_ctx["source_task_id"] == source_id
    assert sum_ctx["artifact_name"] == "big_out"


@pytest.mark.asyncio
async def test_grade_fail_retries_source_excludes_generating_model(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task
    await init_db()
    source_id = await add_task(
        title="source",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({
            "_pending_posthooks": ["grade"],
            "generating_model": "qwen-7b",
        }),
    )
    await update_task(source_id, status="ungraded", worker_attempts=1)

    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_one
    verdict = PostHookVerdict(
        source_task_id=source_id, kind="grade", passed=False,
        raw="reason: hallucinated claim",
    )
    await _apply_one({"id": 999, "mission_id": 1, "agent_type": "grader"}, verdict)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "pending"
    assert refreshed["worker_attempts"] == 2
    ctx = json.loads(refreshed["context"])
    assert "qwen-7b" in ctx["grade_excluded_models"]
    assert ctx["_pending_posthooks"] == []


@pytest.mark.asyncio
async def test_summary_pass_stores_artifact_and_completes_source(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task
    await init_db()
    source_id = await add_task(
        title="source",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({"_pending_posthooks": ["summary:big_out"]}),
    )
    await update_task(source_id, status="ungraded")

    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_one
    verdict = PostHookVerdict(
        source_task_id=source_id, kind="summary:big_out",
        passed=True, raw={"summary": "condensed output", "artifact_name": "big_out"},
    )
    await _apply_one({"id": 998, "mission_id": 1, "agent_type": "artifact_summarizer"}, verdict)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "completed"
    ctx = json.loads(refreshed["context"])
    assert ctx["_pending_posthooks"] == []

    from src.workflows.engine.artifacts import ArtifactStore
    stored = await ArtifactStore().retrieve(1, "big_out_summary")
    assert stored == "condensed output"


@pytest.mark.asyncio
async def test_summary_fail_keeps_structural_and_completes_source(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task
    await init_db()
    source_id = await add_task(
        title="source",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({"_pending_posthooks": ["summary:big_out"]}),
    )
    await update_task(source_id, status="ungraded")

    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_one
    verdict = PostHookVerdict(
        source_task_id=source_id, kind="summary:big_out",
        passed=False, raw={"artifact_name": "big_out"},
    )
    await _apply_one({"id": 997, "mission_id": 1, "agent_type": "artifact_summarizer"}, verdict)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "completed"
