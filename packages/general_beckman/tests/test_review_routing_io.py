"""Real-DB I/O tests for the autonomous reviewer-failure routing path.

Covers `_repend_producer` re-pending an EXISTING producer task row (worker_
attempts carry forward, feedback stamped) and its two refusal paths (exhausted
cap; no matching row). Uses the same temp-sqlite harness as the other
general_beckman lifecycle tests.
"""
import json

import pytest
import aiosqlite


def _reset_db_singleton(db_path: str | None = None):
    import src.infra.db as db_module
    db_module._db_connection = None
    db_module._db_connection_path = None
    if db_path is not None:
        db_module.DB_PATH = db_path


async def _seed_producer(
    db_path: str, *, mission_id: int, step_id: str,
    worker_attempts: int, max_worker_attempts: int = 15,
    status: str = "completed",
) -> int:
    """Insert a producer task row carrying a workflow_step_id in its context."""
    ctx = json.dumps({"workflow_step_id": step_id, "is_workflow_step": True})
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tasks (title, description, mission_id, agent_type, "
            "status, worker_attempts, max_worker_attempts, context) "
            "VALUES (?, '', ?, 'coder', ?, ?, ?, ?)",
            (f"produce {step_id}", mission_id, status,
             worker_attempts, max_worker_attempts, ctx),
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        tid = (await cur.fetchone())[0]
        await db.commit()
    return tid


@pytest.mark.asyncio
async def test_repend_increments_existing_producer(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db_singleton(db_path)
    from src.infra.db import init_db
    await init_db()
    _reset_db_singleton(db_path)

    tid = await _seed_producer(
        db_path, mission_id=1, step_id="3.4", worker_attempts=1,
    )

    from general_beckman.review_routing import _repend_producer
    ok = await _repend_producer(1, "3.4", "Reviewer rejected this artifact. Fix:\n- bad")
    assert ok is True

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (tid,))).fetchone())
    assert row["status"] == "pending"
    assert row["worker_attempts"] == 2
    ctx = json.loads(row["context"])
    # Feedback must be visible to the next retry prompt.
    assert "Reviewer rejected" in (ctx.get("_schema_error") or "")


@pytest.mark.asyncio
async def test_repend_refuses_when_at_cap(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db_singleton(db_path)
    from src.infra.db import init_db
    await init_db()
    _reset_db_singleton(db_path)

    # attempts+1 == max -> exhausted -> refuse.
    tid = await _seed_producer(
        db_path, mission_id=2, step_id="3.4",
        worker_attempts=15, max_worker_attempts=15,
    )

    from general_beckman.review_routing import _repend_producer
    ok = await _repend_producer(2, "3.4", "fix it")
    assert ok is False

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (tid,))).fetchone())
    # Untouched: still the seeded terminal status, attempts not bumped.
    assert row["status"] == "completed"
    assert row["worker_attempts"] == 15


@pytest.mark.asyncio
async def test_repend_refuses_when_no_row(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db_singleton(db_path)
    from src.infra.db import init_db
    await init_db()
    _reset_db_singleton(db_path)

    from general_beckman.review_routing import _repend_producer
    ok = await _repend_producer(99, "9.9", "fix it")
    assert ok is False


@pytest.mark.asyncio
async def test_escalate_to_founder_never_raises(tmp_path, monkeypatch):
    """Safe stub: must mark the reviewer task waiting_human and not raise."""
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db_singleton(db_path)
    from src.infra.db import init_db
    await init_db()
    _reset_db_singleton(db_path)

    rid = await _seed_producer(
        db_path, mission_id=3, step_id="3.11", worker_attempts=1,
        status="ungraded",
    )

    from general_beckman.review_routing import _escalate_to_founder
    # Resolvable reviewer task id -> waiting_human.
    await _escalate_to_founder(
        mission_id=3, reviewer_task_id=rid, reason="no_localisable_target",
    )
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (rid,))).fetchone())
    assert row["status"] == "waiting_human"

    # Unresolvable id -> still must not raise.
    await _escalate_to_founder(mission_id=3, reason="producer_exhausted")


@pytest.mark.asyncio
async def test_assign_unresolved_returns_empty_for_now():
    from general_beckman.review_routing import _assign_unresolved
    out = await _assign_unresolved(
        [{"problem": "x"}], [("3.4", "requirements_spec")],
    )
    assert out == {}


# ---------------------------------------------------------------------------
# apply.py verdict dispatch — verify_review_verdict FAIL routes to producers
# and completes the reviewer; MALFORMED retries the reviewer task.
# ---------------------------------------------------------------------------


class _FakeWF:
    def __init__(self, steps):
        self.steps = steps


def _patch_workflow(monkeypatch, steps, name="i2p_v3"):
    import src.workflows.engine.loader as loader_mod

    def fake_load(workflow_name):
        return _FakeWF(steps)

    monkeypatch.setattr(loader_mod, "load_workflow", fake_load)


async def _seed_mission_with_checkpoint(db_path: str, workflow_name: str) -> int:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')")
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.execute(
            "INSERT INTO workflow_checkpoints (mission_id, workflow_name) "
            "VALUES (?, ?)",
            (mid, workflow_name),
        )
        await db.commit()
    return mid


async def _seed_reviewer(db_path: str, *, mission_id: int, step_id: str) -> int:
    """An ungraded reviewer task carrying a pending verify_review_verdict."""
    ctx = json.dumps({
        "workflow_step_id": step_id,
        "is_workflow_step": True,
        "_pending_posthooks": ["verify_review_verdict"],
    })
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tasks (title, description, mission_id, agent_type, "
            "status, context) VALUES ('review', '', ?, 'reviewer', "
            "'ungraded', ?)",
            (mission_id, ctx),
        )
        tid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()
    return tid


@pytest.mark.asyncio
async def test_verdict_fail_routes_to_producer_and_completes_reviewer(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db_singleton(db_path)
    from src.infra.db import init_db
    await init_db()
    _reset_db_singleton(db_path)

    steps = [
        {"id": "3.4", "output_artifacts": ["requirements_spec"]},
        {"id": "3.11", "input_artifacts": ["requirements_spec"],
         "output_artifacts": ["review_result"]},
    ]
    _patch_workflow(monkeypatch, steps)

    mid = await _seed_mission_with_checkpoint(db_path, "i2p_v3")
    producer_id = await _seed_producer(
        db_path, mission_id=mid, step_id="3.4", worker_attempts=1,
    )
    reviewer_id = await _seed_reviewer(db_path, mission_id=mid, step_id="3.11")

    from general_beckman.apply import _apply_posthook_verdict
    from general_beckman.result_router import PostHookVerdict

    verdict = PostHookVerdict(
        source_task_id=reviewer_id, kind="verify_review_verdict", passed=False,
        raw={
            "verdict_class": "fail",
            "issues": [{"target_artifact": "requirements_spec",
                        "severity": "blocker", "problem": "no traceability"}],
        },
    )
    await _apply_posthook_verdict({"id": reviewer_id}, verdict)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        prod = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (producer_id,))).fetchone())
        rev = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (reviewer_id,))).fetchone())

    # Producer re-pended with feedback.
    assert prod["status"] == "pending"
    assert prod["worker_attempts"] == 2
    assert "no traceability" in json.loads(prod["context"])["_schema_error"]
    # Reviewer completed (it correctly produced a rejection).
    assert rev["status"] == "completed"


@pytest.mark.asyncio
async def test_verdict_malformed_retries_reviewer(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db_singleton(db_path)
    from src.infra.db import init_db
    await init_db()
    _reset_db_singleton(db_path)

    mid = await _seed_mission_with_checkpoint(db_path, "i2p_v3")
    reviewer_id = await _seed_reviewer(db_path, mission_id=mid, step_id="3.11")

    from general_beckman.apply import _apply_posthook_verdict
    from general_beckman.result_router import PostHookVerdict

    verdict = PostHookVerdict(
        source_task_id=reviewer_id, kind="verify_review_verdict", passed=False,
        raw={"verdict_class": "malformed",
             "error": "no parseable review verdict", "issues": []},
    )
    await _apply_posthook_verdict({"id": reviewer_id}, verdict)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        rev = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (reviewer_id,))).fetchone())
    # Reviewer task itself retried (back to pending) — NOT completed, NOT
    # routed to any producer.
    assert rev["status"] in ("pending", "failed")  # retry or DLQ, never completed
    assert rev["status"] != "completed"
