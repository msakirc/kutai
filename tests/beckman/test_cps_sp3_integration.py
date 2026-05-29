"""CPS SP3 — end-to-end integration tests.

Drives the REAL spawn path (``_apply_request_posthook`` →
``_enqueue_posthook_llm_child`` → ``enqueue`` → ``add_task`` → continuations
row) and the REAL fire path (``on_task_finished`` reads the DB-terminal status
and calls ``fire_for_task`` → resume handler → ``_apply_posthook_verdict``)
against an isolated temp DB.

Proves three things:
  1. The grade post-hook spawns a raw_dispatch ``reviewer`` child with a durable
     continuation — NO cap-counted ``grader`` agent task → deadlock closed.
  2. C1 keystone: a transient ``failed`` that re-pends (DB never reaches a
     terminal) does NOT fire on_error; the eventual ``completed`` fires the
     grade resume EXACTLY once → source ends graded, not silently dropped.
  3. CAS idempotency: a double on_task_finished on the same completed child
     fires the resume / applies the verdict exactly once.

Fixture + driving technique mirror tests/beckman/test_continuations_durable.py.
"""
import asyncio
import json
import pytest

import src.infra.db as _db_mod
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp


# A synthetic reviewer output that parse_grade_response resolves to PASS.
GRADER_PASS = (
    "RELEVANT: YES\n"
    "COMPLETE: YES\n"
    "VERDICT: PASS\n"
    "WELL_FORMED: PASS\n"
    "COHERENT: PASS\n"
)

# A non-trivial (>=10 chars), non-degenerate, non-repetitive source result so
# build_grading_spec / dogru_mu_samet does NOT short-circuit to an auto-fail
# verdict (which would skip the child spawn entirely).
GRADEABLE_RESULT = (
    "Implemented the login() function with input validation, password hashing "
    "via bcrypt, and structured error handling for invalid credentials. Added "
    "unit tests covering the happy path and three failure modes."
)


async def _fresh_db(tmp_path, monkeypatch):
    """Isolated temp DB per test (mirrors test_continuations_durable.py)."""
    db_file = tmp_path / "cps.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    monkeypatch.setattr(_cs, "_seeded", False)
    _pp._patterns.clear()
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


async def _add_source(result_text=GRADEABLE_RESULT, agent_type="coder"):
    """Insert a completed source task with a gradeable result."""
    from src.infra.db import add_task
    sid = await add_task(
        title="build login", description="implement auth",
        agent_type=agent_type, context=json.dumps({}),
    )
    await _set_task(sid, "completed", result_text)
    return sid


async def _set_task(task_id, status, result_text=None):
    """Set a task's DB status (+ optional plain-string result column)."""
    db = await _db_mod.get_db()
    if result_text is None:
        await db.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    else:
        await db.execute(
            "UPDATE tasks SET status=?, result=? WHERE id=?",
            (status, result_text, task_id),
        )
    await db.commit()


async def _task_status(task_id):
    db = await _db_mod.get_db()
    cur = await db.execute("SELECT status FROM tasks WHERE id=?", (task_id,))
    row = await cur.fetchone()
    return row[0] if row else None


async def _continuation_row(child_task_id):
    db = await _db_mod.get_db()
    cur = await db.execute(
        "SELECT resume_name, on_error_name, status FROM continuations "
        "WHERE child_task_id=?", (child_task_id,),
    )
    return await cur.fetchone()


async def _agent_types():
    db = await _db_mod.get_db()
    cur = await db.execute("SELECT id, agent_type FROM tasks ORDER BY id")
    return await cur.fetchall()


async def _drive_grade_posthook(source_id):
    """Run the REAL _apply_request_posthook for a 'grade' RequestPostHook and
    return the spawned reviewer child id (the newest task row)."""
    from general_beckman.apply import _apply_request_posthook
    from general_beckman.result_router import RequestPostHook

    before = {row[0] for row in await _agent_types()}
    await _apply_request_posthook(
        await _get_source(source_id),
        RequestPostHook(source_task_id=source_id, kind="grade", source_ctx={}),
    )
    after = await _agent_types()
    new_ids = [row[0] for row in after if row[0] not in before]
    assert len(new_ids) == 1, f"expected exactly one spawned child, got {new_ids}"
    return new_ids[0]


async def _get_source(source_id):
    from src.infra.db import get_task
    return await get_task(source_id)


# ──────────────────────────────────────────────────────────────────────────
# Test 1 — deadlock closure: grade post-hook spawns a reviewer CHILD with a
# durable continuation; NO cap-counted grader agent task is created.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_grade_posthook_enqueues_reviewer_child_no_grader_task(
    tmp_path, monkeypatch,
):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        source_id = await _add_source()
        child_id = await _drive_grade_posthook(source_id)

        # (a) continuation row exists with the grade resume handler name.
        row = await _continuation_row(child_id)
        assert row is not None, "no continuation row written for the child"
        assert row[0] == "posthook.grade.resume", f"resume_name={row[0]}"
        assert row[1] == "posthook.grade.resume_err", f"on_error_name={row[1]}"
        assert row[2] == "pending"

        # (b) the spawned child task is a raw_dispatch reviewer.
        agent_types = {tid: at for tid, at in await _agent_types()}
        assert agent_types[child_id] == "reviewer", (
            f"child agent_type={agent_types[child_id]}"
        )

        # (c) NO cap-counted grader agent task anywhere — the deadlock source.
        assert "grader" not in agent_types.values(), (
            f"a grader task was created: {agent_types}"
        )

        # (d) source parked ungraded, awaiting the reviewer child's verdict.
        assert await _task_status(source_id) == "ungraded"
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# Test 2 — C1 regression (keystone): transient failed → re-pend (no fire) →
# retried completed fires the grade resume EXACTLY once; source ends graded.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_failed_then_retried_completed_fires_grade_resume_once(
    tmp_path, monkeypatch,
):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import on_task_finished
        from general_beckman.posthook_continuations import register_continuations
        import general_beckman.posthook_continuations as _pcmod

        # Ensure the SP3 grade handlers are registered in this process.
        register_continuations()

        # Spy on the resume handler to count fires precisely without depending
        # on telegram / artifact side-effects of the full verdict applier.
        fires = []
        _orig_resume = _pcmod._grade_resume

        async def _counting_resume(child_task_id, result, state):
            fires.append(child_task_id)
            await _orig_resume(child_task_id, result, state)

        monkeypatch.setattr(_pcmod, "_grade_resume", _counting_resume)
        from general_beckman.continuations import register_resume, _HANDLERS
        register_resume("posthook.grade.resume", _counting_resume)

        source_id = await _add_source()
        child_id = await _drive_grade_posthook(source_id)

        # ── Transient failure that RE-PENDS ──────────────────────────────
        # The C1 fix: on_task_finished fires off the DB tasks.status AFTER
        # apply_actions, only on a TRUE terminal (completed/failed). A
        # re-pended child (DB status='pending') must NOT fire on_error.
        await _set_task(child_id, "pending")
        await on_task_finished(child_id, {"status": "failed", "error": "transient"})
        await asyncio.sleep(0.05)
        assert fires == [], (
            f"resume/on_error fired on a re-pended transient failure: {fires}"
        )
        # Continuation still pending; source still parked.
        crow = await _continuation_row(child_id)
        assert crow[2] == "pending", "continuation must stay pending after re-pend"
        assert await _task_status(source_id) == "ungraded"

        # ── Retry succeeds with a valid grader output ────────────────────
        await _set_task(child_id, "completed", GRADER_PASS)
        await on_task_finished(child_id, {"status": "completed", "result": GRADER_PASS})
        await asyncio.sleep(0.05)

        assert fires == [child_id], (
            f"grade resume must fire EXACTLY once on the retry completion: {fires}"
        )
        # The source left 'ungraded' — verdict (PASS) was applied, not dropped.
        assert await _task_status(source_id) == "completed", (
            f"source must end graded(completed), got {await _task_status(source_id)}"
        )
        # Continuation consumed exactly once.
        assert (await _continuation_row(child_id))[2] == "fired"
    finally:
        _HANDLERS.pop("posthook.grade.resume", None)
        register_continuations()  # restore the canonical handler
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# Test 3 — CAS idempotency: two terminal on_task_finished calls for the same
# completed child fire the resume / apply the verdict exactly once.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_double_terminal_fires_resume_once(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import on_task_finished
        from general_beckman.posthook_continuations import register_continuations
        import general_beckman.posthook_continuations as _pcmod
        from general_beckman.continuations import register_resume, _HANDLERS

        register_continuations()

        applied = []
        _orig_apply = _pcmod._apply_posthook_verdict

        async def _counting_apply(child_task, verdict):
            applied.append((child_task.get("id"), verdict.kind, verdict.passed))
            await _orig_apply(child_task, verdict)

        monkeypatch.setattr(_pcmod, "_apply_posthook_verdict", _counting_apply)

        fires = []
        _orig_resume = _pcmod._grade_resume

        async def _counting_resume(child_task_id, result, state):
            fires.append(child_task_id)
            await _orig_resume(child_task_id, result, state)

        monkeypatch.setattr(_pcmod, "_grade_resume", _counting_resume)
        register_resume("posthook.grade.resume", _counting_resume)

        source_id = await _add_source()
        child_id = await _drive_grade_posthook(source_id)

        await _set_task(child_id, "completed", GRADER_PASS)
        # Fire the SAME completed child TWICE.
        await on_task_finished(child_id, {"status": "completed", "result": GRADER_PASS})
        await on_task_finished(child_id, {"status": "completed", "result": GRADER_PASS})
        await asyncio.sleep(0.05)

        # The continuations CAS (pending→fired) guarantees a single dispatch.
        assert fires == [child_id], f"resume must fire exactly once, got {fires}"
        assert len(applied) == 1, f"verdict applied more than once: {applied}"
        assert applied[0][1] == "grade" and applied[0][2] is True
        assert await _task_status(source_id) == "completed"
        assert (await _continuation_row(child_id))[2] == "fired"
    finally:
        _HANDLERS.pop("posthook.grade.resume", None)
        register_continuations()
        await _close_db()
