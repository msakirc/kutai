"""FIX 2.1 — verdict-completion paths must fire the SOURCE's continuation.

A source task that reaches terminal (completed/failed) during ANOTHER task's
on_task_finished — grade/review verdict apply, post-hook chain drain, or the
mechanical post-hook DLQ cascade — never gets its own on_task_finished call,
so its continuation row used to stay 'pending' until the reconcile TTL
(up to an hour late). The chokepoint helper
``continuations.fire_if_terminal`` is invoked from the apply-layer wrappers
(``_apply_posthook_verdict`` / ``_advance_posthook_chain`` /
``_apply_request_posthook``) and after the ``_dlq_write`` post-hook cascade.

Fixture + driving technique mirror tests/beckman/test_cps_sp3_integration.py.
"""
import asyncio
import json
import pytest

import src.infra.db as _db_mod
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp


SOURCE_RESULT = (
    "Generated the requested landing-page hero image with correct dimensions "
    "and stored it under assets/img/hero.png as instructed by the brief."
)


async def _fresh_db(tmp_path, monkeypatch):
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


async def _seed_source_with_continuation(
    *, ctx: dict, resume="t.src.resume", on_error="t.src.err",
    status="ungraded", worker_attempts=0, max_worker_attempts=15,
):
    """Source task parked 'ungraded' that carries its OWN continuation row
    (the swap-chain shape: enqueue(..., on_complete=..., on_error=...))."""
    from src.infra.db import add_task
    sid = await add_task(
        title="image:hero", description="generate hero image",
        agent_type="image", context=json.dumps(ctx),
        on_complete=resume, on_error=on_error,
        cont_state={"pid": "hero"},
    )
    db = await _db_mod.get_db()
    await db.execute(
        "UPDATE tasks SET status=?, result=?, worker_attempts=?, "
        "max_worker_attempts=? WHERE id=?",
        (status, SOURCE_RESULT, worker_attempts, max_worker_attempts, sid),
    )
    await db.commit()
    return sid


def _register_spies(monkeypatch):
    """Register recording resume/on_error handlers; return the call lists."""
    from general_beckman.continuations import _HANDLERS

    resumed, errored = [], []

    async def _resume(task_id, result, state):
        resumed.append((task_id, dict(result or {}), dict(state or {})))

    async def _err(task_id, result, state):
        errored.append((task_id, dict(result or {}), dict(state or {})))

    monkeypatch.setitem(_HANDLERS, "t.src.resume", _resume)
    monkeypatch.setitem(_HANDLERS, "t.src.err", _err)
    return resumed, errored


async def _row_status(child_task_id):
    db = await _db_mod.get_db()
    cur = await db.execute(
        "SELECT status FROM continuations WHERE child_task_id=?",
        (child_task_id,),
    )
    row = await cur.fetchone()
    return row[0] if row else None


async def _task_status(task_id):
    db = await _db_mod.get_db()
    cur = await db.execute("SELECT status FROM tasks WHERE id=?", (task_id,))
    row = await cur.fetchone()
    return row[0] if row else None


def _grade_verdict(source_id, passed, raw=None):
    from general_beckman.result_router import PostHookVerdict
    return PostHookVerdict(
        source_task_id=source_id, kind="grade", passed=passed,
        raw=raw or {"passed": passed, "raw": "VERDICT: PASS" if passed else "bad"},
    )


# ──────────────────────────────────────────────────────────────────────────
# 1. Grade-PASS verdict completes the source → resume fires NOW (not at the
#    reconcile TTL).
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_grade_pass_verdict_fires_source_resume(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.apply import _apply_posthook_verdict

        resumed, errored = _register_spies(monkeypatch)
        sid = await _seed_source_with_continuation(
            ctx={"_pending_posthooks": ["grade"]},
        )
        assert await _row_status(sid) == "pending"

        await _apply_posthook_verdict({"id": sid + 1000},
                                      _grade_verdict(sid, True))
        await asyncio.sleep(0.05)

        assert await _task_status(sid) == "completed"
        assert await _row_status(sid) == "fired", (
            "source continuation must be CLAIMED by the verdict-completion "
            "chokepoint, not left pending for the reconcile TTL"
        )
        assert len(resumed) == 1, f"resume must fire exactly once: {resumed}"
        assert resumed[0][0] == sid
        assert resumed[0][1].get("status") == "completed"
        assert resumed[0][2] == {"pid": "hero"}
        assert errored == []
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# 2. Grade-FAIL at the attempt cap → DLQ (source status 'failed') → the
#    on_error continuation fires (DLQ fires on_error).
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_grade_fail_dlq_fires_source_on_error(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.apply import _apply_posthook_verdict

        resumed, errored = _register_spies(monkeypatch)
        sid = await _seed_source_with_continuation(
            ctx={"_pending_posthooks": ["grade"]},
            worker_attempts=15, max_worker_attempts=15,
        )

        await _apply_posthook_verdict({"id": sid + 1000},
                                      _grade_verdict(sid, False))
        await asyncio.sleep(0.05)

        assert await _task_status(sid) == "failed"
        assert await _row_status(sid) == "fired"
        assert len(errored) == 1, (
            f"DLQ must fire the on_error continuation exactly once: {errored}"
        )
        assert errored[0][0] == sid
        assert errored[0][1].get("status") == "failed"
        assert resumed == []
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# 3. Grade-FAIL below the cap re-pends the source (NOT terminal) → nothing
#    fires, row stays pending for the eventual retry terminal.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_grade_fail_repend_does_not_fire(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.apply import _apply_posthook_verdict

        resumed, errored = _register_spies(monkeypatch)
        sid = await _seed_source_with_continuation(
            ctx={"_pending_posthooks": ["grade"]},
            worker_attempts=0, max_worker_attempts=15,
        )

        await _apply_posthook_verdict({"id": sid + 1000},
                                      _grade_verdict(sid, False))
        await asyncio.sleep(0.05)

        assert await _task_status(sid) == "pending"  # re-pended for retry
        assert await _row_status(sid) == "pending", (
            "a re-pended source must keep its continuation pending"
        )
        assert resumed == [] and errored == []
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# 4. Chain drain — _advance_posthook_chain with an empty queue and no gating
#    pending kind completes the source via _complete_source_if_no_pending →
#    resume fires.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_chain_drain_completion_fires_source_resume(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.apply import _advance_posthook_chain

        resumed, errored = _register_spies(monkeypatch)
        sid = await _seed_source_with_continuation(
            ctx={"_posthook_queue": [], "_pending_posthooks": []},
        )

        await _advance_posthook_chain(sid)
        await asyncio.sleep(0.05)

        assert await _task_status(sid) == "completed"
        assert await _row_status(sid) == "fired"
        assert len(resumed) == 1 and resumed[0][0] == sid
        assert errored == []
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# 5. Mechanical post-hook DLQ cascade — the verifier child DLQs, the cascade
#    flips the SOURCE to failed inside the CHILD's _dlq_write → the source's
#    on_error continuation fires.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_posthook_dlq_cascade_fires_source_on_error(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.apply import _dlq_write
        from src.infra.db import add_task

        resumed, errored = _register_spies(monkeypatch)
        sid = await _seed_source_with_continuation(
            ctx={"_pending_posthooks": ["verify_artifacts"]},
        )
        # The mechanical verifier child carrying source_task_id+posthook_kind.
        child_id = await add_task(
            title="verify artifacts", description="",
            agent_type="mechanical",
            context=json.dumps({
                "source_task_id": sid, "posthook_kind": "verify_artifacts",
            }),
        )
        from src.infra.db import get_task
        child = await get_task(child_id)

        await _dlq_write(child, error="workspace permission denied",
                         category="worker", attempts=15)
        await asyncio.sleep(0.05)

        assert await _task_status(sid) == "failed"
        assert await _row_status(sid) == "fired"
        assert len(errored) == 1 and errored[0][0] == sid
        assert errored[0][1].get("status") == "failed"
        assert resumed == []
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# 6. fire_if_terminal unit behaviour: no row → False; non-terminal → False;
#    terminal → claims and dispatches with the persisted result.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_fire_if_terminal_unit(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import fire_if_terminal
        from src.infra.db import add_task

        resumed, _ = _register_spies(monkeypatch)

        # (a) task with no continuation row → False.
        plain = await add_task(title="p", description="d", agent_type="coder")
        db = await _db_mod.get_db()
        await db.execute(
            "UPDATE tasks SET status='completed' WHERE id=?", (plain,))
        await db.commit()
        assert await fire_if_terminal(plain) is False

        # (b) continuation-bearing task, non-terminal → False, row pending.
        sid = await _seed_source_with_continuation(
            ctx={}, status="ungraded")
        assert await fire_if_terminal(sid) is False
        assert await _row_status(sid) == "pending"

        # (c) terminal → True; handler receives the persisted JSON result.
        await db.execute(
            "UPDATE tasks SET status='completed', result=? WHERE id=?",
            (json.dumps({"content": "img-path"}), sid),
        )
        await db.commit()
        assert await fire_if_terminal(sid) is True
        await asyncio.sleep(0.05)
        assert len(resumed) == 1
        assert resumed[0][1].get("content") == "img-path"
        assert resumed[0][1].get("status") == "completed"

        # (d) idempotent — second call no-ops (row already fired).
        assert await fire_if_terminal(sid) is False
    finally:
        await _close_db()
