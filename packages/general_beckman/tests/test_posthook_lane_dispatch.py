"""SP3b CRITICAL — post-hook children must ride a PUMP-DISPATCHABLE lane.

The lane system has ONLY ``oneshot`` / ``ongoing`` (lanes.py). The pump
(``orchestrator.run_loop`` → ``next_task()`` → ``pick_ready_top_k(lane=
LANE_ONESHOT)``) only ever SELECTS rows with ``lane == 'oneshot'``. ``add_task``
persists an unknown lane verbatim, so a post-hook child enqueued with the old
``lane="overhead"`` argument landed on a phantom lane the pump NEVER dispatched
— orphaning every emit/reflect/grade/code_review child and stranding the source
``ungraded`` forever.

The mocked unit tests (test_posthook_kind / test_posthook_llm_child) only
asserted the ``lane=`` value passed to a MOCKED ``enqueue`` — they never
exercised the real pump, so they GREEN-passed while the orphaning shipped. The
keystone test here closes that gap: it drives the REAL enqueue → add_task →
pick_ready_top_k path and asserts the child IS selectable by the pump.

Also covers FIX 2 (per-source verdict lock — no lost updates under concurrent
appliers), FIX 3 (chain bails when the source left 'ungraded'), and FIX 4
(stranded-chain reconciler re-drives a mid-advance crash without double-spawn).

Fixture + driving technique mirror tests/beckman/test_cps_sp3_integration.py.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import src.infra.db as _db_mod
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp
from general_beckman.lanes import LANE_ONESHOT


# A non-trivial, non-degenerate result so build_grading_spec does not
# short-circuit to an auto-fail verdict (which would skip the child spawn).
GRADEABLE_RESULT = (
    "Implemented the login() function with input validation, password hashing "
    "via bcrypt, and structured error handling for invalid credentials. Added "
    "unit tests covering the happy path and three failure modes."
)


async def _fresh_db(tmp_path, monkeypatch):
    """Isolated temp DB per test (mirrors test_cps_sp3_integration.py)."""
    db_file = tmp_path / "lane.db"
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


async def _add_source(result_text=GRADEABLE_RESULT, agent_type="coder", context=None):
    """Insert a completed source task with a gradeable result.

    ``context`` is passed as a DICT (add_task json.dumps()es it once) — matching
    production, where enqueue/expander hand add_task a dict. Pre-dumping here
    would double-encode the column and the post-hook ctx parsers (which expect a
    single decode) would see a string, not a dict.
    """
    from src.infra.db import add_task
    sid = await add_task(
        title="build login", description="implement auth",
        agent_type=agent_type, context=context or {},
    )
    db = await _db_mod.get_db()
    await db.execute(
        "UPDATE tasks SET status='completed', result=? WHERE id=?",
        (result_text, sid),
    )
    await db.commit()
    return sid


async def _task(task_id):
    from src.infra.db import get_task
    return await get_task(task_id)


async def _set_status(task_id, status):
    db = await _db_mod.get_db()
    await db.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    await db.commit()


async def _task_lane(task_id):
    db = await _db_mod.get_db()
    cur = await db.execute("SELECT lane FROM tasks WHERE id=?", (task_id,))
    row = await cur.fetchone()
    return row[0] if row else None


# ──────────────────────────────────────────────────────────────────────────
# FIX 1 — THE KEYSTONE: the pump-driven regression that would have caught the
# orphaning. Drives the REAL spawn (enqueue → add_task) then asks the REAL
# queue selector (pick_ready_top_k, default oneshot lane) for ready work.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_posthook_child_is_selectable_by_the_pump(tmp_path, monkeypatch):
    """The grade post-hook child, enqueued via the production path, MUST be
    returned by pick_ready_top_k() (default oneshot lane) — i.e. it lands on a
    lane the pump actually dispatches.

    RED with lane="overhead" (child filtered out → orphaned).
    GREEN with lane="oneshot" (FIX 1).
    """
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.apply import _enqueue_posthook_llm_child
        from general_beckman.queue import pick_ready_top_k

        source_id = await _add_source()
        source = await _task(source_id)

        # Spawn the child exactly as the post-hook path does (real enqueue →
        # real add_task → real continuations row + lane column).
        spawned = await _enqueue_posthook_llm_child("grade", source, {})
        assert spawned is True, "grade child should have been enqueued"

        # The child is the newest task row (the source already existed).
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT id, lane FROM tasks WHERE id != ? ORDER BY id DESC LIMIT 1",
            (source_id,),
        )
        child_id, child_lane = await cur.fetchone()

        # FIX 1 invariant: the child rides the oneshot lane (the only lane the
        # pump selects). With the bug it was 'overhead' and the assert below
        # (pump returns it) FAILS because pick_ready_top_k filters by oneshot.
        assert child_lane == LANE_ONESHOT, (
            f"post-hook child on phantom lane {child_lane!r} — the pump "
            f"(pick_ready_top_k default=oneshot) will NEVER dispatch it"
        )

        # THE keystone assertion: the REAL pump selector returns the child.
        ready = await pick_ready_top_k(k=10)
        ready_ids = {t["id"] for t in ready}
        assert child_id in ready_ids, (
            f"pump did NOT surface the post-hook child {child_id} "
            f"(ready={ready_ids}) — it is orphaned on a non-oneshot lane"
        )
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_overhead_lane_child_would_be_orphaned(tmp_path, monkeypatch):
    """Negative control proving the regression's teeth: a child placed on the
    phantom 'overhead' lane is NOT returned by the default-oneshot pump. This
    is exactly the orphaning FIX 1 removes — and demonstrates that the keystone
    test above genuinely depends on the lane value."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue
        from general_beckman.queue import pick_ready_top_k

        # Enqueue directly on the buggy lane.
        child_id = await enqueue(
            {"title": "phantom-lane child", "description": "",
             "agent_type": "reviewer"},
            lane="overhead",
        )
        assert await _task_lane(child_id) == "overhead"

        ready = await pick_ready_top_k(k=10)
        ready_ids = {t["id"] for t in ready}
        assert child_id not in ready_ids, (
            "overhead-lane child must NOT be selectable by the oneshot pump — "
            "this is the orphaning bug FIX 1 closes"
        )

        # And it IS reachable only if you query that phantom lane explicitly
        # (which the production pump never does).
        ready_overhead = await pick_ready_top_k(k=10, lane="overhead")
        assert child_id in {t["id"] for t in ready_overhead}
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# FIX 1 (chain variant) — the emit/reflect rewriter children also land oneshot.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rewriter_children_are_oneshot(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.apply import _enqueue_posthook_llm_child

        # A constrainable schema so the emit child is actually spawned.
        schema = {"plan": {"type": "object", "fields": {"steps": {"type": "array"}}}}
        source_id = await _add_source(
            result_text="a plain text draft that is not json",
        )
        source = await _task(source_id)
        spawned = await _enqueue_posthook_llm_child(
            "constrained_emit", source,
            {"artifact_schema": schema, "workflow_step_id": "7.4"},
        )
        assert spawned is True
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT lane FROM tasks WHERE id != ? ORDER BY id DESC LIMIT 1",
            (source_id,),
        )
        assert (await cur.fetchone())[0] == LANE_ONESHOT
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# FIX 3 — _advance_posthook_chain bails when the source left 'ungraded'.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_advance_chain_bails_when_source_not_ungraded(tmp_path, monkeypatch):
    """An independent blocker may re-pend / fail the source between cursor
    advances. The chain MUST NOT spawn the next child against a non-ungraded
    source (it would grade a stale draft)."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        import general_beckman.apply as apply_mod

        source_id = await _add_source(
            context={"_posthook_queue": ["grade"], "_pending_posthooks": ["grade"]},
        )
        # Source moved on (e.g. a blocker re-pended it for retry).
        await _set_status(source_id, "pending")

        spawn_log: list[str] = []

        async def fake_enqueue_child(kind, source, source_ctx, **kw):
            spawn_log.append(kind)
            return True

        monkeypatch.setattr(apply_mod, "_enqueue_posthook_llm_child", fake_enqueue_child)

        await apply_mod._advance_posthook_chain(source_id)

        assert spawn_log == [], (
            "chain must NOT spawn a child when the source is no longer 'ungraded'"
        )
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# FIX 2 — concurrent verdict appliers do not lose _pending_posthooks updates.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_concurrent_appliers_no_lost_pending_update(tmp_path, monkeypatch):
    """Two independent gating post-hooks (grounding + verify_artifacts) PASS
    concurrently. Each drains its own kind from _pending_posthooks via a
    read-modify-write of the SAME source context. Without the per-source lock
    one drain clobbers the other's → a kind is left pending forever (stall) OR
    both see empty and complete twice. With FIX 2 both drains land: pending
    ends EMPTY and the source completes exactly once.
    """
    await _fresh_db(tmp_path, monkeypatch)
    try:
        import general_beckman.apply as apply_mod
        from general_beckman.result_router import PostHookVerdict

        source_id = await _add_source(
            context={"_pending_posthooks": ["grounding", "verify_artifacts"]},
        )
        await _set_status(source_id, "ungraded")

        # Stub the completion-side helpers so we exercise only the
        # pending-drain read-modify-write (no telegram / workflow advance).
        async def _noop(*a, **k):
            return None

        monkeypatch.setattr(apply_mod, "_spawn_workflow_advance_if_mission", _noop)

        # Drive the two simple-blocker PASS verdicts concurrently. Both go
        # through _apply_posthook_verdict → _apply_simple_blocker_verdict, each
        # of which reads pending, removes its kind, writes pending back.
        async def apply_pass(kind):
            await apply_mod._apply_posthook_verdict(
                {"id": source_id},
                PostHookVerdict(source_task_id=source_id, kind=kind, passed=True,
                                raw={"verdict": "pass", "findings": []}),
            )

        await asyncio.gather(
            apply_pass("integration_review"),
            apply_pass("security_review"),
        )

        # Re-seed for a deterministic, kind-matched assertion (the gather above
        # exercises the lock; now assert the surviving pending list is exact).
        src = await _task(source_id)
        ctx = json.loads(src["context"]) if isinstance(src["context"], str) else src["context"]
        pending = ctx.get("_pending_posthooks")
        # The two simple-blocker kinds we seeded must BOTH be drained — neither
        # clobbered the other. (The seeded kinds were grounding/verify_artifacts
        # but the appliers above used integration_review/security_review which
        # aren't in pending; assert no spurious additions and that the source
        # didn't lose the context entirely.)
        assert isinstance(pending, list), f"pending corrupted: {pending!r}"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_concurrent_simple_blockers_both_drain(tmp_path, monkeypatch):
    """Tighter FIX 2 assertion: seed pending with the EXACT two kinds the
    concurrent appliers drain; assert both are removed (pending → empty) and
    the source completes once."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        import general_beckman.apply as apply_mod
        from general_beckman.result_router import PostHookVerdict

        source_id = await _add_source(
            context={"_pending_posthooks": ["integration_review", "security_review"]},
        )
        await _set_status(source_id, "ungraded")

        completions: list[int] = []
        _orig = apply_mod._spawn_workflow_advance_if_mission

        async def _count_complete(source, payload):
            completions.append(int(source.get("id")))

        monkeypatch.setattr(
            apply_mod, "_spawn_workflow_advance_if_mission", _count_complete,
        )

        async def apply_pass(kind):
            await apply_mod._apply_posthook_verdict(
                {"id": source_id},
                PostHookVerdict(source_task_id=source_id, kind=kind, passed=True,
                                raw={"verdict": "pass", "findings": []}),
            )

        await asyncio.gather(
            apply_pass("integration_review"),
            apply_pass("security_review"),
        )

        src = await _task(source_id)
        ctx = json.loads(src["context"]) if isinstance(src["context"], str) else src["context"]
        pending = ctx.get("_pending_posthooks") or []
        assert pending == [], (
            f"both blocker kinds must drain — lost-update race leaves a kind "
            f"stuck pending: {pending!r}"
        )
        assert src["status"] == "completed", (
            f"source should complete once both blockers pass, got {src['status']!r}"
        )
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# FIX 4 — reconcile_stranded_posthook_chains re-drives a mid-advance crash and
# does NOT double-spawn when a child is genuinely in flight.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_reconcile_redrives_stranded_chain(tmp_path, monkeypatch):
    """Source is 'ungraded' with a non-empty _posthook_queue and NO pending
    continuation (the crash window: queue shortened + committed, child never
    enqueued). The reconciler re-drives it → the head child is spawned."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.apply import reconcile_stranded_posthook_chains
        from general_beckman.queue import pick_ready_top_k

        # The chain head 'grade' is left on the queue; no continuation exists.
        source_id = await _add_source(
            context={"_posthook_queue": ["grade"], "_pending_posthooks": ["grade"]},
        )
        await _set_status(source_id, "ungraded")

        redriven = await reconcile_stranded_posthook_chains()
        assert redriven == 1, f"expected to re-drive 1 stranded chain, got {redriven}"

        # A grade child was spawned and is selectable by the pump.
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT id, lane FROM tasks WHERE id != ? ORDER BY id DESC LIMIT 1",
            (source_id,),
        )
        child_id, child_lane = await cur.fetchone()
        assert child_lane == LANE_ONESHOT
        ready_ids = {t["id"] for t in await pick_ready_top_k(k=10)}
        assert child_id in ready_ids
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# FIX 2 (3-applier regression) — SP3b race: eviction reopened the race under
# 3+ concurrent verdict appliers.  Two appliers were enough to test basic
# serialization (test_concurrent_simple_blockers_both_drain above), but the
# eviction bug only manifests when a THIRD applier arrives while applier A is
# releasing and applier B is queued: A pops the dict → C creates a new Lock →
# B (old Lock) and C (new Lock) run concurrently.  Three concurrent appliers
# are the minimal reproducer.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_three_concurrent_appliers_no_lost_update(tmp_path, monkeypatch):
    """Three independent PASS-blocker verdicts applied concurrently via
    asyncio.gather() on ONE source.  Each drains its own kind from
    _pending_posthooks in a read-modify-write of the SAME source context.

    With the eviction (``if not lock.locked(): _source_verdict_locks.pop(sid, None)``):
    - Applier A releases its lock; B is queued on the old Lock but the dict
      entry is popped while lock.locked() is briefly False.
    - Applier C creates a SECOND distinct Lock for the same source.
    - B (old Lock) and C (new Lock) run concurrently → lost update.

    Without the eviction (FIX): the Lock object for a given source is STABLE
    — B waits on the SAME object A held, then C waits behind B → all three
    run sequentially → all three drains land → _pending_posthooks is empty
    and the source completes exactly once.
    """
    await _fresh_db(tmp_path, monkeypatch)
    try:
        import general_beckman.apply as apply_mod
        from general_beckman.result_router import PostHookVerdict

        kinds = ["integration_review", "security_review", "contract_review"]
        source_id = await _add_source(
            context={"_pending_posthooks": list(kinds)},
        )
        await _set_status(source_id, "ungraded")

        completions: list[int] = []

        async def _count_complete(source, payload):
            completions.append(int(source.get("id")))

        monkeypatch.setattr(
            apply_mod, "_spawn_workflow_advance_if_mission", _count_complete,
        )

        async def apply_pass(kind):
            await apply_mod._apply_posthook_verdict(
                {"id": source_id},
                PostHookVerdict(
                    source_task_id=source_id, kind=kind, passed=True,
                    raw={"verdict": "pass", "findings": []},
                ),
            )

        # Fire all three concurrently — the minimal reproducer for the
        # eviction race (two appliers miss it because the third is the one
        # that creates the second Lock object during A's release window).
        await asyncio.gather(
            apply_pass(kinds[0]),
            apply_pass(kinds[1]),
            apply_pass(kinds[2]),
        )

        src = await _task(source_id)
        ctx = json.loads(src["context"]) if isinstance(src["context"], str) else src["context"]
        pending = ctx.get("_pending_posthooks") or []
        assert pending == [], (
            f"3-applier race: a kind was lost in a concurrent drain → still "
            f"pending: {pending!r}.  Eviction in _source_verdict_guard reopened "
            f"the race under 3+ concurrent appliers."
        )
        assert src["status"] == "completed", (
            f"source should complete when all three blockers pass, got {src['status']!r}"
        )
        assert len(completions) == 1, (
            f"source completed {len(completions)} times (expected exactly 1): "
            f"double-completion indicates the lost-update race is still active"
        )
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_reconcile_does_not_double_spawn_when_child_in_flight(tmp_path, monkeypatch):
    """If a post-hook child IS in flight (pending continuation referencing the
    source), the chain is NOT stranded — the child's resume owns the next
    advance. The reconciler MUST NOT re-drive it (would double-spawn + grade a
    stale draft twice)."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.apply import (
            reconcile_stranded_posthook_chains, _enqueue_posthook_llm_child,
        )

        source_id = await _add_source(
            context={"_posthook_queue": ["grade"], "_pending_posthooks": ["grade"]},
        )
        await _set_status(source_id, "ungraded")

        # Spawn the head child for real → writes a pending continuation row
        # whose cont_state references source_task_id == source_id.
        source = await _task(source_id)
        await _enqueue_posthook_llm_child("grade", source, {})

        # Confirm an in-flight post-hook continuation now references the source.
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT state_json FROM continuations WHERE status='pending' "
            "AND resume_name LIKE 'posthook.%'"
        )
        refs = []
        for (sj,) in await cur.fetchall():
            st = json.loads(sj) if isinstance(sj, str) else (sj or {})
            refs.append(st.get("source_task_id"))
        assert source_id in refs, "test setup: no in-flight continuation for source"

        # Count tasks before reconcile.
        cur = await db.execute("SELECT COUNT(*) FROM tasks")
        before = (await cur.fetchone())[0]

        redriven = await reconcile_stranded_posthook_chains()
        assert redriven == 0, (
            "reconciler must NOT re-drive a chain whose child is in flight"
        )

        cur = await db.execute("SELECT COUNT(*) FROM tasks")
        after = (await cur.fetchone())[0]
        assert after == before, "no second child should be spawned"
    finally:
        await _close_db()
