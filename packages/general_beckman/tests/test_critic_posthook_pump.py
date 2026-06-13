"""SP6 T2 — critic_gate posthook is an admitted LLM child + CPS resume.

Real-DB integration (fixture mirrors test_posthook_lane_dispatch.py):

(a) The production spawn path (``_enqueue_posthook_llm_child("critic_gate", ...)``)
    creates a child task row on the PUMP-DISPATCHABLE oneshot lane with
    agent_type == "critic" (the critic spec from mr_roboto.critic_gate), and the
    child is selectable by pick_ready_top_k.
(b) Feeding a VETO child result through the resume path (_critic_resume) builds a
    failing critic_gate verdict; routed through the EXISTING _apply_posthook_verdict
    rail (critic_gate ∈ _Z1_MECHANICAL_KINDS, blocker) it DLQs/fails the source.
"""
from __future__ import annotations

import json

import pytest

import src.infra.db as _db_mod
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp
from general_beckman.lanes import LANE_ONESHOT


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "critic.db"
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


async def _add_source(context=None, agent_type="coder"):
    from src.infra.db import add_task
    sid = await add_task(
        title="commit the repo", description="git commit",
        agent_type=agent_type, context=context or {},
    )
    db = await _db_mod.get_db()
    await db.execute(
        "UPDATE tasks SET status='completed', result=? WHERE id=?",
        ("staged a commit touching 3 files", sid),
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


# ──────────────────────────────────────────────────────────────────────────
# (a) The admitted critic child rides oneshot + agent_type=critic + is
#     selectable by the pump.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_critic_child_is_oneshot_and_critic_agent(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.apply import _enqueue_posthook_llm_child
        from general_beckman.queue import pick_ready_top_k

        source_id = await _add_source()
        source = await _task(source_id)

        # Spawn the child exactly as the post-hook route does, with the
        # action_name + target_payload the workflow step would carry.
        ctx = {
            "critic_action_name": "git_commit",
            "critic_target_payload": {"files": ["a.py", "b.py"], "msg": "feat: x"},
        }
        spawned = await _enqueue_posthook_llm_child("critic_gate", source, ctx)
        assert spawned is True, "critic child should have been enqueued"

        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT id, lane, agent_type FROM tasks WHERE id != ? "
            "ORDER BY id DESC LIMIT 1",
            (source_id,),
        )
        child_id, child_lane, child_agent = await cur.fetchone()

        assert child_lane == LANE_ONESHOT, (
            f"critic child on phantom lane {child_lane!r} — the pump will "
            f"NEVER dispatch it"
        )
        assert child_agent == "critic", (
            f"critic child must be agent_type='critic', got {child_agent!r}"
        )

        ready_ids = {t["id"] for t in await pick_ready_top_k(k=10)}
        assert child_id in ready_ids, (
            f"pump did NOT surface the critic child {child_id} (ready={ready_ids})"
        )
    finally:
        await _close_db()


# ──────────────────────────────────────────────────────────────────────────
# (b) A veto child result, fed through the resume path, DLQs/fails the source
#     via the EXISTING _Z1_MECHANICAL blocker rail (no new applier branch).
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_veto_resume_dlqs_the_source(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        import general_beckman.posthook_continuations as pc

        # The source is parked 'ungraded' with critic_gate pending (exactly the
        # state _apply_request_posthook leaves it in before the child resolves).
        source_id = await _add_source(
            context={"_pending_posthooks": ["critic_gate"]},
        )
        await _set_status(source_id, "ungraded")

        # Veto child result → resume builds a failing verdict and re-enters the
        # REAL apply-layer verdict applier (no patching of _apply_posthook_verdict).
        veto_result = {"result": {"content": '{"verdict":"veto","reasons":["leaks a token"]}'}}
        state = {"source_task_id": source_id, "action_name": "git_commit", "mission_id": None}

        await pc._critic_resume(child_task_id=99999, result=veto_result, state=state)

        src = await _task(source_id)
        assert src["status"] == "failed", (
            f"a veto must DLQ/fail the source, got status={src['status']!r}"
        )
        assert "critic_gate" in (src.get("error") or ""), (
            f"DLQ error should name critic_gate, got {src.get('error')!r}"
        )
        # I1 end-to-end guard (SP6 T2 FIX 2): the founder-visible DLQ error
        # column must carry the actual veto REASON, not just the kind prefix.
        assert "leaks a token" in (src.get("error") or ""), (
            f"veto reason must reach the DLQ error column, got {src.get('error')!r}"
        )
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_pass_resume_completes_the_source(tmp_path, monkeypatch):
    """A 'pass' verdict drains the pending critic_gate and lets the source
    complete (no other pending post-hooks)."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        import general_beckman.apply as apply_mod
        import general_beckman.posthook_continuations as pc

        async def _noop(*a, **k):
            return None
        monkeypatch.setattr(apply_mod, "_spawn_workflow_advance_if_mission", _noop)

        source_id = await _add_source(
            context={"_pending_posthooks": ["critic_gate"]},
        )
        await _set_status(source_id, "ungraded")

        pass_result = {"result": {"content": '{"verdict":"pass","reasons":[]}'}}
        state = {"source_task_id": source_id, "action_name": "git_commit"}

        await pc._critic_resume(child_task_id=99999, result=pass_result, state=state)

        src = await _task(source_id)
        assert src["status"] == "completed", (
            f"a pass should drain pending and complete the source, got "
            f"status={src['status']!r}"
        )
        ctx = json.loads(src["context"]) if isinstance(src["context"], str) else src["context"]
        assert (ctx.get("_pending_posthooks") or []) == []
    finally:
        await _close_db()
