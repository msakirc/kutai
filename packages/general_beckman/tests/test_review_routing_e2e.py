"""Full-chain real-DB e2e for reviewer-failure routing.

The apply-branch unit tests (test_review_routing_io.py) call
``_apply_posthook_verdict`` directly with a hand-built ``PostHookVerdict``.
This file drives the WHOLE mechanical pipeline a reviewer FAIL actually flows
through in production, exercising every real seam end-to-end:

    reviewer task (completed, FAIL verdict in `result`, `checks[]` declared)
      -> _apply_request_posthook(RequestPostHook(verify_review_verdict))
           [REAL emission entry: _posthook_agent_and_payload injects the
            reviewer's parsed result as payload["review_result"]; enqueues a
            mechanical mr_roboto child; parks reviewer `ungraded`]
      -> mr_roboto.run(child)            [REAL verify_review_verdict verifier]
      -> _mech_action_to_result(action)  [REAL orchestrator Action->result wrap]
      -> on_task_finished(child_id, result)
           route_result -> rewrite_actions (Rule 0c synthesises a
           PostHookVerdict from the mechanical result) -> apply_actions ->
           _apply_posthook_verdict -> _apply_review_verdict ->
           route_review_failure -> producer re-pend / founder-halt.

Nothing in the apply/route/mr_roboto path is stubbed — the only patches are
(a) the workflow loader (same seam the IO tests patch; there is no real i2p
checkpoint in a temp DB) and (b) the Telegram send (so the founder-halt card
doesn't reach a real bot), and we ASSERT on that send to prove escalation.

WIRING NOTE: Case 1 now drives the WHOLE chain from STEP 1 — the real
emission via on_task_finished(reviewer) → route_result → rewrite_actions →
apply_actions — with nothing in the emission→enqueue→verifier→apply→re-pend
chain stubbed (only load_workflow + the Telegram send are patched). This
proves the emission gap is closed: rewrite.py no longer bookkeeps a
workflow-STEP reviewer (agent_type="reviewer" + workflow_step_id, NO
source_task_id/posthook_kind), so its verify_review_verdict RequestPostHook is
actually emitted. Case 2 (systemic/founder-halt) continues to drive from the
apply entry to keep its assertions focused on the route_review_failure branch.
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


# --- workflow loader patch (same shape as test_review_routing_io.py) --------

class _FakeWF:
    def __init__(self, steps):
        self.steps = steps


def _patch_workflow(monkeypatch, steps):
    import src.workflows.engine.loader as loader_mod

    def fake_load(workflow_name):
        return _FakeWF(steps)

    monkeypatch.setattr(loader_mod, "load_workflow", fake_load)


async def _seed_mission_with_checkpoint(
    db_path: str, workflow_name: str = "i2p_v3", *, chat_id: int | None = None,
) -> int:
    ctx = json.dumps({"chat_id": chat_id}) if chat_id is not None else None
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state, context) "
            "VALUES ('m', 'active', ?)",
            (ctx,),
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.execute(
            "INSERT INTO workflow_checkpoints (mission_id, workflow_name) "
            "VALUES (?, ?)",
            (mid, workflow_name),
        )
        await db.commit()
    return mid


async def _seed_producer(
    db_path: str, *, mission_id: int, step_id: str,
    worker_attempts: int, max_worker_attempts: int = 15,
    status: str = "completed",
) -> int:
    ctx = json.dumps({"workflow_step_id": step_id, "is_workflow_step": True})
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tasks (title, description, mission_id, agent_type, "
            "status, worker_attempts, max_worker_attempts, context) "
            "VALUES (?, '', ?, 'coder', ?, ?, ?, ?)",
            (f"produce {step_id}", mission_id, status,
             worker_attempts, max_worker_attempts, ctx),
        )
        tid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()
    return tid


async def _seed_reviewer(
    db_path: str, *, mission_id: int, step_id: str, review_result: dict,
) -> int:
    """A COMPLETED reviewer task whose `result` carries a status-verdict and
    whose context declares the verify_review_verdict mechanical check (exactly
    as build_step_context places it for the 7 i2p reviewer steps)."""
    ctx = json.dumps({
        "workflow_step_id": step_id,
        "is_workflow_step": True,
        "checks": [{"kind": "verify_review_verdict",
                    "payload": {"action": "verify_review_verdict"}}],
    })
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tasks (title, description, mission_id, agent_type, "
            "status, worker_attempts, max_worker_attempts, context, result) "
            "VALUES ('review', '', ?, 'reviewer', 'completed', 0, 15, ?, ?)",
            (mission_id, ctx, json.dumps(review_result)),
        )
        tid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()
    return tid


async def _drive_full_chain(db_path: str, mission_id: int, reviewer_id: int):
    """Drive the real chain from the reviewer's verify_review_verdict request
    down through mr_roboto + apply + route. Returns nothing — assertions read
    the DB rows afterward."""
    from general_beckman.apply import _apply_request_posthook
    from general_beckman.result_router import RequestPostHook
    from src.infra.db import get_task
    import mr_roboto
    from src.core.orchestrator import _mech_action_to_result
    from general_beckman import on_task_finished

    reviewer = await get_task(reviewer_id)
    reviewer_ctx = json.loads(reviewer.get("context") or "{}")

    # 1) REAL emission entry — _posthook_agent_and_payload injects the parsed
    #    review_result into the mechanical payload + enqueues the mr_roboto
    #    child; the reviewer is parked `ungraded`.
    await _apply_request_posthook(
        reviewer,
        RequestPostHook(
            source_task_id=reviewer_id,
            kind="verify_review_verdict",
            source_ctx=reviewer_ctx,
        ),
    )

    # 2) Find the enqueued mechanical child (the only mechanical row).
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        child = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE mission_id=? AND agent_type='mechanical' "
            "ORDER BY id DESC LIMIT 1", (mission_id,))).fetchone())
    child_ctx = json.loads(child["context"])
    child_payload = child_ctx["payload"]
    assert child_payload["action"] == "verify_review_verdict"
    # The reviewer's own verdict must have been injected for mr_roboto to read.
    assert child_payload["review_result"] is not None

    # 3) REAL mr_roboto verifier + REAL orchestrator Action->result wrap.
    #    Mirror the orchestrator's mechanical dispatch prep (run.py): lift the
    #    parsed payload to a top-level key on the task dict before mr_roboto.run
    #    (the DB row's `context` is a JSON string; mr_roboto reads task["payload"]
    #    or a dict context["payload"], never a string).
    mech_task = dict(child)
    if "payload" not in mech_task and "payload" in child_ctx:
        mech_task["payload"] = child_ctx["payload"]
    action = await mr_roboto.run(mech_task)
    result = _mech_action_to_result(action)

    # 4) REAL on_task_finished on the child: route -> rewrite (synthesise the
    #    PostHookVerdict) -> apply -> _apply_review_verdict -> route_review_failure.
    await on_task_finished(child["id"], result)


async def _drive_full_chain_from_step1(db_path: str, mission_id: int, reviewer_id: int):
    """Drive the chain from STEP 1 — the REAL rewrite-layer EMISSION.

    Instead of hand-calling _apply_request_posthook, finish the reviewer task
    through the production entry: on_task_finished(reviewer_id, result). That
    runs route_result -> rewrite_actions (which, post-fix, emits the
    verify_review_verdict RequestPostHook for the workflow-STEP reviewer instead
    of bookkeeping it) -> apply_actions (enqueues the mechanical child). The rest
    of the chain (mr_roboto verifier + on_task_finished on the child) is then
    exercised exactly as _drive_full_chain does — nothing stubbed.
    """
    import json as _json
    import aiosqlite as _aiosqlite
    import mr_roboto
    from src.core.orchestrator import _mech_action_to_result
    from general_beckman import on_task_finished
    from src.infra.db import get_task

    reviewer = await get_task(reviewer_id)
    review_result = _json.loads(reviewer.get("result") or "{}")

    # 1) REAL EMISSION: finish the reviewer through the production entry. The
    #    agent envelope mirrors what the orchestrator hands beckman for a
    #    completed reviewer step — status=completed + the verdict JSON in result.
    await on_task_finished(
        reviewer_id,
        {"status": "completed", "result": _json.dumps(review_result)},
    )

    # 2) The emission must have enqueued the mechanical verifier child. If the
    #    reviewer were (wrongly) bookkept, no child exists and this fetch fails —
    #    which is exactly the regression this STEP-1 drive guards against.
    async with _aiosqlite.connect(db_path) as db:
        db.row_factory = _aiosqlite.Row
        row = await (await db.execute(
            "SELECT * FROM tasks WHERE mission_id=? AND agent_type='mechanical' "
            "AND context LIKE '%verify_review_verdict%' "
            "ORDER BY id DESC LIMIT 1", (mission_id,))).fetchone()
    assert row is not None, (
        "emission gap: finishing the workflow-step reviewer enqueued no "
        "verify_review_verdict mechanical child — rewrite.py still bookkeeps it"
    )
    child = dict(row)
    child_ctx = _json.loads(child["context"])
    child_payload = child_ctx["payload"]
    assert child_payload["action"] == "verify_review_verdict"
    assert child_payload["review_result"] is not None

    # 3) REAL mr_roboto verifier + REAL orchestrator Action->result wrap.
    mech_task = dict(child)
    if "payload" not in mech_task and "payload" in child_ctx:
        mech_task["payload"] = child_ctx["payload"]
    action = await mr_roboto.run(mech_task)
    result = _mech_action_to_result(action)

    # 4) REAL on_task_finished on the child → route -> rewrite (synthesise the
    #    PostHookVerdict) -> apply -> _apply_review_verdict -> route_review_failure.
    await on_task_finished(child["id"], result)


# ---------------------------------------------------------------------------
# Case 1: tagged FAIL -> autonomous producer re-pend + reviewer re-pend,
# no founder escalation.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tagged_fail_full_chain_repends_producer_no_founder(
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

    # Patch the telegram send so we can assert escalation did NOT happen.
    import src.app.telegram_bot as tg_mod

    class _FakeTg:
        def __init__(self):
            self.halt_calls = []

        async def send_review_halt_keyboard(self, **kw):
            self.halt_calls.append(kw)

    fake_tg = _FakeTg()
    monkeypatch.setattr(tg_mod, "get_telegram", lambda: fake_tg)

    mid = await _seed_mission_with_checkpoint(db_path, chat_id=4242)
    producer_id = await _seed_producer(
        db_path, mission_id=mid, step_id="3.4", worker_attempts=1,
    )
    review_result = {
        "status": "fail",
        "issues": [{"target_artifact": "requirements_spec",
                    "severity": "blocker", "problem": "no traceability"}],
    }
    reviewer_id = await _seed_reviewer(
        db_path, mission_id=mid, step_id="3.11", review_result=review_result,
    )

    # Case 1 drives from STEP 1: the REAL rewrite-layer emission via
    # on_task_finished(reviewer). Proves emission→enqueue→verifier→apply→re-pend
    # end-to-end with nothing in that chain stubbed (only load_workflow + tg).
    await _drive_full_chain_from_step1(db_path, mid, reviewer_id)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        prod = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (producer_id,))).fetchone())
        rev = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (reviewer_id,))).fetchone())

    # Producer re-pended with feedback; worker_attempts 1 -> 2.
    assert prod["status"] == "pending"
    assert prod["worker_attempts"] == 2
    assert "no traceability" in json.loads(prod["context"])["_schema_error"]
    # Reviewer RE-PENDED (not completed) so it re-reviews after the producer fix.
    assert rev["status"] == "pending"
    assert rev["status"] != "completed"
    # NO founder escalation: reviewer not parked, halt card never sent.
    assert rev["status"] != "waiting_human"
    assert fake_tg.halt_calls == []


# ---------------------------------------------------------------------------
# Case 2: untagged/systemic FAIL -> founder-halt (reviewer parked, producer
# untouched, halt card sent).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_systemic_fail_full_chain_halts_to_founder(tmp_path, monkeypatch):
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

    # Patch the telegram send so the founder-halt card doesn't reach a real bot
    # AND so we can assert it WAS attempted.
    import src.app.telegram_bot as tg_mod

    class _FakeTg:
        def __init__(self):
            self.halt_calls = []

        async def send_review_halt_keyboard(self, **kw):
            self.halt_calls.append(kw)

    fake_tg = _FakeTg()
    monkeypatch.setattr(tg_mod, "get_telegram", lambda: fake_tg)

    # chat_id required so _resolve_founder_chat_id finds a target and the card
    # is actually attempted.
    mid = await _seed_mission_with_checkpoint(db_path, chat_id=9090)
    producer_id = await _seed_producer(
        db_path, mission_id=mid, step_id="3.4", worker_attempts=1,
    )
    review_result = {
        "status": "fail",
        "issues": [{"target_artifact": None,
                    "severity": "blocker", "problem": "systemic"}],
    }
    reviewer_id = await _seed_reviewer(
        db_path, mission_id=mid, step_id="3.11", review_result=review_result,
    )

    await _drive_full_chain(db_path, mid, reviewer_id)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        prod = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (producer_id,))).fetchone())
        rev = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (reviewer_id,))).fetchone())

    # Reviewer PARKED waiting_human (escalated, must not auto-advance).
    assert rev["status"] == "waiting_human"
    # Producer NOT re-pended — nothing localisable to fix.
    assert prod["status"] == "completed"
    assert prod["worker_attempts"] == 1
    # The founder-halt card WAS attempted.
    assert len(fake_tg.halt_calls) == 1
    halt = fake_tg.halt_calls[0]
    assert halt["chat_id"] == 9090
    assert int(halt["reviewer_task_id"]) == reviewer_id
