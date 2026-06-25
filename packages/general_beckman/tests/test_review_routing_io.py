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
async def test_escalate_persists_review_halt_payload(tmp_path, monkeypatch):
    """Escalation must persist a self-contained ``_review_halt`` payload onto
    the parked reviewer task so the founder-halt card can be re-rendered on
    restart / nudge WITHOUT re-deriving from the (fragile) workflow graph."""
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db_singleton(db_path)
    from src.infra.db import init_db
    await init_db()
    _reset_db_singleton(db_path)

    rid = await _seed_producer(
        db_path, mission_id=3, step_id="1.13", worker_attempts=1,
        status="ungraded",
    )

    from general_beckman.review_routing import _escalate_to_founder
    await _escalate_to_founder(
        mission_id=3, reviewer_id="1.13",
        review_result={"issues": [
            {"severity": "blocker", "problem": "missing boundaries",
             "target_artifact": "product_charter"},
        ]},
        workflow=None, reviewer_task_id=rid, reason="producer_exhausted",
        # The caller (apply.py) hands the reviewer context down; the parking
        # write merges _review_halt into it (no clobber, no extra DB read).
        reviewer_ctx={"workflow_step_id": "1.13", "is_workflow_step": True},
    )

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (rid,))).fetchone())
    assert row["status"] == "waiting_human"
    ctx = json.loads(row["context"])
    # Existing context keys preserved (merge, not clobber).
    assert ctx["workflow_step_id"] == "1.13"
    halt = ctx["_review_halt"]
    assert halt["reviewer_name"] == "1.13"
    assert halt["issues"][0]["problem"] == "missing boundaries"
    # workflow=None -> producers cannot be derived -> empty (card still has
    # the Accept-anyway button; per-producer regen buttons just absent).
    assert halt["producers"] == []


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


async def _seed_mission_with_context(db_path: str, workflow_name: str) -> int:
    """A mission carrying workflow_name in its context JSON but NO
    workflow_checkpoints row (the prod-realistic state — the writer never
    seeds the table). The loader must fall back to mission context."""
    ctx = json.dumps({"workflow_name": workflow_name})
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state, context) "
            "VALUES ('m', 'active', ?)", (ctx,))
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
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
async def test_verdict_fail_routes_to_producer_and_repends_reviewer(
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
    # Reviewer RE-PENDED (not completed) so it re-reviews the fixed artifacts
    # after the producer re-completes. Completing it would let the mission flow
    # past an unsatisfied review (advance.py never re-runs a completed step).
    assert rev["status"] == "pending"
    assert rev["status"] != "completed"


@pytest.mark.asyncio
async def test_verdict_fail_routes_with_no_checkpoint_via_mission_context(
    tmp_path, monkeypatch,
):
    """Class C regression: a mission with NO workflow_checkpoints row but
    `context.workflow_name='i2p_v3'` must still load the graph and re-pend the
    producer — NOT DLQ the reviewer with 'workflow graph unavailable'.
    The prod writer never seeds the checkpoint table, so the loader MUST fall
    back to mission context. Deliberately does NOT pre-seed a checkpoint."""
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

    # No checkpoint row — only mission context carries the workflow name.
    mid = await _seed_mission_with_context(db_path, "i2p_v3")
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

    # Producer re-pended (router resolved the graph from mission context).
    assert prod["status"] == "pending"
    assert prod["worker_attempts"] == 2
    assert "no traceability" in json.loads(prod["context"])["_schema_error"]
    # Reviewer re-pended, NOT DLQ'd with "workflow graph unavailable".
    assert rev["status"] == "pending"
    assert (rev["error"] or "") != "reviewer rejected artifact but workflow graph unavailable"


@pytest.mark.asyncio
async def test_verdict_fail_repends_reviewer_and_does_not_advance(
    tmp_path, monkeypatch,
):
    """Loop-closure: a completed producer + a completed reviewer that just
    emitted FAIL must end with BOTH rows back to `pending` (producer
    worker_attempts bumped; reviewer re-pended so it re-reviews the fix) AND
    the mission must NOT advance past the reviewer."""
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
    # Completed producer (attempts=1) and a completed reviewer that just FAILed.
    producer_id = await _seed_producer(
        db_path, mission_id=mid, step_id="3.4", worker_attempts=1,
        status="completed",
    )
    reviewer_id = await _seed_reviewer(db_path, mission_id=mid, step_id="3.11")

    # Assert the mission does NOT advance: spy on the advance spawn.
    import general_beckman.apply as apply_mod
    advanced = []

    async def _no_advance(source, raw):
        advanced.append(source.get("id"))

    monkeypatch.setattr(apply_mod, "_spawn_workflow_advance_if_mission", _no_advance)

    from general_beckman.apply import _apply_posthook_verdict
    from general_beckman.result_router import PostHookVerdict

    verdict = PostHookVerdict(
        source_task_id=reviewer_id, kind="verify_review_verdict", passed=False,
        raw={
            "verdict_class": "fail",
            "issues": [{"target_artifact": "requirements_spec",
                        "severity": "blocker", "problem": "missing error states"}],
        },
    )
    await _apply_posthook_verdict({"id": reviewer_id}, verdict)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        prod = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (producer_id,))).fetchone())
        rev = dict(await (await db.execute(
            "SELECT * FROM tasks WHERE id=?", (reviewer_id,))).fetchone())

    # Producer re-pended, worker_attempts incremented 1 -> 2.
    assert prod["status"] == "pending"
    assert prod["worker_attempts"] == 2
    # Reviewer re-pended (NOT completed) with its own worker_attempts backstop
    # bump, so it re-runs after the producer re-completes.
    assert rev["status"] == "pending"
    assert rev["status"] != "completed"
    assert int(rev["worker_attempts"] or 0) == 1  # 0 -> 1 backstop bump
    # The mission must NOT have advanced past the reviewer.
    assert advanced == []


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


# ---------------------------------------------------------------------------
# Part (b): the mechanical-check enqueue builder injects the reviewer's
# produced verdict as payload["review_result"]. Without this the mr_roboto
# verifier reads None -> classifies malformed -> wrong DLQ instead of routing.
# ---------------------------------------------------------------------------


def _build_payload(source_result):
    """Run _posthook_agent_and_payload for a verify_review_verdict check whose
    source step produced *source_result*; return the built mechanical payload."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source_ctx = {
        "checks": [{"kind": "verify_review_verdict",
                    "payload": {"action": "verify_review_verdict"}}],
    }
    a = RequestPostHook(
        source_task_id=7, kind="verify_review_verdict", source_ctx=source_ctx)
    source = {"id": 7, "result": source_result}
    agent, spec = _posthook_agent_and_payload(a, source, source_ctx)
    assert agent == "mechanical"
    return spec["payload"]


def test_injects_review_result_from_dict_result():
    review = {"status": "fail",
              "issues": [{"target_artifact": "requirements_spec",
                          "severity": "blocker", "problem": "p"}]}
    payload = _build_payload(review)
    assert payload["action"] == "verify_review_verdict"
    assert payload["review_result"] == review
    assert payload["review_result"]["status"] == "fail"
    assert payload["review_result"]["issues"][0]["target_artifact"] == "requirements_spec"


def test_injects_review_result_from_fenced_string_result():
    review = {"status": "fail",
              "issues": [{"target_artifact": "requirements_spec",
                          "severity": "blocker", "problem": "p"}]}
    fenced = "Here is my verdict:\n```json\n" + json.dumps(review) + "\n```\n"
    payload = _build_payload(fenced)
    assert payload["action"] == "verify_review_verdict"
    assert payload["review_result"]["status"] == "fail"
    assert payload["review_result"]["issues"][0]["problem"] == "p"


def test_injects_review_result_from_bare_json_string():
    review = {"status": "pass", "issues": []}
    payload = _build_payload(json.dumps(review))
    assert payload["review_result"] == review


def test_unparseable_result_leaves_review_result_none():
    # No JSON at all -> classifies malformed downstream, not a wrong route.
    payload = _build_payload("the reviewer narrated prose with no verdict")
    assert payload["action"] == "verify_review_verdict"
    assert payload["review_result"] is None
