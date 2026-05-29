"""SP3b Task 6 — ordered constrained_emit -> self_reflect -> grade chain.

The two result-REWRITING post-hooks (constrained_emit, self_reflect) must run
and land their rewrites on the source BEFORE the terminal grade gate sees the
result. The old fan-out spawned every kind in parallel, so grade could read the
un-rewritten draft. Task 6 turns the rewrite-then-grade trio into a SEQUENTIAL
cursor walk: an ordered ``_posthook_queue`` stashed on the source context; each
kind's resume continuation advances to the next; grade stays the terminal gate.

Two layers are exercised here:

* ``determine_posthooks`` ORDERING — rewriters first, grade last.
* The apply-layer SEQUENCING — only the head of the chain is spawned; each
  resume advances the cursor; a skipped emit/reflect drains without stalling;
  a plain (no-schema, no-reflection) source still spawns ONLY grade, exactly
  as before.

The child-enqueue is stubbed to (a) record the spawned ``kind`` order and
(b) immediately fire its resume continuation — mirroring the db/monkeypatch
pattern used by ``test_posthook_kind.py`` / ``test_posthook_llm_child.py``.
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Layer 1 — determine_posthooks ordering.
# ---------------------------------------------------------------------------

def test_determine_posthooks_orders_rewriters_before_grade():
    """A source with a constrainable artifact_schema AND enable_self_reflection
    yields the kinds in REWRITE-then-GATE order:
        ["constrained_emit", "self_reflect", "grade"].
    """
    from general_beckman.posthooks import determine_posthooks

    task = {"agent_type": "coder", "enable_self_reflection": True}
    # KutAI artifact_schema dialect: top-level keys are artifact names; an
    # object/array value is constrainable (string/markdown is not).
    schema = {
        "plan": {"type": "object", "fields": {"steps": {"type": "array"}}}
    }
    task_ctx = {"artifact_schema": schema, "enable_self_reflection": True}

    kinds = determine_posthooks(task, task_ctx, {})

    assert kinds == ["constrained_emit", "self_reflect", "grade"], kinds


def test_determine_posthooks_emit_then_grade_no_reflection():
    """Schema present, reflection OFF → emit then grade (no reflect)."""
    from general_beckman.posthooks import determine_posthooks

    task = {"agent_type": "coder"}
    schema = {"plan": {"type": "object", "fields": {"a": {"type": "string"}}}}
    task_ctx = {"artifact_schema": schema}

    kinds = determine_posthooks(task, task_ctx, {})

    assert kinds == ["constrained_emit", "grade"], kinds


def test_determine_posthooks_reflect_then_grade_no_schema():
    """Reflection ON, no schema → reflect then grade (no emit)."""
    from general_beckman.posthooks import determine_posthooks

    task = {"agent_type": "coder", "enable_self_reflection": True}
    task_ctx: dict = {}

    kinds = determine_posthooks(task, task_ctx, {})

    assert kinds == ["self_reflect", "grade"], kinds


def test_determine_posthooks_grade_only_unchanged():
    """Regression: a plain source (no schema, no reflection) still yields
    ONLY ["grade"], byte-identical to pre-Task-6 behavior."""
    from general_beckman.posthooks import determine_posthooks

    task = {"agent_type": "coder"}
    task_ctx: dict = {}

    kinds = determine_posthooks(task, task_ctx, {})

    assert kinds == ["grade"], kinds


def test_determine_posthooks_unconstrainable_schema_skips_emit():
    """A markdown/string artifact_schema is NOT constrainable → no emit kind
    is appended (the response_format would be None). Grade still runs."""
    from general_beckman.posthooks import determine_posthooks

    task = {"agent_type": "coder"}
    # All-string schema → schema_response_format returns None → not constrainable.
    task_ctx = {"artifact_schema": {"notes": {"type": "string"}}}

    kinds = determine_posthooks(task, task_ctx, {})

    assert "constrained_emit" not in kinds, kinds
    assert kinds == ["grade"], kinds


def test_determine_posthooks_extras_after_grade():
    """Explicit mechanical extras (verify_artifacts) come AFTER the rewrite
    trio + grade — the chain owns the head, extras keep their tail position."""
    from general_beckman.posthooks import determine_posthooks

    task = {"agent_type": "coder", "enable_self_reflection": True}
    schema = {"plan": {"type": "object", "fields": {"a": {"type": "string"}}}}
    task_ctx = {
        "artifact_schema": schema,
        "enable_self_reflection": True,
        "post_hooks": ["verify_artifacts"],
    }

    kinds = determine_posthooks(task, task_ctx, {})

    assert kinds == [
        "constrained_emit", "self_reflect", "grade", "verify_artifacts",
    ], kinds


# ---------------------------------------------------------------------------
# Layer 2 — apply-layer SEQUENCING (the cursor walk).
# ---------------------------------------------------------------------------

def _install_chain_harness(monkeypatch, store, spawn_log, *, skip=()):
    """Wire a fake DB + a stubbed _enqueue_posthook_llm_child that records the
    spawned kind, then immediately fires that kind's resume continuation.

    ``skip`` is a set of kinds whose child is *skipped* (no spawn / no resume)
    — simulating the emit/reflect early-return path. A skipped kind is recorded
    with a ``(skip)`` suffix so the test can assert the chain still drained.
    """
    import general_beckman.apply as apply_mod

    async def fake_get_task(tid):
        return store.get(int(tid))

    async def fake_update_task(tid, **kwargs):
        row = store.get(int(tid))
        if row is not None:
            row.update(kwargs)
            # Mirror the JSON round-trip: context is written as a JSON string.
            if "context" in kwargs and isinstance(kwargs["context"], str):
                row["context"] = kwargs["context"]
        return None

    monkeypatch.setattr("src.infra.db.get_task", fake_get_task)
    monkeypatch.setattr("src.infra.db.update_task", fake_update_task)
    monkeypatch.setattr("src.infra.db.add_task", _async_noop)

    async def fake_enqueue_child(kind, source, source_ctx, **kw):
        """Record + (for rewriters) fire the resume continuation, which calls
        back into _advance_posthook_chain to spawn the NEXT kind."""
        if kind in skip:
            spawn_log.append(f"{kind}(skip)")
            return False  # no child spawned — caller must drain to next kind
        spawn_log.append(kind)
        if kind == "grade":
            # Terminal gate — does not chain further in this harness.
            return True
        # Rewriter resume: leave source ungraded, advance the cursor.
        await apply_mod._advance_posthook_chain(int(source["id"]))
        return True

    monkeypatch.setattr(
        apply_mod, "_enqueue_posthook_llm_child", fake_enqueue_child,
    )
    return apply_mod


async def _async_noop(*a, **k):
    return None


@pytest.mark.asyncio
async def test_chain_spawns_emit_reflect_grade_in_order(monkeypatch):
    """The full trio spawns SEQUENTIALLY in order: emit, then reflect (after
    emit's resume), then grade LAST — proving the rewriters land before grade."""
    from general_beckman.result_router import RequestPostHook

    store = {
        1: {
            "id": 1, "mission_id": 5, "status": "completed",
            "result": "a plain text draft that is not json",
            "context": "{}",
        }
    }
    spawn_log: list[str] = []
    apply_mod = _install_chain_harness(monkeypatch, store, spawn_log)

    # rewrite.py emits a SINGLE RequestPostHook for the chain head, carrying the
    # ordered queue in source_ctx["_posthook_queue"].
    a = RequestPostHook(
        source_task_id=1,
        kind="constrained_emit",
        source_ctx={
            "_posthook_queue": ["constrained_emit", "self_reflect", "grade"],
        },
    )
    await apply_mod._apply_request_posthook(store[1], a)

    assert spawn_log == ["constrained_emit", "self_reflect", "grade"], spawn_log
    # grade was spawned strictly LAST.
    assert spawn_log[-1] == "grade"


@pytest.mark.asyncio
async def test_chain_skip_emit_still_advances(monkeypatch):
    """Skip-drain (handoff #2): if emit is skipped (draft already conforms /
    empty), the cursor must still advance to reflect then grade — the source
    never stalls in 'ungraded'."""
    from general_beckman.result_router import RequestPostHook

    store = {
        2: {
            "id": 2, "mission_id": 5, "status": "completed",
            "result": "draft", "context": "{}",
        }
    }
    spawn_log: list[str] = []
    apply_mod = _install_chain_harness(
        monkeypatch, store, spawn_log, skip={"constrained_emit"},
    )

    a = RequestPostHook(
        source_task_id=2,
        kind="constrained_emit",
        source_ctx={
            "_posthook_queue": ["constrained_emit", "self_reflect", "grade"],
        },
    )
    await apply_mod._apply_request_posthook(store[2], a)

    assert spawn_log == ["constrained_emit(skip)", "self_reflect", "grade"], spawn_log


@pytest.mark.asyncio
async def test_chain_skip_all_rewriters_lands_on_grade(monkeypatch):
    """Both rewriters skip → chain drains straight to grade. Grade still runs
    exactly once."""
    from general_beckman.result_router import RequestPostHook

    store = {
        3: {
            "id": 3, "mission_id": 5, "status": "completed",
            "result": "draft", "context": "{}",
        }
    }
    spawn_log: list[str] = []
    apply_mod = _install_chain_harness(
        monkeypatch, store, spawn_log, skip={"constrained_emit", "self_reflect"},
    )

    a = RequestPostHook(
        source_task_id=3,
        kind="constrained_emit",
        source_ctx={
            "_posthook_queue": ["constrained_emit", "self_reflect", "grade"],
        },
    )
    await apply_mod._apply_request_posthook(store[3], a)

    assert spawn_log == [
        "constrained_emit(skip)", "self_reflect(skip)", "grade",
    ], spawn_log


@pytest.mark.asyncio
async def test_grade_only_uses_direct_spawn_no_queue(monkeypatch):
    """Regression: a plain grade-only post-hook (no _posthook_queue in ctx)
    spawns grade directly — pending_posthooks holds 'grade', no chain queue is
    stashed, exactly as before Task 6."""
    from general_beckman.result_router import RequestPostHook

    store = {
        4: {
            "id": 4, "mission_id": 5, "status": "completed",
            "result": "draft", "context": "{}",
        }
    }
    spawn_log: list[str] = []
    apply_mod = _install_chain_harness(monkeypatch, store, spawn_log)

    a = RequestPostHook(source_task_id=4, kind="grade", source_ctx={})
    await apply_mod._apply_request_posthook(store[4], a)

    assert spawn_log == ["grade"], spawn_log
    # Source parked ungraded with 'grade' pending; NO chain queue stashed.
    ctx = json.loads(store[4]["context"])
    assert ctx.get("_pending_posthooks") == ["grade"]
    assert "_posthook_queue" not in ctx or ctx["_posthook_queue"] in ([], None)


@pytest.mark.asyncio
async def test_emit_reflect_not_in_pending_posthooks(monkeypatch):
    """The rewriters gate NOTHING — they must NOT be added to
    _pending_posthooks (only grade + non-chain kinds gate the source). If they
    were, grade-PASS would never drain to empty and the source would hang."""
    from general_beckman.result_router import RequestPostHook

    store = {
        5: {
            "id": 5, "mission_id": 5, "status": "completed",
            "result": "draft", "context": "{}",
        }
    }
    spawn_log: list[str] = []
    apply_mod = _install_chain_harness(monkeypatch, store, spawn_log)

    a = RequestPostHook(
        source_task_id=5,
        kind="constrained_emit",
        source_ctx={
            "_posthook_queue": ["constrained_emit", "self_reflect", "grade"],
        },
    )
    await apply_mod._apply_request_posthook(store[5], a)

    ctx = json.loads(store[5]["context"])
    pending = ctx.get("_pending_posthooks") or []
    assert "constrained_emit" not in pending, pending
    assert "self_reflect" not in pending, pending
    assert "grade" in pending, pending


@pytest.mark.asyncio
async def test_grade_less_rewrite_chain_completes_source(monkeypatch):
    """Edge: a rewrite-only chain (no grade — e.g. requires_grading=False) must
    not strand the source in 'ungraded'. When the cursor drains with no gating
    kind pending, the source is completed."""
    from general_beckman.result_router import RequestPostHook

    store = {
        6: {
            "id": 6, "mission_id": 5, "status": "completed",
            "result": "draft", "context": "{}",
        }
    }
    spawn_log: list[str] = []
    apply_mod = _install_chain_harness(monkeypatch, store, spawn_log)

    # Stub completion-side helpers (workflow advance / step progress) so the
    # test exercises only the completion transition.
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(apply_mod, "_spawn_workflow_advance_if_mission", _noop)

    a = RequestPostHook(
        source_task_id=6,
        kind="constrained_emit",
        source_ctx={
            # No grade in the queue — pure rewrite chain.
            "_posthook_queue": ["constrained_emit", "self_reflect"],
        },
    )
    await apply_mod._apply_request_posthook(store[6], a)

    assert spawn_log == ["constrained_emit", "self_reflect"], spawn_log
    # With no gating kind pending and the cursor drained, the source completes.
    assert store[6]["status"] == "completed", store[6]["status"]
