"""SP3b Task 5 — emit/reflect LLM post-hook CHILDREN on husam (CPS rewrite).

``constrained_emit`` and ``self_reflect`` are post-hook kinds that spawn a
raw_dispatch LLM child (running on the husam worker) with a durable
continuation. On the child's terminal state the resume handler maps the
child's response into a ``PostHookVerdict(action="rewrite", new_result=...)``
that REWRITES the source task's result via Task 4's verdict path.

These tests mirror the db/monkeypatch + enqueue-capture pattern used by
``test_posthook_kind.py`` and ``test_posthook_rewrite.py``.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Step 1 (TDD) — child-spec is a raw_dispatch task carrying schema + draft.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_child_spec_is_raw_dispatch(monkeypatch):
    """Calling _enqueue_posthook_llm_child("constrained_emit", ...) builds a
    raw_dispatch=True spec whose messages contain the schema + draft, with a
    response_format set, and enqueues it on the overhead lane with the emit
    resume continuation."""
    import general_beckman.apply as apply_mod

    # KutAI artifact_schema dialect: top-level keys are artifact NAMES; each
    # value is a rule. Only object/array artifacts are constrainable.
    schema = {
        "connection": {
            "type": "object",
            "fields": {"connection_verified": {"type": "boolean"}},
        }
    }
    draft = "the connection is verified, all good"
    source = {"id": 42, "mission_id": 7, "result": draft}
    source_ctx = {"artifact_schema": schema, "workflow_step_id": "7.4"}

    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=901)) as enq:
        await apply_mod._enqueue_posthook_llm_child(
            "constrained_emit", source, source_ctx,
        )

    enq.assert_awaited_once()
    spec = enq.call_args.args[0]
    kwargs = enq.call_args.kwargs

    llm = spec["context"]["llm_call"]
    assert llm["raw_dispatch"] is True
    assert llm["response_format"] is not None
    # The draft text rides in the user message.
    joined = "\n".join(m["content"] for m in llm["messages"])
    assert "connection_verified" in joined  # schema field surfaced
    assert "the connection is verified" in joined  # draft surfaced

    assert kwargs["lane"] == "overhead"
    assert kwargs["on_complete"] == "posthook.constrained_emit.resume"
    assert kwargs["parent_id"] == 42
    # mission_id rides in cont_state, never on the child row.
    assert kwargs["cont_state"]["source_task_id"] == 42
    assert kwargs["cont_state"]["mission_id"] == 7


@pytest.mark.asyncio
async def test_reflect_child_spec_is_raw_dispatch(monkeypatch):
    """self_reflect spawns a raw_dispatch reflection child on the overhead
    lane with the reflect resume continuation."""
    import general_beckman.apply as apply_mod

    draft = (
        "Implemented the auth router with signed session tokens and a "
        "normalized SQLite schema for users and sessions."
    )
    source = {
        "id": 55, "mission_id": 9, "result": draft,
        "title": "build auth", "description": "add login", "agent_type": "coder",
    }
    source_ctx = {"agent_type": "coder"}

    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=902)) as enq:
        await apply_mod._enqueue_posthook_llm_child(
            "self_reflect", source, source_ctx,
        )

    enq.assert_awaited_once()
    spec = enq.call_args.args[0]
    kwargs = enq.call_args.kwargs

    llm = spec["context"]["llm_call"]
    assert llm["raw_dispatch"] is True
    joined = "\n".join(m["content"] for m in llm["messages"])
    assert "signed session tokens" in joined  # draft surfaced to reviewer
    assert kwargs["lane"] == "overhead"
    assert kwargs["on_complete"] == "posthook.self_reflect.resume"
    assert kwargs["parent_id"] == 55


# ---------------------------------------------------------------------------
# Step 1 (TDD) — emit resume produces a rewrite verdict.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_resume_produces_rewrite_verdict(monkeypatch):
    """A child completion carrying emitted JSON → resume applies a
    PostHookVerdict(action="rewrite", new_result=<emitted json>)."""
    import general_beckman.posthook_continuations as pc

    captured: list = []

    async def fake_apply(child_task, verdict):
        captured.append(verdict)

    monkeypatch.setattr(pc, "_apply_posthook_verdict", fake_apply)

    emitted = '{"connection_verified": true}'
    result = {"result": {"content": emitted, "model": "qwen"}}
    state = {"source_task_id": 42, "kind": "constrained_emit", "mission_id": 7}

    await pc._constrained_emit_resume(901, result, state)

    assert len(captured) == 1
    v = captured[0]
    assert v.action == "rewrite"
    assert v.new_result == emitted
    assert v.source_task_id == 42


@pytest.mark.asyncio
async def test_emit_resume_non_json_no_rewrite(monkeypatch):
    """If the emit produced unusable (non-JSON) output, the resume must NOT
    rewrite — never corrupt the source draft."""
    import general_beckman.posthook_continuations as pc

    captured: list = []

    async def fake_apply(child_task, verdict):
        captured.append(verdict)

    monkeypatch.setattr(pc, "_apply_posthook_verdict", fake_apply)

    result = {"result": {"content": "sorry, I cannot do that", "model": "x"}}
    state = {"source_task_id": 42, "kind": "constrained_emit", "mission_id": 7}

    await pc._constrained_emit_resume(901, result, state)

    # No rewrite verdict applied (gate/no-op). Either no verdict, or a
    # non-rewrite verdict — assert nothing rewrites the source.
    rewrites = [v for v in captured if getattr(v, "action", "gate") == "rewrite"]
    assert rewrites == []


# ---------------------------------------------------------------------------
# Step 1 (TDD) — reflect resume: fix rewrites, ok/degenerate no-op.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reflect_resume_fix_rewrites(monkeypatch):
    """child verdict "fix" with a clean corrected_result → rewrite."""
    import general_beckman.posthook_continuations as pc

    captured: list = []

    async def fake_apply(child_task, verdict):
        captured.append(verdict)

    monkeypatch.setattr(pc, "_apply_posthook_verdict", fake_apply)

    corrected = (
        "Implemented the auth router with signed session tokens, added the "
        "missing logout endpoint, and wired CSRF protection on the login form."
    )
    verdict_json = json.dumps({
        "verdict": "fix",
        "issues": "missing logout + csrf",
        "corrected_result": corrected,
    })
    result = {"result": {"content": verdict_json, "model": "qwen"}}
    state = {"source_task_id": 55, "kind": "self_reflect", "mission_id": 9}

    await pc._self_reflect_resume(902, result, state)

    rewrites = [v for v in captured if getattr(v, "action", "gate") == "rewrite"]
    assert len(rewrites) == 1
    assert rewrites[0].new_result == corrected
    assert rewrites[0].source_task_id == 55


@pytest.mark.asyncio
async def test_reflect_resume_ok_noop(monkeypatch):
    """child verdict "ok" → NO rewrite (warning severity never fails source)."""
    import general_beckman.posthook_continuations as pc

    captured: list = []

    async def fake_apply(child_task, verdict):
        captured.append(verdict)

    monkeypatch.setattr(pc, "_apply_posthook_verdict", fake_apply)

    verdict_json = json.dumps({"verdict": "ok"})
    result = {"result": {"content": verdict_json, "model": "qwen"}}
    state = {"source_task_id": 55, "kind": "self_reflect", "mission_id": 9}

    await pc._self_reflect_resume(902, result, state)

    rewrites = [v for v in captured if getattr(v, "action", "gate") == "rewrite"]
    assert rewrites == []


@pytest.mark.asyncio
async def test_reflect_resume_degenerate_corrected_noop(monkeypatch):
    """child verdict "fix" but degenerate corrected_result → NO rewrite."""
    import general_beckman.posthook_continuations as pc

    captured: list = []

    async def fake_apply(child_task, verdict):
        captured.append(verdict)

    async def fake_assess(*a, **k):  # pragma: no cover - replaced below
        raise AssertionError("sync assess expected")

    monkeypatch.setattr(pc, "_apply_posthook_verdict", fake_apply)

    # Force dogru_mu_samet.assess to report degenerate.
    import dogru_mu_samet

    class _Cq:
        is_degenerate = True
        summary = "repetitive"

    monkeypatch.setattr(dogru_mu_samet, "assess", lambda *a, **k: _Cq())

    verdict_json = json.dumps({
        "verdict": "fix",
        "corrected_result": "aaa aaa aaa aaa aaa aaa aaa aaa",
    })
    result = {"result": {"content": verdict_json, "model": "qwen"}}
    state = {"source_task_id": 55, "kind": "self_reflect", "mission_id": 9}

    await pc._self_reflect_resume(902, result, state)

    rewrites = [v for v in captured if getattr(v, "action", "gate") == "rewrite"]
    assert rewrites == []


# ---------------------------------------------------------------------------
# Reflect/emit resume_err — terminal child failure must never corrupt source.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_resume_err_no_rewrite(monkeypatch):
    """on_error: the emit child failed terminally → leave draft (no rewrite)."""
    import general_beckman.posthook_continuations as pc

    captured: list = []

    async def fake_apply(child_task, verdict):
        captured.append(verdict)

    monkeypatch.setattr(pc, "_apply_posthook_verdict", fake_apply)

    result = {"error": "no candidates"}
    state = {"source_task_id": 42, "kind": "constrained_emit", "mission_id": 7}

    await pc._constrained_emit_resume_err(901, result, state)

    rewrites = [v for v in captured if getattr(v, "action", "gate") == "rewrite"]
    assert rewrites == []


@pytest.mark.asyncio
async def test_reflect_resume_err_no_rewrite(monkeypatch):
    """on_error: the reflect child failed terminally → leave draft (no rewrite)."""
    import general_beckman.posthook_continuations as pc

    captured: list = []

    async def fake_apply(child_task, verdict):
        captured.append(verdict)

    monkeypatch.setattr(pc, "_apply_posthook_verdict", fake_apply)

    result = {"error": "infra"}
    state = {"source_task_id": 55, "kind": "self_reflect", "mission_id": 9}

    await pc._self_reflect_resume_err(902, result, state)

    rewrites = [v for v in captured if getattr(v, "action", "gate") == "rewrite"]
    assert rewrites == []


# ---------------------------------------------------------------------------
# Continuation handlers are registered at import time.
# ---------------------------------------------------------------------------

def test_emit_reflect_continuations_registered():
    import general_beckman.posthook_continuations  # noqa: F401
    from general_beckman.continuations import _HANDLERS
    assert "posthook.constrained_emit.resume" in _HANDLERS
    assert "posthook.constrained_emit.resume_err" in _HANDLERS
    assert "posthook.self_reflect.resume" in _HANDLERS
    assert "posthook.self_reflect.resume_err" in _HANDLERS
