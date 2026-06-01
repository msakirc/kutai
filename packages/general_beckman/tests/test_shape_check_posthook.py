"""Shape-check verify verbs as producer post-hooks (the verify-step dead-end fix).

A standalone `.verify` workflow step (source_task_id=null) that fails its check
only blocks its dependents and waits for a human — the PRODUCER stays
`completed` and is never re-run. Founder principle: if a verification says the
source is bad, the source-producing task must be retried.

The fix routes these mechanical shape verbs through the SAME post-hook rail the
existing mechanical checks (grounding, verify_artifacts) use:
  - registered as a named PostHookSpec (blocker)
  - payload built generically from the producer's declared inputs
  - rewrite Rule 0c carries the verb's structured problems into the verdict
  - the verdict re-pends the producer with feedback via _apply_simple_blocker_verdict

`verify_interview_script_shape` is the first verb converted; this pins the rail.
"""
from __future__ import annotations

import json

import pytest

from general_beckman.posthooks import determine_posthooks, POST_HOOK_REGISTRY
from general_beckman.apply import (
    _posthook_agent_and_payload,
    _apply_posthook_verdict_locked,
)
from general_beckman.result_router import Failed, PostHookVerdict, RequestPostHook
from general_beckman.rewrite import rewrite_actions


KIND = "verify_interview_script_shape"


# ── registry + discovery ────────────────────────────────────────────────

def test_kind_registered_as_blocker():
    spec = POST_HOOK_REGISTRY.get(KIND)
    assert spec is not None
    assert spec.default_severity == "blocker"


def test_determine_posthooks_accepts_kind_from_ctx():
    task = {"id": 1, "agent_type": "analyst"}
    ctx = {"post_hooks": [KIND]}
    assert KIND in determine_posthooks(task, ctx, {})


# ── payload builder (generic, from producer inputs) ──────────────────────

def test_payload_built_from_producer_produces():
    a = RequestPostHook(source_task_id=5, kind=KIND, source_ctx={})
    source_ctx = {
        "produces": ["mission_5/.intake/interview_script.md"],
        "payload": {"min_questions": 5, "max_questions": 7},
    }
    agent, spec = _posthook_agent_and_payload(a, {"id": 5}, source_ctx)
    assert agent == "mechanical"
    assert spec["posthook_kind"] == KIND
    assert spec["payload"]["action"] == KIND
    assert spec["payload"]["script_paths"] == ["mission_5/.intake/interview_script.md"]


# ── rewrite Rule 0c carries structured problems (the feedback gap fix) ────

def test_rewrite_checkfail_carries_problems_into_verdict():
    """The verb returns Action(status=failed, result={ok:False,question_problems:[...]}).
    rewrite must carry those problems into verdict.raw so the producer's retry
    feedback can name what to fix — not just a generic error string."""
    res = {
        "ok": False,
        "question_count": 2,
        "question_problems": [
            {"header": "### Q1 — pricing", "missing_fields": ["Probes"]},
        ],
    }
    task = {"id": 600, "mission_id": 9, "context": "{}", "agent_type": "mechanical"}
    ctx = {"source_task_id": 225575, "posthook_kind": KIND}
    raw = {"status": "failed", "error": "shape bad", "result": json.dumps(res)}
    out = rewrite_actions(task, ctx, [Failed(task_id=600, error="shape bad", raw=raw)])
    verdicts = [a for a in out if isinstance(a, PostHookVerdict)]
    assert len(verdicts) == 1
    assert verdicts[0].passed is False
    # the structured problems must survive into the verdict
    assert verdicts[0].raw.get("question_problems")


# ── verdict re-pends the producer with useful feedback ───────────────────

@pytest.mark.asyncio
async def test_verdict_fail_repends_producer_with_problem_feedback(monkeypatch):
    source = {
        "id": 225575, "mission_id": 9, "status": "ungraded",
        "title": "interview_script_generation",
        "worker_attempts": 0, "max_worker_attempts": 5,
        "result": "prose narration, not a script",
        "context": json.dumps({"_pending_posthooks": [KIND]}),
    }
    updates = []

    async def _get(task_id):
        return source

    async def _update(task_id, **kw):
        updates.append((task_id, kw))

    monkeypatch.setattr("src.infra.db.get_task", _get)
    monkeypatch.setattr("src.infra.db.update_task", _update)

    verdict = PostHookVerdict(
        source_task_id=225575, kind=KIND, passed=False,
        raw={
            "ok": False,
            "question_problems": [
                {"header": "### Q1 — pricing", "missing_fields": ["Probes"]},
            ],
        },
    )
    await _apply_posthook_verdict_locked(source, verdict)

    pend = [kw for _, kw in updates if kw.get("status") == "pending"]
    assert pend, "producer must be re-pended on shape failure, not DLQ'd"
    assert pend[0]["worker_attempts"] == 1
    assert pend[0]["error_category"] == "quality"
    saved = json.loads(pend[0]["context"])
    fb = saved.get("_schema_error") or ""
    # feedback must name the actual problem, not be a bare generic string
    assert ("Probes" in fb) or ("Q1" in fb)
    # never cascade straight to failed for a runnable-check fail
    assert not any(kw.get("status") == "failed" for _, kw in updates)


@pytest.mark.asyncio
async def test_verdict_pass_completes_producer(monkeypatch):
    source = {
        "id": 226, "mission_id": 9, "status": "ungraded",
        "title": "interview_script_generation",
        "context": json.dumps({"_pending_posthooks": [KIND]}),
    }
    updates = []

    async def _get(task_id):
        return source

    async def _update(task_id, **kw):
        updates.append((task_id, kw))

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr("src.infra.db.get_task", _get)
    monkeypatch.setattr("src.infra.db.update_task", _update)
    monkeypatch.setattr(
        "general_beckman.apply._spawn_workflow_advance_if_mission", _noop
    )

    verdict = PostHookVerdict(
        source_task_id=226, kind=KIND, passed=True, raw={"ok": True},
    )
    await _apply_posthook_verdict_locked(source, verdict)
    assert any(kw.get("status") == "completed" for _, kw in updates)
