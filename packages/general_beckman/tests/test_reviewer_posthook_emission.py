"""Reviewer-failure routing — the EMISSION step (rewrite.py is_bookkeeping).

A workflow-STEP reviewer (i2p ``"agent": "reviewer"`` → agent_type="reviewer",
context carries workflow_step_id + a ``checks: [{kind: verify_review_verdict}]``
entry, NO source_task_id/posthook_kind) MUST emit its verify_review_verdict
post-hook when it finishes. Before the fix, rewrite.py's ``is_bookkeeping``
predicate classified EVERY agent_type=="reviewer" as bookkeeping — swallowing
the workflow-step reviewer too — so ``determine_posthooks`` never ran and the
whole reviewer-failure-routing subsystem was inert in prod.

These tests pin the rewrite-layer emission contract directly (no DB): a
finished workflow-step reviewer emits RequestPostHook(verify_review_verdict);
a genuine posthook CHILD (carries source_task_id + posthook_kind) does NOT —
that would cause posthook recursion (the SP3 70435cd1 intent we must keep).
"""
import json

from general_beckman.result_router import (
    Complete, RequestPostHook, MissionAdvance,
)
from general_beckman.rewrite import rewrite_actions


def _completed(task_id: int, review_result: dict) -> Complete:
    """A status=completed Complete carrying the reviewer's verdict JSON, shaped
    exactly as route_result produces from the agent envelope."""
    body = json.dumps(review_result)
    raw = {"status": "completed", "result": body}
    return Complete(task_id=task_id, result=body, raw=raw)


def test_workflow_step_reviewer_emits_verify_review_verdict():
    """The headline gap: a FINISHED workflow-step reviewer (FAIL verdict,
    workflow_step_id set, verify_review_verdict in checks, NO source_task_id)
    MUST emit a RequestPostHook(verify_review_verdict)."""
    review_result = {
        "status": "fail",
        "issues": [{"target_artifact": "requirements_spec",
                    "severity": "blocker", "problem": "no traceability"}],
    }
    task_ctx = {
        "workflow_step_id": "3.11",
        "is_workflow_step": True,
        "checks": [{"kind": "verify_review_verdict",
                    "payload": {"action": "verify_review_verdict"}}],
    }
    task = {
        "id": 4321,
        "mission_id": 90,
        "agent_type": "reviewer",
        "context": json.dumps(task_ctx),
    }

    actions = rewrite_actions(task, task_ctx, [_completed(4321, review_result)])

    posthooks = [a for a in actions if isinstance(a, RequestPostHook)]
    kinds = [p.kind for p in posthooks]
    assert "verify_review_verdict" in kinds, (
        "workflow-step reviewer did not emit verify_review_verdict — "
        "reviewer-failure routing is inert. emitted kinds=%r" % kinds
    )


def test_workflow_step_reviewer_is_not_bookkept():
    """A workflow-step reviewer is a real mission step — it must NOT be treated
    as bookkeeping. Beyond the post-hook, Rule 1 emits MissionAdvance for it
    (mission-bearing, non-bookkeeping completion)."""
    review_result = {"status": "fail", "issues": []}
    task_ctx = {
        "workflow_step_id": "3.11",
        "is_workflow_step": True,
        "checks": [{"kind": "verify_review_verdict",
                    "payload": {"action": "verify_review_verdict"}}],
    }
    task = {
        "id": 4322,
        "mission_id": 90,
        "agent_type": "reviewer",
        "context": json.dumps(task_ctx),
    }

    actions = rewrite_actions(task, task_ctx, [_completed(4322, review_result)])

    assert any(isinstance(a, MissionAdvance) for a in actions), (
        "workflow-step reviewer was bookkept — no MissionAdvance emitted"
    )
    assert any(
        isinstance(a, RequestPostHook) and a.kind == "verify_review_verdict"
        for a in actions
    )


def test_posthook_child_reviewer_stays_bookkeeping():
    """GUARD — the SP3 70435cd1 intent must NOT regress. A grade/code_review
    posthook CHILD runs as agent_type=="reviewer" too, but it carries BOTH
    source_task_id AND posthook_kind. It must stay bookkeeping — NOT emit any
    RequestPostHook (that would spawn a recursive grade→review→grade chain) and
    NOT emit MissionAdvance."""
    task_ctx = {
        # the producer's ctx is the source_ctx for the child; it may even
        # carry a workflow_step_id — the source_task_id+posthook_kind pair is
        # the authoritative "this is a posthook child" marker.
        "workflow_step_id": "3.11",
        "is_workflow_step": True,
        "source_task_id": 4000,
        "posthook_kind": "grade",
    }
    task = {
        "id": 4399,
        "mission_id": 90,
        "agent_type": "reviewer",
        "context": json.dumps(task_ctx),
    }
    # The grade child's Complete is a plain completion (no posthook_verdict
    # payload — that path was deleted in SP3; verdict flows via the durable
    # posthook.*.resume continuation, not via this rewrite layer).
    raw = {"status": "completed", "result": "graded ok"}
    complete = Complete(task_id=4399, result="graded ok", raw=raw)

    actions = rewrite_actions(task, task_ctx, [complete])

    assert not any(isinstance(a, RequestPostHook) for a in actions), (
        "posthook CHILD re-emitted a RequestPostHook — recursion guard "
        "regressed (was the SP3 70435cd1 fix)"
    )
    assert not any(isinstance(a, MissionAdvance) for a in actions), (
        "posthook CHILD emitted MissionAdvance — must stay bookkeeping"
    )


def test_bare_summarizer_child_stays_bookkeeping():
    """A summarizer posthook CHILD (no workflow-step markers) also stays
    bookkeeping — neither emit nor MissionAdvance."""
    task_ctx = {
        "source_task_id": 4001,
        "posthook_kind": "summary:phase",
    }
    task = {
        "id": 4400,
        "mission_id": 90,
        "agent_type": "summarizer",
        "context": json.dumps(task_ctx),
    }
    raw = {"status": "completed", "result": "a summary"}
    complete = Complete(task_id=4400, result="a summary", raw=raw)

    actions = rewrite_actions(task, task_ctx, [complete])

    assert not any(isinstance(a, RequestPostHook) for a in actions)
    assert not any(isinstance(a, MissionAdvance) for a in actions)
