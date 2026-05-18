"""Z7 B3 — incident-review gate must receive the drafted status text.

Regression test for the "Critical #3" bug: incident/draft_update returns
the drafted customer-facing text inside its result dict, but the draft
never reached the incident_update_review posthook, so the founder-review
gate silently skipped every time — publishing status updates WITHOUT
founder review.

These tests exercise the REAL rewrite.py -> apply.py seam (no mocking of
the merge): a completed incident/draft_update task whose raw result is a
mr_roboto-style nested JSON blob must produce a RequestPostHook whose
source_ctx carries the draft, and the apply.py payload builder must place
that draft where the review-gate handler reads it.
"""
import json

from general_beckman.result_router import Complete, RequestPostHook
from general_beckman.rewrite import rewrite_actions
from general_beckman.apply import _posthook_agent_and_payload
from general_beckman.posthooks import determine_posthooks


def test_mechanical_task_honours_explicit_post_hooks():
    """A mechanical task that declares post_hooks must still spawn them.

    determine_posthooks skips the default ``grade`` hook for mechanical
    agents, but an EXPLICITLY declared post-hook (e.g. incident_update_review
    on the incident/draft_update step) must not be silently dropped — that
    is the gate that keeps the founder in the loop.
    """
    ctx = {
        "agent_type": "mechanical",
        "payload": {"action": "incident/draft_update"},
        "post_hooks": ["incident_update_review"],
    }
    task = {"id": 1, "mission_id": 9, "agent_type": "mechanical",
            "context": json.dumps(ctx)}
    kinds = determine_posthooks(task, ctx, {})
    assert "incident_update_review" in kinds, (
        "explicit post-hook on a mechanical task was dropped — the "
        "incident-review gate never fires"
    )
    # grade is still skipped for mechanical (no judge-of-judge)
    assert "grade" not in kinds


def _draft_update_task() -> tuple[dict, dict]:
    """A completed mechanical incident/draft_update workflow step."""
    task_ctx = {
        "payload": {"action": "incident/draft_update"},
        "posthook_kind": None,
        # the step declares the review gate as a post-hook
        "post_hooks": ["incident_update_review"],
        "incident_id": 7,
        "product_id": "acme",
        "status_kind": "investigating",
    }
    task = {
        "id": 501,
        "mission_id": 90,
        "agent_type": "mechanical",
        "context": json.dumps(task_ctx),
    }
    return task, task_ctx


def test_draft_reaches_posthook_source_ctx():
    """incident/draft_update result.draft must land in RequestPostHook.source_ctx.

    Exercises the real rewrite_actions merge. The mechanical executor's
    Action.result is wrapped by the orchestrator into
    {"status": "completed", "result": <json-string>} — the draft is nested
    inside that JSON string, NOT a top-level raw key.
    """
    task, task_ctx = _draft_update_task()

    inner_result = {
        "status": "ok",
        "draft": "We are investigating an issue affecting checkout.",
        "incident_id": 7,
        "product_id": "acme",
        "status_kind": "investigating",
        "redaction_applied": True,
    }
    raw = {"status": "completed", "result": json.dumps(inner_result)}
    complete = Complete(task_id=501, result=raw["result"], raw=raw)

    actions = rewrite_actions(task, task_ctx, [complete])

    posthooks = [a for a in actions if isinstance(a, RequestPostHook)]
    review = [p for p in posthooks if p.kind == "incident_update_review"]
    assert review, "incident_update_review posthook was not emitted"
    src_ctx = review[0].source_ctx
    assert src_ctx.get("draft") == inner_result["draft"], (
        "draft text did not reach the review-gate source_ctx — the gate "
        "would silently skip and publish without founder review"
    )


def test_apply_payload_carries_draft_to_review_handler():
    """apply.py's posthook payload builder must surface the merged draft."""
    task, task_ctx = _draft_update_task()
    inner_result = {
        "status": "ok",
        "draft": "We have identified the cause and are deploying a fix.",
        "incident_id": 7,
        "product_id": "acme",
        "status_kind": "identified",
    }
    raw = {"status": "completed", "result": json.dumps(inner_result)}
    complete = Complete(task_id=501, result=raw["result"], raw=raw)

    actions = rewrite_actions(task, task_ctx, [complete])
    review = next(
        a for a in actions
        if isinstance(a, RequestPostHook) and a.kind == "incident_update_review"
    )

    agent_type, payload = _posthook_agent_and_payload(review, task, review.source_ctx)
    assert agent_type == "mechanical"
    assert payload["payload"]["draft"] == inner_result["draft"], (
        "review-gate task payload has empty draft — incident_update_review "
        "handler will return status=skip and the gate is bypassed"
    )


def test_undrafted_review_still_skips():
    """No draft anywhere → review gate has nothing, payload draft stays empty.

    Confirms the gate is not unconditionally open: when the draft genuinely
    is missing, the payload carries an empty draft and the handler skips.
    """
    task, task_ctx = _draft_update_task()
    # mechanical result with NO draft field at all
    inner_result = {"status": "ok", "incident_id": 7, "product_id": "acme"}
    raw = {"status": "completed", "result": json.dumps(inner_result)}
    complete = Complete(task_id=501, result=raw["result"], raw=raw)

    actions = rewrite_actions(task, task_ctx, [complete])
    review = next(
        a for a in actions
        if isinstance(a, RequestPostHook) and a.kind == "incident_update_review"
    )
    _agent, payload = _posthook_agent_and_payload(review, task, review.source_ctx)
    assert payload["payload"]["draft"] == "", (
        "draft should be empty when the executor produced none"
    )
