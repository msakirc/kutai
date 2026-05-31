"""A mechanical post-hook validator that RAN its check and returned a negative
verdict must go TERMINAL — its own task must not be retried/DLQ'd.

mission_79 (2026-05-31): prior_art_min_coverage (#227677) and
verify_interview_script_shape (#225576) returned Action(status="failed",
result={"ok": False, ...}) on a failed check. rewrite Rule 0c synthesised the
producer PostHookVerdict(passed=False) (correct — the producer re-runs with
feedback) BUT also kept the Failed action, so the VALIDATOR task itself went
through normal retry/DLQ handling: it re-ran the same deterministic check
against the same artifact 5× and DLQ'd as "Worker attempts exceeded: 5/6"
noise. The fail is a VERDICT, not a failure of the validator task.

Fix: a mechanical check-fail (verdict-shaped result carrying ok/passed/all_ok)
converts the validator's own action to a terminal Complete; the producer
verdict is unchanged. An executor error (exception / "no paths supplied" — no
verdict-shaped result) stays a retryable Failed.
"""
from __future__ import annotations

import json

from general_beckman.rewrite import rewrite_actions
from general_beckman.result_router import Complete, Failed, PostHookVerdict


def _validator(result_payload, *, error="check failed"):
    task = {"id": 500, "mission_id": 79, "context": "{}", "agent_type": "mechanical"}
    ctx = {"source_task_id": 225583, "posthook_kind": "prior_art_min_coverage"}
    raw = {
        "status": "failed",
        "error": error,
        "result": json.dumps(result_payload) if result_payload is not None else None,
    }
    return task, ctx, [Failed(task_id=500, error=error, raw=raw)]


def test_checkfail_completes_validator_and_verdicts_producer():
    task, ctx, actions = _validator(
        {"ok": False, "problems": ["attempted_solutions[0] is not an object"]}
    )
    out = rewrite_actions(task, ctx, actions)

    # Validator task goes terminal — no self-retry/DLQ.
    assert any(isinstance(a, Complete) and a.task_id == 500 for a in out)
    assert not any(isinstance(a, Failed) for a in out)

    # Producer still gets the fail verdict and re-runs with feedback.
    verdicts = [a for a in out if isinstance(a, PostHookVerdict)]
    assert len(verdicts) == 1
    assert verdicts[0].source_task_id == 225583
    assert verdicts[0].passed is False


def test_executor_error_stays_retryable():
    # status=failed with NO verdict-shaped result = the executor crashed, not a
    # check verdict — keep it retryable (transient input/IO can clear).
    task, ctx, actions = _validator(None, error="no paths supplied")
    out = rewrite_actions(task, ctx, actions)

    assert any(isinstance(a, Failed) and a.task_id == 500 for a in out)
    # Producer still gets a fail verdict so it never stalls 'ungraded'.
    assert any(isinstance(a, PostHookVerdict) and a.passed is False for a in out)


def test_checkfail_passed_verdict_unaffected():
    # A PASSING mechanical check returns Complete (status=completed), handled by
    # the earlier Complete→PostHookVerdict rule — Rule 0c only fires on
    # Failed/Exhausted, so a pass must never be misrouted here.
    task = {"id": 500, "mission_id": 79, "context": "{}", "agent_type": "mechanical"}
    ctx = {"source_task_id": 225583, "posthook_kind": "prior_art_min_coverage"}
    raw = {"status": "completed", "result": json.dumps({"ok": True})}
    out = rewrite_actions(task, ctx, [Complete(task_id=500, result=raw["result"], raw=raw)])
    verdicts = [a for a in out if isinstance(a, PostHookVerdict)]
    assert len(verdicts) == 1 and verdicts[0].passed is True
