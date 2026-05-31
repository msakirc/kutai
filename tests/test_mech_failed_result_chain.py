"""A failed mechanical validator's verdict payload must survive the
orchestrator → route_result → rewrite chain so the check-fail-goes-terminal
rule (rewrite Rule 0c) can actually fire.

mission_79 #225576/#227677 (2026-05-31): rewrite Rule 0c was taught to convert
a mechanical check-fail (Action(status="failed", result={"ok": False, ...})) into
a terminal Complete so the validator task stops self-retrying to DLQ. But the
orchestrator's _mech_action_to_result DROPPED action.result on the failed path
({"status":"failed","error":...}), so rewrite never saw the verdict-shaped
result, the discriminator fell through, and the validator still DLQ'd. The
rewrite unit test passed only because it synthesised a result the production
path never produced. This integration test runs the REAL orchestrator mapper so
the two halves can't drift apart again.
"""
from __future__ import annotations

import json

from mr_roboto.actions import Action
from src.core.orchestrator import _mech_action_to_result


def test_failed_action_carries_result():
    a = Action(status="failed", error="check failed",
               result={"ok": False, "problems": ["attempted_solutions[0] is not an object"]})
    d = _mech_action_to_result(a)
    assert d["status"] == "failed"
    assert "result" in d, "failed mechanical Action must carry its verdict result"
    assert json.loads(d["result"])["ok"] is False


def test_failed_validator_chain_completes_validator_not_retry():
    from general_beckman.result_router import (
        route_result, Complete, Failed, PostHookVerdict,
    )
    from general_beckman.rewrite import rewrite_actions

    a = Action(status="failed", error="prior_art_min_coverage: problems=[...]",
               result={"ok": False, "problems": ["attempted_solutions[0] is not an object"]})
    agent_result = _mech_action_to_result(a)

    task = {"id": 500, "mission_id": 79, "agent_type": "mechanical", "context": "{}"}
    ctx = {"source_task_id": 225583, "posthook_kind": "prior_art_min_coverage"}

    actions = route_result(task, agent_result)
    out = rewrite_actions(task, ctx, actions)

    # Validator goes terminal (no self-retry/DLQ); producer carries the verdict.
    assert any(isinstance(x, Complete) and x.task_id == 500 for x in out)
    assert not any(isinstance(x, Failed) for x in out)
    assert any(isinstance(x, PostHookVerdict) and x.passed is False for x in out)


def test_failed_executor_error_still_no_result_stays_retryable():
    # A genuine executor crash carries an empty result dict — must NOT be
    # mistaken for a check verdict; it stays a retryable Failed.
    from general_beckman.result_router import route_result, Failed
    from general_beckman.rewrite import rewrite_actions

    a = Action(status="failed", error="no paths supplied")  # result defaults to {}
    agent_result = _mech_action_to_result(a)
    task = {"id": 501, "mission_id": 79, "agent_type": "mechanical", "context": "{}"}
    ctx = {"source_task_id": 225583, "posthook_kind": "prior_art_min_coverage"}
    out = rewrite_actions(task, ctx, route_result(task, agent_result))
    assert any(isinstance(x, Failed) and x.task_id == 501 for x in out)
