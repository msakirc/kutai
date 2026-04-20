"""Unit tests for general_beckman.rewrite — pure action-rewriting rules."""
import pytest
from general_beckman.rewrite import rewrite_actions
from general_beckman.result_router import (
    Complete, SpawnSubtasks, RequestClarification, Failed,
    MissionAdvance, CompleteWithReusedAnswer,
)


def _task(**kw):
    base = {"id": 1, "mission_id": None, "context": "{}", "agent_type": "coder"}
    base.update(kw)
    return base


def _ctx(**kw):
    return kw  # parse_context output is a plain dict


def test_mission_task_complete_emits_mission_advance():
    task = _task(id=10, mission_id=5)
    actions = [Complete(task_id=10, result="done", raw={"status": "completed"})]
    out = rewrite_actions(task, _ctx(), actions)
    assert any(isinstance(a, MissionAdvance) for a in out)


def test_non_mission_task_complete_unchanged():
    task = _task(id=10, mission_id=None)
    actions = [Complete(task_id=10, result="done", raw={})]
    out = rewrite_actions(task, _ctx(), actions)
    assert out == actions


def test_workflow_step_blocking_subtask_emission():
    task = _task(id=20, mission_id=1)
    ctx = _ctx(workflow_step=True, mission_id=1)
    actions = [SpawnSubtasks(parent_task_id=20, subtasks=[{"t": "x"}], raw={})]
    out = rewrite_actions(task, ctx, actions)
    assert len(out) == 1
    assert isinstance(out[0], Failed)
    assert "decompose" in out[0].error.lower()


def test_silent_task_clarify_becomes_failed():
    task = _task(id=30)
    ctx = _ctx(silent=True)
    actions = [RequestClarification(task_id=30, question="?", raw={})]
    out = rewrite_actions(task, ctx, actions)
    assert len(out) == 1
    assert isinstance(out[0], Failed)


def test_may_need_clarification_false_clarify_becomes_failed():
    task = _task(id=31)
    ctx = _ctx(may_need_clarification=False)
    actions = [RequestClarification(task_id=31, question="?", raw={})]
    out = rewrite_actions(task, ctx, actions)
    assert len(out) == 1
    assert isinstance(out[0], Failed)


def test_clarification_history_reused():
    task = _task(id=32)
    history = [{"question": "A?", "answer": "B"}]
    ctx = _ctx(clarification_history=history)
    actions = [RequestClarification(task_id=32, question="?", raw={})]
    out = rewrite_actions(task, ctx, actions)
    assert len(out) == 1
    assert isinstance(out[0], CompleteWithReusedAnswer)
    assert "A?" in out[0].result and "B" in out[0].result


from general_beckman.result_router import RequestPostHook


def test_writer_task_complete_emits_request_grade_posthook():
    task = {"id": 100, "mission_id": 5, "agent_type": "writer"}
    ctx = {}
    complete = Complete(task_id=100, result="out", iterations=1, metadata={}, raw={})
    out = rewrite_actions(task, ctx, [complete])
    # Expect: Complete, MissionAdvance, RequestPostHook(grade).
    kinds = [type(a).__name__ for a in out]
    assert "RequestPostHook" in kinds
    posthook = next(a for a in out if isinstance(a, RequestPostHook))
    assert posthook.kind == "grade"
    assert posthook.source_task_id == 100


def test_mechanical_task_complete_emits_no_posthook():
    task = {"id": 200, "mission_id": 5, "agent_type": "mechanical"}
    ctx = {"payload": {"action": "workflow_advance"}}
    complete = Complete(task_id=200, result="out", iterations=1, metadata={}, raw={})
    out = rewrite_actions(task, ctx, [complete])
    kinds = [type(a).__name__ for a in out]
    assert "RequestPostHook" not in kinds
    # workflow_advance mechanical also skips MissionAdvance (pre-existing guard).
    assert "MissionAdvance" not in kinds


from general_beckman.result_router import Complete, PostHookVerdict


def test_grader_task_complete_emits_posthook_verdict():
    task = {"id": 500, "mission_id": 1, "agent_type": "grader"}
    ctx = {}
    raw = {
        "status": "completed",
        "result": "grade json",
        "posthook_verdict": {
            "kind": "grade",
            "source_task_id": 100,
            "passed": True,
            "raw": {"score": 0.95},
        },
    }
    complete = Complete(task_id=500, result="grade json", iterations=1, metadata={}, raw=raw)
    out = rewrite_actions(task, ctx, [complete])
    kinds = [type(a).__name__ for a in out]
    assert "Complete" in kinds
    assert "PostHookVerdict" in kinds
    # Bookkeeping → no MissionAdvance.
    assert "MissionAdvance" not in kinds
    verdict = next(a for a in out if isinstance(a, PostHookVerdict))
    assert verdict.kind == "grade"
    assert verdict.source_task_id == 100
    assert verdict.passed is True
