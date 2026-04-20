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
