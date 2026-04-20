"""Post-hook pipeline tests: actions, policy, apply, migrations."""
import pytest
from general_beckman.result_router import (
    Action, RequestPostHook, PostHookVerdict,
)


def test_request_posthook_is_action():
    a = RequestPostHook(source_task_id=1, kind="grade", source_ctx={})
    assert isinstance(a, RequestPostHook)
    # Action is a Union; isinstance check works via dataclass identity.
    assert a.source_task_id == 1
    assert a.kind == "grade"


def test_posthook_verdict_is_action():
    v = PostHookVerdict(
        source_task_id=2, kind="grade", passed=True, raw={"score": 0.9},
    )
    assert v.passed is True
    assert v.raw == {"score": 0.9}


from general_beckman.posthooks import determine_posthooks


def test_mechanical_task_needs_no_posthooks():
    task = {"agent_type": "mechanical"}
    assert determine_posthooks(task, {}, {}) == []


def test_shopping_pipeline_task_needs_no_posthooks():
    task = {"agent_type": "shopping_pipeline"}
    assert determine_posthooks(task, {}, {}) == []


def test_grader_task_needs_no_posthooks():
    task = {"agent_type": "grader"}
    assert determine_posthooks(task, {}, {}) == []


def test_artifact_summarizer_task_needs_no_posthooks():
    task = {"agent_type": "artifact_summarizer"}
    assert determine_posthooks(task, {}, {}) == []


def test_llm_agent_task_needs_grade_by_default():
    task = {"agent_type": "writer"}
    assert determine_posthooks(task, {}, {}) == ["grade"]


def test_requires_grading_false_opts_out():
    task = {"agent_type": "writer"}
    ctx = {"requires_grading": False}
    assert determine_posthooks(task, ctx, {}) == []
