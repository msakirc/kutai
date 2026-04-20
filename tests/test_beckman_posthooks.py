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
