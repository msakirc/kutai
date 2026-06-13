import pytest
from unittest.mock import AsyncMock, patch
import general_beckman.posthook_continuations as pc


@pytest.mark.asyncio
async def test_critic_resume_veto_builds_failing_verdict():
    captured = {}
    async def _fake_apply(child_task, verdict):
        captured["verdict"] = verdict
    result = {"result": {"content": '{"verdict": "veto", "reasons": ["leaks a token"]}'}}
    state = {"source_task_id": 42, "action_name": "git_commit", "mission_id": 7}
    with patch.object(pc, "_apply_posthook_verdict", _fake_apply), \
         patch.object(pc, "_persist_critic_log", AsyncMock()):
        await pc._critic_resume(child_task_id=99, result=result, state=state)
    v = captured["verdict"]
    assert v.kind == "critic_gate"
    assert v.passed is False
    assert v.source_task_id == 42


@pytest.mark.asyncio
async def test_critic_resume_pass_builds_passing_verdict():
    captured = {}
    async def _fake_apply(child_task, verdict):
        captured["verdict"] = verdict
    result = {"result": {"content": '{"verdict": "pass", "reasons": []}'}}
    state = {"source_task_id": 42, "action_name": "git_commit"}
    with patch.object(pc, "_apply_posthook_verdict", _fake_apply), \
         patch.object(pc, "_persist_critic_log", AsyncMock()):
        await pc._critic_resume(child_task_id=99, result=result, state=state)
    assert captured["verdict"].passed is True


@pytest.mark.asyncio
async def test_critic_resume_err_fail_closed():
    captured = {}
    async def _fake_apply(child_task, verdict):
        captured["verdict"] = verdict
    state = {"source_task_id": 42, "action_name": "notify_user"}
    with patch.object(pc, "_apply_posthook_verdict", _fake_apply), \
         patch.object(pc, "_persist_critic_log", AsyncMock()):
        await pc._critic_resume_err(child_task_id=99, result={"error": "no candidates"}, state=state)
    assert captured["verdict"].kind == "critic_gate"
    assert captured["verdict"].passed is False
