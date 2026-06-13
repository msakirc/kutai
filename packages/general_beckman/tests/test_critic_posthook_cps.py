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


# ──────────────────────────────────────────────────────────────────────────
# SP6 T2 FIX 1 (B1 regression guard) — a completed critic child whose output
# is GARBAGE / non-enum / empty must FAIL CLOSED (veto), not default to pass.
# These MUST fail before parse_verdict_strict is wired in.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json at all", '{"foo": 1}'])
async def test_critic_resume_garbage_fails_closed(content):
    captured = {}
    async def _fake_apply(child_task, verdict):
        captured["verdict"] = verdict
    result = {"result": {"content": content}}
    state = {"source_task_id": 42, "action_name": "git_commit"}
    with patch.object(pc, "_apply_posthook_verdict", _fake_apply), \
         patch.object(pc, "_persist_critic_log", AsyncMock()):
        await pc._critic_resume(child_task_id=99, result=result, state=state)
    assert captured["verdict"].kind == "critic_gate"
    assert captured["verdict"].passed is False, (
        f"garbage critic output {content!r} must fail CLOSED, not proceed unreviewed"
    )


@pytest.mark.asyncio
async def test_critic_resume_empty_fails_closed():
    captured = {}
    async def _fake_apply(child_task, verdict):
        captured["verdict"] = verdict
    result = {"result": {"content": ""}}
    state = {"source_task_id": 42, "action_name": "git_commit"}
    with patch.object(pc, "_apply_posthook_verdict", _fake_apply), \
         patch.object(pc, "_persist_critic_log", AsyncMock()):
        await pc._critic_resume(child_task_id=99, result=result, state=state)
    assert captured["verdict"].passed is False


# ──────────────────────────────────────────────────────────────────────────
# SP6 T2 FIX 2 — veto reasons must reach the founder-visible DLQ error.
# _apply_z1_mechanical_verdict builds error_detail from raw.get("error"),
# never raw.get("reasons"); the verdict must therefore carry reasons under
# "error" too so the reason text survives into the DLQ row.
# ──────────────────────────────────────────────────────────────────────────
def test_critic_veto_reason_in_error():
    v = pc._make_critic_verdict(1, False, ["leaks a token"])
    assert "leaks a token" in (v.raw.get("error") or ""), (
        f"veto reason lost from DLQ-visible error: {v.raw!r}"
    )


def test_critic_pass_has_no_error_key():
    v = pc._make_critic_verdict(1, True, [])
    assert not v.raw.get("error"), "a passing critic verdict must not carry an error"
