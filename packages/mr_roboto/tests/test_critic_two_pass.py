import json
import pytest
from unittest.mock import AsyncMock, patch
import mr_roboto.critic_continuations as cc


@pytest.mark.asyncio
async def test_verdict_done_stamps_and_repends():
    updates = []
    async def _fake_update(task_id, **kw):
        updates.append((task_id, kw))
    async def _fake_get_task(tid):
        return {"id": tid, "context": json.dumps({"action": "git_commit"})}
    result = {"result": {"content": '{"verdict": "pass", "reasons": []}'}}
    state = {"gated_task_id": 55, "action_name": "git_commit",
             "mission_id": 3, "payload_hash": "abc123"}
    with patch.object(cc, "update_task", _fake_update), \
         patch.object(cc, "get_task", _fake_get_task), \
         patch.object(cc, "_persist_critic_log", AsyncMock()):
        await cc._verdict_done(child_task_id=99, result=result, state=state)
    assert updates
    tid, kw = updates[0]
    assert tid == 55
    assert kw["status"] == "pending"
    ctx = json.loads(kw["context"])
    assert ctx["critic_verdict"]["verdict"] == "pass"
    assert ctx["critic_verdict"]["payload_hash"] == "abc123"


@pytest.mark.asyncio
async def test_verdict_done_garbage_fails_closed():
    """Carry-over from T2: a garbage child verdict must stamp VETO, not pass."""
    updates = []
    async def _fake_update(task_id, **kw):
        updates.append((task_id, kw))
    async def _fake_get_task(tid):
        return {"id": tid, "context": json.dumps({})}
    result = {"result": {"content": "not json at all"}}
    state = {"gated_task_id": 55, "action_name": "git_commit", "payload_hash": "h"}
    with patch.object(cc, "update_task", _fake_update), \
         patch.object(cc, "get_task", _fake_get_task), \
         patch.object(cc, "_persist_critic_log", AsyncMock()):
        await cc._verdict_done(child_task_id=99, result=result, state=state)
    _, kw = updates[0]
    assert json.loads(kw["context"])["critic_verdict"]["verdict"] == "veto"


@pytest.mark.asyncio
async def test_verdict_err_fails_closed_gated_task():
    updates = []
    async def _fake_update(task_id, **kw):
        updates.append((task_id, kw))
    state = {"gated_task_id": 55, "action_name": "git_commit"}
    with patch.object(cc, "update_task", _fake_update), \
         patch.object(cc, "_persist_critic_log", AsyncMock()):
        await cc._verdict_err(child_task_id=99, result={"error": "no candidates"}, state=state)
    tid, kw = updates[0]
    assert tid == 55
    assert kw["status"] == "failed"
    assert "blocked" in kw["error"].lower()


import mr_roboto


@pytest.mark.asyncio
async def test_git_commit_pass1_parks_and_enqueues_critic(monkeypatch, tmp_path):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    enq = AsyncMock(return_value=1234)
    upd = AsyncMock()
    monkeypatch.setattr("mr_roboto.enqueue", enq, raising=False)
    monkeypatch.setattr("mr_roboto.update_task", upd, raising=False)
    monkeypatch.setattr("src.tools.git_ops._resolve_repo", lambda p: str(tmp_path))
    monkeypatch.setattr("src.tools.git_ops._run_git", AsyncMock(return_value=(0, "stat", "")))
    monkeypatch.setattr("src.tools.git_ops.ensure_git_repo", AsyncMock())
    auto = AsyncMock()
    monkeypatch.setattr("mr_roboto.auto_commit", auto, raising=False)
    task = {"id": 55, "title": "t", "mission_id": 3, "context": json.dumps({}),
            "payload": {"action": "git_commit", "workspace_path": str(tmp_path)}}
    action = await mr_roboto.run(task)
    assert action.status == "needs_clarification"
    assert enq.await_count == 1
    assert any(c.kwargs.get("status") == "waiting_human" for c in upd.await_args_list)
    auto.assert_not_called()


@pytest.mark.asyncio
async def test_git_commit_pass2_veto_blocks_commit(monkeypatch, tmp_path):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    monkeypatch.setattr("src.tools.git_ops._resolve_repo", lambda p: str(tmp_path))
    monkeypatch.setattr("src.tools.git_ops._run_git", AsyncMock(return_value=(0, "stat", "")))
    monkeypatch.setattr("src.tools.git_ops.ensure_git_repo", AsyncMock())
    auto = AsyncMock()
    monkeypatch.setattr("mr_roboto.auto_commit", auto, raising=False)
    ctx = {"critic_verdict": {"verdict": "veto", "reasons": ["leaks token"], "payload_hash": ""}}
    task = {"id": 55, "title": "t", "mission_id": 3, "context": json.dumps(ctx),
            "payload": {"action": "git_commit", "workspace_path": str(tmp_path)}}
    action = await mr_roboto.run(task)
    assert action.status == "failed"
    auto.assert_not_called()
