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
    # SP6 T3: a matching payload_hash makes the drift guard pass so this test
    # exercises the *confirm* path (veto honoured), NOT the re-gate path. An
    # empty payload_hash now fail-closes into re-gate (see
    # test_git_commit_pass2_empty_hash_regate), so we stub _hash_payload to a
    # fixed value and set the context hash to match.
    monkeypatch.setattr("mr_roboto.critic_gate._hash_payload", lambda p: "MATCH")
    ctx = {"critic_verdict": {"verdict": "veto", "reasons": ["leaks token"], "payload_hash": "MATCH"}}
    task = {"id": 55, "title": "t", "mission_id": 3, "context": json.dumps(ctx),
            "payload": {"action": "git_commit", "workspace_path": str(tmp_path)}}
    action = await mr_roboto.run(task)
    assert action.status == "failed"
    assert "veto" in (action.error or "").lower()
    auto.assert_not_called()


@pytest.mark.asyncio
async def test_git_commit_pass2_approve_commits(monkeypatch, tmp_path):
    """SP6 T3 TEST 1: pass-2 with a matching-hash PASS verdict → LLM-free
    confirm_gate returns pass → real commit happens."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    monkeypatch.setattr("src.tools.git_ops._resolve_repo", lambda p: str(tmp_path))
    monkeypatch.setattr("src.tools.git_ops._run_git", AsyncMock(return_value=(0, "stat", "")))
    monkeypatch.setattr("src.tools.git_ops.ensure_git_repo", AsyncMock())
    auto = AsyncMock(return_value={"committed": True, "empty": False, "commit_sha": ""})
    monkeypatch.setattr("mr_roboto.auto_commit", auto, raising=False)
    # Force the drift guard to PASS: _capture's recomputed hash == context hash.
    monkeypatch.setattr("mr_roboto.critic_gate._hash_payload", lambda p: "MATCH")
    ctx = {"critic_verdict": {"verdict": "pass", "reasons": [], "payload_hash": "MATCH"}}
    task = {"id": 55, "title": "t", "mission_id": 3, "context": json.dumps(ctx),
            "payload": {"action": "git_commit", "workspace_path": str(tmp_path)}}
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    auto.assert_awaited_once()


@pytest.mark.asyncio
async def test_git_commit_pass2_drift_regate_increments(monkeypatch, tmp_path):
    """SP6 T3 TEST 2(a): hash drift with critic_regate_n absent → re-pend with
    critic_regate_n==1, critic_verdict popped, needs_clarification; no commit."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    monkeypatch.setattr("src.tools.git_ops._resolve_repo", lambda p: str(tmp_path))
    monkeypatch.setattr("src.tools.git_ops._run_git", AsyncMock(return_value=(0, "stat", "")))
    monkeypatch.setattr("src.tools.git_ops.ensure_git_repo", AsyncMock())
    auto = AsyncMock()
    monkeypatch.setattr("mr_roboto.auto_commit", auto, raising=False)
    upd = AsyncMock()
    monkeypatch.setattr("mr_roboto.update_task", upd, raising=False)
    # _capture recomputes "NEW"; context carries "OLD" → mismatch → re-gate.
    monkeypatch.setattr("mr_roboto.critic_gate._hash_payload", lambda p: "NEW")
    ctx = {"critic_verdict": {"verdict": "pass", "reasons": [], "payload_hash": "OLD"}}
    task = {"id": 55, "title": "t", "mission_id": 3, "context": json.dumps(ctx),
            "payload": {"action": "git_commit", "workspace_path": str(tmp_path)}}
    action = await mr_roboto.run(task)
    assert action.status == "needs_clarification"
    auto.assert_not_called()
    # update_task must have re-pended with the incremented re-gate counter and
    # the stale verdict removed.
    repend = [c for c in upd.await_args_list if c.kwargs.get("status") == "pending"]
    assert repend, "expected a pending re-pend update"
    new_ctx = json.loads(repend[-1].kwargs["context"])
    assert new_ctx["critic_regate_n"] == 1
    assert "critic_verdict" not in new_ctx


@pytest.mark.asyncio
async def test_git_commit_pass2_drift_regate_exhausts(monkeypatch, tmp_path):
    """SP6 T3 TEST 2(b): hash drift with critic_regate_n already at 2 →
    termination: failed with 're-gate exhausted', git reset issued, no commit.
    Proves the loop terminates at n==2 instead of re-pending forever."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    monkeypatch.setattr("src.tools.git_ops._resolve_repo", lambda p: str(tmp_path))
    run_git = AsyncMock(return_value=(0, "stat", ""))
    monkeypatch.setattr("src.tools.git_ops._run_git", run_git)
    monkeypatch.setattr("src.tools.git_ops.ensure_git_repo", AsyncMock())
    auto = AsyncMock()
    monkeypatch.setattr("mr_roboto.auto_commit", auto, raising=False)
    upd = AsyncMock()
    monkeypatch.setattr("mr_roboto.update_task", upd, raising=False)
    monkeypatch.setattr("mr_roboto.critic_gate._hash_payload", lambda p: "NEW")
    ctx = {"critic_verdict": {"verdict": "pass", "reasons": [], "payload_hash": "OLD"},
           "critic_regate_n": 2}
    task = {"id": 55, "title": "t", "mission_id": 3, "context": json.dumps(ctx),
            "payload": {"action": "git_commit", "workspace_path": str(tmp_path)}}
    action = await mr_roboto.run(task)
    assert action.status == "failed"
    assert "re-gate exhausted" in (action.error or "")
    auto.assert_not_called()
    # `git reset` must have been issued to unstage the kept-changing tree.
    reset_calls = [c for c in run_git.await_args_list if "reset" in c.args[0]]
    assert reset_calls, "expected `git reset` rollback on re-gate exhaustion"
    # NOT re-pended — terminal.
    assert not [c for c in upd.await_args_list if c.kwargs.get("status") == "pending"]


@pytest.mark.asyncio
async def test_git_commit_pass2_empty_hash_regate(monkeypatch, tmp_path):
    """SP6 T3 TEST 3: pass-2 with an EMPTY payload_hash while the gate is
    enabled must FAIL CLOSED into re-gate (not silently trust → confirm/commit).
    """
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    monkeypatch.setattr("src.tools.git_ops._resolve_repo", lambda p: str(tmp_path))
    monkeypatch.setattr("src.tools.git_ops._run_git", AsyncMock(return_value=(0, "stat", "")))
    monkeypatch.setattr("src.tools.git_ops.ensure_git_repo", AsyncMock())
    auto = AsyncMock()
    monkeypatch.setattr("mr_roboto.auto_commit", auto, raising=False)
    upd = AsyncMock()
    monkeypatch.setattr("mr_roboto.update_task", upd, raising=False)
    # Even a PASS verdict with an empty anchor must NOT commit — empty hash is
    # treated as drift.
    ctx = {"critic_verdict": {"verdict": "pass", "reasons": [], "payload_hash": ""}}
    task = {"id": 55, "title": "t", "mission_id": 3, "context": json.dumps(ctx),
            "payload": {"action": "git_commit", "workspace_path": str(tmp_path)}}
    action = await mr_roboto.run(task)
    assert action.status == "needs_clarification"
    auto.assert_not_called()
    repend = [c for c in upd.await_args_list if c.kwargs.get("status") == "pending"]
    assert repend, "empty hash must re-pend (fail-closed), not confirm"
    new_ctx = json.loads(repend[-1].kwargs["context"])
    assert new_ctx["critic_regate_n"] == 1
    assert "critic_verdict" not in new_ctx
