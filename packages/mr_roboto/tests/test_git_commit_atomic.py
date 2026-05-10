"""Z10 T1A — atomic git_commit ↔ git_push tests.

Behaviour matrix:

| push | push_ok | allow_orphan | expected status | committed | pushed | reset called |
|------|---------|--------------|-----------------|-----------|--------|--------------|
| F    | n/a     | F            | completed       | True      | False  | False        |
| T    | T       | F            | completed       | True      | True   | False        |
| T    | F       | F            | failed          | False     | False  | True         |
| T    | F       | T            | partial         | True      | False  | False        |
"""
import pytest
from unittest.mock import AsyncMock, patch

from mr_roboto.git_commit import auto_commit


def _patches(commit_out="[main abc1234] msg",
             push_ret=(0, "", ""),
             head_sha="abc1234",
             reset_ret=(0, "", "")):
    return (
        patch("mr_roboto.git_commit.ensure_git_repo", new_callable=AsyncMock),
        patch("mr_roboto.git_commit.git_commit", new_callable=AsyncMock,
              return_value=commit_out),
        patch("mr_roboto.git_commit.git_push", new_callable=AsyncMock,
              return_value=push_ret),
        patch("mr_roboto.git_commit.git_reset_soft_one", new_callable=AsyncMock,
              return_value=reset_ret),
        patch("mr_roboto.git_commit.git_head_sha", new_callable=AsyncMock,
              return_value=head_sha),
        patch("mr_roboto.git_commit.get_mission_workspace_relative",
              return_value="missions/1"),
    )


@pytest.mark.asyncio
async def test_no_push_keeps_legacy_completed_behavior():
    p1, p2, p3, p4, p5, p6 = _patches()
    with p1, p2 as commit, p3 as push, p4 as reset, p5, p6:
        out = await auto_commit({"id": 1, "mission_id": 1, "title": "t"}, {})
    assert out["status"] == "completed"
    assert out["committed"] is True
    assert out["pushed"] is False
    assert out["commit_sha"] == "abc1234"
    push.assert_not_awaited()
    reset.assert_not_awaited()


@pytest.mark.asyncio
async def test_push_succeeds_marks_pushed_true():
    p1, p2, p3, p4, p5, p6 = _patches(push_ret=(0, "ok", ""))
    task = {"id": 2, "mission_id": 1, "title": "t",
            "payload": {"action": "git_commit", "push": True}}
    with p1, p2, p3 as push, p4 as reset, p5, p6:
        out = await auto_commit(task, {})
    assert out["status"] == "completed"
    assert out["pushed"] is True
    assert out["committed"] is True
    push.assert_awaited_once()
    reset.assert_not_awaited()


@pytest.mark.asyncio
async def test_push_fails_default_rolls_back_local_commit():
    p1, p2, p3, p4, p5, p6 = _patches(push_ret=(1, "", "auth failure"))
    task = {"id": 3, "mission_id": 1, "title": "t",
            "payload": {"action": "git_commit", "push": True}}
    with p1, p2, p3 as push, p4 as reset, p5, p6:
        out = await auto_commit(task, {})
    push.assert_awaited_once()
    reset.assert_awaited_once()
    assert out["status"] == "failed"
    assert out["pushed"] is False
    assert out["committed"] is False
    assert "auth failure" in (out["error"] or "")
    # commit_sha is preserved so the caller can audit which commit was reverted
    assert out["commit_sha"] == "abc1234"


@pytest.mark.asyncio
async def test_push_fails_with_allow_orphan_keeps_local_commit():
    p1, p2, p3, p4, p5, p6 = _patches(push_ret=(1, "", "network down"))
    task = {"id": 4, "mission_id": 1, "title": "t",
            "payload": {"action": "git_commit", "push": True, "allow_orphan": True}}
    with p1, p2, p3 as push, p4 as reset, p5, p6:
        out = await auto_commit(task, {})
    push.assert_awaited_once()
    reset.assert_not_awaited()
    assert out["status"] == "partial"
    assert out["pushed"] is False
    assert out["committed"] is True
    assert out["commit_sha"] == "abc1234"
    assert "network down" in (out["error"] or "")


@pytest.mark.asyncio
async def test_push_fails_and_rollback_fails_marks_hard_failure():
    p1, p2, p3, p4, p5, p6 = _patches(
        push_ret=(1, "", "auth"),
        reset_ret=(1, "", "reset broke"),
    )
    task = {"id": 5, "mission_id": 1, "title": "t",
            "payload": {"action": "git_commit", "push": True}}
    with p1, p2, p3, p4, p5, p6:
        out = await auto_commit(task, {})
    assert out["status"] == "failed"
    assert "rollback failed" in (out["error"] or "")


@pytest.mark.asyncio
async def test_empty_diff_returns_completed_no_push_attempted():
    p1, p2, p3, p4, p5, p6 = _patches(
        commit_out="Nothing to commit, working tree clean",
    )
    task = {"id": 6, "mission_id": 1, "title": "t",
            "payload": {"action": "git_commit", "push": True}}
    with p1, p2, p3 as push, p4, p5, p6:
        out = await auto_commit(task, {})
    assert out["status"] == "completed"
    assert out["empty"] is True
    assert out["committed"] is False
    push.assert_not_awaited()
