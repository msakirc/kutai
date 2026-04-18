import pytest
from unittest.mock import AsyncMock, patch

import salako


@pytest.mark.asyncio
async def test_run_workspace_snapshot_happy_path():
    task = {
        "id": 1,
        "mission_id": 2,
        "payload": {"action": "workspace_snapshot", "workspace_path": "/ws"},
    }
    with patch("salako.snapshot_workspace", new_callable=AsyncMock) as mock_snap:
        mock_snap.return_value = {"hashes": {}, "commit_sha": "abc", "branch": "main"}
        action = await salako.run(task)
    assert action.status == "completed"
    assert action.result["commit_sha"] == "abc"
    mock_snap.assert_awaited_once_with(
        mission_id=2, task_id=1, workspace_path="/ws", repo_path=None
    )


@pytest.mark.asyncio
async def test_run_workspace_snapshot_failure_returns_failed_action():
    task = {
        "id": 1,
        "mission_id": 2,
        "payload": {"action": "workspace_snapshot", "workspace_path": "/ws"},
    }
    with patch("salako.snapshot_workspace", new_callable=AsyncMock) as mock_snap:
        mock_snap.return_value = None  # snapshot swallows errors → None
        action = await salako.run(task)
    assert action.status == "failed"
    assert action.error == "snapshot failed"


@pytest.mark.asyncio
async def test_run_git_commit_happy_path():
    task = {
        "id": 3,
        "mission_id": 4,
        "title": "commit me",
        "payload": {"action": "git_commit", "result": {"ok": True}},
    }
    with patch("salako.auto_commit", new_callable=AsyncMock) as mock_commit:
        action = await salako.run(task)
    assert action.status == "completed"
    mock_commit.assert_awaited_once()
    args, _ = mock_commit.call_args
    assert args[0] is task
    assert args[1] == {"ok": True}


@pytest.mark.asyncio
async def test_run_unknown_action_returns_failed():
    action = await salako.run({"id": 1, "payload": {"action": "not_a_thing"}})
    assert action.status == "failed"
    assert "unknown mechanical action" in (action.error or "")


@pytest.mark.asyncio
async def test_run_missing_payload_returns_failed():
    action = await salako.run({"id": 1})
    assert action.status == "failed"
