import pytest
from unittest.mock import AsyncMock, patch
from salako.git_commit import auto_commit


@pytest.mark.asyncio
async def test_auto_commit_runs_git_commit_on_success():
    """Happy path: ensure_git_repo + git_commit invoked, info log on success."""
    with patch("salako.git_commit.ensure_git_repo", new_callable=AsyncMock) as mock_ensure, \
         patch("salako.git_commit.git_commit", new_callable=AsyncMock) as mock_commit, \
         patch("salako.git_commit.get_mission_workspace_relative", return_value="missions/7"):
        mock_commit.return_value = "[main abc1234] Task #42: title"
        task = {"id": 42, "mission_id": 7, "title": "do the thing"}
        await auto_commit(task, {})
        mock_ensure.assert_awaited_once_with("missions/7")
        mock_commit.assert_awaited_once()
        args, kwargs = mock_commit.call_args
        assert args[0].startswith("Task #42: do the thing")
        assert kwargs["path"] == "missions/7"


@pytest.mark.asyncio
async def test_auto_commit_skips_on_nothing_to_commit():
    """When git reports 'Nothing to commit', skip the info log (no-op happy path)."""
    with patch("salako.git_commit.ensure_git_repo", new_callable=AsyncMock), \
         patch("salako.git_commit.git_commit", new_callable=AsyncMock) as mock_commit, \
         patch("salako.git_commit.get_mission_workspace_relative", return_value="missions/1"), \
         patch("salako.git_commit.logger") as mock_logger:
        mock_commit.return_value = "Nothing to commit, working tree clean"
        await auto_commit({"id": 1, "mission_id": 1, "title": "x"}, {})
        mock_logger.info.assert_not_called()


@pytest.mark.asyncio
async def test_auto_commit_swallows_exceptions():
    """Any exception must be swallowed — auto_commit is best-effort."""
    with patch("salako.git_commit.ensure_git_repo", new_callable=AsyncMock,
               side_effect=RuntimeError("boom")), \
         patch("salako.git_commit.get_mission_workspace_relative", return_value="missions/1"):
        # Must not raise
        await auto_commit({"id": 1, "mission_id": 1, "title": "x"}, {})


@pytest.mark.asyncio
async def test_auto_commit_uses_mission_workspace_path_when_mission_id_present():
    """If mission_id is present, path comes from get_mission_workspace_relative.
    If missing, path is empty string (repo root)."""
    with patch("salako.git_commit.ensure_git_repo", new_callable=AsyncMock) as mock_ensure, \
         patch("salako.git_commit.git_commit", new_callable=AsyncMock) as mock_commit, \
         patch("salako.git_commit.get_mission_workspace_relative",
               return_value="missions/42") as mock_path:
        mock_commit.return_value = "ok"
        await auto_commit({"id": 1, "mission_id": 42, "title": "t"}, {})
        mock_path.assert_called_once_with(42)
        mock_ensure.assert_awaited_once_with("missions/42")

    with patch("salako.git_commit.ensure_git_repo", new_callable=AsyncMock) as mock_ensure2, \
         patch("salako.git_commit.git_commit", new_callable=AsyncMock) as mock_commit2, \
         patch("salako.git_commit.get_mission_workspace_relative") as mock_path2:
        mock_commit2.return_value = "ok"
        await auto_commit({"id": 1, "title": "t"}, {})  # no mission_id
        mock_path2.assert_not_called()
        mock_ensure2.assert_awaited_once_with("")
