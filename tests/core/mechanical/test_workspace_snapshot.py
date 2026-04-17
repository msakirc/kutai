import pytest
from unittest.mock import AsyncMock, patch
from src.core.mechanical.workspace_snapshot import snapshot_workspace


@pytest.mark.asyncio
async def test_snapshot_returns_dict_with_hashes_and_sha():
    """snapshot_workspace wraps the three underlying calls and returns a flat dict."""
    with patch("src.core.mechanical.workspace_snapshot.compute_workspace_hashes") as mock_hashes, \
         patch("src.core.mechanical.workspace_snapshot.get_commit_sha", new_callable=AsyncMock) as mock_sha, \
         patch("src.core.mechanical.workspace_snapshot.get_current_branch", new_callable=AsyncMock) as mock_branch, \
         patch("src.core.mechanical.workspace_snapshot.save_workspace_snapshot", new_callable=AsyncMock) as mock_save:
        mock_hashes.return_value = {"a.py": "abc"}
        mock_sha.return_value = "deadbeef"
        mock_branch.return_value = "main"
        result = await snapshot_workspace(mission_id=1, task_id=42, workspace_path="/tmp/ws")
        assert result["hashes"] == {"a.py": "abc"}
        assert result["commit_sha"] == "deadbeef"
        assert result["branch"] == "main"
        mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_returns_none_on_exception():
    """Snapshot failures are non-fatal — return None, caller continues."""
    with patch("src.core.mechanical.workspace_snapshot.compute_workspace_hashes",
               side_effect=OSError("disk gone")):
        result = await snapshot_workspace(mission_id=1, task_id=42, workspace_path="/nonexistent")
        assert result is None
