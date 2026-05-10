"""Tests for mr_roboto.kill_preview_url — stop the tunnel + remove surface files."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from mr_roboto.kill_preview_url import kill_preview_url


@pytest.fixture
def workspace_with_tunnel(tmp_path):
    pid_file = tmp_path / ".tunnel.pid"
    pid_file.write_text("12345\n", encoding="utf-8")
    url_file = tmp_path / "preview_url.txt"
    url_file.write_text("https://example.trycloudflare.com\n", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.asyncio
async def test_kill_removes_pidfile_and_url(workspace_with_tunnel):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        res = await kill_preview_url(mission_id=42, workspace_path=workspace_with_tunnel)

    assert res["ok"] is True
    assert res["killed_pid"] == 12345
    assert not os.path.exists(os.path.join(workspace_with_tunnel, ".tunnel.pid"))
    assert not os.path.exists(os.path.join(workspace_with_tunnel, "preview_url.txt"))
    mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_kill_no_pidfile_is_soft_success(tmp_path):
    """Nothing to kill — still a soft success; just removes preview_url.txt if present."""
    ws = str(tmp_path)
    with patch("subprocess.run") as mock_run:
        res = await kill_preview_url(mission_id=42, workspace_path=ws)
    assert res["ok"] is True
    assert res["killed_pid"] is None
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_kill_via_run_dispatcher(workspace_with_tunnel):
    import mr_roboto

    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "kill_preview_url",
            "workspace_path": workspace_with_tunnel,
        },
    }
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["ok"] is True
