"""Tests for mr_roboto.kill_preview_url — stop tunnel + http server + surface files.

Two-PID model: kill_preview_url now terminates BOTH .tunnel.pid (cloudflared)
and .httpserver.pid (python -m http.server), plus removes preview_url.txt.
All kills are best-effort and idempotent — missing pidfile is a no-op.
"""
from __future__ import annotations

import os
from unittest.mock import call, patch

import pytest

import importlib as _importlib
_kpu_mod = _importlib.import_module("mr_roboto.kill_preview_url")
from mr_roboto.kill_preview_url import kill_preview_url, _terminate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace_with_both_pids(tmp_path):
    """Workspace seeded with both pidfiles + preview_url.txt."""
    (tmp_path / ".tunnel.pid").write_text("12345\n", encoding="utf-8")
    (tmp_path / ".httpserver.pid").write_text("11111\n", encoding="utf-8")
    (tmp_path / "preview_url.txt").write_text(
        "https://example.trycloudflare.com\n", encoding="utf-8"
    )
    return str(tmp_path)


@pytest.fixture
def workspace_with_tunnel(tmp_path):
    """Legacy fixture: only .tunnel.pid + preview_url.txt (no .httpserver.pid)."""
    (tmp_path / ".tunnel.pid").write_text("12345\n", encoding="utf-8")
    (tmp_path / "preview_url.txt").write_text(
        "https://example.trycloudflare.com\n", encoding="utf-8"
    )
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Teardown: both pidfiles present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kill_removes_both_pidfiles_and_url(workspace_with_both_pids):
    """Both pidfiles present → _terminate called for both pids; both pidfiles removed."""
    terminate_calls = []

    def fake_terminate(pid):
        terminate_calls.append(pid)
        return 0

    with patch.object(_kpu_mod, "_terminate", side_effect=fake_terminate):
        res = await kill_preview_url(
            mission_id=42, workspace_path=workspace_with_both_pids, _silent=True
        )

    assert res["ok"] is True
    # Both pids terminated.
    assert set(terminate_calls) == {12345, 11111}
    # Both pidfiles removed.
    assert not os.path.exists(os.path.join(workspace_with_both_pids, ".tunnel.pid"))
    assert not os.path.exists(os.path.join(workspace_with_both_pids, ".httpserver.pid"))
    # URL surface file removed.
    assert not os.path.exists(os.path.join(workspace_with_both_pids, "preview_url.txt"))
    # killed_pid reports the tunnel pid.
    assert res["killed_pid"] == 12345


@pytest.mark.asyncio
async def test_kill_returns_tunnel_pid_in_result(workspace_with_both_pids):
    """killed_pid in result is the cloudflared tunnel pid (12345), not the server pid."""
    with patch.object(_kpu_mod, "_terminate", return_value=0):
        res = await kill_preview_url(
            mission_id=42, workspace_path=workspace_with_both_pids, _silent=True
        )
    assert res["killed_pid"] == 12345


# ---------------------------------------------------------------------------
# Teardown: backward compat — only tunnel pid present (no .httpserver.pid)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kill_tunnel_only_no_httpserver_pidfile(workspace_with_tunnel):
    """Legacy: .httpserver.pid absent → still kills tunnel, no error."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        res = await kill_preview_url(
            mission_id=42, workspace_path=workspace_with_tunnel, _silent=True
        )

    assert res["ok"] is True
    assert res["killed_pid"] == 12345
    assert not os.path.exists(os.path.join(workspace_with_tunnel, ".tunnel.pid"))
    assert not os.path.exists(os.path.join(workspace_with_tunnel, "preview_url.txt"))


# ---------------------------------------------------------------------------
# Idempotency: no pidfiles at all
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kill_no_pidfile_is_soft_success(tmp_path):
    """Nothing to kill — still a soft success; just removes preview_url.txt if present."""
    ws = str(tmp_path)
    with patch("subprocess.run") as mock_run:
        res = await kill_preview_url(mission_id=42, workspace_path=ws, _silent=True)
    assert res["ok"] is True
    assert res["killed_pid"] is None
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Backward-compat: original single-pid test (subprocess.run path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kill_removes_pidfile_and_url(workspace_with_tunnel):
    """Original test: single .tunnel.pid removed + url removed."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        res = await kill_preview_url(
            mission_id=42, workspace_path=workspace_with_tunnel, _silent=True
        )

    assert res["ok"] is True
    assert res["killed_pid"] == 12345
    assert not os.path.exists(os.path.join(workspace_with_tunnel, ".tunnel.pid"))
    assert not os.path.exists(os.path.join(workspace_with_tunnel, "preview_url.txt"))
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Smoke: run() dispatcher routes kill_preview_url
# ---------------------------------------------------------------------------

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
