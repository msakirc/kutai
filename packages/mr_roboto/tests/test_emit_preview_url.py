"""Tests for mr_roboto.emit_preview_url — Z1 T4C tunneled-preview EMIT-ONLY.

Z1 owns the surface: write `preview_url.txt` to mission workspace. Z2 owns
hosting. When `KUTAI_PREVIEW_PROVIDER=cloudflared` and the binary is on PATH
we spawn a tunnel subprocess; otherwise we fail-soft with a `pending:` line.

All `subprocess.Popen` and PATH lookups are mocked.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from mr_roboto.emit_preview_url import emit_preview_url


@pytest.fixture
def workspace(tmp_path):
    proto = tmp_path / ".prototype"
    proto.mkdir()
    (proto / "index.html").write_text("<html></html>", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.asyncio
async def test_emit_pending_when_env_unset(workspace, monkeypatch):
    """No env var set → write a 'pending:' placeholder, never spawn anything."""
    monkeypatch.delenv("KUTAI_PREVIEW_PROVIDER", raising=False)

    with patch("subprocess.Popen") as mock_popen:
        res = await emit_preview_url(mission_id=42, workspace_path=workspace)

    assert res["ok"] is True
    assert res["pending"] is True
    assert res["provider"] in (None, "")
    url_file = os.path.join(workspace, "preview_url.txt")
    body = open(url_file, encoding="utf-8").read()
    assert body.startswith("pending:")
    assert "Z2" in body or "deferred" in body.lower()
    mock_popen.assert_not_called()
    # No pidfile when nothing was spawned.
    assert not os.path.exists(os.path.join(workspace, ".tunnel.pid"))


@pytest.mark.asyncio
async def test_emit_pending_when_binary_missing(workspace, monkeypatch):
    """Env var set but cloudflared not on PATH → still fail-soft."""
    monkeypatch.setenv("KUTAI_PREVIEW_PROVIDER", "cloudflared")

    with patch("shutil.which", return_value=None) as mock_which, \
         patch("subprocess.Popen") as mock_popen:
        res = await emit_preview_url(mission_id=42, workspace_path=workspace)

    assert res["ok"] is True
    assert res["pending"] is True
    body = open(os.path.join(workspace, "preview_url.txt"), encoding="utf-8").read()
    assert body.startswith("pending:")
    mock_popen.assert_not_called()
    mock_which.assert_called()


@pytest.mark.asyncio
async def test_emit_spawns_cloudflared_and_captures_url(workspace, monkeypatch):
    """Env var set + binary present → spawn tunnel, write captured URL, persist pidfile."""
    monkeypatch.setenv("KUTAI_PREVIEW_PROVIDER", "cloudflared")

    fake_proc = MagicMock()
    fake_proc.pid = 12345
    fake_proc.poll.return_value = None
    # Mock stdout: emit a cloudflared-shaped trycloudflare URL line then EOF.
    fake_lines = iter([
        b"2026-05-10 INFO  Cloudflared something\n",
        b"2026-05-10 INFO  https://random-words-1234.trycloudflare.com\n",
        b"",  # readline returning b"" signals EOF in the consumer
    ])
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = lambda: next(fake_lines, b"")

    with patch("shutil.which", return_value="/usr/bin/cloudflared"), \
         patch("subprocess.Popen", return_value=fake_proc) as mock_popen:
        res = await emit_preview_url(mission_id=42, workspace_path=workspace)

    assert res["ok"] is True
    assert res["pending"] is False
    assert res["url"].startswith("https://")
    assert "trycloudflare.com" in res["url"]
    assert res["pid"] == 12345
    # File contents should be the live URL only (not the pending placeholder).
    body = open(os.path.join(workspace, "preview_url.txt"), encoding="utf-8").read()
    assert body.strip().startswith("https://")
    assert "trycloudflare.com" in body
    pid_file = os.path.join(workspace, ".tunnel.pid")
    assert os.path.exists(pid_file)
    assert open(pid_file, encoding="utf-8").read().strip() == "12345"
    mock_popen.assert_called_once()


@pytest.mark.asyncio
async def test_emit_idempotent_kills_prior_tunnel(workspace, monkeypatch):
    """Re-running with a stale .tunnel.pid kills the old process before re-emitting."""
    # Pre-seed a stale pidfile.
    pid_file = os.path.join(workspace, ".tunnel.pid")
    with open(pid_file, "w", encoding="utf-8") as f:
        f.write("99999\n")

    monkeypatch.delenv("KUTAI_PREVIEW_PROVIDER", raising=False)

    killed = {"called": False}

    def fake_kill(mid, ws):
        killed["called"] = True
        # Simulate kill removing the pidfile.
        if os.path.exists(pid_file):
            os.remove(pid_file)
        return {"ok": True, "killed": True}

    import importlib
    _epu_mod = importlib.import_module("mr_roboto.emit_preview_url")
    with patch.object(_epu_mod, "_kill_prior_tunnel", side_effect=fake_kill):
        res = await emit_preview_url(mission_id=42, workspace_path=workspace)

    assert killed["called"] is True
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_emit_via_run_dispatcher(workspace, monkeypatch):
    """Smoke test: run() dispatches `emit_preview_url` through the action router."""
    import mr_roboto

    monkeypatch.delenv("KUTAI_PREVIEW_PROVIDER", raising=False)
    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "emit_preview_url",
            "workspace_path": workspace,
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["ok"] is True
