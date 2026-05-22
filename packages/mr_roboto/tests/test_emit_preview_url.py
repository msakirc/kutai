"""Tests for mr_roboto.emit_preview_url — Z1 T4C tunneled-preview EMIT-ONLY.

Z1 owns the surface: write `preview_url.txt` to mission workspace. Z2 owns
hosting. When `KUTAI_PREVIEW_PROVIDER=cloudflared` and the binary is on PATH
we spawn a local static HTTP server, then a cloudflared tunnel; otherwise we
fail-soft with a `pending:` line.

All subprocess.Popen, shutil.which, socket, and port-wait calls are mocked.
No real servers, tunnels, or sockets are opened.
"""
from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, call, patch

import pytest

# Import the module objects (not the functions) so monkeypatch.setattr works.
_epu_mod = importlib.import_module("mr_roboto.emit_preview_url")
_kpu_mod = importlib.import_module("mr_roboto.kill_preview_url")
from mr_roboto.emit_preview_url import (
    _resolve_preview_root,
    emit_preview_url,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace_prototype(tmp_path):
    """Workspace with a .prototype/index.html (mobile track)."""
    proto = tmp_path / ".prototype"
    proto.mkdir()
    (proto / "index.html").write_text("<html></html>", encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def workspace_web(tmp_path):
    """Workspace with a .web/ dir but no .prototype/index.html (web track)."""
    web = tmp_path / ".web"
    web.mkdir()
    (web / "index.html").write_text("<html></html>", encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def workspace_both(tmp_path):
    """Workspace with BOTH .prototype/index.html AND .web/ — .prototype wins."""
    proto = tmp_path / ".prototype"
    proto.mkdir()
    (proto / "index.html").write_text("<html></html>", encoding="utf-8")
    web = tmp_path / ".web"
    web.mkdir()
    (web / "index.html").write_text("<html></html>", encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def workspace_empty(tmp_path):
    """Workspace with neither .prototype/index.html nor .web/ content."""
    return str(tmp_path)


# ---------------------------------------------------------------------------
# _resolve_preview_root unit tests
# ---------------------------------------------------------------------------

def test_resolve_returns_prototype_when_index_present(workspace_prototype):
    root = _resolve_preview_root(workspace_prototype)
    assert root is not None
    assert root.endswith(".prototype")


def test_resolve_returns_web_when_no_prototype(workspace_web):
    root = _resolve_preview_root(workspace_web)
    assert root is not None
    assert root.endswith(".web")


def test_resolve_prototype_preferred_over_web(workspace_both):
    root = _resolve_preview_root(workspace_both)
    assert root is not None
    assert root.endswith(".prototype")


def test_resolve_none_when_no_root(workspace_empty):
    root = _resolve_preview_root(workspace_empty)
    assert root is None


def test_resolve_none_when_prototype_dir_exists_but_no_index(tmp_path):
    """A .prototype dir without index.html should not count."""
    proto = tmp_path / ".prototype"
    proto.mkdir()
    (proto / "style.css").write_text("body {}", encoding="utf-8")
    root = _resolve_preview_root(str(tmp_path))
    assert root is None


def test_resolve_none_when_web_dir_is_empty(tmp_path):
    """An empty .web/ dir should not resolve."""
    web = tmp_path / ".web"
    web.mkdir()
    root = _resolve_preview_root(str(tmp_path))
    assert root is None


# ---------------------------------------------------------------------------
# Pending: env unset
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_pending_when_env_unset(workspace_prototype, monkeypatch):
    """No env var set → write a 'pending:' placeholder, never spawn anything."""
    monkeypatch.delenv("KUTAI_PREVIEW_PROVIDER", raising=False)

    with patch("subprocess.Popen") as mock_popen:
        res = await emit_preview_url(mission_id=42, workspace_path=workspace_prototype)

    assert res["ok"] is True
    assert res["pending"] is True
    assert res["provider"] in (None, "")
    url_file = os.path.join(workspace_prototype, "preview_url.txt")
    body = open(url_file, encoding="utf-8").read()
    assert body.startswith("pending:")
    assert "Z2" in body or "deferred" in body.lower()
    mock_popen.assert_not_called()
    assert not os.path.exists(os.path.join(workspace_prototype, ".tunnel.pid"))
    assert not os.path.exists(os.path.join(workspace_prototype, ".httpserver.pid"))


# ---------------------------------------------------------------------------
# Pending: binary missing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_pending_when_binary_missing(workspace_prototype, monkeypatch):
    """Env var set but cloudflared not on PATH → still fail-soft."""
    monkeypatch.setenv("KUTAI_PREVIEW_PROVIDER", "cloudflared")

    with patch("shutil.which", return_value=None) as mock_which, \
         patch("subprocess.Popen") as mock_popen:
        res = await emit_preview_url(mission_id=42, workspace_path=workspace_prototype)

    assert res["ok"] is True
    assert res["pending"] is True
    body = open(os.path.join(workspace_prototype, "preview_url.txt"), encoding="utf-8").read()
    assert body.startswith("pending:")
    mock_popen.assert_not_called()
    mock_which.assert_called()


# ---------------------------------------------------------------------------
# Pending: no preview root
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_pending_when_no_root(workspace_empty, monkeypatch):
    """Neither .prototype nor .web present → pending, no spawn."""
    monkeypatch.setenv("KUTAI_PREVIEW_PROVIDER", "cloudflared")

    with patch("subprocess.Popen") as mock_popen:
        res = await emit_preview_url(mission_id=42, workspace_path=workspace_empty)

    assert res["ok"] is True
    assert res["pending"] is True
    body = open(os.path.join(workspace_empty, "preview_url.txt"), encoding="utf-8").read()
    assert "no preview root" in body or "pending:" in body
    mock_popen.assert_not_called()
    assert not os.path.exists(os.path.join(workspace_empty, ".tunnel.pid"))
    assert not os.path.exists(os.path.join(workspace_empty, ".httpserver.pid"))


# ---------------------------------------------------------------------------
# THE FIX: success path — two Popen calls (http server, then cloudflared)
# ---------------------------------------------------------------------------

@pytest.fixture
def _patched_success(monkeypatch):
    """Return a context-manager factory for a successful emit.

    Yields (mock_popen, fake_server_proc, fake_cf_proc).
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx(workspace_path, proto_or_web="prototype"):
        # Fixed port so we can assert on it.
        monkeypatch.setattr(_epu_mod, "_pick_free_port", lambda: 54321)
        # _wait_port_ready → instant True so no actual socket polling.
        monkeypatch.setattr(_epu_mod, "_wait_port_ready", lambda port, timeout_s=3.0: True)
        monkeypatch.setenv("KUTAI_PREVIEW_PROVIDER", "cloudflared")

        # Fake static server process (first Popen call).
        fake_server = MagicMock()
        fake_server.pid = 11111
        fake_server.poll.return_value = None
        fake_server.stdout = MagicMock()
        fake_server.stdout.readline.return_value = b""  # server stdout ignored

        # Fake cloudflared process (second Popen call).
        fake_cf = MagicMock()
        fake_cf.pid = 12345
        fake_cf.poll.return_value = None
        cf_lines = iter([
            b"2026-05-22 INFO  Starting tunnel\n",
            b"2026-05-22 INFO  https://random-words-1234.trycloudflare.com\n",
            b"",
        ])
        fake_cf.stdout = MagicMock()
        fake_cf.stdout.readline.side_effect = lambda: next(cf_lines, b"")

        mock_popen = MagicMock(side_effect=[fake_server, fake_cf])

        with patch("shutil.which", return_value="/usr/bin/cloudflared"), \
             patch("subprocess.Popen", mock_popen):
            yield mock_popen, fake_server, fake_cf

    return _ctx


@pytest.mark.asyncio
async def test_emit_spawns_two_procs_and_captures_url(
    workspace_prototype, _patched_success, monkeypatch
):
    """THE FIX: Popen called twice — http.server then cloudflared."""
    with _patched_success(workspace_prototype) as (mock_popen, fake_server, fake_cf):
        res = await emit_preview_url(mission_id=42, workspace_path=workspace_prototype)

    assert res["ok"] is True
    assert res["pending"] is False
    assert res["url"] == "https://random-words-1234.trycloudflare.com"
    assert res["pid"] == 12345

    # Exactly two Popen calls.
    assert mock_popen.call_count == 2

    # First call: static server — argv contains http.server, port 54321, --directory.
    first_argv = mock_popen.call_args_list[0][0][0]
    assert "http.server" in first_argv
    assert "54321" in first_argv
    assert "--directory" in first_argv
    # root must be .prototype (has index.html).
    assert first_argv[first_argv.index("--directory") + 1].endswith(".prototype")

    # Second call: cloudflared — argv contains http://127.0.0.1:54321,
    # must NOT contain file://.
    second_argv = mock_popen.call_args_list[1][0][0]
    assert "cloudflared" in second_argv
    origin_arg = " ".join(second_argv)
    assert "http://127.0.0.1:54321" in origin_arg
    assert "file://" not in origin_arg

    # Both pidfiles written.
    ws = workspace_prototype
    server_pid_file = os.path.join(ws, ".httpserver.pid")
    tunnel_pid_file = os.path.join(ws, ".tunnel.pid")
    assert os.path.exists(server_pid_file)
    assert open(server_pid_file, encoding="utf-8").read().strip() == "11111"
    assert os.path.exists(tunnel_pid_file)
    assert open(tunnel_pid_file, encoding="utf-8").read().strip() == "12345"

    # URL file contains the live URL.
    body = open(os.path.join(ws, "preview_url.txt"), encoding="utf-8").read()
    assert body.strip() == "https://random-words-1234.trycloudflare.com"

    # last_preview_url.txt also written.
    last_url_file = os.path.join(ws, ".preview", "last_preview_url.txt")
    assert os.path.exists(last_url_file)
    assert open(last_url_file, encoding="utf-8").read().strip() == \
        "https://random-words-1234.trycloudflare.com"


@pytest.mark.asyncio
async def test_emit_resolves_web_dir_when_no_prototype(
    workspace_web, _patched_success, monkeypatch
):
    """Web-track workspace (.web, no .prototype) resolves and spawns correctly."""
    with _patched_success(workspace_web, proto_or_web="web") as (mock_popen, _, __):
        res = await emit_preview_url(mission_id=7, workspace_path=workspace_web)

    assert res["ok"] is True
    assert res["pending"] is False
    assert mock_popen.call_count == 2

    # Static server must be pointed at .web.
    first_argv = mock_popen.call_args_list[0][0][0]
    assert first_argv[first_argv.index("--directory") + 1].endswith(".web")


# ---------------------------------------------------------------------------
# Orphan safety: cloudflared yields no URL → http server terminated
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_orphan_safety_no_url(workspace_prototype, monkeypatch):
    """No URL captured from cloudflared → static server proc is terminated."""
    monkeypatch.setenv("KUTAI_PREVIEW_PROVIDER", "cloudflared")
    monkeypatch.setattr(_epu_mod, "_pick_free_port", lambda: 54321)
    monkeypatch.setattr(_epu_mod, "_wait_port_ready", lambda port, timeout_s=3.0: True)

    fake_server = MagicMock()
    fake_server.pid = 11111
    fake_server.stdout = MagicMock()
    fake_server.stdout.readline.return_value = b""

    # cloudflared stdout yields nothing → no URL captured.
    fake_cf = MagicMock()
    fake_cf.pid = 12345
    fake_cf.stdout = MagicMock()
    fake_cf.stdout.readline.return_value = b""  # EOF immediately

    mock_popen = MagicMock(side_effect=[fake_server, fake_cf])

    terminate_calls = []

    def fake_terminate(pid):
        terminate_calls.append(pid)
        return 0

    with patch("shutil.which", return_value="/usr/bin/cloudflared"), \
         patch("subprocess.Popen", mock_popen), \
         patch.object(_kpu_mod, "_terminate", side_effect=fake_terminate):
        res = await emit_preview_url(mission_id=42, workspace_path=workspace_prototype)

    assert res["ok"] is True
    assert res["pending"] is True

    # The static server (pid 11111) must have been terminated.
    assert 11111 in terminate_calls

    # The .httpserver.pid file must be gone (cleaned up).
    assert not os.path.exists(os.path.join(workspace_prototype, ".httpserver.pid"))


# ---------------------------------------------------------------------------
# Idempotency: prior tunnel killed before re-emitting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_idempotent_kills_prior_tunnel(workspace_prototype, monkeypatch):
    """Re-running with a stale .tunnel.pid kills the old process before re-emitting."""
    pid_file = os.path.join(workspace_prototype, ".tunnel.pid")
    with open(pid_file, "w", encoding="utf-8") as f:
        f.write("99999\n")

    monkeypatch.delenv("KUTAI_PREVIEW_PROVIDER", raising=False)

    killed = {"called": False}

    def fake_kill(mid, ws):
        killed["called"] = True
        if os.path.exists(pid_file):
            os.remove(pid_file)
        return {"ok": True, "killed": True}

    with patch.object(_epu_mod, "_kill_prior_tunnel", side_effect=fake_kill):
        res = await emit_preview_url(mission_id=42, workspace_path=workspace_prototype)

    assert killed["called"] is True
    assert res["ok"] is True


# ---------------------------------------------------------------------------
# Smoke: run() dispatcher routes emit_preview_url
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_via_run_dispatcher(workspace_prototype, monkeypatch):
    """Smoke test: run() dispatches `emit_preview_url` through the action router."""
    import mr_roboto

    monkeypatch.delenv("KUTAI_PREVIEW_PROVIDER", raising=False)
    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "emit_preview_url",
            "workspace_path": workspace_prototype,
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["ok"] is True
