"""Spawn-flag regression for the Claude Code remote-control button.

Production incident 2026-06-22: pressing Yaşar Usta's "🖥️ Claude Code"
button stopped starting sessions. The child was launched with
``DETACHED_PROCESS``, which strips the console. Claude Code CLI v2.1.179
needs a console to bring up its remote-control server (older versions
tolerated detachment — old session logs DID capture output), so the
detached child produced a 0-byte log and never registered a session.

Faithful repro proved ``CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW``
starts and connects a session (window-less but console-backed), while
``... | DETACHED_PROCESS`` yields 0 bytes / no session. CREATE_NEW_PROCESS_GROUP
is retained so the session still survives a guard restart (detached from
the guard's Ctrl-C group); only DETACHED_PROCESS is dropped.
"""

import sys

import pytest

from yasar_usta import remote


class _FakeStdin:
    def write(self, _data):  # noqa: D401
        return None

    def flush(self):
        return None

    def close(self):
        return None


class _FakeProc:
    def __init__(self, *args, **kwargs):
        self.captured_kwargs = kwargs
        self.pid = 4242
        self.stdin = _FakeStdin()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows spawn flags only")
@pytest.mark.asyncio
async def test_remote_spawn_uses_no_window_not_detached(monkeypatch):
    captured = {}

    def _fake_popen(*args, **kwargs):
        proc = _FakeProc(*args, **kwargs)
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(remote._sp, "Popen", _fake_popen)

    # session_dir=None → no log file / no URL poll; we only assert spawn flags.
    pid, _url = await remote.start_claude_remote("claude.cmd", session_dir=None)

    assert pid == 4242
    flags = captured["kwargs"].get("creationflags", 0)
    assert flags & remote._sp.CREATE_NO_WINDOW, "must give the child a (hidden) console"
    assert not (flags & remote._sp.DETACHED_PROCESS), (
        "DETACHED_PROCESS strips the console and breaks session startup"
    )
    assert flags & remote._sp.CREATE_NEW_PROCESS_GROUP, (
        "keep new process group so the session survives a guard restart"
    )


import inspect
from yasar_usta import remote as _remote_mod


def test_start_claude_remote_accepts_session_label():
    sig = inspect.signature(_remote_mod.start_claude_remote)
    assert "session_label" in sig.parameters
