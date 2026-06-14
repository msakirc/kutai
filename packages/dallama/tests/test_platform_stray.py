"""Tests for port-aware stray-server cleanup.

`kill_stray_servers(keep_port)` must kill every llama-server EXCEPT the one
(if any) listening on keep_port. This preserves a healthy server on the
configured port (honoring "never kill the good llama-server") while clearing
wrong-port orphans that occupy VRAM — the 2026-06-14 incident.
"""
from dallama.platform import PlatformHelper


def _helper(monkeypatch, pids, keeper):
    ph = PlatformHelper()
    killed = []
    monkeypatch.setattr(ph, "_llama_server_pids", lambda *a, **k: set(pids))
    monkeypatch.setattr(ph, "_pid_on_port", lambda port: keeper if port == 8081 else None)
    monkeypatch.setattr(ph, "_kill_pid", lambda pid: killed.append(pid))
    return ph, killed


def test_preserves_keeper_kills_strays(monkeypatch):
    ph, killed = _helper(monkeypatch, {100, 200, 300}, keeper=300)
    n = ph.kill_stray_servers(8081)
    assert set(killed) == {100, 200}
    assert n == 2


def test_no_keeper_kills_all(monkeypatch):
    ph, killed = _helper(monkeypatch, {100, 200}, keeper=None)
    n = ph.kill_stray_servers(8081)
    assert set(killed) == {100, 200}
    assert n == 2


def test_only_keeper_kills_none(monkeypatch):
    ph, killed = _helper(monkeypatch, {300}, keeper=300)
    n = ph.kill_stray_servers(8081)
    assert killed == []
    assert n == 0


def test_no_servers_kills_none(monkeypatch):
    ph, killed = _helper(monkeypatch, set(), keeper=None)
    n = ph.kill_stray_servers(8081)
    assert killed == []
    assert n == 0
