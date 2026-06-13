# packages/clair_obscur/tests/test_server_lifecycle.py
import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer


def _cfg(tmp_path, idle=60):
    exe = tmp_path / "fake_exe"; exe.write_text("x")
    return ClairObscurConfig(
        backend="comfyui", host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        exe_path=str(exe), idle_release_seconds=idle,
    )


def test_status_when_not_started(tmp_path):
    s = ImageServer(_cfg(tmp_path))
    st = s.status()
    assert st["resident"] is False and st["pid"] is None


@pytest.mark.asyncio
async def test_start_polls_health_until_ready(monkeypatch, tmp_path):
    s = ImageServer(_cfg(tmp_path))
    calls = {"n": 0}

    async def _fake_launch(): s._pid = 12345
    async def _fake_health(*_a):
        calls["n"] += 1
        return calls["n"] >= 3
    monkeypatch.setattr(s, "_launch_process", _fake_launch)
    monkeypatch.setattr(s, "_health_probe", _fake_health)
    monkeypatch.setattr(s, "_acquire_lock", lambda: None)
    monkeypatch.setattr(s, "_reconcile_orphan", lambda: None)
    monkeypatch.setattr(s, "_notify_nerd_herd_resident", lambda vram_mb=4500: None)
    s._health_poll_interval = 0.01

    url = await s.start()
    assert url == "http://127.0.0.1:8188"
    assert calls["n"] >= 3
    assert s.status()["resident"] is True


@pytest.mark.asyncio
async def test_start_is_idempotent_and_clears_release_hint(monkeypatch, tmp_path):
    """Repeat start() when already resident clears any pending release hint —
    so an in-flight idle backstop window resets when a new image arrives."""
    s = ImageServer(_cfg(tmp_path))
    s._pid = 555
    s._resident = True
    s._release_hint_at = 100.0  # pretend a hint was recorded

    url = await s.start()
    assert url == "http://127.0.0.1:8188"
    assert s._release_hint_at is None, "start() must clear pending release hint"


@pytest.mark.asyncio
async def test_start_times_out_when_health_never_up(monkeypatch, tmp_path):
    s = ImageServer(_cfg(tmp_path))
    async def _fake_launch(): s._pid = 99
    async def _fake_health(*_a): return False
    monkeypatch.setattr(s, "_launch_process", _fake_launch)
    monkeypatch.setattr(s, "_health_probe", _fake_health)
    monkeypatch.setattr(s, "_acquire_lock", lambda: None)
    monkeypatch.setattr(s, "_reconcile_orphan", lambda: None)
    monkeypatch.setattr(s, "_notify_nerd_herd_resident", lambda vram_mb=0: None)
    monkeypatch.setattr(s, "_kill_own_pid", lambda pid: None)
    s._health_timeout_seconds = 0.3
    s._health_poll_interval = 0.05

    with pytest.raises(TimeoutError):
        await s.start()


@pytest.mark.asyncio
async def test_stop_releases_and_notifies(monkeypatch, tmp_path):
    s = ImageServer(_cfg(tmp_path))
    s._pid = 12345; s._resident = True
    killed = {"pid": None}
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("pid", pid))
    monkeypatch.setattr(s, "_notify_nerd_herd_resident", lambda vram_mb=0: None)
    monkeypatch.setattr(s, "_release_lock", lambda: None)

    await s.stop()
    assert killed["pid"] == 12345
    st = s.status()
    assert st["resident"] is False and st["pid"] is None
