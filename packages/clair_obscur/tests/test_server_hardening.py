# packages/clair_obscur/tests/test_server_hardening.py
"""FIX 4.2 — start/stop lock, dead-process health abort, Popen cwd."""
import asyncio
import time

import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer


def _cfg(tmp_path, *, backend="comfyui", working_dir="", idle=60):
    exe = tmp_path / "fake_exe"; exe.write_text("x")
    return ClairObscurConfig(
        backend=backend, host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(exe), working_dir=working_dir,
        idle_release_seconds=idle,
    )


# ── (a) start()/stop() serialization ─────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_start_spawns_one_process(monkeypatch, tmp_path):
    """Two concurrent start() calls (cloud-failover storm / beckman lane cap)
    must spawn exactly ONE backend: the second caller waits on the lock, then
    sees _resident and returns the base_url. Without the lock, B's
    _reconcile_orphan can kill A's just-launched backend."""
    s = ImageServer(_cfg(tmp_path))
    launches = {"n": 0}
    reconciles = {"n": 0}

    async def _fake_launch():
        launches["n"] += 1
        await asyncio.sleep(0.05)  # widen the cold-start window
        s._pid = 12345

    async def _fake_health():
        return True

    monkeypatch.setattr(s, "_launch_process", _fake_launch)
    monkeypatch.setattr(s, "_health_probe", _fake_health)
    monkeypatch.setattr(s, "_acquire_lock", lambda: None)
    monkeypatch.setattr(
        s, "_reconcile_orphan",
        lambda: reconciles.__setitem__("n", reconciles["n"] + 1),
    )
    monkeypatch.setattr(s, "_notify_nerd_herd_resident",
                        lambda vram_mb=4500: None)
    s._health_poll_interval = 0.01

    urls = await asyncio.gather(s.start(), s.start())
    assert launches["n"] == 1, "second start() must not relaunch"
    assert reconciles["n"] == 1, "second start() must not orphan-reconcile"
    assert set(urls) == {"http://127.0.0.1:8188"}
    assert s.status()["resident"] is True


@pytest.mark.asyncio
async def test_stop_serializes_with_start(monkeypatch, tmp_path):
    """stop() takes the same lock as start(): a stop issued while a start is
    mid-launch waits for the start to finish, then tears down — no interleaved
    mutation of _pid/_resident."""
    s = ImageServer(_cfg(tmp_path))
    order = []

    async def _fake_launch():
        order.append("launch")
        await asyncio.sleep(0.05)
        s._pid = 777

    async def _fake_health():
        return True

    killed = {"pid": None}
    monkeypatch.setattr(s, "_launch_process", _fake_launch)
    monkeypatch.setattr(s, "_health_probe", _fake_health)
    monkeypatch.setattr(s, "_acquire_lock", lambda: None)
    monkeypatch.setattr(s, "_reconcile_orphan", lambda: None)
    monkeypatch.setattr(s, "_notify_nerd_herd_resident", lambda vram_mb=0: None)
    monkeypatch.setattr(s, "_release_lock", lambda: None)
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("pid", pid))
    s._health_poll_interval = 0.01

    async def _stop_soon():
        await asyncio.sleep(0.01)  # let start() acquire the lock first
        order.append("stop_called")
        await s.stop()
        order.append("stop_done")

    await asyncio.gather(s.start(), _stop_soon())
    # stop() ran strictly AFTER start() completed (lock serialization), so it
    # saw the launched pid and tore it down.
    assert killed["pid"] == 777
    assert s.status()["resident"] is False


@pytest.mark.asyncio
async def test_idle_backstop_stop_does_not_deadlock(monkeypatch, tmp_path):
    """The watcher task calls public stop() (which takes the lock) — must
    complete, not deadlock, even though stop() cancels the watcher itself."""
    s = ImageServer(_cfg(tmp_path, idle=0.05))
    s._resident = True
    s._pid = 555
    monkeypatch.setattr(s, "_kill_own_pid", lambda pid: None)
    monkeypatch.setattr(s, "_notify_nerd_herd_resident", lambda vram_mb=0: None)
    monkeypatch.setattr(s, "_release_lock", lambda: None)

    s._arm_idle_backstop()
    s.record_release_hint()
    await asyncio.sleep(0.3)
    assert s.status()["resident"] is False, "backstop stop() must have fired"
    assert not s._lock.locked(), "lock must be released after watcher stop()"


# ── (b) dead-process abort during health wait ────────────────────────────

class _DeadProc:
    pid = 99
    returncode = 1

    def poll(self):
        return 1


@pytest.mark.asyncio
async def test_health_wait_aborts_fast_when_process_dies(monkeypatch, tmp_path):
    """If the spawned backend exits during the health wait, start() must
    abort immediately (log stderr tail, raise) instead of burning the full
    health timeout."""
    s = ImageServer(_cfg(tmp_path))

    async def _fake_launch():
        s._pid = 99
        s._proc = _DeadProc()

    async def _fake_health():
        return False

    monkeypatch.setattr(s, "_launch_process", _fake_launch)
    monkeypatch.setattr(s, "_health_probe", _fake_health)
    monkeypatch.setattr(s, "_acquire_lock", lambda: None)
    monkeypatch.setattr(s, "_reconcile_orphan", lambda: None)
    monkeypatch.setattr(s, "_release_lock", lambda: None)
    s._health_timeout_seconds = 10.0
    s._health_poll_interval = 0.05

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        await s.start()
    assert time.monotonic() - t0 < 2.0, "must abort fast, not burn timeout"
    assert s.status()["resident"] is False
    assert s._pid is None


# ── (c) Popen cwd semantics ──────────────────────────────────────────────

class _FakeProc:
    pid = 4242

    def poll(self):
        return None


@pytest.mark.asyncio
async def test_launch_passes_cwd_when_working_dir_set(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIR_OBSCUR_LOG_DIR", str(tmp_path / "logs"))
    s = ImageServer(_cfg(tmp_path, working_dir=str(tmp_path)))
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr("clair_obscur.server.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("clair_obscur.server._assign_to_job",
                        lambda job, pid: None)
    await s._launch_process()
    assert captured["cwd"] == str(tmp_path)
    assert s._pid == 4242


@pytest.mark.asyncio
async def test_comfyui_without_working_dir_warns_and_passes_no_cwd(
    monkeypatch, tmp_path, caplog,
):
    """ComfyUI's cmd references main.py relative to the checkout dir — with
    working_dir unset we cannot infer it from the python exe path, so cwd is
    None and a WARNING is logged."""
    import logging
    monkeypatch.setenv("CLAIR_OBSCUR_LOG_DIR", str(tmp_path / "logs"))
    s = ImageServer(_cfg(tmp_path, backend="comfyui", working_dir=""))
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr("clair_obscur.server.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("clair_obscur.server._assign_to_job",
                        lambda job, pid: None)
    with caplog.at_level(logging.WARNING, logger="clair_obscur.server"):
        await s._launch_process()
    assert captured["cwd"] is None
    assert any("CLAIR_OBSCUR_DIR" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a1111_without_working_dir_defaults_to_exe_dir(
    monkeypatch, tmp_path,
):
    """A1111's launcher lives inside the webui checkout — default cwd is the
    launcher's own directory."""
    monkeypatch.setenv("CLAIR_OBSCUR_LOG_DIR", str(tmp_path / "logs"))
    s = ImageServer(_cfg(tmp_path, backend="a1111", working_dir=""))
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr("clair_obscur.server.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("clair_obscur.server._assign_to_job",
                        lambda job, pid: None)
    await s._launch_process()
    assert captured["cwd"] == str(tmp_path)


def test_load_config_reads_working_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIR_OBSCUR_DIR", str(tmp_path))
    monkeypatch.delenv("CLAIR_OBSCUR_BACKEND", raising=False)
    from clair_obscur.config import load_config
    cfg = load_config()
    assert cfg.working_dir == str(tmp_path)


def test_load_config_working_dir_defaults_empty(monkeypatch):
    monkeypatch.delenv("CLAIR_OBSCUR_DIR", raising=False)
    monkeypatch.delenv("CLAIR_OBSCUR_BACKEND", raising=False)
    from clair_obscur.config import load_config
    cfg = load_config()
    assert cfg.working_dir == ""
