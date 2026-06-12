# packages/clair_obscur/tests/test_idle_backstop.py
import asyncio
import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer


def _cfg(tmp_path, idle=0.2):
    exe = tmp_path / "fake_exe"; exe.write_text("x")
    return ClairObscurConfig(
        backend="comfyui", host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(exe), idle_release_seconds=idle,
    )


@pytest.mark.asyncio
async def test_release_hint_then_idle_triggers_stop(monkeypatch, tmp_path):
    """After record_release_hint(), the watcher fires stop() once the idle
    window elapses. This is the normal lane-switch release path."""
    s = ImageServer(_cfg(tmp_path, idle=0.2))
    s._resident = True
    s._pid = 555
    stops = {"n": 0}
    async def _fake_stop():
        stops["n"] += 1
        s._resident = False; s._pid = None
    monkeypatch.setattr(s, "stop", _fake_stop)

    s._arm_idle_backstop()
    s.record_release_hint()
    await asyncio.sleep(0.5)
    assert stops["n"] == 1, "backstop must call stop() after hint + idle"


@pytest.mark.asyncio
async def test_no_release_hint_means_no_stop(monkeypatch, tmp_path):
    """Without record_release_hint(), the watcher must NOT stop the server.
    This is the warm-batch case: beckman never hinted, so we hold."""
    s = ImageServer(_cfg(tmp_path, idle=0.2))
    s._resident = True
    s._pid = 555
    stops = {"n": 0}
    async def _fake_stop():
        stops["n"] += 1; s._resident = False
    monkeypatch.setattr(s, "stop", _fake_stop)

    s._arm_idle_backstop()
    await asyncio.sleep(0.5)
    assert stops["n"] == 0
    s._resident = False  # let watcher exit


@pytest.mark.asyncio
async def test_hint_cleared_by_idempotent_start_extends_window(
    monkeypatch, tmp_path,
):
    """If a new image task arrives mid-window, the dispatcher's idempotent
    start() clears the hint (set in Task 2). The watcher must then NOT
    stop the server, allowing the warm batch to continue."""
    s = ImageServer(_cfg(tmp_path, idle=0.2))
    s._resident = True
    s._pid = 555
    stops = {"n": 0}
    async def _fake_stop():
        stops["n"] += 1; s._resident = False
    monkeypatch.setattr(s, "stop", _fake_stop)

    s._arm_idle_backstop()
    s.record_release_hint()
    await asyncio.sleep(0.05)
    # Mid-window, dispatcher calls start() for the next image — idempotent
    # branch clears the hint:
    await s.start()  # idempotent branch (resident=True already)
    await asyncio.sleep(0.5)
    assert stops["n"] == 0
