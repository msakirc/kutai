"""Test: clair_obscur ImageServer start/stop push state through nerd_herd (Task 6)."""
import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer


def _cfg(tmp_path):
    exe = tmp_path / "fake_exe"; exe.write_text("x")
    return ClairObscurConfig(
        backend="comfyui", host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(exe), idle_release_seconds=60,
    )


@pytest.mark.asyncio
async def test_start_pushes_resident_true(monkeypatch, tmp_path):
    import nerd_herd
    nerd_herd.record_image_server_state(resident=False, vram_mb=0)

    s = ImageServer(_cfg(tmp_path))

    async def _fake_launch(): s._pid = 111
    async def _fake_health(): return True
    monkeypatch.setattr(s, "_launch_process", _fake_launch)
    monkeypatch.setattr(s, "_health_probe", _fake_health)
    monkeypatch.setattr(s, "_acquire_lock", lambda: None)
    monkeypatch.setattr(s, "_reconcile_orphan", lambda: None)

    await s.start()
    # Read back via the singleton that _notify_nerd_herd_resident writes to.
    snap = nerd_herd._get_singleton().snapshot()
    assert snap.image_server_resident is True
    assert snap.image_server_vram_mb > 0


@pytest.mark.asyncio
async def test_stop_pushes_resident_false(monkeypatch, tmp_path):
    import nerd_herd
    nerd_herd.record_image_server_state(resident=True, vram_mb=4500)

    s = ImageServer(_cfg(tmp_path))
    s._pid = 222; s._resident = True
    monkeypatch.setattr(s, "_kill_own_pid", lambda pid: None)
    monkeypatch.setattr(s, "_release_lock", lambda: None)

    await s.stop()
    snap = nerd_herd._get_singleton().snapshot()
    assert snap.image_server_resident is False
    assert snap.image_server_vram_mb == 0
