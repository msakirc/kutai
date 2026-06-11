# packages/clair_obscur/tests/test_orphan_reconcile.py
import os
from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer, _LOCK_PATH


def _cfg(tmp_path, backend="comfyui"):
    exe = tmp_path / "fake_exe"; exe.write_text("x")
    return ClairObscurConfig(
        backend=backend, host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(exe), idle_release_seconds=60,
    )


def _write_lock(pid: int, backend: str):
    os.makedirs(os.path.dirname(_LOCK_PATH) or ".", exist_ok=True)
    with open(_LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(f"{pid}\n{backend}\n")


def _clear_lock():
    try: os.remove(_LOCK_PATH)
    except FileNotFoundError: pass


def test_orphan_killed_when_lock_points_at_our_backend(monkeypatch, tmp_path):
    _clear_lock(); _write_lock(12345, "comfyui")
    killed = {"pid": None}
    s = ImageServer(_cfg(tmp_path))
    monkeypatch.setattr(s, "_is_own_backend_pid", lambda pid: pid == 12345)
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("pid", pid))
    s._reconcile_orphan()
    assert killed["pid"] == 12345
    assert not os.path.exists(_LOCK_PATH)


def test_orphan_skipped_when_pid_is_not_our_backend(monkeypatch, tmp_path):
    """SAFETY: if the PID in image_server.lock no longer belongs to a
    ComfyUI/A1111 process (e.g. it now belongs to llama-server or any
    other tenant — PID-reuse case), the sweep MUST NOT kill it."""
    _clear_lock(); _write_lock(67890, "comfyui")
    killed = {"called": False}
    s = ImageServer(_cfg(tmp_path))
    monkeypatch.setattr(s, "_is_own_backend_pid", lambda pid: False)
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("called", True))
    s._reconcile_orphan()
    assert killed["called"] is False
    assert not os.path.exists(_LOCK_PATH)


def test_orphan_skipped_when_backend_in_lock_does_not_match(monkeypatch, tmp_path):
    """Lock says 'a1111' but config says 'comfyui' — refuse to touch."""
    _clear_lock(); _write_lock(11111, "a1111")
    killed = {"called": False}
    s = ImageServer(_cfg(tmp_path, backend="comfyui"))
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("called", True))
    s._reconcile_orphan()
    assert killed["called"] is False


def test_no_lock_no_action(tmp_path):
    _clear_lock()
    s = ImageServer(_cfg(tmp_path))
    s._reconcile_orphan()  # must not raise


def test_corrupt_lock_clears_quietly(tmp_path):
    """Lock with garbage content (e.g. half-written from a previous crash)
    must clear the file without raising or killing anything."""
    os.makedirs(os.path.dirname(_LOCK_PATH) or ".", exist_ok=True)
    with open(_LOCK_PATH, "w", encoding="utf-8") as f:
        f.write("not-a-pid\n")
    s = ImageServer(_cfg(tmp_path))
    s._reconcile_orphan()
    assert not os.path.exists(_LOCK_PATH)
