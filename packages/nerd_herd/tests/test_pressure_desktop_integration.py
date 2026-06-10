from types import SimpleNamespace
from nerd_herd.types import SystemSnapshot, LocalModelState

LOCAL = SimpleNamespace(name="loc", is_local=True, is_loaded=True, is_free=False,
                        provider="", cap_score=5.0, size_mb=4000)


def _snap(**kw):
    base = dict(vram_available_mb=8000,
                local=LocalModelState(model_name="loc", idle_seconds=120.0))
    base.update(kw)
    return SystemSnapshot(**base)


def test_away_full_mode_no_desktop_pressure():
    snap = _snap(user_idle_s=1e9, load_mode="full")
    assert snap.pressure_for(LOCAL).scalar >= 0.0


def test_present_heavy_pushes_local_negative():
    snap = _snap(user_idle_s=1.0, load_mode="heavy")
    assert snap.pressure_for(LOCAL).scalar < 0.0


def test_full_mode_silences_presence_even_when_present():
    snap = _snap(user_idle_s=1.0, load_mode="full")
    assert snap.pressure_for(LOCAL).scalar >= 0.0


def test_fullscreen_pegs_minus_one_in_heavy():
    snap = _snap(user_idle_s=1.0, foreground_fullscreen=True, load_mode="heavy")
    assert snap.pressure_for(LOCAL).scalar == -1.0


def test_external_gpu_veto_in_heavy():
    snap = _snap(user_idle_s=1e9, external_gpu_fraction=0.7, load_mode="heavy")
    assert snap.pressure_for(LOCAL).scalar == -1.0
