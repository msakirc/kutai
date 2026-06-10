from nerd_herd.types import SystemSnapshot


def test_snapshot_has_desktop_fields_with_safe_defaults():
    s = SystemSnapshot()
    assert s.load_mode == "full"
    assert s.user_idle_s >= 1e8
    assert s.foreground_fullscreen is False
    assert s.ram_available_mb == 0
    assert s.ram_total_mb == 0
    assert s.external_gpu_fraction == 0.0
