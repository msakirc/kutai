from nerd_herd.nerd_herd import NerdHerd


def test_snapshot_carries_desktop_signals(monkeypatch):
    nh = NerdHerd(metrics_port=0)
    monkeypatch.setattr(nh._presence, "collect",
                        lambda: {"user_idle_s": 5.0, "foreground_fullscreen": True})
    from nerd_herd.types import SystemState
    monkeypatch.setattr(nh._gpu, "system_state",
                        lambda: SystemState(ram_total_mb=32000, ram_available_mb=4000))
    monkeypatch.setattr(nh._load, "get_external_gpu_fraction", lambda: 0.7)
    nh._load.set_load_mode("balanced", source="user")

    snap = nh.snapshot()
    assert snap.load_mode == "balanced"
    assert snap.user_idle_s == 5.0
    assert snap.foreground_fullscreen is True
    assert snap.ram_available_mb == 4000
    assert snap.ram_total_mb == 32000
    assert snap.external_gpu_fraction == 0.7
