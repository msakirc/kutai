from types import SimpleNamespace
from nerd_herd.signals.s14_contention import s14_contention

LOCAL = SimpleNamespace(is_local=True)
CLOUD = SimpleNamespace(is_local=False)


def test_cloud_always_zero():
    assert s14_contention(CLOUD, ram_available_mb=10, ram_total_mb=32000,
                          external_gpu_fraction=0.9) == 0.0


def test_external_gpu_heavy_hard_veto():
    assert s14_contention(LOCAL, ram_available_mb=16000, ram_total_mb=32000,
                          external_gpu_fraction=0.7) == -10.0


def test_low_ram_pressure_is_zero():
    assert s14_contention(LOCAL, ram_available_mb=16000, ram_total_mb=32000,
                          external_gpu_fraction=0.0) == 0.0


def test_high_ram_pressure_graded_negative():
    v = s14_contention(LOCAL, ram_available_mb=2000, ram_total_mb=32000,
                       external_gpu_fraction=0.0)
    assert -1.0 < v < 0.0


def test_critical_ram_pegs_minus_one():
    v = s14_contention(LOCAL, ram_available_mb=300, ram_total_mb=32000,
                       external_gpu_fraction=0.0)
    assert v == -1.0


def test_zero_total_ram_is_safe():
    assert s14_contention(LOCAL, ram_available_mb=0, ram_total_mb=0,
                          external_gpu_fraction=0.0) == 0.0
