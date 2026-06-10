from types import SimpleNamespace
from nerd_herd.signals.s13_presence import s13_presence, PRESENT_IDLE_S

LOCAL = SimpleNamespace(is_local=True)
CLOUD = SimpleNamespace(is_local=False)


def test_cloud_always_zero():
    assert s13_presence(CLOUD, user_idle_s=0.0, foreground_fullscreen=True) == 0.0


def test_away_is_zero():
    assert s13_presence(LOCAL, user_idle_s=10_000.0, foreground_fullscreen=False) == 0.0


def test_fullscreen_hard_veto():
    assert s13_presence(LOCAL, user_idle_s=1.0, foreground_fullscreen=True) == -10.0


def test_present_normal_is_graded_negative():
    v = s13_presence(LOCAL, user_idle_s=1.0, foreground_fullscreen=False)
    assert -0.6 <= v <= -0.3


def test_just_past_present_threshold_fades_to_zero():
    v = s13_presence(LOCAL, user_idle_s=PRESENT_IDLE_S, foreground_fullscreen=False)
    assert -0.05 <= v <= 0.0
