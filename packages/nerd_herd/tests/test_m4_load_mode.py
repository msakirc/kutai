from nerd_herd.modifiers import M4_load_mode_weights


def test_full_silences_desktop_signals():
    w = M4_load_mode_weights(mode="full")
    assert w["S13"] == 0.0 and w["S14"] == 0.0


def test_heavy_amplifies():
    w = M4_load_mode_weights(mode="heavy")
    assert w["S13"] >= 1.0 and w["S14"] >= 1.0


def test_shared_amplifies_more_than_heavy():
    assert M4_load_mode_weights(mode="shared")["S13"] > M4_load_mode_weights(mode="heavy")["S13"]


def test_unknown_mode_is_passthrough():
    w = M4_load_mode_weights(mode="bogus")
    assert w["S13"] == 1.0 and w["S14"] == 1.0


def test_only_touches_s13_s14():
    w = M4_load_mode_weights(mode="shared")
    assert set(w.keys()) == {"S13", "S14"}
