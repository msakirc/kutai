from nerd_herd.modifiers import M4_load_mode_weights


def test_full_silences_desktop_signals():
    w = M4_load_mode_weights(mode="full")
    assert w["S13"] == 0.0 and w["S14"] == 0.0


def test_balanced_amplifies():
    w = M4_load_mode_weights(mode="balanced")
    assert w["S13"] == 2.0 and w["S14"] == 2.0


def test_minimal_passthrough():
    w = M4_load_mode_weights(mode="minimal")
    assert w["S13"] == 1.0 and w["S14"] == 1.0


def test_unknown_mode_passthrough():
    w = M4_load_mode_weights(mode="bogus")
    assert w["S13"] == 1.0 and w["S14"] == 1.0


def test_only_touches_s13_s14():
    w = M4_load_mode_weights(mode="balanced")
    assert set(w.keys()) == {"S13", "S14"}
