from nerd_herd.presence import PresenceCollector


def test_degrades_to_away_when_apis_unavailable(monkeypatch):
    c = PresenceCollector()
    monkeypatch.setattr(c, "_idle_seconds_impl", lambda: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(c, "_fullscreen_impl", lambda: (_ for _ in ()).throw(OSError()))
    state = c.collect()
    assert state["user_idle_s"] >= 1e8
    assert state["foreground_fullscreen"] is False


def test_collect_returns_floats_and_bool(monkeypatch):
    c = PresenceCollector()
    monkeypatch.setattr(c, "_idle_seconds_impl", lambda: 12.5)
    monkeypatch.setattr(c, "_fullscreen_impl", lambda: True)
    state = c.collect()
    assert state["user_idle_s"] == 12.5
    assert state["foreground_fullscreen"] is True
