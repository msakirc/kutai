from nerd_herd.presence import PresenceCollector, _idle_seconds_from_ticks


def test_idle_normal():
    # 5s ago, no wrap
    assert _idle_seconds_from_ticks(now_ms=2_000_000, last_ms=1_995_000) == 5.0


def test_idle_high_uptime_no_false_present():
    # ~25.5 days uptime; last input 5s ago. Naive signed read would floor to 0.
    now = 2_200_000_000
    assert _idle_seconds_from_ticks(now_ms=now, last_ms=now - 5000) == 5.0


def test_idle_across_32bit_wrap():
    # now wrapped past 2^32; last input 1s before the wrap
    now = 500            # wrapped (i.e. true now = 2^32 + 500)
    last = (2**32 - 500) # 1000 ms before wrap point
    assert _idle_seconds_from_ticks(now_ms=now, last_ms=last) == 1.0


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
