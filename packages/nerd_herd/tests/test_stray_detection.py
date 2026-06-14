"""nerd_herd must turn a silent metrics blackout into a loud signal.

In the 2026-06-14 incident, llama-server ran on the wrong port: nerd_herd's
metrics fetch failed forever and it silently reported "no model", which a
monitor read as "VRAM full + no model = leak". The collector now warns once
after N consecutive metric failures (caller cross-checks that a llama-server
process actually exists → wrong-port stray, not a VRAM leak).
"""
from nerd_herd.inference import InferenceCollector


def _collector():
    return InferenceCollector(llama_server_url="http://127.0.0.1:8081")


def test_warns_exactly_once_after_threshold():
    c = _collector()
    assert c._should_warn_stray() is False  # fail 1
    assert c._should_warn_stray() is False  # fail 2
    assert c._should_warn_stray() is True   # fail 3 → warn
    assert c._should_warn_stray() is False  # already warned


def test_success_resets_counter():
    c = _collector()
    c._should_warn_stray()
    c._should_warn_stray()
    c._note_fetch_success()
    assert c._should_warn_stray() is False  # back to fail 1
    assert c._should_warn_stray() is False  # fail 2
    assert c._should_warn_stray() is True   # fail 3 → warn again
