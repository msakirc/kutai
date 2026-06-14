"""nerd_herd must turn a silent metrics blackout into a loud signal.

In the 2026-06-14 incident, llama-server ran on the wrong port: nerd_herd's
metrics fetch failed forever and it silently reported "no model", which a
monitor read as "VRAM full + no model = leak". The collector now warns once,
after N consecutive metric failures, ONLY when a llama-server process actually
exists (wrong-port stray, not a real VRAM leak). The warn latch must not be
consumed unless the warning is actually emitted, so a transient process-probe
miss at the threshold cannot permanently suppress the alarm.
"""
import nerd_herd.inference as inference_mod
from nerd_herd.inference import InferenceCollector


class _SpyLogger:
    def __init__(self):
        self.warnings = 0

    def warning(self, *a, **k):
        self.warnings += 1

    def debug(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


def _collector(monkeypatch, *, process_present=True):
    c = InferenceCollector(llama_server_url="http://127.0.0.1:8081")
    monkeypatch.setattr(c, "_llama_server_running", lambda: process_present)
    spy = _SpyLogger()
    monkeypatch.setattr(inference_mod, "logger", spy)
    return c, spy


def test_warns_once_after_threshold_when_process_present(monkeypatch):
    c, spy = _collector(monkeypatch, process_present=True)
    for _ in range(5):
        c._handle_fetch_failure("boom")
    assert spy.warnings == 1


def test_no_warn_below_threshold(monkeypatch):
    c, spy = _collector(monkeypatch, process_present=True)
    c._handle_fetch_failure("boom")
    c._handle_fetch_failure("boom")
    assert spy.warnings == 0


def test_no_warn_when_process_absent(monkeypatch):
    c, spy = _collector(monkeypatch, process_present=False)
    for _ in range(5):
        c._handle_fetch_failure("boom")
    assert spy.warnings == 0


def test_latch_not_consumed_until_warning_emitted(monkeypatch):
    """Process absent at the threshold must NOT burn the one-shot latch —
    the warning still fires once the process is later detected."""
    c, spy = _collector(monkeypatch, process_present=False)
    for _ in range(4):  # past threshold, but process absent → no warn, no latch
        c._handle_fetch_failure("boom")
    assert spy.warnings == 0
    monkeypatch.setattr(c, "_llama_server_running", lambda: True)
    c._handle_fetch_failure("boom")  # now detectable → warn exactly once
    assert spy.warnings == 1


def test_success_resets_counter(monkeypatch):
    c, spy = _collector(monkeypatch, process_present=True)
    c._handle_fetch_failure("boom")
    c._handle_fetch_failure("boom")
    c._note_fetch_success()
    c._handle_fetch_failure("boom")
    c._handle_fetch_failure("boom")
    assert spy.warnings == 0  # only 2 fails since reset
    c._handle_fetch_failure("boom")
    assert spy.warnings == 1
