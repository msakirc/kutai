import time
from pathlib import Path

from fatih_hoca.cloud.alert_throttle import AlertThrottle


def test_first_failure_alerts(tmp_path: Path):
    t = AlertThrottle(tmp_path / "throttle.json")
    assert t.should_alert("groq", current_state="auth_fail") is True


def test_repeat_failure_within_24h_suppressed(tmp_path: Path):
    t = AlertThrottle(tmp_path / "throttle.json")
    assert t.should_alert("groq", current_state="auth_fail") is True
    assert t.should_alert("groq", current_state="auth_fail") is False


def test_repeat_failure_after_24h_alerts(tmp_path: Path):
    t = AlertThrottle(tmp_path / "throttle.json")
    t.should_alert("groq", current_state="auth_fail")
    # Rewind last alert by 25h.
    t._state["groq"]["last_alert_unix"] = time.time() - (25 * 3600)
    t._save()
    assert t.should_alert("groq", current_state="auth_fail") is True


def test_state_flip_always_alerts(tmp_path: Path):
    t = AlertThrottle(tmp_path / "throttle.json")
    t.should_alert("groq", current_state="auth_fail")
    # Recovery transition is always alerted.
    assert t.should_alert("groq", current_state="ok") is True
    # Re-fail right after recovery is a transition too — alert.
    assert t.should_alert("groq", current_state="auth_fail") is True


def test_independent_per_provider(tmp_path: Path):
    t = AlertThrottle(tmp_path / "throttle.json")
    assert t.should_alert("groq", current_state="auth_fail") is True
    assert t.should_alert("openai", current_state="auth_fail") is True
