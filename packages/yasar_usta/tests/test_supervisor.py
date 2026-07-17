import asyncio
import pytest
from yasar_usta.config import GuardConfig
from yasar_usta.supervisor import TargetSupervisor


def _sup(tmp_path):
    cfg = GuardConfig(name="web", app_name="web", command=["python"],
                      log_dir=str(tmp_path / "logs"), backoff_steps=[1])
    sent = []
    async def notify(text, reply_markup=None):
        sent.append((text, reply_markup))
    sup = TargetSupervisor("demo", cfg, notify=notify,
                           reply_keyboard={"keyboard": [[{"text": "X"}]]})
    return sup, sent


def test_request_restart_sets_flag_and_signal(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.request_restart()
    assert sup._restart_requested is True
    assert (tmp_path / "logs" / "shutdown.signal").read_text() == "restart"


def test_request_stop_sets_flag_and_signal(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.request_stop()
    assert sup._stop_requested is True
    assert (tmp_path / "logs" / "shutdown.signal").read_text() == "stop"


def test_status_snapshot_shape(tmp_path):
    sup, _ = _sup(tmp_path)
    s = sup.status()
    assert s["project_id"] == "demo"
    assert s["running"] is False
    assert "heartbeat_age" in s and "total_crashes" in s


@pytest.mark.asyncio
async def test_crash_notification_carries_injected_keyboard(tmp_path):
    """Guards review finding #3: _notify_crash must NOT call a missing _kb();
    it attaches the injected reply_keyboard."""
    sup, sent = _sup(tmp_path)
    sup.subprocess.stderr_tail = ["boom"]
    await sup._notify_crash(1)
    assert sent, "crash notification not sent"
    text, kb = sent[-1]
    assert kb == {"keyboard": [[{"text": "X"}]]}  # injected keyboard, no AttributeError
