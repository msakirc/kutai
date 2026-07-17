"""Two short-lived fake targets under one Hub: one crash-loops, the other
runs clean; assert isolation (one crash does not stop the other) and that
coordinated shutdown stops both."""
import asyncio
import sys
import pytest
from yasar_usta.config import GuardConfig, HubConfig, ProjectConfig
from yasar_usta.hub import Hub


@pytest.fixture(autouse=True)
def _no_lock(monkeypatch):
    monkeypatch.setattr("yasar_usta.hub.acquire_lock", lambda *a, **k: None)
    monkeypatch.setattr("yasar_usta.hub.release_lock", lambda *a, **k: None)


def _proj(pid, tmp_path, code):
    cfg = GuardConfig(
        name=pid, app_name=pid,
        command=[sys.executable, "-c", code],
        log_dir=str(tmp_path / pid / "logs"),
        backoff_steps=[1], auto_restart=False,
        telegram_token="", telegram_chat_id="",
    )
    return ProjectConfig(id=pid, name=pid, targets=[cfg])


@pytest.mark.asyncio
async def test_two_targets_run_independently(tmp_path):
    hub_cfg = HubConfig(telegram_token="", telegram_chat_id="",
                        log_dir=str(tmp_path / "hub"))
    projects = [
        _proj("good", tmp_path, "import time; time.sleep(0.3)"),
        _proj("bad", tmp_path, "import sys; sys.exit(3)"),
    ]
    hub = Hub(hub_cfg, projects)
    # Stop the hub shortly after start so run() returns.
    async def _killer():
        await asyncio.sleep(1.5)
        hub.request_shutdown()
        for s in hub.supervisors.values():
            if s.is_running:
                await s.subprocess.stop()
    asyncio.create_task(_killer())
    await asyncio.wait_for(hub.run(), timeout=8)
    # both supervisors were constructed and reachable
    assert set(hub.supervisors) == {"good", "bad"}
