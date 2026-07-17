"""Characterization tests: lock ProcessGuard.run() branch behavior BEFORE the
Hub/TargetSupervisor split. These assert observable effects (notifications sent,
app (re)started, intent flags consumed) for each exit-code branch."""
import asyncio
import types
import pytest

from yasar_usta.config import GuardConfig
from yasar_usta.guard import ProcessGuard


@pytest.fixture(autouse=True)
def _no_lock(monkeypatch):
    """run() calls acquire_lock (module-global handle + atexit, never released on
    normal shutdown). No-op it so tests don't leak locked files on tmp dirs."""
    monkeypatch.setattr("yasar_usta.guard.acquire_lock", lambda *a, **k: None)


def _guard(tmp_path, **over):
    cfg = GuardConfig(
        name="T", app_name="App",
        command=["python", "-c", "pass"],
        log_dir=str(tmp_path / "logs"),
        telegram_token="", telegram_chat_id="",  # telegram disabled → no network
        backoff_steps=[1], auto_restart=True,
        **over,
    )
    return ProcessGuard(cfg)


class _FakeSub:
    """Scriptable stand-in for SubprocessManager."""
    def __init__(self, exit_codes):
        self._codes = list(exit_codes)
        self.running = False
        self.started = 0
        self.stopped = 0
        self.stderr_tail = []
        self.command = ["python"]
        self.process = None
    async def start(self):
        self.started += 1
        self.running = bool(self._codes)
    async def stop(self, timeout=None):
        self.stopped += 1
        self.running = False
    async def wait_for_exit(self):
        if not self._codes:
            self.running = False
            return 0
        code = self._codes.pop(0)
        self.running = False
        return code
    def heartbeat_age(self):
        return 0.0


@pytest.mark.asyncio
async def test_run_restart_exit_code_restarts_app(tmp_path):
    g = _guard(tmp_path)
    # exit 42 (restart), then a clean 0 that we use to end the loop
    g.subprocess = _FakeSub([42, 0])
    sent = []
    g._send = lambda text, reply_markup=None: sent.append(text) or asyncio.sleep(0)
    g._notify_started = lambda: asyncio.sleep(0)
    g._start_signal_watcher = lambda: asyncio.sleep(0)
    # end the loop after two exits
    orig_wait = g.subprocess.wait_for_exit
    async def wait():
        code = await orig_wait()
        if not g.subprocess._codes:
            g._shutdown = True
        return code
    g.subprocess.wait_for_exit = wait
    await g.run()
    assert g.subprocess.started >= 2  # initial + restart


@pytest.mark.asyncio
async def test_run_restart_flag_consumed(tmp_path):
    g = _guard(tmp_path)
    g.subprocess = _FakeSub([0])
    g._restart_requested = True
    g._notify_started = lambda: asyncio.sleep(0)
    g._start_signal_watcher = lambda: asyncio.sleep(0)
    g._send = lambda *a, **k: asyncio.sleep(0)
    async def wait():
        g._shutdown = True
        g.subprocess.running = False
        return 0
    g.subprocess.wait_for_exit = wait
    await g.run()
    assert g._restart_requested is False  # flag consumed by the loop
