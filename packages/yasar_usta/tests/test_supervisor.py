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


def test_status_sidecars_is_snapshot_not_managers(tmp_path):
    sup, _ = _sup(tmp_path)
    # a supervisor with no sidecars → empty dict; values (if any) must be plain dicts
    scs = sup.status()["sidecars"]
    assert isinstance(scs, dict)
    for v in scs.values():
        assert isinstance(v, dict) and "name" in v


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


# ── run() loop coverage (parked-wait / intent honoring — B3) ─────────────

class _FakeSub:
    """Scriptable stand-in for SubprocessManager (mirrors the guard
    characterization harness). ``running`` is True only while scripted exit
    codes remain, so once the script is exhausted the loop parks."""
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
        self.running = bool(self._codes)  # running only if more scripted exits remain

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


def _run_sup(tmp_path, fake, **cfg_over):
    """Build a supervisor wired to a _FakeSub with no-op notify/watcher, so
    run() branch behavior can be observed deterministically."""
    cfg_over.setdefault("backoff_steps", [1])
    cfg = GuardConfig(name="web", app_name="web", command=["python"],
                      log_dir=str(tmp_path / "logs"), **cfg_over)
    sent = []
    async def notify(text, reply_markup=None):
        sent.append((text, reply_markup))
    sup = TargetSupervisor("demo", cfg, notify=notify,
                           reply_keyboard={"keyboard": [[{"text": "X"}]]})
    sup.subprocess = fake
    # Silence side-effecting coroutines the loop calls on (re)start.
    async def _noop(*a, **k):
        return None
    sup._notify_started = _noop
    sup._start_signal_watcher = _noop
    return sup, sent


def _end_loop_when_script_exhausted(sup):
    """Wrap wait_for_exit so the loop ends after the last scripted exit code
    (sets _shutdown once no codes remain) — keeps tests from looping forever."""
    fake = sup.subprocess
    orig_wait = fake.wait_for_exit
    async def wait():
        code = await orig_wait()
        if not fake._codes:
            sup._shutdown = True
        return code
    fake.wait_for_exit = wait


@pytest.mark.asyncio
async def test_restart_while_parked_is_not_lost(tmp_path):
    """B3 regression: app exits CLEAN (exit 0) with no intent set, so the post-exit
    _restart_requested check sees False and run() enters the parked wait. The Hub
    poller then fires a restart WHILE PARKED (injected at park-entry). The app MUST
    be (re)started. Before the fix the park subloop only woke on subprocess.running
    and never re-checked the flag → the restart was silently lost (test hangs / no
    2nd start); after the fix _park_until_wake honors the flag and starts the app."""
    fake = _FakeSub([0])           # one clean exit → then parks
    sup, _ = _run_sup(tmp_path, fake)
    # Observable start: count each start; the SECOND start (the one from the
    # parked wait) marks running True AND ends the loop so the test terminates.
    async def start():
        fake.started += 1
        if fake.started >= 2:      # the restart triggered from the parked wait
            fake.running = True
            sup._shutdown = True   # let run() exit cleanly after proving restart
    sup._start_app = start
    # Inject the poller's restart intent AT park entry — i.e. AFTER the post-exit
    # _restart_requested check has already run and seen False. This is the exact
    # B3 window: a restart requested while the app is parked/stopped.
    orig_notify_stopped = sup._notify_stopped
    async def notify_stopped_then_request_restart():
        await orig_notify_stopped()
        sup._restart_requested = True   # poller fires WHILE parked
    sup._notify_stopped = notify_stopped_then_request_restart
    await sup.run()
    assert fake.started >= 2, "restart requested while parked was lost (B3)"
    assert sup._restart_requested is False, "restart intent flag not consumed"


@pytest.mark.asyncio
async def test_start_while_parked(tmp_path):
    """A /start requested while parked (poller sets _start_requested) starts the
    app from the parked wait."""
    fake = _FakeSub([0])
    sup, _ = _run_sup(tmp_path, fake)
    async def start():
        fake.started += 1
        if fake.started >= 2:      # the start from the parked wait
            fake.running = True
            sup._shutdown = True   # let run() exit cleanly after proving start
    sup._start_app = start
    sup._start_requested = True
    await sup.run()
    assert fake.started >= 2, "app was not started from the parked wait"
    assert sup._start_requested is False, "start intent flag not consumed"


@pytest.mark.asyncio
async def test_restart_exit_code_restarts(tmp_path):
    """Running-app restart path still works: an exit with restart_exit_code (42)
    restarts unconditionally (no park, no intent flag needed)."""
    fake = _FakeSub([42, 0])       # restart-code exit, then clean 0 to end
    sup, _ = _run_sup(tmp_path, fake, restart_exit_code=42)
    _end_loop_when_script_exhausted(sup)
    await sup.run()
    assert fake.started >= 2, "restart_exit_code did not restart the app"


@pytest.mark.asyncio
async def test_hung_minus_one_restarts(tmp_path):
    """Hung path (-1) with no recent crash (last_crash_time unset) sends the hung
    notification and restarts after the timed backoff — not a parked wait."""
    fake = _FakeSub([-1, 0])       # hung, then clean 0 to end
    sup, sent = _run_sup(tmp_path, fake)
    assert sup.backoff.last_crash_time == 0  # unset → skip the rapid-crash park
    _end_loop_when_script_exhausted(sup)
    await sup.run()
    assert fake.started >= 2, "hung app was not restarted"
    assert any("hung" in (t or "").lower() or "5" in (t or "")
               for t, _ in sent), "hung notification not sent"


@pytest.mark.asyncio
async def test_crash_backoff_then_restart(tmp_path):
    """Crash path (exit 1) sends a crash notification then restarts after the
    (zero-length) backoff delay when auto_restart is on."""
    fake = _FakeSub([1, 0])        # crash, then clean 0 to end
    sup, sent = _run_sup(tmp_path, fake, backoff_steps=[0], auto_restart=True)
    _end_loop_when_script_exhausted(sup)
    await sup.run()
    assert fake.started >= 2, "crashed app was not restarted after backoff"
    assert sent, "crash notification not sent"
