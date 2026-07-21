import asyncio
from pathlib import Path
import pytest
from yasar_usta.config import GuardConfig, HubConfig, ProjectConfig
from yasar_usta.hub import Hub
from yasar_usta.singleton import ERROR_ALREADY_EXISTS
from yasar_usta.singleton import _win32_create_mutex as _real_mutex


class _FakeMutex:
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def __call__(self, qualified_name):
        self.calls.append(qualified_name)
        return self._results.pop(0)


def _project(pid, tmp_path):
    cfg = GuardConfig(name=pid, app_name=pid, command=["python"],
                      log_dir=str(tmp_path / pid / "logs"), backoff_steps=[1])
    return ProjectConfig(id=pid, name=pid.title(), targets=[cfg])


def _hub(tmp_path, pids):
    hub_cfg = HubConfig(name="Hub", telegram_token="", telegram_chat_id="",
                        log_dir=str(tmp_path / "hublogs"))
    projects = [_project(p, tmp_path) for p in pids]
    return Hub(hub_cfg, projects)


@pytest.fixture(autouse=True)
def _no_lock(monkeypatch):
    monkeypatch.setattr("yasar_usta.hub.acquire_lock", lambda *a, **k: None)
    monkeypatch.setattr("yasar_usta.hub.release_lock", lambda *a, **k: None)


def test_hub_builds_one_supervisor_per_target(tmp_path):
    hub = _hub(tmp_path, ["kutai", "foo"])
    assert set(hub.supervisors.keys()) == {"kutai", "foo"}


def test_single_target_display_name_is_project_name(tmp_path):
    # _project sets ProjectConfig.name = pid.title(), cfg.name = pid.
    hub = _hub(tmp_path, ["kutai"])
    assert hub.supervisors["kutai"].display_name == "Kutai"


def test_multi_target_display_name_includes_target(tmp_path):
    cfg1 = GuardConfig(name="backend", app_name="Backend", command=["python"],
                       log_dir=str(tmp_path / "l1"), backoff_steps=[1])
    cfg2 = GuardConfig(name="bot", app_name="Bot", command=["python"],
                       log_dir=str(tmp_path / "l2"), backoff_steps=[1])
    proj = ProjectConfig(id="p2", name="MyApp", targets=[cfg1, cfg2])
    hub_cfg = HubConfig(name="Hub", telegram_token="", telegram_chat_id="",
                        log_dir=str(tmp_path / "h"))
    hub = Hub(hub_cfg, [proj])
    dn = {rid: s.display_name for rid, s in hub.supervisors.items()}
    assert dn["p2:backend"] == "MyApp · Backend"
    assert dn["p2:bot"] == "MyApp · Bot"


@pytest.mark.asyncio
async def test_hub_restart_button_asks_confirmation(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    sent = []
    hub.telegram.send = lambda text, reply_markup=None: (
        sent.append((text, reply_markup)) or asyncio.sleep(0))
    ran = {"n": 0}
    async def _do():
        ran["n"] += 1
    hub._do_restart_hub = _do
    await hub._route_callback("restart_hub", cb_msg_id=None)
    assert ran["n"] == 0  # a mis-tap must NOT restart the whole hub immediately
    assert "confirm_restart_hub" in str(sent)


@pytest.mark.asyncio
async def test_confirm_restart_hub_executes(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    hub.telegram.delete = lambda *a, **k: asyncio.sleep(0)
    hub._notify = lambda *a, **k: asyncio.sleep(0)
    ran = {"n": 0}
    async def _do():
        ran["n"] += 1
    hub._do_restart_hub = _do
    await hub._route_callback("confirm_restart_hub", cb_msg_id=5)
    assert ran["n"] == 1


def test_singleton_gate_exits_zero_when_another_hub_owns_it(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    calls = []
    hub._create_mutex = _FakeMutex([(1, ERROR_ALREADY_EXISTS)])
    hub._singleton_exit = lambda c: calls.append(c)
    hub._acquire_singleton()
    assert calls == [0]  # a second hub must exit, not proceed


def test_singleton_gate_proceeds_when_owned(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    calls = []
    hub._create_mutex = _FakeMutex([(1, 0)])
    hub._singleton_exit = lambda c: calls.append(c)
    hub._acquire_singleton()
    assert calls == []  # we own it → run
    assert hub._create_mutex.calls == ["Global\\YasarUstaHub"]


def test_hub_alive_path_under_log_dir(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    ap = hub._hub_alive_path()
    assert ap.endswith("hub.alive")
    assert "hublogs" in ap  # cfg.log_dir = tmp_path/hublogs


@pytest.mark.asyncio
async def test_run_writes_hub_alive(tmp_path):
    """run() must write hub.alive (the watchdog's liveness signal) from a task
    decoupled from the crash/backoff loop."""
    hub = _hub(tmp_path, ["kutai"])
    hub._acquire_singleton = lambda: None

    async def _boom():
        raise RuntimeError("stop")

    for sup in hub.supervisors.values():
        sup.run = _boom
    hub._stop_poller = lambda: asyncio.sleep(0)
    await asyncio.wait_for(hub.run(), timeout=5)
    assert (Path(hub.cfg.log_dir) / "hub.alive").exists()


def test_tests_never_touch_the_real_global_mutex(tmp_path):
    """Regression guard: the autouse conftest fixture must neutralize the real
    Win32 CreateMutexW so no test acquires/holds the machine-global prod mutex
    (Global\\YasarUstaHub). Without it, a test run holds the prod mutex → the
    live hub can't start, and post-restart integration tests sys.exit(0)."""
    hub = _hub(tmp_path, ["kutai"])
    assert hub._create_mutex is not _real_mutex


@pytest.mark.asyncio
async def test_confirm_restart_targets_only_named_project(tmp_path):
    hub = _hub(tmp_path, ["kutai", "foo"])
    hub.telegram.delete = lambda *a, **k: asyncio.sleep(0)
    hub._notify = lambda *a, **k: asyncio.sleep(0)
    called = {"kutai": 0, "foo": 0}
    for pid, sup in hub.supervisors.items():
        sup.request_restart = lambda p=pid: called.__setitem__(p, called[p] + 1)
        sup.do_restart_now = lambda: asyncio.sleep(0)
    # confirm_restart is the action; bare restart only opens a dialog
    await hub._route_callback("confirm_restart:foo", cb_msg_id=None)
    assert called == {"kutai": 0, "foo": 1}


@pytest.mark.asyncio
async def test_bare_restart_sends_confirm_dialog_no_action(tmp_path):
    hub = _hub(tmp_path, ["kutai", "foo"])
    sent = []
    hub.telegram.send = lambda text, reply_markup=None: sent.append(reply_markup) or asyncio.sleep(0)
    acted = {"n": 0}
    hub.supervisors["foo"].request_restart = lambda: acted.__setitem__("n", 1)
    await hub._route_callback("restart:foo", cb_msg_id=None)
    assert acted["n"] == 0  # no action yet
    assert "confirm_restart:foo" in str(sent)  # dialog carries the pid


def test_bare_verb_rejected_when_multiple_projects(tmp_path):
    hub = _hub(tmp_path, ["kutai", "foo"])
    assert hub._resolve_bare_target() is None  # ambiguous → None


def test_bare_verb_resolves_when_single_project(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    assert hub._resolve_bare_target().project_id == "kutai"


@pytest.mark.asyncio
async def test_poll_status_command_sends_dashboard(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    hub.cfg.telegram_chat_id = "42"
    hub.telegram.token = "x"; hub.telegram.chat_id = "42"
    dash = {"n": 0}
    async def _send_dash(edit_message_id=None):
        dash["n"] += 1
    hub._send_dashboard = _send_dash
    calls = {"n": 0}
    async def get_updates(offset=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"update_id": 1, "message": {"text": "/status", "chat": {"id": 42}}}]
        raise asyncio.CancelledError()
    hub.telegram.get_updates = get_updates
    try:
        await hub._poll_loop(0)
    except asyncio.CancelledError:
        pass
    assert dash["n"] == 1


@pytest.mark.asyncio
async def test_kutai_aliases_route(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    hub.cfg.telegram_chat_id = "42"
    hits = {"dash": 0, "start": 0}
    async def _dash(edit_message_id=None):
        hits["dash"] += 1
    hub._send_dashboard = _dash
    async def _bare(verb):
        if verb == "start":
            hits["start"] += 1
    hub._for_bare_target = _bare
    await hub._route_text("/kutai_status")
    await hub._route_text("/kutai_start")
    assert hits == {"dash": 1, "start": 1}


@pytest.mark.asyncio
async def test_supervisor_crash_does_not_kill_hub(tmp_path):
    """A supervisor run() raising must NOT propagate out of Hub.run(); the hub
    must still clean up the poller."""
    hub = _hub(tmp_path, ["kutai", "foo"])
    hub._acquire_singleton = lambda: None  # don't touch the real Win32 mutex
    monkey_stopped = {"n": 0}
    async def _boom():
        raise RuntimeError("boom")
    for sup in hub.supervisors.values():
        sup.run = _boom
    # stop the poller cleanly regardless
    async def _stop():
        monkey_stopped["n"] += 1
    hub._stop_poller = _stop
    # no telegram → poller not started; run() should return without raising
    await asyncio.wait_for(hub.run(), timeout=5)
    assert monkey_stopped["n"] == 1  # _stop_poller ran in finally


@pytest.mark.asyncio
async def test_shutdown_watcher_fans_out_stop(tmp_path):
    hub = _hub(tmp_path, ["kutai", "foo"])
    stopped = {"kutai": 0, "foo": 0}
    for pid, sup in hub.supervisors.items():
        async def _s(p=pid):
            stopped[p] += 1
        sup.do_stop_now = _s
    hub._shutdown = True  # already requested
    await hub._shutdown_watcher()  # returns immediately, fans out
    assert stopped == {"kutai": 1, "foo": 1}


@pytest.mark.asyncio
async def test_route_restart_sidecar(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    hub._notify = lambda *a, **k: asyncio.sleep(0)
    sup = hub.supervisors["kutai"]
    calls = {"stop": 0, "start": 0}
    class _SC:
        async def stop(self): calls["stop"] += 1
        async def start(self): calls["start"] += 1
    sup.sidecars = {"yazbunu": _SC()}
    await hub._route_callback("restart_sidecar:kutai:yazbunu", cb_msg_id=None)
    assert calls == {"stop": 1, "start": 1}


@pytest.mark.asyncio
async def test_logs_command_parses_count_and_links_yazbunu(tmp_path, monkeypatch):
    import yasar_usta.hub as hubmod
    hub = _hub(tmp_path, ["kutai"])
    hub.cfg.telegram_chat_id = "42"
    sent = []
    hub._notify = lambda text, **k: sent.append(text) or asyncio.sleep(0)
    captured = {"n": None}
    def _fmt(path, n):
        captured["n"] = n
        return "log-line"
    monkeypatch.setattr(hubmod, "format_log_entries", _fmt)
    # yazbunu sidecar alive with a health_url
    class _Yaz:
        health_url = "http://127.0.0.1:9880/health"
        async def http_alive(self): return True
    hub.supervisors["kutai"].sidecars = {"yazbunu": _Yaz()}
    await hub._route_text("/logs 35")
    assert captured["n"] == 35
    assert any("Yazbunu Log Viewer" in s for s in sent)


@pytest.mark.asyncio
async def test_send_dashboard_offloads_blocking_build(tmp_path, monkeypatch):
    """The dashboard text build (which does blocking tasklist calls) must run
    via asyncio.to_thread, not directly on the event loop."""
    import yasar_usta.hub as hubmod
    hub = _hub(tmp_path, ["kutai"])
    hub.telegram.send = lambda *a, **k: asyncio.sleep(0)  # returns coroutine
    used = {"to_thread": 0}
    real_to_thread = asyncio.to_thread
    async def _tt(fn, *a, **k):
        used["to_thread"] += 1
        return await real_to_thread(fn, *a, **k)
    monkeypatch.setattr(hubmod.asyncio, "to_thread", _tt)
    await hub._send_dashboard()
    assert used["to_thread"] == 1


@pytest.mark.asyncio
async def test_notify_retries_plain_on_markdown_fail(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    calls = []
    async def _send(text, reply_markup=None, parse_mode="Markdown"):
        calls.append(parse_mode)
        return {"ok": False} if len(calls) == 1 else {"ok": True}
    hub.telegram.send = _send
    await hub._notify("boom ``` unbalanced")
    assert calls == ["Markdown", None]  # retried as plain text


@pytest.mark.asyncio
async def test_sidecar_health_concurrent_and_no_double_http(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    class _SC:
        def __init__(self, name): self.name = name; self.http = 0
        async def http_alive(self): self.http += 1; return True
        def pid_alive(self): return 111
    sc = _SC("yazbunu")
    hub.supervisors["kutai"].sidecars = {"yazbunu": sc}
    hub.telegram.send = lambda *a, **k: asyncio.sleep(0)
    await hub._send_dashboard()
    assert sc.http == 1  # exactly one http check, no redundant is_alive() call
