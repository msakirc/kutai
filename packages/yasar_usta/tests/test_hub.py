import asyncio
import pytest
from yasar_usta.config import GuardConfig, HubConfig, ProjectConfig
from yasar_usta.hub import Hub


def _project(pid, tmp_path):
    cfg = GuardConfig(name=pid, app_name=pid, command=["python"],
                      log_dir=str(tmp_path / pid / "logs"), backoff_steps=[1])
    return ProjectConfig(id=pid, name=pid.title(), targets=[cfg])


def _hub(tmp_path, pids):
    hub_cfg = HubConfig(name="Hub", telegram_token="", telegram_chat_id="",
                        log_dir=str(tmp_path / "hublogs"))
    projects = [_project(p, tmp_path) for p in pids]
    return Hub(hub_cfg, projects)


def test_hub_builds_one_supervisor_per_target(tmp_path):
    hub = _hub(tmp_path, ["kutai", "foo"])
    assert set(hub.supervisors.keys()) == {"kutai", "foo"}


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
