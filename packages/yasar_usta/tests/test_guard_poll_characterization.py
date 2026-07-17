"""Characterize the callback dispatch of _telegram_poll_loop BEFORE the split.
We drive one batch of updates through the loop, then cancel it, asserting the
side effects (restart flag set, shutdown signal written, subprocess.stop called)."""
import asyncio
import pytest

from yasar_usta.config import GuardConfig
from yasar_usta.guard import ProcessGuard


def _guard(tmp_path):
    cfg = GuardConfig(
        name="T", app_name="App", command=["python"],
        log_dir=str(tmp_path / "logs"),
        telegram_token="x", telegram_chat_id="42",
    )
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return ProcessGuard(cfg)


class _Sub:
    def __init__(self):
        self.running = True
        self.stopped = 0
        self.process = None
    async def stop(self, timeout=None):
        self.stopped += 1
        self.running = False


async def _run_one_batch(g, updates):
    """Feed exactly one getUpdates batch, then make the loop exit."""
    calls = {"n": 0}
    async def get_updates(offset=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return updates
        raise asyncio.CancelledError()
    g.telegram.get_updates = get_updates
    g.telegram.answer_callback = lambda *a, **k: asyncio.sleep(0)
    g.telegram.delete = lambda *a, **k: asyncio.sleep(0)
    g.telegram.send = lambda *a, **k: asyncio.sleep(0)
    g._send = lambda *a, **k: asyncio.sleep(0)
    try:
        await g._telegram_poll_loop(0)
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_confirm_restart_sets_flag_and_stops(tmp_path):
    g = _guard(tmp_path)
    g.subprocess = _Sub()
    upd = [{"update_id": 1, "callback_query": {
        "id": "c", "data": "confirm_restart",
        "message": {"message_id": 9, "chat": {"id": 42}}}}]
    await _run_one_batch(g, upd)
    assert g._restart_requested is True
    assert g.subprocess.stopped == 1
    assert (tmp_path / "logs" / "shutdown.signal").read_text() == "restart"


@pytest.mark.asyncio
async def test_confirm_stop_sets_flag_and_stops(tmp_path):
    g = _guard(tmp_path)
    g.subprocess = _Sub()
    upd = [{"update_id": 1, "callback_query": {
        "id": "c", "data": "confirm_stop",
        "message": {"message_id": 9, "chat": {"id": 42}}}}]
    await _run_one_batch(g, upd)
    assert g._stop_requested is True
    assert g.subprocess.stopped == 1
    assert (tmp_path / "logs" / "shutdown.signal").read_text() == "stop"
