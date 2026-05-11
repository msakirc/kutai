"""Z6 T1D — /actions, /action_done, and inline-button handler."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z6_td.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()
    return db_mod, fa


def _make_iface():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._reply = AsyncMock()
    iface.app = MagicMock()
    iface.app.bot = MagicMock()
    return iface


@pytest.mark.asyncio
async def test_cmd_actions_empty(tmp_path, monkeypatch):
    _, _ = await _setup_db(tmp_path, monkeypatch)
    iface = _make_iface()
    update = MagicMock()
    ctx = MagicMock()
    ctx.args = []
    await iface.cmd_actions(update, ctx)
    msgs = [c.args[1] for c in iface._reply.await_args_list]
    assert any("All clear" in m for m in msgs)


@pytest.mark.asyncio
async def test_cmd_actions_lists_pending(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    await fa.create(
        mid, "credential_paste", "paste stripe", "why",
        ["go to dashboard"], blocking_step_id="13.12",
        notify_telegram=False,
    )
    iface = _make_iface()
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    ctx = MagicMock()
    ctx.args = []
    await iface.cmd_actions(update, ctx)
    # Index message first.
    idx_msg = iface._reply.await_args_list[0].args[1]
    assert "Pending founder_actions" in idx_msg
    assert "[credential_paste]" in idx_msg
    # Card should be sent via reply_text.
    update.message.reply_text.assert_awaited()


@pytest.mark.asyncio
async def test_cmd_action_done_resolves(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a = await fa.create(
        mid, "generic", "t", "w", [], notify_telegram=False,
    )
    iface = _make_iface()
    update = MagicMock()
    ctx = MagicMock()
    ctx.args = [str(a.id)]
    await iface.cmd_action_done(update, ctx)
    refreshed = await fa.get(a.id)
    assert refreshed.status == "done"


@pytest.mark.asyncio
async def test_cmd_action_done_with_payload(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a = await fa.create(
        mid, "generic", "t", "w", [], notify_telegram=False,
    )
    iface = _make_iface()
    update = MagicMock()
    ctx = MagicMock()
    ctx.args = [str(a.id), '{"order_id":', '"X-123"}']
    await iface.cmd_action_done(update, ctx)
    refreshed = await fa.get(a.id)
    assert refreshed.status == "done"
    assert refreshed.response_payload == {"order_id": "X-123"}


@pytest.mark.asyncio
async def test_callback_done_flips_status(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a = await fa.create(
        mid, "generic", "t", "w", [], notify_telegram=False,
    )
    iface = _make_iface()
    update = MagicMock()
    update.callback_query = MagicMock()
    update.callback_query.data = f"fa_done_{a.id}"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    ctx = MagicMock()
    await iface._handle_founder_action_callback(update, ctx)
    refreshed = await fa.get(a.id)
    assert refreshed.status == "done"


@pytest.mark.asyncio
async def test_callback_inprogress(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a = await fa.create(
        mid, "generic", "t", "w", [], notify_telegram=False,
    )
    iface = _make_iface()
    update = MagicMock()
    update.callback_query = MagicMock()
    update.callback_query.data = f"fa_inprogress_{a.id}"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    ctx = MagicMock()
    await iface._handle_founder_action_callback(update, ctx)
    refreshed = await fa.get(a.id)
    assert refreshed.status == "in_progress"


@pytest.mark.asyncio
async def test_callback_block(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a = await fa.create(
        mid, "generic", "t", "w", [], notify_telegram=False,
    )
    iface = _make_iface()
    update = MagicMock()
    update.callback_query = MagicMock()
    update.callback_query.data = f"fa_block_{a.id}"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    ctx = MagicMock()
    await iface._handle_founder_action_callback(update, ctx)
    refreshed = await fa.get(a.id)
    assert refreshed.status == "blocked"
