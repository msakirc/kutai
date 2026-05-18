"""Z6 T7B — founder_action UX polish: /missions count badge,
/mission detail pending-actions section, /actions empty-state tail."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z6_t7b.db"
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
async def test_cmd_missions_shows_action_badge(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("My mission", "describe")
    await fa.create(
        mid, "credential_paste", "Paste stripe", "why",
        ["dashboard"], notify_telegram=False,
    )
    await fa.create(
        mid, "vendor_enroll", "Enroll vercel", "why2",
        ["enroll"], notify_telegram=False,
    )
    iface = _make_iface()
    update = MagicMock()
    ctx = MagicMock()
    ctx.args = []
    await iface.cmd_missions(update, ctx)
    msg = iface._reply.await_args_list[0].args[1]
    assert "#" in msg
    assert "My mission" in msg
    assert "2 action(s) pending" in msg


@pytest.mark.asyncio
async def test_cmd_missions_no_badge_when_clear(tmp_path, monkeypatch):
    db_mod, _ = await _setup_db(tmp_path, monkeypatch)
    await db_mod.add_mission("Clean mission", "no actions")
    iface = _make_iface()
    update = MagicMock()
    ctx = MagicMock()
    ctx.args = []
    await iface.cmd_missions(update, ctx)
    msg = iface._reply.await_args_list[0].args[1]
    assert "Clean mission" in msg
    assert "action(s) pending" not in msg


@pytest.mark.asyncio
async def test_mission_view_shows_pending_section(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Detail mission", "desc")
    await fa.create(
        mid, "vendor_enroll", "Enroll vercel", "why",
        ["go"], notify_telegram=False,
    )
    await fa.create(
        mid, "credential_paste", "Paste stripe", "why2",
        ["go"], notify_telegram=False,
    )
    from src.app.mission_view import format_mission_view
    body = await format_mission_view(mid)
    assert "Pending founder_actions: 2" in body
    assert "Enroll vercel" in body
    assert "Paste stripe" in body


@pytest.mark.asyncio
async def test_mission_view_caps_titles_at_5(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Many", "desc")
    for i in range(8):
        await fa.create(
            mid, "generic", f"Action #{i}", "why",
            ["go"], notify_telegram=False,
        )
    from src.app.mission_view import format_mission_view
    body = await format_mission_view(mid)
    assert "Pending founder_actions: 8" in body
    assert "and 3 more" in body
    assert f"/actions {mid}" in body


@pytest.mark.asyncio
async def test_actions_empty_state_shows_last_resolved(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Mission", "")
    a = await fa.create(
        mid, "generic", "Done thing", "why",
        ["x"], notify_telegram=False,
    )
    await fa.resolve(a.id, response_payload={"ack": True})
    iface = _make_iface()
    update = MagicMock()
    ctx = MagicMock()
    ctx.args = []
    await iface.cmd_actions(update, ctx)
    msg = iface._reply.await_args_list[0].args[1]
    assert "All clear" in msg
    assert "Last resolved" in msg
    assert "Done thing" in msg


@pytest.mark.asyncio
async def test_actions_empty_state_no_tail_when_never_resolved(
    tmp_path, monkeypatch,
):
    _, _ = await _setup_db(tmp_path, monkeypatch)
    iface = _make_iface()
    update = MagicMock()
    ctx = MagicMock()
    ctx.args = []
    await iface.cmd_actions(update, ctx)
    msg = iface._reply.await_args_list[0].args[1]
    assert "All clear" in msg
    assert "Last resolved" not in msg
