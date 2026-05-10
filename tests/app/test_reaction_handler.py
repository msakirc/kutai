"""Z10 T2B (D3) — reaction handler resolves confirmations + events."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "react.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


def _iface_skeleton():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface.app = MagicMock()
    iface.app.bot = MagicMock()
    iface.app.bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    return iface


def _mock_update(callback_data: str):
    update = MagicMock()
    update.effective_chat.id = 42
    update.callback_query.data = callback_data
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.message.text = "old text"
    return update


@pytest.mark.asyncio
async def test_confirm_approve_resolves(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    cid = await db_mod.request_confirmation(
        task_id=1, verb="x", reversibility="partial", payload_summary="y",
    )
    iface = _iface_skeleton()
    update = _mock_update(f"confirm:approve:{cid}")
    await iface._handle_mission_event_callback(update, MagicMock())
    state = await db_mod.check_confirmation(cid)
    assert state["verdict"] == "approved"
    update.callback_query.edit_message_text.assert_awaited()


@pytest.mark.asyncio
async def test_confirm_reject_resolves(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    cid = await db_mod.request_confirmation(
        task_id=1, verb="x", reversibility="partial", payload_summary="y",
    )
    iface = _iface_skeleton()
    update = _mock_update(f"confirm:reject:{cid}")
    await iface._handle_mission_event_callback(update, MagicMock())
    state = await db_mod.check_confirmation(cid)
    assert state["verdict"] == "rejected"


@pytest.mark.asyncio
async def test_event_reject_posts_blocker_followup(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Demo", "x")

    # Seed an 'asking' event manually.
    db = await db_mod.get_db()
    cur = await db.execute(
        "INSERT INTO mission_events (mission_id, kind, payload) VALUES (?, ?, ?)",
        (mid, "asking", '{"question": "do X?"}'),
    )
    await db.commit()
    event_id = cur.lastrowid

    iface = _iface_skeleton()
    update = _mock_update(f"event:reject:{event_id}")

    # Patch post_event so we can verify the follow-up blocker.
    with patch(
        "src.app.mission_events.post_event", new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = 999
        await iface._handle_mission_event_callback(update, MagicMock())

    # Event row resolved.
    cur = await db.execute(
        "SELECT resolution FROM mission_events WHERE id = ?", (event_id,),
    )
    row = await cur.fetchone()
    assert row[0] == "reject"

    # Follow-up blocker posted via post_event mock.
    assert mock_post.await_count == 1
    call_kwargs = mock_post.await_args
    assert call_kwargs.args[2] == "blocker"


@pytest.mark.asyncio
async def test_event_answer_records_option(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Demo", "x")
    db = await db_mod.get_db()
    cur = await db.execute(
        "INSERT INTO mission_events (mission_id, kind, payload) VALUES (?, ?, ?)",
        (mid, "asking", '{"question": "?", "options": ["A", "B"]}'),
    )
    await db.commit()
    event_id = cur.lastrowid

    iface = _iface_skeleton()
    update = _mock_update(f"event:answer:{event_id}:1")
    await iface._handle_mission_event_callback(update, MagicMock())

    cur = await db.execute(
        "SELECT resolution FROM mission_events WHERE id = ?", (event_id,),
    )
    row = await cur.fetchone()
    assert row[0] == "answer:1"
