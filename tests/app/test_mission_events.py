"""Z10 T2B (D2) — typed mission events: post_event + formatters."""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, MagicMock


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "events.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


def test_format_milestone():
    from src.app.mission_events import format_event
    text, kb = format_event("milestone", {"summary": "merged X"})
    assert "🎯" in text and "milestone" in text and "merged X" in text
    assert kb is None


def test_format_blocker():
    from src.app.mission_events import format_event
    text, _ = format_event("blocker", {"reason": "API key missing"})
    assert "🚧" in text and "blocker" in text and "API key missing" in text


def test_format_asking_with_options():
    from src.app.mission_events import format_event
    text, kb = format_event(
        "asking",
        {"question": "Pick a", "options": ["A", "B"]},
    )
    assert "❓" in text and "Pick a" in text
    assert kb is not None
    # 2 rows × 1 button each.
    flat = [b for row in kb.inline_keyboard for b in row]
    assert len(flat) == 2
    assert all("event:answer:__EID__:" in b.callback_data for b in flat)


def test_format_confirmation_required():
    from src.app.mission_events import format_event
    text, kb = format_event(
        "confirmation_required",
        {"verb": "rm_rf", "reversibility": "irreversible",
         "payload_summary": "rm /tmp", "confirmation_id": 5},
    )
    assert "⚠️" in text and "rm_rf" in text and "irreversible" in text
    assert kb is not None
    buttons = [b for row in kb.inline_keyboard for b in row]
    assert any("confirm:approve:5" in b.callback_data for b in buttons)
    assert any("confirm:reject:5" in b.callback_data for b in buttons)


def test_format_cost_alert():
    from src.app.mission_events import format_event
    text, kb = format_event(
        "cost_alert",
        {"mission_id": 7, "threshold_pct": 80, "total": 1.23, "ceiling": 1.50},
    )
    assert "💸" in text and "Mission 7" in text and "80%" in text


@pytest.mark.asyncio
async def test_post_event_writes_row_and_sends(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Demo", "x")

    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=777))

    from src.app.mission_events import post_event
    event_id = await post_event(
        bot, mid, "milestone", {"summary": "X"}, chat_id=-1001,
    )
    assert event_id > 0
    bot.send_message.assert_awaited_once()

    # Row persisted with telegram_message_id stamped.
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT mission_id, kind, payload, telegram_message_id FROM mission_events "
        "WHERE id = ?", (event_id,),
    )
    row = await cur.fetchone()
    assert row[0] == mid
    assert row[1] == "milestone"
    assert json.loads(row[2])["summary"] == "X"
    assert row[3] == 777


@pytest.mark.asyncio
async def test_post_event_unknown_kind_raises():
    from src.app.mission_events import post_event
    bot = MagicMock()
    with pytest.raises(ValueError):
        await post_event(bot, 1, "bogus", {}, chat_id=-1)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_post_event_asking_rewrites_event_id(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Demo", "x")

    captured = {}

    async def _send(*, chat_id, text, reply_markup=None, **kw):
        captured["text"] = text
        captured["reply_markup"] = reply_markup
        return MagicMock(message_id=42)

    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=_send)

    from src.app.mission_events import post_event
    event_id = await post_event(
        bot, mid, "asking",
        {"question": "?", "options": ["A", "B"]},
        chat_id=-1,
    )
    kb = captured["reply_markup"]
    flat = [b for row in kb.inline_keyboard for b in row]
    # __EID__ must be rewritten to the actual event_id.
    assert all(f"event:answer:{event_id}:" in b.callback_data for b in flat)
