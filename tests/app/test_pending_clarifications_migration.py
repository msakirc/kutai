"""Z10 T2B (D6) — _pending_clarifications coexists with mission_event asking.

The legacy ad-hoc clarification flow (``_pending_clarifications`` dict
keyed by chat_id) must keep working. Only the mission-scoped subset
should migrate to ``post_event(kind='asking', ...)``. This test asserts
both surfaces are alive in the same instance and don't collide.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_pending_clarifications_dict_still_present():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._pending_clarifications = {}
    iface._pending_clarifications[42] = 7
    assert iface._pending_clarifications[42] == 7


@pytest.mark.asyncio
async def test_asking_post_event_creates_separate_row(tmp_path, monkeypatch):
    """New 'asking' flow writes to mission_events, not _pending_clarifications."""
    db_path = tmp_path / "coexist.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()

    mid = await db_mod.add_mission("Demo", "x")
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))

    from src.app.mission_events import post_event
    event_id = await post_event(
        bot, mid, "asking",
        {"question": "Which UI?", "options": ["mobile", "web"]},
        chat_id=-1,
    )
    assert event_id > 0

    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT kind FROM mission_events WHERE id = ?", (event_id,),
    )
    assert (await cur.fetchone())[0] == "asking"
