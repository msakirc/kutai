"""Restart-restore must re-surface PARKED REVIEW HALTS, not just clarifications.

The only startup restore (``restore_clarification_state``) handled mechanical
clarifications only and ``break``-ed after the first waiting_human row — so a
parked reviewer halt (agent_type='reviewer', escalated review) was silently
dropped on every restart, and a second waiting_human task never got looked at.

After the fix:
  - ALL parked reviewer halts are re-rendered (stateless callbacks → no
    single-slot limit), via ``resurface_review_halt``.
  - Exactly ONE clarification is restored into the single in-memory slot.
  - The unconditional ``break`` is gone, so a clarification sitting behind a
    reviewer halt is no longer dropped.
"""
from __future__ import annotations

import json

import pytest
import aiosqlite
from unittest.mock import AsyncMock


def _reset_db(db_path: str):
    import src.infra.db as m
    m._db_connection = None
    m._db_connection_path = None
    m.DB_PATH = db_path


async def _insert(db_path, *, agent_type, ctx, title):
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tasks (title, status, agent_type, context) "
            "VALUES (?, 'waiting_human', ?, ?)",
            (title, agent_type, json.dumps(ctx)),
        )
        tid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()
    return tid


def _make_iface(monkeypatch):
    import src.app.telegram_bot as tb
    monkeypatch.setattr(tb, "TELEGRAM_ADMIN_CHAT_ID", "42", raising=False)
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface._pending_clarifications = {}
    iface.resurface_review_halt = AsyncMock(return_value=True)
    iface.request_clarification = AsyncMock()
    iface._recover_question_from_child = AsyncMock(return_value="")
    iface._send_next_clarification_question = AsyncMock()
    iface.send_notification = AsyncMock()
    return iface


@pytest.mark.asyncio
async def test_restore_surfaces_review_halt_and_clarification(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db(db_path)
    from src.infra.db import init_db
    await init_db()
    _reset_db(db_path)

    rid = await _insert(
        db_path, agent_type="reviewer", title="[1.13] research_quality_review",
        ctx={"_review_halt": {"reviewer_name": "1.13",
                              "issues": [{"severity": "blocker", "problem": "p"}],
                              "producers": ["0.0z"]}},
    )
    cid = await _insert(
        db_path, agent_type="mechanical", title="[0.6a] non_goals_confirm",
        ctx={"_clarification_question": "Confirm the non-goals draft?"},
    )

    iface = _make_iface(monkeypatch)
    await iface.restore_clarification_state()

    # The parked review halt is re-rendered (was silently dropped before).
    iface.resurface_review_halt.assert_awaited_once()
    assert iface.resurface_review_halt.await_args.args[0]["id"] == rid
    # The clarification is ALSO restored — the old unconditional break dropped
    # whichever waiting_human row came second.
    iface.request_clarification.assert_awaited_once()
    assert iface.request_clarification.await_args.args[0] == cid


@pytest.mark.asyncio
async def test_restore_surfaces_all_review_halts(tmp_path, monkeypatch):
    """Two parked review halts -> both re-rendered (stateless, no slot limit)."""
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db(db_path)
    from src.infra.db import init_db
    await init_db()
    _reset_db(db_path)

    await _insert(db_path, agent_type="reviewer", title="[1.13] a",
                  ctx={"_review_halt": {"reviewer_name": "1.13", "issues": [],
                                        "producers": []}})
    await _insert(db_path, agent_type="reviewer", title="[3.11] b",
                  ctx={"_review_halt": {"reviewer_name": "3.11", "issues": [],
                                        "producers": []}})

    iface = _make_iface(monkeypatch)
    await iface.restore_clarification_state()

    assert iface.resurface_review_halt.await_count == 2
    iface.request_clarification.assert_not_awaited()
