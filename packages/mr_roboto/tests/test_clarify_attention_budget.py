"""Z1 T5A (A5) integration — low attention budget defers clarify.

When `missions.founder_attention_budget_minutes - SUM(debits) < reserve`,
the clarify executor must NOT fire on Telegram; it must write the
question to ``deferred_questions.md`` and return ``status='deferred'``.
"""
from __future__ import annotations

import os
from unittest.mock import patch, AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
async def _db_reset():
    import src.infra.db as _dbmod
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None
    yield
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None


async def _setup(tmp_path, monkeypatch, budget: int | None, debits: list[int] | None = None):
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db, get_db
    await init_db()
    db = await get_db()
    if budget is None:
        await db.execute(
            "INSERT INTO missions (id, title, status) VALUES (?, ?, ?)",
            (1, "m1", "active"),
        )
    else:
        await db.execute(
            "INSERT INTO missions (id, title, status, "
            "founder_attention_budget_minutes) VALUES (?, ?, ?, ?)",
            (1, "m1", "active", budget),
        )
    if debits:
        from mr_roboto.attention_check import attention_debit
        for d in debits:
            await attention_debit(
                mission_id=1, step_id="prior", action="clarify", minutes_debited=d,
            )
    await db.commit()


@pytest.mark.asyncio
async def test_low_budget_defers_clarify_no_telegram(tmp_path, monkeypatch):
    """Budget=10, prior debits=8, reserve=5 → ok=False → deferred (no Telegram)."""
    await _setup(tmp_path, monkeypatch, budget=10, debits=[8])

    # Workspace points at tmp_path so the deferred_questions.md lands there.
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace",
        lambda mid: str(tmp_path / f"mission_{mid}"),
    )

    from mr_roboto.clarify import clarify

    fake_tg = MagicMock()
    fake_tg.request_clarification = AsyncMock()

    with patch("mr_roboto.clarify.get_telegram", return_value=fake_tg):
        task = {
            "id": 99,
            "mission_id": 1,
            "title": "Pick a tagline",
            "context": {"workflow_step_id": "0.5"},
            "payload": {"question": "Pick a tagline option (A/B/C)?"},
        }
        result = await clarify(task)

    assert result["status"] == "deferred"
    assert result["reason"] == "attention_budget_exhausted"
    fake_tg.request_clarification.assert_not_called()

    deferred = tmp_path / "mission_1" / "deferred_questions.md"
    assert deferred.exists()
    body = deferred.read_text(encoding="utf-8")
    assert "Pick a tagline" in body


@pytest.mark.asyncio
async def test_unbounded_budget_proceeds_normally(tmp_path, monkeypatch):
    """No budget set → clarify fires on Telegram as before."""
    await _setup(tmp_path, monkeypatch, budget=None)

    from mr_roboto.clarify import clarify
    fake_tg = MagicMock()
    fake_tg.request_clarification = AsyncMock()

    with patch("mr_roboto.clarify.get_telegram", return_value=fake_tg):
        task = {
            "id": 99,
            "mission_id": 1,
            "title": "Hello",
            "payload": {"question": "Pick option?"},
        }
        result = await clarify(task)

    assert result.get("sent") is True
    fake_tg.request_clarification.assert_awaited_once()


@pytest.mark.asyncio
async def test_attention_skip_payload_bypasses_check(tmp_path, monkeypatch):
    """Caller can opt out of the gate via payload.attention_skip=True."""
    await _setup(tmp_path, monkeypatch, budget=10, debits=[10])

    from mr_roboto.clarify import clarify
    fake_tg = MagicMock()
    fake_tg.request_clarification = AsyncMock()

    with patch("mr_roboto.clarify.get_telegram", return_value=fake_tg):
        task = {
            "id": 99,
            "mission_id": 1,
            "title": "Critical",
            "payload": {"question": "Critical question?", "attention_skip": True},
        }
        result = await clarify(task)

    assert result.get("sent") is True
    fake_tg.request_clarification.assert_awaited_once()
