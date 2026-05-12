"""Z8 T5E — support_tier1 RAG + confidence escalation."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "support.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


# ── Sentiment detection ──────────────────────────────────────────────────────


def test_detect_sentiment_neutral():
    from src.ops.support_rag import detect_sentiment

    assert detect_sentiment("How do I change my password?") == "neutral"
    assert detect_sentiment("") == "neutral"


def test_detect_sentiment_angry():
    from src.ops.support_rag import detect_sentiment

    assert detect_sentiment("This is a scam, I want a refund now!") == "angry"
    assert detect_sentiment("Cancel my subscription immediately") == "angry"
    assert detect_sentiment("I'll get my lawyer involved") == "angry"


def test_detect_sentiment_urgent():
    from src.ops.support_rag import detect_sentiment

    assert detect_sentiment("URGENT — site is down") == "urgent"
    assert detect_sentiment("I can't login asap please") == "urgent"


# ── Escalation predicate ─────────────────────────────────────────────────────


def test_needs_escalation_low_confidence():
    from src.ops.support_rag import needs_escalation

    assert needs_escalation(0.5, "neutral") is True
    assert needs_escalation(0.69, "neutral") is True


def test_needs_escalation_high_confidence_neutral():
    from src.ops.support_rag import needs_escalation

    assert needs_escalation(0.9, "neutral") is False
    assert needs_escalation(0.71, "neutral") is False


def test_needs_escalation_angry_overrides_confidence():
    from src.ops.support_rag import needs_escalation

    assert needs_escalation(0.99, "angry") is True
    assert needs_escalation(0.99, "urgent") is True


def test_needs_escalation_none_confidence():
    from src.ops.support_rag import needs_escalation

    assert needs_escalation(None, "neutral") is True


# ── Tickets table + save/update ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tickets_table_exists(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tickets'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None, "tickets migration didn't run"


@pytest.mark.asyncio
async def test_save_ticket_roundtrip(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from src.ops.support_rag import save_ticket, update_ticket
    from src.infra.db import get_db

    tid = await save_ticket(
        user_id="42",
        question="How do I reset my password?",
        sentiment="neutral",
    )
    assert tid > 0

    await update_ticket(tid, answer="Click the reset link.", confidence=0.85, status="closed")

    db = await get_db()
    async with db.execute(
        "SELECT user_id, question, answer, confidence, status FROM tickets WHERE id = ?",
        (tid,),
    ) as cur:
        row = await cur.fetchone()
    assert tuple(row) == ("42", "How do I reset my password?", "Click the reset link.", 0.85, "closed")


@pytest.mark.asyncio
async def test_escalate_if_needed_creates_founder_action(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from src.ops.support_rag import escalate_if_needed, save_ticket
    from src.infra.db import get_db

    # Seed a fake mission so block_mission_if_needed has something to read.
    db = await get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status) VALUES (1, 'support', 'active')"
    )
    await db.commit()

    tid = await save_ticket(
        user_id="99",
        question="I cannot login, urgent!",
        sentiment="urgent",
    )

    action_id = await escalate_if_needed(
        tid,
        mission_id=1,
        user_id="99",
        question="I cannot login, urgent!",
        answer=None,
        confidence=0.2,
        sentiment="urgent",
    )
    assert action_id is not None and action_id > 0

    async with db.execute(
        "SELECT escalated_to_founder, founder_action_id, status FROM tickets WHERE id = ?",
        (tid,),
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == 1
    assert row[1] == action_id
    assert row[2] == "escalated"

    async with db.execute(
        "SELECT kind, urgent FROM founder_actions WHERE id = ?",
        (action_id,),
    ) as cur:
        fa = await cur.fetchone()
    assert fa[0] == "support_escalation"
    assert fa[1] == 1  # urgent flag set


@pytest.mark.asyncio
async def test_escalate_if_needed_skips_high_confidence(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from src.ops.support_rag import escalate_if_needed, save_ticket

    tid = await save_ticket(
        user_id="3", question="hours?", sentiment="neutral",
    )
    out = await escalate_if_needed(
        tid,
        mission_id=None,
        user_id="3",
        question="hours?",
        answer="9-5",
        confidence=0.95,
        sentiment="neutral",
    )
    assert out is None


# ── retrieve_docs falls back to [] cleanly ───────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_docs_empty_question_returns_empty():
    from src.ops.support_rag import retrieve_docs

    assert await retrieve_docs("") == []
    assert await retrieve_docs("   ") == []
