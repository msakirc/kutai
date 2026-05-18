"""Z8 T5E — support tier-1 RAG + confidence-based escalation.

Public API
----------
``retrieve_docs(question, top_k=3)`` — embed question, query ``support_docs``
ChromaDB collection, return top-k hits.

``detect_sentiment(question)`` — keyword-based v1 sentiment detection. Returns
``"angry"`` / ``"urgent"`` / ``"neutral"``. Angry/urgent always triggers
escalation regardless of confidence.

``save_ticket(...)`` — insert row into ``tickets`` table; returns ticket_id.

``escalate_if_needed(ticket_id, mission_id, question, answer, confidence,
sentiment)`` — when ``confidence < CONFIDENCE_THRESHOLD`` or sentiment is
``"angry"``/``"urgent"``, create a ``founder_action(kind='support_escalation')``
and flip ``tickets.escalated_to_founder=1`` + ``founder_action_id``. Returns
the founder_action_id (or ``None`` if no escalation needed).

The Telegram ``/ask`` inlet (in ``src/app/telegram_bot.py``) drives this
module — retrieve → enqueue a task with ``agent_type='support_tier1'`` and
the docs in context.payload → on completion, save_ticket + escalate_if_needed.
"""
from __future__ import annotations

import re
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("ops.support_rag")

# Tunable: tickets below this confidence force founder escalation.
CONFIDENCE_THRESHOLD: float = 0.7

# Keyword-based v1 sentiment. Pre-compiled for speed.
_ANGRY_PATTERNS = re.compile(
    r"\b(angry|furious|terrible|awful|hate|worst|scam|fraud|cancel\s*"
    r"(my\s*)?(subscription|account)|refund|lawyer|sue|legal\s*action)\b",
    re.IGNORECASE,
)
_URGENT_PATTERNS = re.compile(
    r"\b(urgent|asap|immediately|right now|emergency|critical|broken|"
    r"can'?t (login|access|use)|outage|down|locked out)\b",
    re.IGNORECASE,
)


def detect_sentiment(question: str) -> str:
    """Return ``'angry'`` / ``'urgent'`` / ``'neutral'``.

    Angry takes precedence over urgent — both signal escalation, but the
    sentiment field on the ticket reflects the stronger signal for analytics.
    """
    q = question or ""
    if _ANGRY_PATTERNS.search(q):
        return "angry"
    if _URGENT_PATTERNS.search(q):
        return "urgent"
    return "neutral"


async def retrieve_docs(question: str, top_k: int = 3) -> list[dict]:
    """Embed ``question`` and return top-k hits from ``support_docs``.

    Returns ``[]`` on any retrieval failure (vector store not initialized,
    embedder failure, etc.) — the agent then answers with no grounding and
    will self-report low confidence, triggering escalation downstream.
    """
    if not question or not question.strip():
        return []
    try:
        from src.memory.vector_store import query
        return await query(question, collection="support_docs", top_k=top_k)
    except Exception as e:  # noqa: BLE001
        logger.warning("support_docs retrieve failed: %s", e)
        return []


async def index_doc(
    text: str,
    metadata: dict | None = None,
    doc_id: str | None = None,
) -> str | None:
    """Store a FAQ / policy / how-to passage into ``support_docs``.

    Called by the weekly cluster-and-propose job once a founder approves
    proposed FAQ additions.
    """
    try:
        from src.memory.vector_store import embed_and_store
        return await embed_and_store(
            text=text,
            metadata=metadata or {},
            collection="support_docs",
            doc_id=doc_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("support_docs index failed: %s", e)
        return None


async def save_ticket(
    user_id: str,
    question: str,
    *,
    mission_id: int | None = None,
    answer: str | None = None,
    confidence: float | None = None,
    sentiment: str | None = None,
    status: str = "open",
) -> int:
    """Insert a new row into ``tickets``. Returns the new ticket_id."""
    from src.infra.db import get_db

    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO tickets "
        "(mission_id, user_id, question, answer, confidence, sentiment, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            mission_id,
            str(user_id),
            question,
            answer,
            float(confidence) if confidence is not None else None,
            sentiment,
            status,
        ),
    )
    await db.commit()
    return int(cursor.lastrowid or 0)


async def update_ticket(
    ticket_id: int,
    *,
    answer: str | None = None,
    confidence: float | None = None,
    status: str | None = None,
    sentiment: str | None = None,
    escalated_to_founder: bool | None = None,
    founder_action_id: int | None = None,
) -> None:
    """Patch a ticket row in place."""
    sets: list[str] = []
    params: list[Any] = []
    if answer is not None:
        sets.append("answer = ?")
        params.append(answer)
    if confidence is not None:
        sets.append("confidence = ?")
        params.append(float(confidence))
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    if sentiment is not None:
        sets.append("sentiment = ?")
        params.append(sentiment)
    if escalated_to_founder is not None:
        sets.append("escalated_to_founder = ?")
        params.append(1 if escalated_to_founder else 0)
    if founder_action_id is not None:
        sets.append("founder_action_id = ?")
        params.append(int(founder_action_id))
    if not sets:
        return

    from src.infra.db import get_db

    db = await get_db()
    params.append(int(ticket_id))
    await db.execute(
        f"UPDATE tickets SET {', '.join(sets)} WHERE id = ?",
        params,
    )
    await db.commit()


def needs_escalation(confidence: float | None, sentiment: str | None) -> bool:
    """Pure predicate — confidence below threshold OR angry/urgent sentiment."""
    if sentiment in ("angry", "urgent"):
        return True
    if confidence is None:
        return True
    return float(confidence) < CONFIDENCE_THRESHOLD


async def escalate_if_needed(
    ticket_id: int,
    *,
    mission_id: int | None,
    user_id: str,
    question: str,
    answer: str | None,
    confidence: float | None,
    sentiment: str | None,
) -> int | None:
    """When confidence/sentiment force escalation, create a founder_action.

    Returns the founder_action_id (or ``None`` when no escalation was needed).
    """
    if not needs_escalation(confidence, sentiment):
        return None

    try:
        from src.founder_actions import create as fa_create
    except Exception as e:  # noqa: BLE001
        logger.warning("support_rag: founder_actions import failed: %s", e)
        return None

    reason_bits = []
    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        reason_bits.append(f"low confidence ({confidence:.2f})")
    if sentiment in ("angry", "urgent"):
        reason_bits.append(f"sentiment={sentiment}")
    why = "Support tier-1 escalation: " + (
        ", ".join(reason_bits) or "policy"
    )

    action = await fa_create(
        mission_id=int(mission_id) if mission_id is not None else 0,
        kind="support_escalation",
        title=f"Tier-1 escalation: ticket #{ticket_id}",
        why=why,
        instructions=[
            f"User ({user_id}) asked: {question}",
            f"Tier-1 answer (low-confidence): {answer or '<no answer>'}",
            "Review and respond directly to the user.",
        ],
        urgent=(sentiment in ("angry", "urgent")),
        notify_telegram=True,
    )
    action_id = int(getattr(action, "id", 0) or 0)
    await update_ticket(
        ticket_id,
        escalated_to_founder=True,
        founder_action_id=action_id or None,
        status="escalated",
    )
    return action_id or None
