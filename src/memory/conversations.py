# memory/conversations.py
"""
Phase 11.4 — Conversation Continuity

Embeds user messages and AI responses into the conversations
collection.  Provides follow-up detection via embedding similarity
(replacing the simple user_last_task_id hack).

Public API:
    await store_exchange(chat_id, user_msg, ai_response, task_id)
    await find_followup_context(chat_id, new_message, top_k=3)
    await get_recent_exchanges(chat_id, limit=5)
"""
import logging
import time
from typing import Optional

from .vector_store import embed_and_store, query, is_ready

logger = logging.getLogger(__name__)


async def store_exchange(
    chat_id: int | str,
    user_message: str,
    ai_response: str,
    task_id: int | str | None = None,
    task_title: str = "",
) -> Optional[str]:
    """
    Store a user-AI exchange in the conversations collection.

    Args:
        chat_id:      Telegram chat ID.
        user_message:  What the user said.
        ai_response:   What the AI responded (first 500 chars).
        task_id:       Task ID that was created/processed, if any.
        task_title:    Task title, if any.

    Returns:
        Document ID if stored, None otherwise.
    """
    if not is_ready():
        return None

    response_preview = (ai_response or "")[:500]

    text = (
        f"User: {user_message}\n"
        f"Response: {response_preview}"
    )
    if task_title:
        text = f"Task: {task_title}\n{text}"

    metadata = {
        "chat_id": str(chat_id),
        "user_message": user_message[:300],
        "response_preview": response_preview[:200],
        "task_id": str(task_id) if task_id else "",
        "task_title": task_title[:200] if task_title else "",
        "timestamp": time.time(),
        "type": "conversation",
    }

    doc_id = f"conv-{chat_id}-{int(time.time() * 1000)}"

    return await embed_and_store(
        text=text,
        metadata=metadata,
        collection="conversations",
        doc_id=doc_id,
    )


async def get_recent_exchanges(
    chat_id: int | str,
    limit: int = 5,
) -> list[dict]:
    """
    Get the most recent exchanges for a chat.

    Returns list of dicts with keys:
        user_message, response_preview, task_id, task_title, timestamp
    """
    if not is_ready():
        return []

    # Query with a broad text to get all, filtered by chat_id
    results = await query(
        text="recent conversation",
        collection="conversations",
        top_k=limit,
        where={"chat_id": str(chat_id)},
    )

    exchanges = []
    for r in results:
        meta = r.get("metadata", {})
        exchanges.append({
            "user_message": meta.get("user_message", ""),
            "response_preview": meta.get("response_preview", ""),
            "task_id": meta.get("task_id", ""),
            "task_title": meta.get("task_title", ""),
            "timestamp": meta.get("timestamp", 0),
        })

    # Sort by timestamp (most recent first)
    exchanges.sort(key=lambda x: -x.get("timestamp", 0))
    return exchanges[:limit]


async def find_followup_context(
    chat_id: int | str,
    new_message: str,
    top_k: int = 3,
) -> dict:
    """
    Detect if a new message is a follow-up to a previous conversation
    and return the relevant context.

    Uses embedding similarity to find the most relevant previous
    exchange, which is more robust than tracking user_last_task_id.

    Args:
        chat_id:     Telegram chat ID.
        new_message: The new user message.
        top_k:       Number of similar exchanges to consider.

    Returns:
        Dict with:
          - is_followup: bool
          - parent_task_id: str or None
          - context: list of recent exchange dicts
          - best_match: dict with the most relevant previous exchange
    """
    if not is_ready():
        return {
            "is_followup": False,
            "parent_task_id": None,
            "context": [],
            "best_match": None,
        }

    # Get recent exchanges for this chat
    recent = await get_recent_exchanges(chat_id, limit=5)

    # Find semantically similar past conversations
    similar = await query(
        text=new_message,
        collection="conversations",
        top_k=top_k,
        where={"chat_id": str(chat_id)},
    )

    best_match = None
    parent_task_id = None
    is_followup = False

    if similar:
        best = similar[0]
        distance = best.get("distance", 1.0)
        meta = best.get("metadata", {})

        # If the best match is very recent and fairly similar,
        # it's likely a follow-up
        age = time.time() - meta.get("timestamp", 0)
        is_recent = age < 3600  # within last hour
        is_similar = distance < 0.8  # reasonable similarity threshold

        if is_recent or is_similar:
            is_followup = True
            parent_task_id = meta.get("task_id") or None
            best_match = {
                "user_message": meta.get("user_message", ""),
                "response_preview": meta.get("response_preview", ""),
                "task_id": meta.get("task_id", ""),
                "task_title": meta.get("task_title", ""),
                "distance": distance,
            }

    return {
        "is_followup": is_followup,
        "parent_task_id": parent_task_id,
        "context": recent,
        "best_match": best_match,
    }


def format_recent_context(exchanges: list[dict], limit: int = 3) -> list[dict]:
    """
    Format recent exchanges for injection into task context
    as the 'recent_conversation' field.

    Returns list of dicts compatible with base.py _build_context.
    """
    formatted = []
    for ex in exchanges[:limit]:
        formatted.append({
            "user_asked": ex.get("user_message", "?"),
            "result": ex.get("response_preview", ""),
        })
    return formatted
