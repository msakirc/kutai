# memory/preferences.py
"""
Phase 11.7 — User Preference Learning

Tracks how users interact with task results to learn preferences:
  - Accepted (no follow-up) → positive signal
  - Modified (follow-up correction) → partial signal + preference update
  - Rejected (explicit negative) → negative signal

Detects patterns like:
  - Preferred languages & frameworks
  - Naming conventions
  - Verbosity level
  - Risk tolerance
  - Tool preferences

Public API:
    await record_feedback(task, feedback_type, details)
    await store_preference(preference, category, chat_id, confidence)
    prefs = await get_user_preferences(chat_id)
    prompt_block = format_preferences(preferences)
"""
import time
from typing import Optional

from src.infra.logging_config import get_logger
from src.memory.vector_store import is_ready, embed_and_store, query

logger = get_logger("memory.preferences")


# ─── Feedback Types ──────────────────────────────────────────────────────────

FEEDBACK_ACCEPTED = "accepted"
FEEDBACK_MODIFIED = "modified"
FEEDBACK_REJECTED = "rejected"


# ─── Record Feedback ────────────────────────────────────────────────────────

async def record_feedback(
    task: dict,
    feedback_type: str,
    details: str = "",
    chat_id: int | str = "default",
) -> Optional[str]:
    """
    Record user feedback on a task result.

    This builds the interaction history for preference learning.

    Args:
        task:          Task dict with id, title, description, agent_type.
        feedback_type: One of: "accepted", "modified", "rejected".
        details:       Additional context (e.g., the correction text).
        chat_id:       User identifier.

    Returns:
        Document ID if stored, None otherwise.
    """
    if not is_ready():
        return None

    if feedback_type not in (FEEDBACK_ACCEPTED, FEEDBACK_MODIFIED, FEEDBACK_REJECTED):
        logger.warning(f"Unknown feedback type: {feedback_type}")
        return None

    title = task.get("title", "Untitled")
    description = task.get("description", "")
    agent_type = task.get("agent_type", "unknown")
    task_id = task.get("id", "?")

    text = (
        f"Feedback on task: {title}\n"
        f"Type: {feedback_type}\n"
        f"Agent: {agent_type}\n"
        f"Description: {description[:300]}"
    )
    if details:
        text += f"\nUser correction: {details[:500]}"

    metadata = {
        "task_id": str(task_id),
        "title": title[:200],
        "agent_type": agent_type,
        "feedback_type": feedback_type,
        "chat_id": str(chat_id),
        "timestamp": time.time(),
        "type": "user_feedback",
        "importance": 7,  # feedback is important for learning
    }

    if details:
        metadata["correction_preview"] = details[:200]

    doc_id = f"feedback-{task_id}-{int(time.time())}"

    return await embed_and_store(
        text=text,
        metadata=metadata,
        collection="semantic",
        doc_id=doc_id,
    )


# ─── Store Preference ───────────────────────────────────────────────────────

async def store_preference(
    preference: str,
    category: str = "general",
    chat_id: int | str = "default",
    confidence: float = 0.7,
) -> Optional[str]:
    """
    Directly store a learned user preference.

    Args:
        preference:  The preference text (e.g., "Prefers snake_case naming").
        category:    Preference category (see PREFERENCE_CATEGORIES).
        chat_id:     User identifier.
        confidence:  Confidence level 0.0 - 1.0.

    Returns:
        Document ID if stored, None otherwise.
    """
    if not is_ready():
        return None

    text = f"User preference ({category}): {preference}"

    metadata = {
        "type": "user_preference",
        "category": category,
        "chat_id": str(chat_id),
        "confidence": confidence,
        "preference_text": preference[:300],
        "timestamp": time.time(),
        "importance": 9,  # preferences are high importance (protected from decay)
    }

    # Use a deterministic ID so repeated detections update rather than duplicate
    import hashlib
    doc_id = "pref-" + hashlib.sha256(
        f"{chat_id}:{category}:{preference[:100]}".encode()
    ).hexdigest()[:16]

    return await embed_and_store(
        text=text,
        metadata=metadata,
        collection="semantic",
        doc_id=doc_id,
    )


# ─── Get Preferences ────────────────────────────────────────────────────────

async def get_user_preferences(
    chat_id: int | str = "default",
    limit: int = 20,
) -> list[dict]:
    """
    Retrieve all stored preferences for a user.

    Returns list of dicts with: preference_text, category, confidence, timestamp.
    """
    if not is_ready():
        return []

    results = await query(
        text="user preferences and conventions",
        collection="semantic",
        top_k=limit,
        where={"type": "user_preference"},
    )

    prefs = []
    for r in results:
        meta = r.get("metadata", {})
        # Filter by chat_id if we have one
        if str(chat_id) != "default" and meta.get("chat_id", "default") != str(chat_id):
            continue
        prefs.append({
            "preference_text": meta.get("preference_text", ""),
            "category": meta.get("category", "general"),
            "confidence": meta.get("confidence", 0.5),
            "timestamp": meta.get("timestamp", 0),
        })

    # Sort by confidence (highest first)
    prefs.sort(key=lambda x: -x.get("confidence", 0))
    return prefs


# ─── Format for Prompt Injection ─────────────────────────────────────────────

def format_preferences(preferences: list[dict]) -> str:
    """
    Format user preferences into a prompt-friendly text block.

    Suitable for injection into agent system prompts.

    Args:
        preferences: List of preference dicts from get_user_preferences().

    Returns:
        Formatted string like:
          ## User Preferences
          - Prefers Python for implementation
          - Prefers concise output
          - Always wants tests
    """
    if not preferences:
        return ""

    lines = ["## User Preferences"]
    seen = set()

    for p in preferences:
        text = p.get("preference_text", "")
        if not text or text in seen:
            continue
        seen.add(text)

        confidence = p.get("confidence", 0.5)
        if confidence >= 0.7:
            lines.append(f"- {text}")
        elif confidence >= 0.5:
            lines.append(f"- {text} (tentative)")

    if len(lines) <= 1:
        return ""

    return "\n".join(lines)
