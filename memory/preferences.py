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
    await detect_preferences(chat_id)
    prefs = await get_user_preferences(chat_id)
    prompt_block = format_preferences(preferences)
"""
import logging
import time
from typing import Optional

from memory.vector_store import embed_and_store, query, is_ready

logger = logging.getLogger(__name__)


# ─── Feedback Types ──────────────────────────────────────────────────────────

FEEDBACK_ACCEPTED = "accepted"
FEEDBACK_MODIFIED = "modified"
FEEDBACK_REJECTED = "rejected"


# ─── Preference Categories ──────────────────────────────────────────────────

PREFERENCE_CATEGORIES = [
    "language",           # programming language preferences
    "framework",          # framework / library preferences
    "style",              # code style / naming conventions
    "verbosity",          # concise vs detailed responses
    "testing",            # testing preferences (framework, coverage)
    "risk_tolerance",     # conservative vs experimental
    "communication",      # communication style preferences
    "tools",              # preferred tools (linter, formatter, etc.)
    "general",            # catch-all
]


# ─── Record Feedback ────────────────────────────────────────────────────────

async def record_feedback(
    task: dict,
    feedback_type: str,
    details: str = "",
    chat_id: int | str = "default",
) -> Optional[str]:
    """
    Record user feedback on a task result.

    This builds the interaction history that detect_preferences()
    analyzes to learn user preferences.

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


# ─── Detect Preferences from Feedback ───────────────────────────────────────

async def detect_preferences(
    chat_id: int | str = "default",
) -> list[dict]:
    """
    Analyze feedback history to detect user preferences.

    Looks at patterns in:
      - Tasks that were modified (what did the user change?)
      - Tasks that were rejected (what did the user dislike?)
      - Accepted patterns (what does the user prefer?)

    Returns list of detected preferences (each is a dict with
    preference_text, category, confidence).
    """
    if not is_ready():
        return []

    # Get recent feedback entries
    modifications = await query(
        text="user modified task correction",
        collection="semantic",
        top_k=20,
        where={"feedback_type": "modified"},
    )

    rejections = await query(
        text="user rejected task result",
        collection="semantic",
        top_k=10,
        where={"feedback_type": "rejected"},
    )

    detected: list[dict] = []

    # Analyze modification patterns
    corrections: list[str] = []
    for r in modifications:
        meta = r.get("metadata", {})
        correction = meta.get("correction_preview", "")
        if correction:
            corrections.append(correction)

    if corrections:
        # Look for common themes in corrections
        detected.extend(_extract_patterns(corrections, "modification"))

    # Analyze rejection patterns
    rejection_titles: list[str] = []
    for r in rejections:
        meta = r.get("metadata", {})
        title = meta.get("title", "")
        if title:
            rejection_titles.append(title)

    if rejection_titles:
        detected.extend(_extract_patterns(rejection_titles, "rejection"))

    # Store detected preferences
    for pref in detected:
        await store_preference(
            preference=pref["preference_text"],
            category=pref["category"],
            chat_id=chat_id,
            confidence=pref["confidence"],
        )

    return detected


def _extract_patterns(
    texts: list[str],
    source: str,
) -> list[dict]:
    """
    Extract preference patterns from a collection of feedback texts.

    Uses simple keyword/pattern matching to detect preferences.
    This is a heuristic approach — future versions could use LLM analysis.
    """
    patterns: list[dict] = []
    combined = " ".join(texts).lower()

    # ── Language preferences ──
    language_signals = {
        "python": "Prefers Python for implementation",
        "typescript": "Prefers TypeScript over JavaScript",
        "javascript": "Prefers JavaScript",
        "rust": "Prefers Rust for systems work",
        "go": "Prefers Go for backend services",
    }
    for lang, pref_text in language_signals.items():
        count = combined.count(lang)
        if count >= 2:
            patterns.append({
                "preference_text": pref_text,
                "category": "language",
                "confidence": min(0.5 + count * 0.1, 0.9),
            })

    # ── Style preferences ──
    style_signals = {
        "snake_case": ("Prefers snake_case naming convention", "style"),
        "camelcase": ("Prefers camelCase naming convention", "style"),
        "camel_case": ("Prefers camelCase naming convention", "style"),
        "concise": ("Prefers concise, brief responses", "verbosity"),
        "detailed": ("Prefers detailed, thorough responses", "verbosity"),
        "verbose": ("Prefers verbose, explanatory responses", "verbosity"),
        "brief": ("Prefers brief, to-the-point responses", "verbosity"),
        "comments": ("Prefers well-commented code", "style"),
        "docstring": ("Prefers docstrings on functions", "style"),
        "type hint": ("Prefers type hints / type annotations", "style"),
    }
    for signal, (pref_text, category) in style_signals.items():
        if signal in combined:
            count = combined.count(signal)
            patterns.append({
                "preference_text": pref_text,
                "category": category,
                "confidence": min(0.5 + count * 0.1, 0.9),
            })

    # ── Framework preferences ──
    framework_signals = {
        "fastapi": "Prefers FastAPI for web APIs",
        "flask": "Prefers Flask for web development",
        "django": "Prefers Django for web development",
        "react": "Prefers React for frontend",
        "vue": "Prefers Vue.js for frontend",
        "pytest": "Prefers pytest for testing",
        "unittest": "Prefers unittest for testing",
    }
    for fw, pref_text in framework_signals.items():
        if fw in combined:
            count = combined.count(fw)
            patterns.append({
                "preference_text": pref_text,
                "category": "framework",
                "confidence": min(0.5 + count * 0.1, 0.9),
            })

    # ── Testing preferences ──
    if "test" in combined:
        if "always" in combined and "test" in combined:
            patterns.append({
                "preference_text": "Always wants tests written for new code",
                "category": "testing",
                "confidence": 0.7,
            })

    return patterns


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
