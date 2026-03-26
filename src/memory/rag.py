# memory/rag.py
"""
Phase 11.3 — RAG Pipeline for Agent Context

Retrieves relevant context from vector store collections and formats
it for injection into agent prompts.

Main function:
    context_block = await retrieve_context(task, agent_type)

The returned text block is injected into BaseAgent._build_context()
between the task description and tool descriptions.
"""
import time
from typing import Optional

from src.infra.logging_config import get_logger
from src.memory.vector_store import is_ready, query, embed_and_store

logger = get_logger("memory.rag")


# ─── Configuration ──────────────────────────────────────────────────────────

RAG_MIN_BUDGET = 2000
RAG_MAX_BUDGET = 12000
RAG_BUDGET_FRACTION = 0.15  # of available context window
RAG_MIN_RELEVANCE = 0.3     # cosine distance threshold (lower = more similar)
RAG_DEDUP_THRESHOLD = 0.85  # embedding similarity for dedup

# Task-type budgets (when model context window is unknown)
_TASK_TYPE_BUDGETS = {
    "code": 6000,
    "research": 4000,
    "shopping": 4000,
    "default": 4000,
}


def _compute_rag_budget(
    task: dict, model_context_window: int | None = None
) -> int:
    """
    Compute dynamic RAG token budget based on task type and model context.

    If model_context_window is provided, uses up to 15% of available space.
    Otherwise falls back to task-type-based defaults.
    """
    if model_context_window:
        # Reserve space for system prompt, tools, task, conversation
        reserved = 10000
        available = model_context_window - reserved
        budget = int(available * RAG_BUDGET_FRACTION)
        return max(RAG_MIN_BUDGET, min(RAG_MAX_BUDGET, budget))

    # Infer from task type
    task_type = task.get("type", "default")
    description = (task.get("description", "") + task.get("title", "")).lower()

    if any(kw in description for kw in ("code", "implement", "fix", "debug", "refactor")):
        task_type = "code"
    elif any(kw in description for kw in ("research", "search", "find", "analyze")):
        task_type = "research"
    elif any(kw in description for kw in ("shop", "buy", "price", "product", "compare")):
        task_type = "shopping"

    return _TASK_TYPE_BUDGETS.get(task_type, _TASK_TYPE_BUDGETS["default"])


# ─── Scoring Helpers ──────────────────────────────────────────────────────────

def _recency_weight(timestamp: float, half_life_days: float = 7.0) -> float:
    """
    Compute a recency weight that decays exponentially.

    A memory from `half_life_days` ago gets weight 0.5.
    A fresh memory gets weight ~1.0.
    """
    if not timestamp:
        return 0.5
    age_days = (time.time() - timestamp) / 86400.0
    if age_days < 0:
        return 1.0
    import math
    return math.pow(0.5, age_days / half_life_days)


def _importance_weight(metadata: dict) -> float:
    """
    Derive importance from metadata fields.

    High importance: error patterns, user preferences
    Medium: task results, code context
    Low: old conversations
    """
    doc_type = metadata.get("type", "")
    importance = metadata.get("importance", 5)

    # Type-based floor
    if doc_type in ("error_recovery", "user_preference"):
        importance = max(importance, 8)
    elif doc_type == "task_result":
        # Failed tasks are more important (we learn from failures)
        if not metadata.get("success", True):
            importance = max(importance, 7)
        else:
            importance = max(importance, 5)

    return importance / 10.0  # normalize to 0-1


def _rank_results(results: list[dict]) -> list[dict]:
    """
    Rank query results by: relevance * recency * importance.

    Filters out results below minimum relevance threshold.
    """
    scored = []
    for r in results:
        meta = r.get("metadata", {})
        distance = r.get("distance", 1.0)

        # Filter by relevance threshold (cosine distance)
        if distance > (1.0 - RAG_MIN_RELEVANCE):
            continue

        # Convert distance to relevance (cosine distance: 0 = identical)
        relevance = max(0.0, 1.0 - distance)

        recency = _recency_weight(meta.get("timestamp", 0))
        importance = _importance_weight(meta)

        # Composite score (rebalanced weights)
        score = (
            relevance * 0.4
            + recency * 0.25
            + importance * 0.2
            + meta.get("access_count", 0) * 0.001 * 0.15  # access frequency
        )

        scored.append({**r, "_score": score})

    scored.sort(key=lambda x: -x["_score"])
    return scored


def _deduplicate(results: list[dict], threshold: float = 0.9) -> list[dict]:
    """
    Remove near-duplicate results based on text overlap.

    Uses a simple set-of-words Jaccard similarity.
    """
    deduped: list[dict] = []
    seen_word_sets: list[set[str]] = []

    for r in results:
        text = r.get("text", "")
        words = set(text.lower().split())

        is_dup = False
        for seen in seen_word_sets:
            if not words or not seen:
                continue
            overlap = len(words & seen) / max(len(words | seen), 1)
            if overlap > threshold:
                is_dup = True
                break

        if not is_dup:
            deduped.append(r)
            seen_word_sets.append(words)

    return deduped


# ─── Token Estimation ────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    return len(text) // 4


# ─── Main RAG Function ───────────────────────────────────────────────────────

async def retrieve_context(
    task: dict,
    agent_type: str | None = None,
    max_tokens: int | None = None,
    model_context_window: int | None = None,
) -> str:
    """
    Retrieve relevant context from all vector store collections.

    Pipeline:
      1. Compute dynamic token budget
      2. Query episodic, semantic, errors, shopping, web_knowledge
      3. Rank by recency * relevance * importance
      4. Filter by minimum relevance threshold
      5. Deduplicate
      6. Format within token budget

    Args:
        task:                 Task dict with title, description, etc.
        agent_type:           Agent type (for filtering episodic results).
        max_tokens:           Override token budget (None = auto-compute).
        model_context_window: Model's context window size for budget calc.

    Returns:
        Formatted text block ready for prompt injection.
        Empty string if nothing relevant found.
    """
    if not is_ready():
        return ""

    title = task.get("title", "")
    description = task.get("description", "")
    query_text = f"{title}: {description[:400]}"

    if not query_text.strip():
        return ""

    # Compute budget
    budget = max_tokens or _compute_rag_budget(task, model_context_window)

    # ── 1. Query core collections ──
    episodic_results = await query(
        text=query_text, collection="episodic", top_k=5,
    )
    semantic_results = await query(
        text=query_text, collection="semantic", top_k=5,
    )
    error_results = await query(
        text=query_text, collection="errors", top_k=3,
    )

    # ── 2. Query domain-specific collections ──
    shopping_results = []
    task_desc_lower = (title + description).lower()
    if any(kw in task_desc_lower for kw in ("shop", "buy", "price", "product", "compare")):
        shopping_results = await query(
            text=query_text, collection="shopping", top_k=5,
        )

    web_results = await query(
        text=query_text, collection="web_knowledge", top_k=3,
    )

    # ── 3. Rank all results together (includes relevance filtering) ──
    all_results = _rank_results(
        episodic_results + semantic_results + error_results
        + shopping_results + web_results
    )

    # ── 4. Deduplicate ──
    deduped = _deduplicate(all_results)

    if not deduped:
        return ""

    # ── 5. Format within token budget ──
    sections: list[str] = []
    total_tokens = 0

    # Group by source collection
    episodic_items: list[dict] = []
    semantic_items: list[dict] = []
    error_items: list[dict] = []
    shopping_items: list[dict] = []
    web_items: list[dict] = []

    for r in deduped:
        meta = r.get("metadata", {})
        doc_type = meta.get("type", "")
        data_type = meta.get("data_type", "")

        if doc_type == "task_result":
            episodic_items.append(r)
        elif doc_type == "error_recovery":
            error_items.append(r)
        elif data_type in ("product", "review", "shopping_session"):
            shopping_items.append(r)
        elif meta.get("source_url") or data_type == "web_result":
            web_items.append(r)
        else:
            semantic_items.append(r)

    # Priority order: errors > episodic > code > semantic > shopping > web
    # Format error warnings first (highest priority)
    if error_items:
        lines = ["### Known Issues"]
        for item in error_items[:3]:
            meta = item.get("metadata", {})
            hint = meta.get("prevention_hint") or meta.get("fix_applied", "")
            if hint:
                lines.append(
                    f"- **{meta.get('error_signature', 'Error')}**: {hint[:150]}"
                )

        block = "\n".join(lines)
        block_tokens = _estimate_tokens(block)
        if total_tokens + block_tokens <= budget:
            sections.append(block)
            total_tokens += block_tokens

    # Format episodic memories
    if episodic_items:
        lines = ["### Past Experience"]
        for item in episodic_items[:4]:
            meta = item.get("metadata", {})
            status = "succeeded" if meta.get("success") else "FAILED"
            line = f"- \"{meta.get('title', '?')}\" {status}"
            if meta.get("model_used") and meta["model_used"] != "?":
                line += f" (model: {meta['model_used']})"
            if not meta.get("success") and meta.get("error_preview"):
                line += f"\n  Warning: {meta['error_preview'][:100]}"
            lines.append(line)

        block = "\n".join(lines)
        block_tokens = _estimate_tokens(block)
        if total_tokens + block_tokens <= budget:
            sections.append(block)
            total_tokens += block_tokens

    # Format semantic knowledge
    if semantic_items:
        lines = ["### Relevant Knowledge"]
        for item in semantic_items[:5]:
            text = item.get("text", "")
            remaining = budget - total_tokens
            max_chars = remaining * 4
            if max_chars < 50:
                break
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            lines.append(f"- {text[:500]}")
            total_tokens += _estimate_tokens(text[:500])

        if len(lines) > 1:
            sections.append("\n".join(lines))

    # Format shopping knowledge
    if shopping_items:
        lines = ["### Shopping Knowledge"]
        for item in shopping_items[:3]:
            text = item.get("text", "")
            remaining = budget - total_tokens
            max_chars = remaining * 4
            if max_chars < 50:
                break
            lines.append(f"- {text[:400]}")
            total_tokens += _estimate_tokens(text[:400])

        if len(lines) > 1:
            sections.append("\n".join(lines))

    # Format web knowledge
    if web_items:
        lines = ["### Web Knowledge"]
        for item in web_items[:3]:
            text = item.get("text", "")
            meta = item.get("metadata", {})
            remaining = budget - total_tokens
            max_chars = remaining * 4
            if max_chars < 50:
                break
            source = meta.get("source_url", "")
            entry = f"- {text[:300]}"
            if source:
                entry += f" [source: {source[:80]}]"
            lines.append(entry)
            total_tokens += _estimate_tokens(entry)

        if len(lines) > 1:
            sections.append("\n".join(lines))

    if not sections:
        return ""

    return "## Retrieved Knowledge\n\n" + "\n\n".join(sections)


# ─── Convenience: Store Semantic Fact ─────────────────────────────────────────

async def store_fact(
    text: str,
    category: str = "general",
    source: str = "",
    importance: int = 5,
) -> Optional[str]:
    """
    Store a semantic fact/knowledge entry.

    Args:
        text:       The fact or knowledge to store.
        category:   Category tag (e.g., "user_preference", "project_info").
        source:     Where this knowledge came from.
        importance: 1-10 importance score.

    Returns:
        Document ID if stored, None otherwise.
    """
    metadata = {
        "type": "fact",
        "category": category,
        "source": source,
        "importance": importance,
        "timestamp": time.time(),
    }

    return await embed_and_store(
        text=text,
        metadata=metadata,
        collection="semantic",
    )
