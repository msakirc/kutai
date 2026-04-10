# memory/episodic.py
"""
Phase 11.2 — Episodic Memory (Task History)

Stores task outcomes (success/failure) in the vector store so that
agents can learn from past experience.

Public API:
    await store_task_result(task, result, model, cost, duration)
    await recall_similar_tasks(title, description, top_k=3)
"""
import time

from src.infra.logging_config import get_logger
from .vector_store import embed_and_store, query, is_ready

logger = get_logger("memory.episodic")


# ─── Store Task Result ────────────────────────────────────────────────────────

async def store_task_result(
    task: dict,
    result: str,
    model: str = "unknown",
    cost: float = 0.0,
    duration: float = 0.0,
    success: bool = True,
) -> str | None:
    """
    Embed and store the outcome of a completed or failed task.

    This creates a searchable memory of past task executions that
    agents can query before starting similar work.

    Args:
        task:     Task dict with id, title, description, agent_type, etc.
        result:   The result text (first 500 chars stored).
        model:    LLM model used.
        cost:     Cost in dollars.
        duration: Duration in seconds.
        success:  True if task completed successfully.

    Returns:
        Document ID if stored, None otherwise.
    """
    if not is_ready():
        return None

    title = task.get("title", "Untitled")
    description = task.get("description", "")
    agent_type = task.get("agent_type", "unknown")
    task_id = task.get("id", "?")

    # Text to embed — combines title + description + result summary
    result_summary = (result or "")[:500]
    text = (
        f"Task: {title}\n"
        f"Description: {description[:300]}\n"
        f"Agent: {agent_type}\n"
        f"Outcome: {'success' if success else 'failure'}\n"
        f"Result: {result_summary}"
    )

    metadata = {
        "task_id": str(task_id),
        "title": title[:200],
        "agent_type": agent_type,
        "model_used": model,
        "success": success,
        "cost": cost,
        "duration": duration,
        "timestamp": time.time(),
        "type": "task_result",
    }

    if not success and result:
        metadata["error_preview"] = result[:200]

    doc_id = f"task-{task_id}-{int(time.time())}"

    stored = await embed_and_store(
        text=text,
        metadata=metadata,
        collection="episodic",
        doc_id=doc_id,
    )

    return stored


# ─── Recall Similar Tasks ───────────────────────────────────────────────────

async def recall_similar_tasks(
    title: str,
    description: str,
    top_k: int = 3,
    agent_type: str | None = None,
) -> list[dict]:
    """
    Find similar past tasks from episodic memory.

    Returns a list of dicts with keys:
        title, agent_type, success, result_preview, model_used, distance

    Args:
        title:       Task title to search for.
        description: Task description.
        top_k:       Number of similar tasks to return.
        agent_type:  Optional filter by agent type.

    Returns:
        List of similar task records, sorted by relevance.
    """
    if not is_ready():
        return []

    query_text = f"Task: {title}\nDescription: {description[:300]}"

    where = None
    if agent_type:
        where = {"agent_type": agent_type}

    results = await query(
        text=query_text,
        collection="episodic",
        top_k=top_k,
        where=where,
    )

    similar: list[dict] = []
    for r in results:
        meta = r.get("metadata", {})
        similar.append({
            "title": meta.get("title", "Unknown"),
            "agent_type": meta.get("agent_type", "?"),
            "success": meta.get("success", True),
            "model_used": meta.get("model_used", "?"),
            "error_preview": meta.get("error_preview", ""),
            "distance": r.get("distance", 0),
            "timestamp": meta.get("timestamp", 0),
        })

    return similar


async def recall_error_patterns(
    title: str,
    description: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Find similar past error patterns from the errors collection.

    Returns a list of dicts with keys:
        error_signature, root_cause, fix_applied, prevention_hint, distance
    """
    if not is_ready():
        return []

    query_text = f"Task: {title}\nDescription: {description[:300]}"

    results = await query(
        text=query_text,
        collection="errors",
        top_k=top_k,
    )

    patterns: list[dict] = []
    for r in results:
        meta = r.get("metadata", {})
        patterns.append({
            "error_signature": meta.get("error_signature", ""),
            "root_cause": meta.get("root_cause", ""),
            "fix_applied": meta.get("fix_applied", ""),
            "prevention_hint": meta.get("prevention_hint", ""),
            "distance": r.get("distance", 0),
        })

    return patterns


# ─── Format for Agent Context ────────────────────────────────────────────────

def format_similar_tasks(similar: list[dict]) -> str:
    """Format similar task results for injection into agent context."""
    if not similar:
        return ""

    lines = ["## Relevant Past Experience"]
    for s in similar:
        status = "succeeded" if s.get("success") else "FAILED"
        line = f"- Similar task \"{s['title']}\" {status}"
        if s.get("model_used") and s["model_used"] != "?":
            line += f" (model: {s['model_used']})"
        if not s.get("success") and s.get("error_preview"):
            line += f"\n  Warning: failed with: {s['error_preview'][:100]}"
        lines.append(line)

    return "\n".join(lines)


def format_error_warnings(patterns: list[dict]) -> str:
    """Format error pattern warnings for injection into agent context."""
    if not patterns:
        return ""

    lines = ["## Known Issues (from past failures)"]
    for p in patterns:
        if p.get("prevention_hint"):
            lines.append(
                f"- **{p['error_signature']}**: {p['prevention_hint']}"
            )
        elif p.get("fix_applied"):
            lines.append(
                f"- **{p['error_signature']}**: "
                f"Previously fixed by: {p['fix_applied']}"
            )

    return "\n".join(lines)


# ─── Cross-Agent Insight Extraction (Phase D) ───────────────────────────────

async def store_insight(
    insight_text: str,
    agent_type: str,
    task_id: int,
    task_title: str = "",
) -> str | None:
    """Store a grader-extracted insight in the semantic collection.

    Unlike the old extract_and_store_insight, this receives real LLM-extracted
    insight text, not a reformatted task title. Called from apply_grade_result.
    """
    if not is_ready() or not insight_text:
        return None

    metadata = {
        "type": "cross_agent_insight",
        "agent_type": agent_type,
        "task_title": task_title[:200],
        "source": "grader_extraction",
        "importance": 7,
        "timestamp": time.time(),
    }

    doc_id = f"insight-{task_id}-{int(time.time())}"

    return await embed_and_store(
        text=insight_text,
        metadata=metadata,
        collection="semantic",
        doc_id=doc_id,
    )
