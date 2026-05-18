"""Z7 T4 A8 — documentation_gap_detect posthook handler.

Fires on every support escalation (ticket with escalated_to_founder=1 or
low confidence). Semantic-searches the user's question against existing
``support_docs`` docs (per-language collection via lang_collection_name).
If no doc matches (score below threshold), writes a ``docs_gap_log`` row.

The weekly ``faq_regen`` job surfaces gap clusters in A0 briefing and
drafts FAQ entries to fill the gaps.

Handler contract
----------------
``handle(task, result) -> dict``

Returns one of:
  - ``{"status": "gap_logged", "gap_id": N, "question": ...}`` — gap written
  - ``{"status": "covered", "matched_doc_id": str, ...}``      — doc found
  - ``{"status": "skip", "reason": str}``                       — no question/product_id
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.documentation_gap_detect")

# Semantic-search hit threshold: a score above this means the question IS covered.
MATCH_THRESHOLD: float = 0.6


# ---------------------------------------------------------------------------
# Injected helpers (testable via monkeypatch)
# ---------------------------------------------------------------------------


async def retrieve_docs(
    question: str,
    collection_name: str = "support_docs",
    top_k: int = 1,
) -> list[dict]:
    """Semantic search against a per-language support_docs Chroma collection.

    Uses ``src.memory.vector_store.query`` directly so that the
    ``collection_name`` parameter (e.g. ``support_docs_tr``) is honoured —
    ``src.ops.support_rag.retrieve_docs`` always searches ``support_docs``
    and has no collection parameter.

    Returns list of ``{"id": str, "score": float, "document": str}`` dicts.
    Falls back to empty list on import / runtime failure.
    """
    try:
        from src.memory.vector_store import query as vs_query, COLLECTIONS
        # If the per-language collection is not registered, fall back to the
        # base "support_docs" collection rather than silently no-oping.
        if collection_name not in COLLECTIONS:
            logger.info(
                "documentation_gap_detect: collection not registered, falling back to support_docs",
                requested=collection_name,
            )
            collection_name = "support_docs"
        hits = await vs_query(question, collection=collection_name, top_k=top_k)
        # vs_query returns {id, text, metadata, distance}; normalise to
        # the {id, score, document} shape expected by callers.
        results = []
        for h in hits:
            distance = h.get("distance", 1.0)
            # Convert cosine distance to a 0–1 similarity score (clamped).
            # The collection uses hnsw:space=cosine, so distance ∈ [0, 2]
            # and 1.0 - distance maps to a similarity score in [-1, 1], clamped to [0, 1].
            score = max(0.0, 1.0 - float(distance))
            results.append({
                "id": h.get("id", ""),
                "score": score,
                "document": h.get("text", ""),
            })
        return results
    except Exception as exc:
        logger.debug(
            "documentation_gap_detect: retrieve_docs failed (treating as no-match)",
            error=str(exc),
        )
        return []


async def _write_gap_row(
    *,
    product_id: str,
    question: str,
    matched_doc_id: "str | None",
) -> int | None:
    """Write a docs_gap_log row. Returns the new gap_id or None on failure."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "INSERT INTO docs_gap_log "
            "(product_id, question, matched_doc_id, logged_at) "
            "VALUES (?, ?, ?, strftime('%Y-%m-%d %H:%M:%S','now'))",
            (product_id, question, matched_doc_id),
        )
        await db.commit()
        return cur.lastrowid
    except Exception as exc:
        logger.error(
            "documentation_gap_detect: _write_gap_row failed",
            product_id=product_id,
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


async def handle(task: dict, result: dict) -> dict[str, Any]:
    """documentation_gap_detect posthook handler."""
    task_id = task.get("id")

    # Parse task context
    ctx_raw = task.get("context", "{}")
    if isinstance(ctx_raw, str):
        try:
            ctx: dict = json.loads(ctx_raw)
        except Exception:
            ctx = {}
    elif isinstance(ctx_raw, dict):
        ctx = ctx_raw
    else:
        ctx = {}

    # Extract the user's question and product_id from context.
    # The support_tier1 agent stores these in context.payload.
    payload = ctx.get("payload") or ctx
    question = payload.get("question") or ctx.get("question") or ""
    product_id = payload.get("product_id") or ctx.get("product_id") or ""

    if not question:
        logger.debug(
            "documentation_gap_detect: no question in task context — skip",
            task_id=task_id,
        )
        return {"status": "skip", "reason": "no question in task context"}

    if not product_id:
        # Degrade gracefully: use a sentinel product_id so the gap is still logged
        product_id = "__unknown__"

    # Determine the per-language collection to search
    try:
        from src.util.lang import detect_language, lang_collection_name
        lang = detect_language(question, default="en")
        collection = lang_collection_name("support_docs", lang)
    except Exception:
        collection = "support_docs"

    # Semantic search
    hits = await retrieve_docs(question, collection_name=collection, top_k=1)

    # Check if any hit clears the match threshold
    matched_doc_id: str | None = None
    for hit in hits:
        score = hit.get("score", 0.0)
        if score >= MATCH_THRESHOLD:
            matched_doc_id = str(hit.get("id", "unknown"))
            break

    if matched_doc_id is not None:
        logger.info(
            "documentation_gap_detect: question covered by existing doc",
            task_id=task_id,
            product_id=product_id,
            matched_doc_id=matched_doc_id,
        )
        return {
            "status": "covered",
            "matched_doc_id": matched_doc_id,
            "product_id": product_id,
        }

    # No match — write the gap row
    gap_id = await _write_gap_row(
        product_id=product_id,
        question=question,
        matched_doc_id=None,
    )
    logger.info(
        "documentation_gap_detect: gap logged",
        task_id=task_id,
        product_id=product_id,
        gap_id=gap_id,
        question=question[:80],
    )
    return {
        "status": "gap_logged",
        "gap_id": gap_id,
        "product_id": product_id,
        "question": question,
    }
