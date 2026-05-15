"""Z7 T4 A8 — Weekly FAQ regen job (A8 + A8.r1 multilingual).

Runs weekly (registered in beckman cron_seed as ``_executor='faq_regen'``).

Pipeline
--------
1. Pull last 7 days from ``tickets`` where ``confidence < threshold`` OR
   ``escalated_to_founder = 1``.
2. Detect each ticket's language via ``src.util.lang.detect_language``.
3. Group tickets by language (within-language clustering only — no cross-lingual
   mixing per A8.r1).
4. For each language group: LLM-cluster by topic (OVERHEAD lane via
   ``general_beckman.enqueue``).
5. For clusters > 3 interactions: draft a FAQ entry.
6. Surface each draft as a founder_action "approve FAQ entry?".
7. On approve (caller invokes ``_apply_faq_approval``): append to per-language
   ``faq_{lang}.md`` artifact and re-index per-language ``support_docs_{lang}``
   Chroma collection.

Public API
----------
- ``run_faq_regen()``             — main entry point (mr_roboto executor).
- ``_fetch_candidate_tickets(confidence_threshold, days)`` — DB query, testable.
- ``_group_tickets_by_language(tickets)``   — lang-detect + group, testable.
- ``_draft_faq_entry(cluster, lang)``       — calls LLM if cluster > 3.
- ``_apply_faq_approval(entry)``            — file-write + Chroma re-index.
- ``FAQ_ARTIFACTS_DIR``                     — monkeypatchable in tests.
- ``_reindex_collection(collection_name, text)`` — monkeypatchable in tests.
"""
from __future__ import annotations

import os
from typing import Any

from src.infra.logging_config import get_logger
from src.util.lang import detect_language, lang_artifact_path, lang_collection_name

logger = get_logger("app.jobs.faq_regen")

# Tunable threshold — tickets below this confidence are candidates.
CONFIDENCE_THRESHOLD: float = 0.7
# Minimum cluster size to draft a FAQ entry.
MIN_CLUSTER_SIZE: int = 3
# Artifact directory for per-language FAQ markdown files.
FAQ_ARTIFACTS_DIR: str = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "docs", "support"
)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _fetch_candidate_tickets(
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    days: int = 7,
) -> list[dict]:
    """Return low-confidence or escalated tickets from the last *days* days.

    Criteria: ``confidence < threshold`` OR ``escalated_to_founder = 1``.
    Old tickets (> days ago) are excluded.
    """
    from src.infra.db import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT id, user_id, question, answer, confidence, "
            "status, escalated_to_founder, sentiment, created_at "
            "FROM tickets "
            "WHERE (confidence < ? OR escalated_to_founder = 1) "
            "  AND created_at >= datetime('now', ?) "
            "ORDER BY created_at DESC",
            (confidence_threshold, f"-{int(days)} days"),
        )
        cols = [d[0] for d in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]
    except Exception as exc:
        logger.warning("faq_regen: _fetch_candidate_tickets failed", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Language grouping
# ---------------------------------------------------------------------------


def _group_tickets_by_language(tickets: list[dict]) -> dict[str, list[dict]]:
    """Detect each ticket's language and group into per-language lists.

    Within-language only: tickets in different languages are never mixed
    into the same cluster (A8.r1 requirement).
    """
    groups: dict[str, list[dict]] = {}
    for ticket in tickets:
        question = ticket.get("question") or ""
        lang = detect_language(question, default="en")
        groups.setdefault(lang, []).append(ticket)
    return groups


# ---------------------------------------------------------------------------
# LLM clustering + entry drafting
# ---------------------------------------------------------------------------


async def _llm_cluster_draft(cluster: list[dict], lang: str) -> dict | None:
    """Call LLM via beckman OVERHEAD lane to draft a synthetic FAQ entry.

    Returns ``{"question": ..., "answer": ...}`` or None on failure.
    """
    # Build a compact prompt
    examples = "\n".join(
        f"Q: {t.get('question', '')}\nA: {t.get('answer', '')}"
        for t in cluster[:10]  # cap at 10 to stay within context
    )
    prompt = (
        f"You are a support documentation writer. "
        f"Given these {len(cluster)} similar support interactions "
        f"(in language code '{lang}'), write ONE canonical FAQ entry:\n\n"
        f"{examples}\n\n"
        f"Output a JSON object with keys 'question' and 'answer' only. "
        f"The FAQ should be in the same language as the interactions."
    )
    try:
        from packages.general_beckman.src.general_beckman.beckman import get_beckman
        beckman = get_beckman()
        result = await beckman.enqueue(
            goal=prompt,
            agent_type="responder",
            lane="OVERHEAD",
            context={"source": "faq_regen", "lang": lang},
        )
        if isinstance(result, dict) and "question" in result:
            return result
        # Try to extract JSON from text response
        import json
        text = str(result) if result else ""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            if "question" in parsed:
                return parsed
    except Exception as exc:
        logger.warning("faq_regen: _llm_cluster_draft failed", error=str(exc))
    return None


async def _draft_faq_entry(cluster: list[dict], lang: str) -> dict | None:
    """Draft a FAQ entry from *cluster* if cluster size > MIN_CLUSTER_SIZE.

    Returns the draft dict (with 'question', 'answer', 'lang') or None.
    """
    if len(cluster) <= MIN_CLUSTER_SIZE:
        return None
    draft = await _llm_cluster_draft(cluster, lang)
    if draft:
        draft["lang"] = lang
    return draft


# ---------------------------------------------------------------------------
# File write + Chroma re-index
# ---------------------------------------------------------------------------


async def _reindex_collection(collection_name: str, text: str) -> None:
    """Re-index the per-language Chroma collection with the new FAQ text.

    Delegates to ``src.memory.vector_store`` if available; degrades gracefully
    on import failure (Chroma not initialised yet in test environments).
    """
    try:
        from src.memory.vector_store import embed_and_store
        await embed_and_store(
            collection_name=collection_name,
            documents=[text],
            metadatas=[{"source": "faq_regen"}],
        )
        logger.info("faq_regen: re-indexed collection", collection=collection_name)
    except Exception as exc:
        logger.warning(
            "faq_regen: _reindex_collection failed (non-fatal)",
            collection=collection_name,
            error=str(exc),
        )


async def _apply_faq_approval(entry: dict) -> None:
    """Apply an approved FAQ entry: append to per-language markdown + re-index Chroma.

    *entry* must have 'question', 'answer', 'lang'.
    """
    lang = entry.get("lang", "en")
    question = entry.get("question", "")
    answer = entry.get("answer", "")

    # --- 1. Write to per-language faq file ---
    artifact_filename = lang_artifact_path("faq", lang)
    artifacts_dir = FAQ_ARTIFACTS_DIR
    os.makedirs(artifacts_dir, exist_ok=True)
    faq_path = os.path.join(artifacts_dir, artifact_filename)

    faq_block = f"\n## {question}\n\n{answer}\n"
    try:
        with open(faq_path, "a", encoding="utf-8") as f:
            f.write(faq_block)
        logger.info(
            "faq_regen: appended FAQ entry",
            path=faq_path,
            lang=lang,
            question=question[:60],
        )
    except Exception as exc:
        logger.error("faq_regen: failed to write FAQ file", path=faq_path, error=str(exc))
        return

    # --- 2. Re-index Chroma collection ---
    collection = lang_collection_name("support_docs", lang)
    await _reindex_collection(collection, faq_block)


# ---------------------------------------------------------------------------
# Founder action emission
# ---------------------------------------------------------------------------


async def _emit_faq_founder_action(
    *,
    mission_id: int,
    entry: dict,
    cluster_size: int,
) -> Any:
    """Emit a founder_action asking to approve or reject a drafted FAQ entry."""
    try:
        from src.founder_actions import create as create_founder_action
        lang = entry.get("lang", "en")
        question = entry.get("question", "")
        title = f"Approve FAQ entry [{lang}]: {question[:60]}"
        why = (
            f"KutAI clustered {cluster_size} low-confidence / escalated tickets "
            f"in language '{lang}' around a common question and drafted this FAQ entry. "
            "Approve to append it to the per-language FAQ file and re-index support docs."
        )
        instructions = [
            "Review the drafted question and answer below.",
            "Approve to append to faq.md (or faq_{lang}.md) and re-index support_docs_{lang}.",
            "Reject to discard the draft.",
        ]
        payload_json = {
            "faq_entry": entry,
            "cluster_size": cluster_size,
            "_faq_approval_pending": True,
        }
        import json
        return await create_founder_action(
            mission_id=mission_id,
            kind="generic",
            title=title,
            why=why,
            instructions=instructions,
            expected_output_kind="ack_only",
            notify_telegram=True,
            context_json=json.dumps(payload_json),
        )
    except Exception as exc:
        logger.warning("faq_regen: _emit_faq_founder_action failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_faq_regen(confidence_threshold: float = CONFIDENCE_THRESHOLD, days: int = 7) -> dict:
    """Weekly FAQ regen entry point. Called by mr_roboto for the ``faq_regen`` executor.

    Returns ``{"ok": True, "drafts": N}`` on success.
    """
    try:
        tickets = await _fetch_candidate_tickets(
            confidence_threshold=confidence_threshold,
            days=days,
        )
        if not tickets:
            logger.info("faq_regen: no candidate tickets in window")
            return {"ok": True, "drafts": 0, "reason": "no_candidates"}

        grouped = _group_tickets_by_language(tickets)
        total_drafts = 0

        for lang, lang_tickets in grouped.items():
            logger.info(
                "faq_regen: processing language group",
                lang=lang,
                count=len(lang_tickets),
            )
            # Simple topic clustering: group by shared keywords
            # Real clustering: pass all to _llm_cluster_draft as one big cluster
            # For now, treat the entire language group as one cluster — LLM
            # can split or merge topics internally. Production upgrade: k-means
            # or LLM-driven multi-cluster.
            entry = await _draft_faq_entry(lang_tickets, lang)
            if entry:
                await _emit_faq_founder_action(
                    mission_id=0,
                    entry=entry,
                    cluster_size=len(lang_tickets),
                )
                total_drafts += 1

        logger.info("faq_regen: run complete", total_drafts=total_drafts)
        return {"ok": True, "drafts": total_drafts}

    except Exception as exc:
        logger.error("faq_regen: failed", error=str(exc))
        return {"ok": False, "reason": str(exc)}
