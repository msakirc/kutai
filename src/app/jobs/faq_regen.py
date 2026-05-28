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


async def enqueue_cluster_draft(cluster: list[dict], lang: str) -> int:
    """Enqueue the cluster-draft LLM call via CPS (SP2 Task 5).

    Returns the child task id immediately. The draft is parsed and the
    founder_action emitted by :func:`_draft_persist_resume` when the
    child completes.
    """
    import time
    import uuid
    import general_beckman
    from general_beckman.lanes import LANE_ONESHOT

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
    messages = [{"role": "user", "content": prompt}]
    _suffix = f"{time.monotonic_ns() % 1_000_000:06d}-{uuid.uuid4().hex[:6]}"

    return await general_beckman.enqueue(
        {
            "title": f"faq_regen:cluster:{lang}:{_suffix}",
            "description": "Draft synthetic FAQ entry from clustered support tickets.",
            "agent_type": "responder",
            "kind": "overhead",
            "priority": 2,
            "context": {
                "source": "faq_regen",
                "lang": lang,
                "llm_call": {
                    "raw_dispatch": True,
                    "call_category": "overhead",
                    "task": "responder",
                    "agent_type": "responder",
                    "difficulty": 3,
                    "messages": messages,
                    "failures": [],
                    "estimated_input_tokens": 500,
                    "estimated_output_tokens": 150,
                },
            },
        },
        lane=LANE_ONESHOT,
        on_complete="faq_regen.draft_persist_resume",
        on_error="faq_regen.draft_persist_err",
        cont_state={"lang": lang, "cluster_size": len(cluster)},
    )


def _extract_faq_entry_from_content(content) -> dict | None:
    """Pull a ``{"question": ..., "answer": ...}`` dict out of arbitrary
    LLM content. Handles dict-shaped result, JSON-string, or JSON embedded
    in surrounding prose. Returns None on failure (matches the pre-CPS
    silent-no-emit behaviour)."""
    import json
    if content is None:
        return None
    # Dict already.
    if isinstance(content, dict) and "question" in content:
        return content
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    text = str(content)
    if not text.strip():
        return None
    # Try whole-string JSON first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "question" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    # Embedded JSON.
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, dict) and "question" in parsed:
                return parsed
        except json.JSONDecodeError:
            return None
    return None


async def _draft_persist_resume(
    child_task_id: int, result: dict, state: dict
) -> None:
    """CPS resume for ``enqueue_cluster_draft`` (SP2 Task 5).

    Parses the LLM response and emits a founder_action when the draft is
    usable. Mirrors the pre-CPS ``run_faq_regen`` per-language loop.
    """
    lang = state.get("lang") or "en"
    cluster_size = int(state.get("cluster_size") or 0)

    # Normal terminal: ``result["result"]["content"]`` (dispatcher envelope).
    # Restart-reconcile: ``tasks.result`` is reconstructed by
    # ``continuations.reconcile_continuations`` as ``dict(parsed)`` of
    # whatever was persisted — so the envelope is flat, e.g.
    # ``{"content": "...", "status": "completed"}``. Handle both shapes.
    result = result or {}
    inner = result.get("result")
    if isinstance(inner, dict):
        content = (
            inner.get("content")
            or inner.get("text")
            or inner.get("response")
        )
    elif inner is not None:
        content = inner
    else:
        content = (
            result.get("content")
            or result.get("text")
            or result.get("response")
        )

    entry = _extract_faq_entry_from_content(content)
    if entry is None:
        logger.info(
            "faq_regen: resume parsed no usable entry (silent skip)",
            child_task_id=child_task_id, lang=lang,
        )
        return
    entry["lang"] = lang
    await _emit_faq_founder_action(
        mission_id=0,
        entry=entry,
        cluster_size=cluster_size,
    )


async def _draft_persist_err(
    child_task_id: int, result: dict, state: dict
) -> None:
    """CPS on_error for ``enqueue_cluster_draft`` (SP2 Task 5).

    Mirrors pre-CPS ``_llm_cluster_draft`` returning None on LLM failure —
    no founder action is emitted; the cluster is silently retried next
    weekly run.
    """
    logger.warning(
        "faq_regen: cluster-draft LLM failed; no founder action emitted",
        child_task_id=child_task_id,
        lang=state.get("lang"),
        cluster_size=state.get("cluster_size"),
        error=(result or {}).get("error"),
    )


def register_continuations() -> None:
    """Register faq_regen CPS handlers (SP2). Idempotent."""
    try:
        from general_beckman.continuations import register_resume
        register_resume("faq_regen.draft_persist_resume", _draft_persist_resume)
        register_resume("faq_regen.draft_persist_err",    _draft_persist_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("faq_regen continuation registration deferred",
                     error=str(exc))


# Register at import so the handler is present for restart reconcile.
register_continuations()


# ---------------------------------------------------------------------------
# File write + Chroma re-index
# ---------------------------------------------------------------------------


async def _reindex_collection(collection_name: str, text: str) -> None:
    """Re-index the per-language Chroma collection with the new FAQ text.

    Delegates to ``src.memory.vector_store`` if available; degrades gracefully
    on import failure (Chroma not initialised yet in test environments).
    If the per-language collection is not registered, falls back to the base
    ``support_docs`` collection rather than silently no-oping.
    """
    try:
        from src.memory.vector_store import embed_and_store, COLLECTIONS
        # Fall back to the base collection if the per-language one is not registered.
        if collection_name not in COLLECTIONS:
            logger.info(
                "faq_regen: collection not registered, falling back to support_docs",
                requested=collection_name,
            )
            collection_name = "support_docs"
        await embed_and_store(
            text,
            {"source": "faq_regen"},
            collection=collection_name,
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
        payload = {
            "faq_entry": entry,
            "cluster_size": cluster_size,
            "_faq_approval_pending": True,
        }
        return await create_founder_action(
            mission_id=mission_id,
            kind="generic",
            title=title,
            why=why,
            instructions=instructions,
            expected_output_kind="ack_only",
            expected_output_schema=payload,
            notify_telegram=True,
        )
    except Exception as exc:
        logger.warning("faq_regen: _emit_faq_founder_action failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_faq_regen(confidence_threshold: float = CONFIDENCE_THRESHOLD, days: int = 7) -> dict:
    """Weekly FAQ regen entry point. Called by mr_roboto for the ``faq_regen`` executor.

    SP2: per-language clusters are enqueued via CPS; founder_actions are
    emitted by ``_draft_persist_resume`` when each child completes.
    Returns ``{"ok": True, "queued": N}`` immediately.
    """
    try:
        tickets = await _fetch_candidate_tickets(
            confidence_threshold=confidence_threshold,
            days=days,
        )
        if not tickets:
            logger.info("faq_regen: no candidate tickets in window")
            return {"ok": True, "queued": 0, "reason": "no_candidates"}

        grouped = _group_tickets_by_language(tickets)
        total_queued = 0

        for lang, lang_tickets in grouped.items():
            logger.info(
                "faq_regen: processing language group",
                lang=lang,
                count=len(lang_tickets),
            )
            if len(lang_tickets) <= MIN_CLUSTER_SIZE:
                continue
            try:
                await enqueue_cluster_draft(lang_tickets, lang)
                total_queued += 1
            except Exception as exc:
                logger.warning(
                    "faq_regen: enqueue_cluster_draft failed",
                    lang=lang, error=str(exc),
                )

        logger.info("faq_regen: run complete", total_queued=total_queued)
        return {"ok": True, "queued": total_queued}

    except Exception as exc:
        logger.error("faq_regen: failed", error=str(exc))
        return {"ok": False, "reason": str(exc)}
