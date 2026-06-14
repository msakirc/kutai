"""SP4b — reviews CPS mechanical sinks (continuation handlers).

These are MECHANICAL: no LLM call, no dispatcher import. They receive the
already-produced LLM output (reviews.*.resume) or fire on_error with a
heuristic/canned fallback (reviews.*.resume_err), validate/persist, route
side-effects, and enforce the never-auto-post contract. Registered in
``general_beckman.continuations._HANDLER_MODULES`` so the handlers are present
after a restart (else continuation rows stay pending — silent correctness bug).

Handler signature (canonical, from posthook_continuations.py):
    async def handler(child_task_id: int, result: dict, state: dict) -> None
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

# Mechanical helpers reused from the (now LLM-free) verb modules.
from mr_roboto.reviews_classify import (
    VALID_SENTIMENTS, VALID_THEMES, LOW_STAR_THRESHOLD,
    _heuristic_classify, _parse_llm_response,
    _emit_low_star_founder_action, _enqueue_bug_investigation,
)
from mr_roboto.reviews_draft_reply import _fallback_draft

logger = get_logger("mr_roboto.executors.reviews_continuations")


def _extract_content(result: dict) -> str:
    """Dual-shape decode (normal terminal vs restart-reconcile)."""
    result = result or {}
    inner = result.get("result")
    if isinstance(inner, dict):
        content = inner.get("content", "")
    elif inner is not None:
        content = inner
    else:
        content = result.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


# ── classify sink ───────────────────────────────────────────────────────────

async def _persist_classification(review_id, sentiment, theme_tag) -> None:
    from dabidabi import get_db
    db = await get_db()
    await db.execute(
        "UPDATE external_reviews SET sentiment=?, theme_tag=? WHERE review_id=?",
        (sentiment, theme_tag, review_id),
    )
    await db.commit()


async def _route_classify_sideeffects(state: dict, sentiment: str, theme_tag: str) -> None:
    review_id = state["review_id"]
    rating = int(state.get("rating") or 0)
    if rating <= LOW_STAR_THRESHOLD:
        await _emit_low_star_founder_action(
            review_id=review_id, platform=state.get("platform") or "",
            author=state.get("author") or "Unknown", rating=rating,
            body_md=state.get("body_md") or "", product_id=state.get("product_id") or "",
            theme_tag=theme_tag,
        )
    if theme_tag == "bug":
        await _enqueue_bug_investigation({
            "title": f"[BUG] Investigate report from {state.get('platform')} review",
            "description": (
                f"Review on {state.get('platform')} by {state.get('author')!r} "
                f"classified as bug. Body: {(state.get('body_md') or '')[:200]}..."
            ),
            "agent_type": "mechanical", "kind": "overhead",
            "context": {"review_id": review_id, "platform": state.get("platform"),
                        "product_id": state.get("product_id"),
                        "body_md": (state.get("body_md") or "")[:500]},
        })


async def _apply_classify(state: dict, sentiment: str, theme_tag: str) -> None:
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "neutral"
    if theme_tag not in VALID_THEMES:
        theme_tag = "generic-negative"
    await _persist_classification(state["review_id"], sentiment, theme_tag)
    await _route_classify_sideeffects(state, sentiment, theme_tag)


async def _classify_resume(child_task_id: int, result: dict, state: dict) -> None:
    raw = _extract_content(result).strip()
    c = _parse_llm_response(raw, state.get("body_md") or "", int(state.get("rating") or 0))
    await _apply_classify(state, c.get("sentiment", "neutral"), c.get("theme_tag", "generic-negative"))


async def _classify_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    logger.warning("reviews classify child failed (%s) — heuristic fallback",
                   (result or {}).get("error"))
    c = _heuristic_classify(state.get("body_md") or "", int(state.get("rating") or 0))
    await _apply_classify(state, c["sentiment"], c["theme_tag"])


# ── draft_reply sink (never auto-posts) ──────────────────────────────────────

async def _surface_draft(state: dict, draft: str) -> None:
    """Mechanical: surface the draft to the founder. NEVER auto-posts —
    replied_at / reply_body_md stay NULL until the founder manually confirms."""
    from src.founder_actions import create as fa_create
    review_id = state["review_id"]
    platform = state.get("platform") or ""
    await fa_create(
        mission_id=None, kind="generic",
        title=f"Draft reply ready for {platform} review (id={review_id}) — review before posting",
        why=("A reply draft was generated. NEVER auto-posted — review/edit, then "
             "post manually via the platform. Mark done when sent."),
        instructions=[f"Draft:\n\n{draft[:1000]}",
                      "Edit if needed, then post manually on the platform.",
                      "NEVER reply automatically — this is a draft only."],
        expected_output_kind="ack_only", notify_telegram=True,
    )


async def _draft_reply_resume(child_task_id: int, result: dict, state: dict) -> None:
    draft = _extract_content(result).strip()
    if not draft:
        draft = _fallback_draft(state.get("platform") or "", state.get("author") or "Anonymous",
                                int(state.get("rating") or 3))
    await _surface_draft(state, draft)


async def _draft_reply_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    logger.warning("reviews draft_reply child failed (%s) — fallback draft",
                   (result or {}).get("error"))
    draft = _fallback_draft(state.get("platform") or "", state.get("author") or "Anonymous",
                            int(state.get("rating") or 3))
    await _surface_draft(state, draft)


def register_continuations() -> None:
    """Register reviews CPS sinks. Idempotent."""
    try:
        from general_beckman.continuations import register_resume
        register_resume("reviews.classify.resume", _classify_resume)
        register_resume("reviews.classify.resume_err", _classify_resume_err)
        register_resume("reviews.draft_reply.resume", _draft_reply_resume)
        register_resume("reviews.draft_reply.resume_err", _draft_reply_resume_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("reviews continuation registration deferred: %s", exc)


register_continuations()
