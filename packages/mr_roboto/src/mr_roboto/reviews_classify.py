"""Z7 T5 B8 — reviews/classify mechanical executor.

LLM-bound classifier for external reviews.

Assigns:
  sentiment : 'positive' | 'negative' | 'neutral'
  theme_tag : 'UX' | 'pricing' | 'bug' | 'feature-request' |
              'support' | 'generic-positive' | 'generic-negative'

Side-effects after classification:
  1-2-star  → _emit_low_star_founder_action (review + decide reply)
  5-star    → (no automatic action; daily job may surface for A4 quotes)
  bug-tagged → _enqueue_bug_investigation (investigation task in backlog)

Public API
----------
  run(payload) -> dict
      payload keys: review_id (int), product_id (str)

  _call_llm_classify(body_md, rating) -> {"sentiment": str, "theme_tag": str}
      Internal LLM call via beckman.enqueue OVERHEAD lane — mocked in tests.

  _emit_low_star_founder_action(**kwargs) -> Any
      Internal founder_action emitter — mocked in tests.

  _enqueue_bug_investigation(spec, **kwargs) -> int
      Internal beckman.enqueue call — mocked in tests.
"""
from __future__ import annotations

import asyncio
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.reviews_classify")

VALID_SENTIMENTS = frozenset({"positive", "negative", "neutral"})
VALID_THEMES = frozenset({
    "UX", "pricing", "bug", "feature-request",
    "support", "generic-positive", "generic-negative",
})

# Ratings at or below this threshold trigger a low-star founder_action.
LOW_STAR_THRESHOLD = 2


# ---------------------------------------------------------------------------
# LLM call (OVERHEAD lane via beckman.enqueue)
# ---------------------------------------------------------------------------

async def _call_llm_classify(body_md: str, rating: int) -> dict:
    """Call LLM (OVERHEAD lane) to classify the review.

    Returns {"sentiment": str, "theme_tag": str}.
    Falls back to heuristic on timeout / error.
    """
    from general_beckman import enqueue
    from general_beckman.lanes import LANE_OVERHEAD

    prompt = (
        "You are classifying a product review. Return JSON only.\n\n"
        f"Rating: {rating}/5\n"
        f"Review: {body_md[:800]}\n\n"
        'Respond ONLY with: {"sentiment": "<positive|negative|neutral>", '
        '"theme_tag": "<UX|pricing|bug|feature-request|support|generic-positive|generic-negative>"}\n'
        "Pick the single most relevant theme_tag."
    )

    result_holder: list[str] = []
    done_event = asyncio.Event()

    async def _on_finish(task_result: dict) -> None:
        output = task_result.get("output") or task_result.get("result") or ""
        result_holder.append(str(output))
        done_event.set()

    await enqueue(
        {
            "title": "reviews_classify:llm",
            "description": "Classify review sentiment + theme.",
            "agent_type": "assistant",
            "kind": "overhead",
            "context": {
                "prompt": prompt,
                "_callback": _on_finish,
            },
        },
        lane=LANE_OVERHEAD,
    )

    try:
        await asyncio.wait_for(done_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("reviews_classify: LLM timed out; using heuristic fallback")
        return _heuristic_classify(body_md, rating)

    raw = result_holder[0] if result_holder else ""
    return _parse_llm_response(raw, body_md, rating)


def _parse_llm_response(raw: str, body_md: str, rating: int) -> dict:
    """Parse LLM JSON response with fallback."""
    import json as _json
    import re

    # Try to extract JSON from the raw output
    try:
        match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if match:
            parsed = _json.loads(match.group(0))
            sentiment = parsed.get("sentiment", "")
            theme_tag = parsed.get("theme_tag", "")
            if sentiment in VALID_SENTIMENTS and theme_tag in VALID_THEMES:
                return {"sentiment": sentiment, "theme_tag": theme_tag}
    except Exception:
        pass
    return _heuristic_classify(body_md, rating)


def _heuristic_classify(body_md: str, rating: int) -> dict:
    """Simple heuristic fallback when LLM is unavailable."""
    body_lower = body_md.lower()
    # Theme detection
    if any(w in body_lower for w in ("crash", "bug", "error", "broken", "doesn't work", "not working")):
        theme_tag = "bug"
    elif any(w in body_lower for w in ("price", "pricing", "expensive", "cost", "cheap")):
        theme_tag = "pricing"
    elif any(w in body_lower for w in ("ui", "ux", "design", "interface", "usability", "confusing")):
        theme_tag = "UX"
    elif any(w in body_lower for w in ("feature", "missing", "wish", "would be nice")):
        theme_tag = "feature-request"
    elif any(w in body_lower for w in ("support", "help", "response", "team")):
        theme_tag = "support"
    elif rating >= 4:
        theme_tag = "generic-positive"
    else:
        theme_tag = "generic-negative"

    # Sentiment
    if rating >= 4:
        sentiment = "positive"
    elif rating <= 2:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"sentiment": sentiment, "theme_tag": theme_tag}


# ---------------------------------------------------------------------------
# Side-effect helpers (mocked in tests)
# ---------------------------------------------------------------------------

async def _emit_low_star_founder_action(
    *,
    review_id: int,
    platform: str,
    author: str,
    rating: int,
    body_md: str,
    product_id: str,
    theme_tag: str,
) -> Any:
    """Emit a founder_action card for a 1-2-star review.

    Founder reviews the bad review and decides whether/how to reply.
    NEVER auto-replies. Draft reply can be generated separately via
    reviews/draft_reply.
    """
    try:
        from src.founder_actions import create as fa_create
        return await fa_create(
            mission_id=None,
            kind="generic",
            title=(
                f"[{rating}-star review on {platform}] Review from {author!r} — "
                f"theme: {theme_tag}"
            ),
            why=(
                f"A {rating}-star review was received on {platform}. "
                f"Theme classification: {theme_tag}. "
                "Founder should review and decide whether to reply."
            ),
            instructions=[
                f"Read the {rating}-star review from {author!r} (review_id={review_id}).",
                "Decide: ignore, respond, or escalate as a product issue.",
                "Use reviews/draft_reply to generate a draft reply if needed.",
                "NEVER reply directly — always use the draft for founder approval first.",
            ],
            expected_output_kind="ack_only",
            notify_telegram=True,
        )
    except Exception as exc:
        logger.warning("reviews_classify._emit_low_star_founder_action failed: %s", exc)
        return None


async def _enqueue_bug_investigation(spec: dict, **kwargs) -> int:
    """Enqueue a bug investigation task in the mission backlog via beckman."""
    try:
        from general_beckman import enqueue
        from general_beckman.lanes import LANE_OVERHEAD
        return await enqueue(spec, lane=LANE_OVERHEAD, **kwargs)
    except Exception as exc:
        logger.warning("reviews_classify._enqueue_bug_investigation failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

async def run(payload: dict) -> dict:
    """mr_roboto executor: reviews/classify.

    Classifies the review at ``review_id`` and updates external_reviews
    with sentiment + theme_tag. Triggers side-effects based on rating/theme.

    payload keys:
        review_id  (int) — required
        product_id (str) — required
    """
    review_id_raw = payload.get("review_id")
    product_id = str(payload.get("product_id") or "")

    if review_id_raw is None:
        return {"status": "error", "error": "review_id is required"}
    if not product_id:
        return {"status": "error", "error": "product_id is required"}

    review_id = int(review_id_raw)

    # Load the review
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT review_id, platform, external_id, author, rating, body_md "
            "FROM external_reviews WHERE review_id=?",
            (review_id,),
        )
        row = await cur.fetchone()
    except Exception as exc:
        return {"status": "error", "error": f"DB read failed: {exc}"}

    if row is None:
        return {"status": "error", "error": f"review_id={review_id} not found"}

    _, platform, external_id, author, rating, body_md = row
    rating = int(rating or 0)
    body_md = body_md or ""

    # Call LLM classifier
    try:
        classification = await _call_llm_classify(body_md, rating)
    except Exception as exc:
        logger.warning("reviews_classify: LLM call failed; using heuristic: %s", exc)
        classification = _heuristic_classify(body_md, rating)

    sentiment = classification.get("sentiment", "neutral")
    theme_tag = classification.get("theme_tag", "generic-negative")

    # Validate
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "neutral"
    if theme_tag not in VALID_THEMES:
        theme_tag = "generic-negative"

    # Persist classification
    try:
        await db.execute(
            "UPDATE external_reviews SET sentiment=?, theme_tag=? WHERE review_id=?",
            (sentiment, theme_tag, review_id),
        )
        await db.commit()
    except Exception as exc:
        return {"status": "error", "error": f"DB update failed: {exc}"}

    # Side-effect: 1-2-star → founder_action
    if rating <= LOW_STAR_THRESHOLD:
        await _emit_low_star_founder_action(
            review_id=review_id,
            platform=platform,
            author=author or "Unknown",
            rating=rating,
            body_md=body_md,
            product_id=product_id,
            theme_tag=theme_tag,
        )

    # Side-effect: bug-tagged → enqueue investigation
    if theme_tag == "bug":
        bug_spec = {
            "title": f"[BUG] Investigate report from {platform} review",
            "description": (
                f"Review on {platform} (id={external_id}) by {author!r} "
                f"classified as bug. Body: {body_md[:200]}..."
            ),
            "agent_type": "mechanical",
            "kind": "overhead",
            "context": {
                "review_id": review_id,
                "platform": platform,
                "product_id": product_id,
                "body_md": body_md[:500],
            },
        }
        await _enqueue_bug_investigation(bug_spec)

    logger.info(
        "reviews_classify: review_id=%d platform=%s sentiment=%s theme=%s",
        review_id, platform, sentiment, theme_tag,
    )

    return {
        "status": "ok",
        "review_id": review_id,
        "sentiment": sentiment,
        "theme_tag": theme_tag,
        "low_star_action": rating <= LOW_STAR_THRESHOLD,
        "bug_investigation_queued": theme_tag == "bug",
    }


__all__ = [
    "run",
    "_call_llm_classify",
    "_heuristic_classify",
    "_emit_low_star_founder_action",
    "_enqueue_bug_investigation",
    "VALID_SENTIMENTS",
    "VALID_THEMES",
    "LOW_STAR_THRESHOLD",
]
