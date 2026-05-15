"""Z7 T5 B8 — reviews/draft_reply mechanical executor.

LLM-bound reply drafter for external reviews.

Critical contract: NEVER auto-posts. Produces a draft that the founder
approves and manually posts through the platform's native interface.
replied_at and reply_body_md in external_reviews remain NULL until the
founder manually marks the reply as sent.

Public API
----------
  run(payload) -> dict
      payload keys: review_id (int), product_id (str)
      Returns: {"status": "ok", "reply_draft": str, "review_id": int}

  _call_llm_draft_reply(platform, author, rating, body_md, product_id) -> str
      Internal LLM call via beckman.enqueue OVERHEAD lane — mocked in tests.
"""
from __future__ import annotations

import asyncio

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.reviews_draft_reply")

# Platform-specific conventions for reply tone
_PLATFORM_CONVENTIONS: dict[str, str] = {
    "g2": (
        "G2 replies are read by B2B buyers. Be professional, specific, "
        "and acknowledge the feedback. Offer to connect privately for issues."
    ),
    "appstore": (
        "AppStore replies are brief (up to 1000 chars). Be warm, thank the user, "
        "and for negative reviews offer a specific path to resolution (email/support)."
    ),
    "playstore": (
        "PlayStore replies are brief. Acknowledge the feedback, be empathetic, "
        "and direct to support channel for technical issues."
    ),
    "producthunt": (
        "ProductHunt replies are community-facing. Be authentic, founder-to-user, "
        "conversational. For criticism, acknowledge openly and describe what's next."
    ),
    "trustpilot": (
        "Trustpilot replies are public and formal. Use company name, be professional, "
        "address the specific complaint, offer resolution path."
    ),
    "capterra": (
        "Capterra is B2B. Professional tone, acknowledge the point, highlight roadmap "
        "or workarounds for feature requests."
    ),
}

_DEFAULT_CONVENTION = (
    "Be polite, genuine, and specific to the review content. "
    "For negative reviews, acknowledge the issue and offer a resolution path. "
    "For positive reviews, thank the reviewer warmly."
)


# ---------------------------------------------------------------------------
# LLM call (OVERHEAD lane via beckman.enqueue)
# ---------------------------------------------------------------------------

async def _call_llm_draft_reply(
    platform: str,
    author: str,
    rating: int,
    body_md: str,
    product_id: str,
) -> str:
    """Call LLM (OVERHEAD lane) to draft a reply.

    Returns draft reply text. Falls back to a generic template on error.
    """
    from general_beckman import enqueue
    from general_beckman.lanes import LANE_OVERHEAD

    convention = _PLATFORM_CONVENTIONS.get(platform, _DEFAULT_CONVENTION)
    star_label = f"{rating}/5 star{'s' if rating != 1 else ''}"

    prompt = (
        f"You are drafting a reply to a {star_label} review on {platform}.\n"
        f"Reviewer: {author}\n"
        f"Review content: {body_md[:600]}\n\n"
        f"Platform conventions:\n{convention}\n\n"
        "Write a reply in first person from the product founder's perspective.\n"
        "Rules:\n"
        "- Do NOT make promises about specific features or timelines.\n"
        "- Do NOT offer refunds unless the review mentions billing issues.\n"
        "- Keep it concise: 2-4 sentences for positive reviews, 3-6 for negative.\n"
        "- Do NOT start with 'Hi,' or 'Dear,'\n"
        "Draft the reply only — no meta-commentary."
    )

    result_holder: list[str] = []
    done_event = asyncio.Event()

    async def _on_finish(task_result: dict) -> None:
        output = task_result.get("output") or task_result.get("result") or ""
        result_holder.append(str(output))
        done_event.set()

    await enqueue(
        {
            "title": "reviews_draft_reply:llm",
            "description": f"Draft reply for {platform} review.",
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
        logger.warning("reviews_draft_reply: LLM timed out; returning fallback draft")
        return _fallback_draft(platform, author, rating)

    return result_holder[0].strip() if result_holder else _fallback_draft(platform, author, rating)


def _fallback_draft(platform: str, author: str, rating: int) -> str:
    """Generic fallback draft when LLM is unavailable."""
    if rating >= 4:
        return (
            f"Thank you so much for your kind review, {author}! "
            "We're really glad you're enjoying the product. "
            "Feel free to reach out if there's anything we can improve."
        )
    else:
        return (
            f"Thank you for your honest feedback, {author}. "
            "We take every review seriously and would love to hear more "
            "about your experience so we can make things right. "
            "Please reach out to us at support@example.com."
        )


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

async def run(payload: dict) -> dict:
    """mr_roboto executor: reviews/draft_reply.

    Generates a draft reply for the review at ``review_id``.
    NEVER auto-posts: replied_at and reply_body_md remain NULL.
    The draft is returned to the caller for founder review.

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
            "SELECT review_id, platform, author, rating, body_md "
            "FROM external_reviews WHERE review_id=?",
            (review_id,),
        )
        row = await cur.fetchone()
    except Exception as exc:
        return {"status": "error", "error": f"DB read failed: {exc}"}

    if row is None:
        return {"status": "error", "error": f"review_id={review_id} not found"}

    _, platform, author, rating, body_md = row
    rating = int(rating or 3)
    author = author or "Anonymous"
    body_md = body_md or ""

    # Generate draft reply (LLM-bound, mocked in tests)
    try:
        reply_draft = await _call_llm_draft_reply(
            platform=platform,
            author=author,
            rating=rating,
            body_md=body_md,
            product_id=product_id,
        )
    except Exception as exc:
        logger.warning("reviews_draft_reply: LLM failed: %s", exc)
        reply_draft = _fallback_draft(platform, author, rating)

    logger.info(
        "reviews_draft_reply: review_id=%d platform=%s draft_len=%d",
        review_id, platform, len(reply_draft),
    )

    # IMPORTANT: Do NOT write to replied_at or reply_body_md.
    # Those fields are only set when the founder manually confirms the reply was sent.
    return {
        "status": "ok",
        "review_id": review_id,
        "platform": platform,
        "reply_draft": reply_draft,
        "auto_posted": False,  # Explicit contract: never auto-posted
    }


__all__ = [
    "run",
    "_call_llm_draft_reply",
    "_fallback_draft",
]
