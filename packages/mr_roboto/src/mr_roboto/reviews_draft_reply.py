"""Z7 T5 B8 — reviews/draft_reply mechanical helper.

SP4b: the reply-drafting LLM was extracted out of mr_roboto. The draft LLM
hop is now an admitted Beckman producer task (``src.reviews.producers.
enqueue_draft_reply``) whose ``reviews.draft_reply.resume`` continuation
(``mr_roboto.executors.reviews_continuations``) surfaces the draft to the
founder. Platform conventions + the prompt now live in the producer.

Critical contract: NEVER auto-posts. The sink surfaces a draft the founder
approves and manually posts; replied_at / reply_body_md stay NULL until the
founder marks the reply as sent. This module retains only the mechanical
``_fallback_draft`` (used by the sink on LLM failure).
"""
from __future__ import annotations

from yazbunu import get_logger

logger = get_logger("mr_roboto.reviews_draft_reply")


def _fallback_draft(platform: str, author: str, rating: int) -> str:
    """Generic fallback draft when the LLM is unavailable."""
    if rating >= 4:
        return (
            f"Thank you so much for your kind review, {author}! "
            "We're really glad you're enjoying the product. "
            "Feel free to reach out if there's anything we can improve."
        )
    return (
        f"Thank you for your honest feedback, {author}. "
        "We take every review seriously and would love to hear more "
        "about your experience so we can make things right. "
        "Please reach out to us at support@example.com."
    )


async def run(payload: dict) -> dict:
    """DEPRECATED direct entry. Enqueues the CPS draft_reply producer and
    returns its task id. NEVER auto-posts. Kept for legacy callers; new
    callers use the router action (reviews/draft_reply)."""
    review_id_raw = payload.get("review_id")
    product_id = str(payload.get("product_id") or "")
    if review_id_raw is None:
        return {"status": "error", "error": "review_id is required"}
    if not product_id:
        return {"status": "error", "error": "product_id is required"}
    from src.reviews.producers import enqueue_draft_reply
    tid = await enqueue_draft_reply(review_id=int(review_id_raw), product_id=product_id)
    if tid is None:
        return {"status": "error", "error": f"review_id={review_id_raw} not found"}
    return {"status": "ok", "enqueued": tid, "auto_posted": False}


__all__ = [
    "run",
    "_fallback_draft",
]
