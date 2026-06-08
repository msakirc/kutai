"""SP4b — reviews CPS producers (LLM extracted out of mr_roboto).

Each function loads the review, builds the verb-specific prompt + a
raw_dispatch OVERHEAD spec, and enqueues it as an admitted pump task with a
durable continuation (on_complete -> mechanical sink). NO await_inline.
The prompts live HERE (outside mr_roboto), never in the mechanical verb.
"""
from __future__ import annotations

import time
import uuid

from general_beckman import enqueue  # module-level for test patching
from general_beckman.lanes import LANE_ONESHOT
from src.infra.logging_config import get_logger

logger = get_logger("reviews.producers")

_CLASSIFY_PROMPT = (
    "You are classifying a product review. Return JSON only.\n\n"
    "Rating: {rating}/5\n"
    "Review: {body}\n\n"
    'Respond ONLY with: {{"sentiment": "<positive|negative|neutral>", '
    '"theme_tag": "<UX|pricing|bug|feature-request|support|generic-positive|generic-negative>"}}\n'
    "Pick the single most relevant theme_tag."
)

# Platform-specific reply conventions (moved out of mr_roboto's draft verb).
_PLATFORM_CONVENTIONS = {
    "g2": "G2 replies are read by B2B buyers. Be professional, specific, acknowledge feedback. Offer to connect privately for issues.",
    "appstore": "AppStore replies are brief (<=1000 chars). Be warm, thank the user; for negatives offer a specific resolution path (email/support).",
    "playstore": "PlayStore replies are brief. Acknowledge feedback, be empathetic, direct to support for technical issues.",
    "producthunt": "ProductHunt replies are community-facing. Authentic, founder-to-user; for criticism acknowledge openly and describe what's next.",
    "trustpilot": "Trustpilot replies are public and formal. Use company name, address the complaint, offer a resolution path.",
    "capterra": "Capterra is B2B. Professional, acknowledge the point, highlight roadmap/workarounds for feature requests.",
}
_DEFAULT_CONVENTION = (
    "Be polite, genuine, specific to the review. For negatives acknowledge the issue "
    "and offer a resolution path; for positives thank the reviewer warmly."
)


async def _load_review(review_id: int):
    """Return (review_id, platform, author, rating, body_md) or None."""
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT review_id, platform, author, rating, body_md "
        "FROM external_reviews WHERE review_id=?",
        (review_id,),
    )
    return await cur.fetchone()


def _suffix() -> str:
    return f"{time.monotonic_ns() % 1_000_000:06d}-{uuid.uuid4().hex[:6]}"


async def enqueue_classify(*, review_id: int, product_id: str) -> int | None:
    """Enqueue an admitted classify producer; the reviews.classify.resume
    continuation does the mechanical persist + side-effects. Returns task id."""
    row = await _load_review(review_id)
    if row is None:
        logger.warning("enqueue_classify: review_id=%s not found", review_id)
        return None
    _, platform, author, rating, body_md = row
    rating = int(rating or 0)
    body_md = body_md or ""

    prompt = _CLASSIFY_PROMPT.format(rating=rating, body=body_md[:800])
    spec = {
        "title": f"reviews_classify:llm:{_suffix()}",
        "description": "Classify review sentiment + theme.",
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 2,
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "reviewer",
            "agent_type": "reviewer",
            "difficulty": 3,
            "messages": [{"role": "user", "content": prompt}],
            "failures": [],
            "estimated_input_tokens": 250,
            "estimated_output_tokens": 50,
        }},
    }
    return await enqueue(
        spec,
        lane=LANE_ONESHOT,
        on_complete="reviews.classify.resume",
        on_error="reviews.classify.resume_err",
        cont_state={
            "review_id": review_id, "product_id": product_id,
            "platform": platform, "author": author or "Unknown",
            "rating": rating, "body_md": body_md,
        },
    )


async def enqueue_draft_reply(*, review_id: int, product_id: str) -> int | None:
    """Enqueue an admitted draft-reply producer; the reviews.draft_reply.resume
    continuation surfaces the draft via a founder_action (NEVER auto-posts)."""
    row = await _load_review(review_id)
    if row is None:
        logger.warning("enqueue_draft_reply: review_id=%s not found", review_id)
        return None
    _, platform, author, rating, body_md = row
    rating = int(rating or 3)
    author = author or "Anonymous"
    body_md = body_md or ""
    convention = _PLATFORM_CONVENTIONS.get(platform, _DEFAULT_CONVENTION)
    star = f"{rating}/5 star{'s' if rating != 1 else ''}"
    prompt = (
        f"You are drafting a reply to a {star} review on {platform}.\n"
        f"Reviewer: {author}\nReview content: {body_md[:600]}\n\n"
        f"Platform conventions:\n{convention}\n\n"
        "Write a reply in first person from the product founder's perspective.\n"
        "Rules:\n- No promises about specific features/timelines.\n"
        "- No refunds unless the review mentions billing.\n"
        "- Concise: 2-4 sentences positive, 3-6 negative.\n"
        "- Do NOT start with 'Hi,' or 'Dear,'\nDraft the reply only."
    )
    spec = {
        "title": f"reviews_draft_reply:llm:{_suffix()}",
        "description": f"Draft reply for {platform} review.",
        "agent_type": "reviewer", "kind": "overhead", "priority": 2,
        "context": {"llm_call": {
            "raw_dispatch": True, "call_category": "overhead",
            "task": "reviewer", "agent_type": "reviewer", "difficulty": 3,
            "messages": [{"role": "user", "content": prompt}], "failures": [],
            "estimated_input_tokens": 350, "estimated_output_tokens": 150,
        }},
    }
    return await enqueue(
        spec, lane=LANE_ONESHOT,
        on_complete="reviews.draft_reply.resume",
        on_error="reviews.draft_reply.resume_err",
        cont_state={"review_id": review_id, "product_id": product_id,
                    "platform": platform, "author": author, "rating": rating},
    )
