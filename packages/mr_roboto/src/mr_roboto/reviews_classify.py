"""Z7 T5 B8 — reviews/classify mechanical helpers.

SP4b: the LLM classifier was extracted out of mr_roboto. The classify LLM
hop is now an admitted Beckman producer task (``src.reviews.producers.
enqueue_classify``) whose ``reviews.classify.resume`` continuation
(``mr_roboto.executors.reviews_continuations``) does the mechanical work
below. This module retains ONLY the mechanical pieces the sink reuses:
parsing, heuristic fallback, enum constants, and the two side-effect
emitters. NO LLM call lives here.

Assigns:
  sentiment : 'positive' | 'negative' | 'neutral'
  theme_tag : 'UX' | 'pricing' | 'bug' | 'feature-request' |
              'support' | 'generic-positive' | 'generic-negative'

Side-effects (routed by the sink after classification):
  1-2-star  → _emit_low_star_founder_action (review + decide reply)
  bug-tagged → _enqueue_bug_investigation (investigation task in backlog)
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.reviews_classify")

VALID_SENTIMENTS = frozenset({"positive", "negative", "neutral"})
VALID_THEMES = frozenset({
    "UX", "pricing", "bug", "feature-request",
    "support", "generic-positive", "generic-negative",
})

# Ratings at or below this threshold trigger a low-star founder_action.
LOW_STAR_THRESHOLD = 2


# ---------------------------------------------------------------------------
# LLM response parsing + heuristic fallback (consumed by the CPS sink)
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str, body_md: str, rating: int) -> dict:
    """Parse LLM JSON response with heuristic fallback."""
    import json as _json
    import re

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
    """Simple heuristic fallback when the LLM is unavailable."""
    body_lower = body_md.lower()
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

    if rating >= 4:
        sentiment = "positive"
    elif rating <= 2:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"sentiment": sentiment, "theme_tag": theme_tag}


# ---------------------------------------------------------------------------
# Side-effect helpers (mocked in tests; called by the CPS sink)
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
    """Enqueue a bug investigation task in the mission backlog via beckman.

    Intentionally NOT await_inline — this is a fire-and-forget backlog task
    that should run asynchronously, not block anything.
    """
    try:
        from general_beckman import enqueue
        from general_beckman.lanes import LANE_ONESHOT
        return await enqueue(spec, lane=LANE_ONESHOT, **kwargs)
    except Exception as exc:
        logger.warning("reviews_classify._enqueue_bug_investigation failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Deprecated direct entry — delegates to the CPS producer
# ---------------------------------------------------------------------------

async def run(payload: dict) -> dict:
    """DEPRECATED direct entry. Enqueues the CPS classify producer and returns
    its task id. Kept for legacy callers; new callers use the router action
    (reviews/classify) or the daily cron, both of which drive the producer."""
    review_id_raw = payload.get("review_id")
    product_id = str(payload.get("product_id") or "")
    if review_id_raw is None:
        return {"status": "error", "error": "review_id is required"}
    if not product_id:
        return {"status": "error", "error": "product_id is required"}
    from src.reviews.producers import enqueue_classify
    tid = await enqueue_classify(review_id=int(review_id_raw), product_id=product_id)
    if tid is None:
        return {"status": "error", "error": f"review_id={review_id_raw} not found"}
    return {"status": "ok", "enqueued": tid}


__all__ = [
    "run",
    "_heuristic_classify",
    "_parse_llm_response",
    "_emit_low_star_founder_action",
    "_enqueue_bug_investigation",
    "VALID_SENTIMENTS",
    "VALID_THEMES",
    "LOW_STAR_THRESHOLD",
]
