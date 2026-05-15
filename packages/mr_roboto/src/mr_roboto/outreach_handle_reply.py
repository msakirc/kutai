"""Z7 T6 A7.r1 — outreach/handle_reply: ESP reply webhook handler.

When the ESP fires a reply-event matched by Reply-To + X-Send-ID header,
this verb:

  1. Looks up the outreach_sends row by send_id.
  2. Classifies the reply: positive_interest | negative | unsubscribe_request |
     out_of_office | bounce | question.
  3. If unsubscribe_request → INSERT OR IGNORE into email_suppression.
  4. Logs the interaction to the CRM via crm.log_interaction (interactions table).
  5. If positive_interest → surface a beckman enqueue for a follow-up draft.
  6. Updates outreach_sends.replied_at.
  7. Returns a classification dict.

Public API
----------
  run_outreach_handle_reply(
      product_id, send_id, reply_body, reply_from
  ) -> dict

Internal hooks (patched in tests)
----------------------------------
  _classify_reply(reply_body: str) -> str
  _log_to_crm(product_id, contact_handle, summary, mission_id) -> int
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.outreach_handle_reply")

_VALID_CLASSIFICATIONS = frozenset({
    "positive_interest",
    "negative",
    "unsubscribe_request",
    "out_of_office",
    "bounce",
    "question",
})

# Heuristic keyword mappings for the stub classifier
_UNSUB_KEYWORDS = frozenset({
    "unsubscribe", "remove me", "remove from", "opt out", "opt-out",
    "stop emailing", "take me off", "do not contact", "stop contact",
})
_OOO_KEYWORDS = frozenset({
    "out of office", "on vacation", "away from", "will return", "auto-reply",
    "automatic reply",
})
_POSITIVE_KEYWORDS = frozenset({
    "interested", "tell me more", "schedule a call", "let's talk", "sounds good",
    "would like to", "how does this", "pricing", "demo", "meeting",
})


async def _classify_reply(reply_body: str) -> str:
    """Heuristic/stub reply classifier.

    In production this would call an LLM via beckman.enqueue; for now the
    stub uses keyword matching so the mechanical path works without LLM.
    Tests patch this with an AsyncMock.
    """
    body_lower = reply_body.lower()

    for kw in _UNSUB_KEYWORDS:
        if kw in body_lower:
            return "unsubscribe_request"

    for kw in _OOO_KEYWORDS:
        if kw in body_lower:
            return "out_of_office"

    for kw in _POSITIVE_KEYWORDS:
        if kw in body_lower:
            return "positive_interest"

    return "question"


async def _log_to_crm(
    product_id: str,
    contact_handle: str,
    summary: str,
    mission_id: int | None = None,
) -> int:
    """Log interaction via CRM module; returns interaction_id."""
    try:
        from src.app.crm import add_contact, get_contact_by_handle, log_interaction

        # Ensure contact exists
        contact = await get_contact_by_handle(product_id, contact_handle)
        if contact is None:
            contact_id = await add_contact(
                product_id=product_id,
                handle=contact_handle,
                display_name=contact_handle,
                category="prospect",
            )
        else:
            contact_id = contact["contact_id"]

        return await log_interaction(
            product_id=product_id,
            contact_id=contact_id,
            kind="email",
            summary=summary,
            mission_id=mission_id,
        )
    except Exception as exc:
        logger.warning("_log_to_crm failed", error=str(exc))
        return -1


async def run_outreach_handle_reply(
    product_id: str,
    send_id: int,
    reply_body: str,
    reply_from: str,
    *,
    mission_id: int | None = None,
) -> dict[str, Any]:
    """Handle an inbound reply to a cold outreach email.

    Returns a dict with at minimum:
      {"status": "ok", "classification": <str>, "send_id": <int>}
    Or:
      {"status": "error", "error": <str>}
    """
    from src.infra.db import get_db
    db = await get_db()

    # Verify send_id exists and belongs to product_id
    cur = await db.execute(
        "SELECT target_email, replied_at FROM outreach_sends "
        "WHERE send_id=? AND product_id=?",
        (send_id, product_id),
    )
    send_row = await cur.fetchone()
    if send_row is None:
        logger.warning(
            "outreach_handle_reply: send_id not found",
            send_id=send_id,
            product_id=product_id,
        )
        return {
            "status": "error",
            "error": f"send_id {send_id} not found for product {product_id}",
        }

    # Classify
    classification = await _classify_reply(reply_body)

    suppressed = False

    # Auto-suppress on unsubscribe
    if classification == "unsubscribe_request":
        await db.execute(
            "INSERT OR IGNORE INTO email_suppression "
            "(product_id, email, reason) VALUES (?, ?, ?)",
            (product_id, reply_from, "unsub"),
        )
        suppressed = True

    # Update replied_at on send row
    await db.execute(
        "UPDATE outreach_sends SET replied_at=datetime('now') WHERE send_id=?",
        (send_id,),
    )
    await db.commit()

    # Log to CRM interactions
    summary = (
        f"Outreach reply (send_id={send_id}): [{classification}] "
        f"{reply_body[:200]}"
    )
    await _log_to_crm(
        product_id=product_id,
        contact_handle=reply_from,
        summary=summary,
        mission_id=mission_id,
    )

    logger.info(
        "outreach_handle_reply: classified",
        product_id=product_id,
        send_id=send_id,
        classification=classification,
        suppressed=suppressed,
    )

    result: dict[str, Any] = {
        "status": "ok",
        "classification": classification,
        "send_id": send_id,
        "suppressed": suppressed,
    }

    # For positive_interest: note that a follow-up draft should be queued
    # (actual LLM enqueue deferred to caller / workflow step)
    if classification == "positive_interest":
        result["follow_up_needed"] = True

    return result
