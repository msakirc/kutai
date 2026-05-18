"""Z7 T2A — Postmark paid-tier adapter stub.

Postmark is a paid service (no free tier beyond 100 test emails).
Per the Z7 T2A founder decision: no paid email until a product earns revenue.

When a product flips its tier to 'paid' and selects 'postmark' as provider,
this adapter will be used.  Until a real implementation is warranted, every
method raises NotImplementedError with an instructive message.

To wire a real implementation:
  1. Set product_email_config.tier = 'paid' and provider = 'postmark'.
  2. Store a credential via /credential add postmark_<product_id> {"api_key": "..."}
  3. Replace the body of send() with the Postmark Messages API call.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger
from src.integrations.email.base import EmailProvider

logger = get_logger("integrations.email.postmark")


class PostmarkProvider(EmailProvider):
    """Postmark paid-tier adapter — stub until revenue justifies it."""

    def __init__(self, api_key: str, from_domain: str) -> None:
        super().__init__(api_key, from_domain)

    async def send(
        self,
        to: str,
        subject: str,
        body_md: str,
        headers: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "paid tier — wire when revenue justifies. "
            "See src/integrations/email/providers/postmark.py for instructions."
        )

    def parse_webhook_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Basic Postmark webhook normalisation (enough for future wiring)."""
        record_type = payload.get("RecordType", "")
        recipient = (
            payload.get("Email")
            or payload.get("Recipient")
            or ""
        )
        message_id = payload.get("MessageID")

        if record_type == "Open":
            event_type = "open"
            should_suppress = False
        elif record_type == "Click":
            event_type = "click"
            should_suppress = False
        elif record_type == "Bounce":
            bounce_type = payload.get("Type", "")
            event_type = "bounce"
            should_suppress = bounce_type == "HardBounce"
        elif record_type == "SpamComplaint":
            event_type = "complaint"
            should_suppress = True
        elif record_type == "SubscriptionChange":
            event_type = "unsub"
            should_suppress = True
        elif record_type == "Delivery":
            event_type = "delivery"
            should_suppress = False
        else:
            event_type = record_type.lower() or "unknown"
            should_suppress = False

        return {
            "event_type": event_type,
            "recipient": recipient,
            "message_id": message_id,
            "should_suppress": should_suppress,
            "raw": payload,
        }
