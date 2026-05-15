"""Z7 T2A — AWS SES paid-tier adapter stub.

AWS SES requires an AWS account, domain verification, and billing setup.
Per the Z7 T2A founder decision: no paid email until a product earns revenue.

SES is the cheapest paid option at $0.10/1000 emails but requires AWS setup
overhead that is only worth it at scale.

To wire a real implementation:
  1. Set product_email_config.tier = 'paid' and provider = 'ses'.
  2. Store AWS credentials via /credential add ses_<product_id>
     {"access_key_id": "...", "secret_access_key": "...", "region": "us-east-1"}
  3. Install boto3 (add to requirements.txt) and implement send() using
     boto3.client('ses').send_email() or the SES v2 API.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger
from src.integrations.email.base import EmailProvider

logger = get_logger("integrations.email.ses")


class SESProvider(EmailProvider):
    """AWS SES paid-tier adapter — stub until revenue justifies it."""

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
            "See src/integrations/email/providers/ses.py for instructions."
        )

    def parse_webhook_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Basic SES SNS notification normalisation.

        SES delivers events via SNS -> HTTP endpoint. The SNS wrapper contains
        a JSON string in the 'Message' field; the inner message has a 'notificationType'
        field ('Bounce', 'Complaint', 'Delivery').
        """
        import json

        # Unwrap SNS envelope if present
        if "Message" in payload and isinstance(payload["Message"], str):
            try:
                inner = json.loads(payload["Message"])
            except (ValueError, TypeError):
                inner = payload
        else:
            inner = payload

        notification_type = inner.get("notificationType", "")
        mail = inner.get("mail", {}) or {}
        destination = mail.get("destination", [])
        recipient = destination[0] if destination else ""
        message_id = mail.get("messageId")

        if notification_type == "Delivery":
            event_type = "delivery"
            should_suppress = False
        elif notification_type == "Bounce":
            bounce = inner.get("bounce", {}) or {}
            bounce_type = bounce.get("bounceType", "")
            event_type = "bounce"
            should_suppress = bounce_type == "Permanent"
        elif notification_type == "Complaint":
            event_type = "complaint"
            should_suppress = True
        else:
            event_type = notification_type.lower() or "unknown"
            should_suppress = False

        return {
            "event_type": event_type,
            "recipient": recipient,
            "message_id": message_id,
            "should_suppress": should_suppress,
            "raw": payload,
        }
