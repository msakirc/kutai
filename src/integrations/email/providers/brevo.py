"""Z7 T2A — Brevo (formerly Sendinblue) free-tier adapter.

Free tier: 300 emails/day, forever free.
API docs: https://developers.brevo.com/reference/sendtransacemail

Brevo webhook event types (v3 API):
    delivered, soft_bounce, hard_bounce, invalid_email,
    deferred, spam, opened, clicks, unsubscribed,
    complaint, blocked, error
"""
from __future__ import annotations

import json
import os
from typing import Any

import aiohttp

from src.infra.logging_config import get_logger
from src.integrations.email.base import EmailProvider

logger = get_logger("integrations.email.brevo")

_BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"

# Webhook event types that map to "bounce"
_BOUNCE_EVENTS = frozenset({"hard_bounce", "soft_bounce", "invalid_email"})
# Webhook events that require suppression
_SUPPRESS_EVENTS = frozenset({"hard_bounce", "invalid_email", "complaint", "unsubscribed"})


class BrevoProvider(EmailProvider):
    """Brevo (Sendinblue) transactional email adapter."""

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
        """Send a transactional email via Brevo API.

        body_md is used as both the text/plain content and a simple HTML body
        (Brevo requires at least one of htmlContent / textContent).
        """
        from_email = f"noreply@{self.from_domain}"
        payload: dict[str, Any] = {
            "sender": {"email": from_email},
            "to": [{"email": to}],
            "subject": subject,
            "textContent": body_md,
            "htmlContent": f"<pre>{body_md}</pre>",
        }
        if headers:
            payload["headers"] = headers

        request_headers: dict[str, str] = {
            "accept": "application/json",
            "content-type": "application/json",
            "api-key": self.api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    _BREVO_API_URL,
                    headers=request_headers,
                    data=json.dumps(payload),
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    body = await resp.json(content_type=None)
                    if resp.status in (200, 201):
                        msg_id = body.get("messageId")
                        logger.info(
                            "brevo send OK",
                            to=to,
                            subject=subject,
                            message_id=msg_id,
                        )
                        return {
                            "status": "sent",
                            "provider": "brevo",
                            "message_id": msg_id,
                        }
                    else:
                        logger.error(
                            "brevo send failed",
                            status=resp.status,
                            body=body,
                            to=to,
                        )
                        return {
                            "status": "error",
                            "provider": "brevo",
                            "message_id": None,
                            "error": str(body),
                        }
        except Exception as exc:
            logger.error("brevo send exception", exc=str(exc), to=to)
            return {
                "status": "error",
                "provider": "brevo",
                "message_id": None,
                "error": str(exc),
            }

    def parse_webhook_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalise a Brevo webhook payload to the canonical event dict."""
        event_type_raw = payload.get("event", "")
        recipient = payload.get("email", "")
        message_id = payload.get("messageId")

        if event_type_raw == "opened":
            event_type = "open"
            should_suppress = False
        elif event_type_raw == "clicks":
            event_type = "click"
            should_suppress = False
        elif event_type_raw == "unsubscribed":
            event_type = "unsub"
            should_suppress = True
        elif event_type_raw == "complaint" or event_type_raw == "spam":
            event_type = "complaint"
            should_suppress = True
        elif event_type_raw in _BOUNCE_EVENTS:
            event_type = "bounce"
            should_suppress = event_type_raw in _SUPPRESS_EVENTS
        elif event_type_raw == "delivered":
            event_type = "delivery"
            should_suppress = False
        else:
            event_type = event_type_raw or "unknown"
            should_suppress = False

        return {
            "event_type": event_type,
            "recipient": recipient,
            "message_id": message_id,
            "should_suppress": should_suppress,
            "raw": payload,
        }
