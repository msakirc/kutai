"""Z7 T2A — Resend free-tier adapter.

Free tier: 3,000 emails/month, 100/day.
API docs: https://resend.com/docs/api-reference/emails/send-email

Resend webhook event types:
    email.sent, email.delivered, email.delivery_delayed,
    email.complained, email.bounced, email.opened, email.clicked
"""
from __future__ import annotations

import json
from typing import Any

import aiohttp

from src.infra.logging_config import get_logger
from src.integrations.email.base import EmailProvider

logger = get_logger("integrations.email.resend")

_RESEND_API_URL = "https://api.resend.com/emails"

_BOUNCE_TYPES = frozenset({"email.bounced"})
_COMPLAINT_TYPES = frozenset({"email.complained"})
_SUPPRESS_TYPES = frozenset({"email.bounced", "email.complained"})


class ResendProvider(EmailProvider):
    """Resend transactional email adapter."""

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
        """Send a transactional email via Resend API."""
        from_email = f"noreply@{self.from_domain}"
        payload: dict[str, Any] = {
            "from": from_email,
            "to": [to],
            "subject": subject,
            "text": body_md,
        }
        if headers:
            payload["headers"] = headers

        request_headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if idempotency_key:
            request_headers["Idempotency-Key"] = idempotency_key

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    _RESEND_API_URL,
                    headers=request_headers,
                    data=json.dumps(payload),
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    body = await resp.json(content_type=None)
                    if resp.status in (200, 201):
                        msg_id = body.get("id")
                        logger.info(
                            "resend send OK",
                            to=to,
                            subject=subject,
                            message_id=msg_id,
                        )
                        return {
                            "status": "sent",
                            "provider": "resend",
                            "message_id": msg_id,
                        }
                    else:
                        logger.error(
                            "resend send failed",
                            status=resp.status,
                            body=body,
                            to=to,
                        )
                        return {
                            "status": "error",
                            "provider": "resend",
                            "message_id": None,
                            "error": str(body),
                        }
        except Exception as exc:
            logger.error("resend send exception", exc=str(exc), to=to)
            return {
                "status": "error",
                "provider": "resend",
                "message_id": None,
                "error": str(exc),
            }

    def parse_webhook_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalise a Resend webhook payload to the canonical event dict."""
        event_type_raw = payload.get("type", "")
        data = payload.get("data", {}) or {}

        # Recipient: "to" is a list in Resend payloads
        to_field = data.get("to", [])
        if isinstance(to_field, list):
            recipient = to_field[0] if to_field else ""
        else:
            recipient = str(to_field)

        message_id = data.get("email_id") or data.get("id")

        if event_type_raw == "email.opened":
            event_type = "open"
            should_suppress = False
        elif event_type_raw == "email.clicked":
            event_type = "click"
            should_suppress = False
        elif event_type_raw in _BOUNCE_TYPES:
            event_type = "bounce"
            should_suppress = True
        elif event_type_raw in _COMPLAINT_TYPES:
            event_type = "complaint"
            should_suppress = True
        elif event_type_raw == "email.delivered":
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
