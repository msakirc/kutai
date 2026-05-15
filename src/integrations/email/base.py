"""Z7 T2A — EmailProvider ABC.

Every email provider adapter implements this interface.  The ABC enforces:
  - async send() → returns a result dict with at least {"status": ..., "provider": ...}
  - parse_webhook_event() → normalises vendor-specific webhook payloads
    into a canonical dict used by service.handle_webhook_event().

Webhook event canonical shape:
    {
        "event_type":    "open" | "click" | "bounce" | "unsub" | "complaint" | "delivery",
        "recipient":     str,                   # email address
        "message_id":    str | None,
        "should_suppress": bool,               # True for bounce/complaint/unsub
        "raw":           dict,                 # original payload
    }
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EmailProvider(ABC):
    """Base class for all email provider adapters."""

    def __init__(self, api_key: str, from_domain: str) -> None:
        self.api_key = api_key
        self.from_domain = from_domain

    @abstractmethod
    async def send(
        self,
        to: str,
        subject: str,
        body_md: str,
        headers: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Send an email.

        Returns a dict with at minimum:
            {"status": "sent"|"error", "provider": str, "message_id": str|None}
        """

    @abstractmethod
    def parse_webhook_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Parse a provider-specific webhook payload into the canonical shape.

        Returns the canonical webhook event dict (see module docstring).
        """
