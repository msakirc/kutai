"""Z8 T3A — placeholder; per-vendor verifiers land in T3B.

For T3A the listener imports ``verify_signature``; tests monkeypatch it.
T3B replaces this body with sentry/stripe/github/betterstack/twilio HMAC
implementations.
"""
from __future__ import annotations


async def verify_signature(
    integration_id: str,
    raw: bytes,
    headers: dict,
    secret: str | None = None,
) -> bool:
    """T3A stub. T3B overwrites with real verifiers."""
    return False
