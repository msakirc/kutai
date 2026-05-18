"""Z8 T3B — per-vendor webhook signature verification.

Each verifier returns a bool synchronously; ``verify_signature`` (async) is
the public entry — it loads the per-integration secret from
``IntegrationRegistry`` config when not passed explicitly. Header keys are
expected lower-cased (the listener normalises them in ``webhook_inbound``).

Vendors covered v1:
  * sentry        — HMAC-SHA256 of raw body, header ``sentry-hook-signature``
  * stripe        — ``stripe-signature`` ``t=...,v1=...`` (5-min replay window)
  * github        — ``x-hub-signature-256`` ``sha256=<hex>``
  * betterstack   — HMAC-SHA256, header ``x-betterstack-signature``
  * twilio        — ``x-twilio-signature`` (sha1+base64 over URL+params,
                    listener must inject ``x-twilio-url`` header before call)

Unknown integrations are rejected (return False with a logged warning) so a
mis-routed POST cannot bypass verification by simply omitting a header.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
from typing import Callable

logger = logging.getLogger("kutai.webhook.signing")


async def verify_signature(
    integration_id: str,
    raw: bytes,
    headers: dict,
    secret: str | None = None,
) -> bool:
    """Dispatch to the right verifier; load secret from registry if absent."""
    if secret is None:
        secret = await _load_webhook_secret(integration_id)
        if not secret:
            logger.warning(
                "no webhook_secret for %s — rejecting", integration_id,
            )
            return False

    verifiers: dict[str, Callable[[bytes, dict, str], bool]] = {
        "sentry": _verify_sentry,
        "stripe": _verify_stripe,
        "github": _verify_github,
        "betterstack": _verify_betterstack,
        "twilio": _verify_twilio,
        # Z9 T3A — growth signal intake.
        "intercom": _verify_intercom,
        "zendesk": _verify_zendesk,
        "posthog": _verify_posthog,
    }
    fn = verifiers.get(integration_id)
    if fn is None:
        logger.warning(
            "no verifier registered for %s — rejecting", integration_id,
        )
        return False
    try:
        return fn(raw, headers, secret)
    except Exception as exc:  # defensive — never let a verifier raise
        logger.warning(
            "verifier crash for %s: %s — rejecting", integration_id, exc,
        )
        return False


# ---------------------------------------------------------------------------
# Per-vendor verifiers (sync; headers expected lower-cased by listener)
# ---------------------------------------------------------------------------


def _verify_sentry(raw: bytes, headers: dict, secret: str) -> bool:
    sig = headers.get("sentry-hook-signature", "")
    if not sig:
        return False
    expected = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def _verify_stripe(raw: bytes, headers: dict, secret: str) -> bool:
    header = headers.get("stripe-signature", "")
    if not header:
        return False
    parts: dict[str, str] = {}
    for kv in header.split(","):
        if "=" in kv:
            k, _, v = kv.partition("=")
            parts[k.strip()] = v.strip()
    t = parts.get("t", "")
    v1 = parts.get("v1", "")
    if not t or not v1:
        return False
    try:
        ts = int(t)
    except ValueError:
        return False
    # 5-minute replay window — Stripe's recommended tolerance.
    if abs(time.time() - ts) > 300:
        return False
    signed_payload = (str(ts) + ".").encode() + raw
    expected = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(v1, expected)


def _verify_github(raw: bytes, headers: dict, secret: str) -> bool:
    sig = headers.get("x-hub-signature-256", "")
    if not sig.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), raw, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(sig, expected)


def _verify_betterstack(raw: bytes, headers: dict, secret: str) -> bool:
    sig = headers.get("x-betterstack-signature", "")
    if not sig:
        return False
    expected = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def _verify_twilio(raw: bytes, headers: dict, secret: str) -> bool:
    """Twilio signs (URL + sorted form-param concatenation).

    For JSON-mode webhooks Twilio still signs URL + raw body; the listener
    is responsible for injecting ``x-twilio-url`` with the absolute URL the
    request hit.
    """
    sig = headers.get("x-twilio-signature", "")
    url = headers.get("x-twilio-url", "")
    if not sig or not url:
        return False
    expected = hmac.new(
        secret.encode(), (url + raw.decode("utf-8", "replace")).encode(),
        hashlib.sha1,
    ).digest()
    expected_b64 = base64.b64encode(expected).decode()
    return hmac.compare_digest(sig, expected_b64)


# ---------------------------------------------------------------------------
# Z9 T3A — growth signal intake verifiers
# ---------------------------------------------------------------------------


def _verify_intercom(raw: bytes, headers: dict, secret: str) -> bool:
    """Intercom signs the raw body with HMAC-SHA1, header ``x-hub-signature``.

    The header value is ``sha1=<hex>`` (mirrors GitHub's older scheme).
    """
    sig = headers.get("x-hub-signature", "")
    if not sig.startswith("sha1="):
        return False
    expected = "sha1=" + hmac.new(
        secret.encode(), raw, hashlib.sha1
    ).hexdigest()
    return hmac.compare_digest(sig, expected)


def _verify_zendesk(raw: bytes, headers: dict, secret: str) -> bool:
    """Zendesk signs ``timestamp + raw_body`` with HMAC-SHA256, base64.

    Headers: ``x-zendesk-webhook-signature`` (base64 digest) and
    ``x-zendesk-webhook-signature-timestamp``.
    """
    sig = headers.get("x-zendesk-webhook-signature", "")
    ts = headers.get("x-zendesk-webhook-signature-timestamp", "")
    if not sig or not ts:
        return False
    signed_payload = ts.encode() + raw
    expected = base64.b64encode(
        hmac.new(secret.encode(), signed_payload, hashlib.sha256).digest()
    ).decode()
    return hmac.compare_digest(sig, expected)


def _verify_posthog(raw: bytes, headers: dict, secret: str) -> bool:
    """PostHog webhook delivery — HMAC-SHA256 of the raw body.

    PostHog's hook system does not ship a first-party signing header, so the
    intake convention is a configured shared secret echoed in
    ``x-posthog-signature`` as ``sha256=<hex>`` (set when registering the
    webhook). This keeps verification mandatory and uniform with the other
    providers rather than accepting unsigned posts.
    """
    sig = headers.get("x-posthog-signature", "")
    if not sig.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), raw, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(sig, expected)


# ---------------------------------------------------------------------------
# Secret resolution from IntegrationRegistry
# ---------------------------------------------------------------------------


async def _load_webhook_secret(integration_id: str) -> str | None:
    """Read ``webhook_secret`` from the integration config (T3C wires this)."""
    try:
        from src.integrations.registry import get_integration_registry
    except Exception as exc:
        logger.debug("IntegrationRegistry import failed: %s", exc)
        return None
    reg = get_integration_registry()
    integ = reg.get(integration_id)
    if integ is None:
        return None
    # HttpIntegration stores raw config under ``_config``; access defensively.
    config = getattr(integ, "_config", None) or {}
    secret = config.get("webhook_secret")
    if not secret:
        return None
    # Resolve ``${ENV_VAR}`` placeholders so configs can env-ref secrets.
    return _resolve_env_ref(secret)


def _resolve_env_ref(value: str) -> str | None:
    """If value looks like ``${VAR}``, return os.environ[VAR] (or None).

    Otherwise return the literal value (lets configs hard-code in tests)."""
    import os
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_key = value[2:-1]
        return os.environ.get(env_key)
    return value
