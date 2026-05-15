"""analytics_instrumentation/v1 — posthog-python server shim (backend).

A thin ``track_event(name, properties)`` helper over the posthog Python SDK,
mirroring the web client shim. Use it for server-side events that the browser
cannot observe reliably: ``checkout_completed`` (webhook-confirmed),
``subscription_created`` / ``subscription_cancelled`` (Stripe webhook),
``signup_completed`` (after the DB row commits), etc.

RECIPE_PARAM markers:
  # RECIPE_PARAM:POSTHOG_API_KEY_ENV=POSTHOG_API_KEY
  # RECIPE_PARAM:POSTHOG_HOST_ENV=POSTHOG_HOST
  # RECIPE_PARAM:POSTHOG_HOST_DEFAULT=https://us.i.posthog.com

Leave the RECIPE_PARAM comment markers intact — they must survive an
ast.parse() call so the syntax checker passes before substitution.

Env contract (read at init):
  POSTHOG_API_KEY  — PostHog project API key
  POSTHOG_HOST     — PostHog ingestion host

Usage:
    from analytics import init_analytics, track_event, set_analytics_context

    init_analytics()
    set_analytics_context(mission_id="m-42", feature_id="checkout",
                          business_model="b2c")
    track_event("checkout_completed", distinct_id=user.id,
                properties={"plan": "pro", "amount": 19, "currency": "USD"})
"""
from __future__ import annotations

import os
from typing import Any, Optional

try:
    import posthog as _posthog  # type: ignore[import]
except ImportError:  # pragma: no cover - dependency declared in recipe.yaml
    _posthog = None  # type: ignore[assignment]


# Metadata auto-attached to every event. Set once via set_analytics_context.
_context: dict[str, Any] = {
    "mission_id": "unknown",
    "feature_id": "unknown",
}
_initialized = False

# Z9 T5A — optional cohort gate. When the analytics_instrumentation recipe
# declares `segment_predicate`, the recipe applier substitutes it here. When
# set, track_event tags every event with the active segment AND only emits
# for users whose segment matches the predicate. None -> no gate.
# RECIPE_PARAM:SEGMENT_PREDICATE=null
SEGMENT_PREDICATE: Optional[str] = None


def _segment_matches() -> bool:
    """True when the active segment satisfies SEGMENT_PREDICATE.

    A None predicate always matches (no cohort gate). The predicate is
    matched literally against ``_context['segment']``; recipe appliers may
    replace this body with a richer expression for custom cohorts.
    """
    if SEGMENT_PREDICATE is None:
        return True
    return _context.get("segment") == SEGMENT_PREDICATE


def set_analytics_context(
    *,
    mission_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    variant: Optional[str] = None,
    segment: Optional[str] = None,
    business_model: Optional[str] = None,
    account_id: Optional[str] = None,
) -> None:
    """Populate the metadata block attached to every subsequent track_event.

    ``account_id`` (Z9 T5B) is the B2B account/org identifier — attached to
    every event alongside ``distinct_id`` when ``business_model == 'b2b'``.
    """
    if mission_id is not None:
        _context["mission_id"] = mission_id
    if feature_id is not None:
        _context["feature_id"] = feature_id
    if variant is not None:
        _context["variant"] = variant
    if segment is not None:
        _context["segment"] = segment
    if business_model is not None:
        _context["business_model"] = business_model
    if account_id is not None:
        _context["account_id"] = account_id


def init_analytics() -> None:
    """Initialize the PostHog SDK from env. Idempotent.

    No-ops with a warning when the API key is absent so dev/test runs do not
    crash; in that mode track_event silently drops events.
    """
    global _initialized
    if _initialized:
        return
    if _posthog is None:
        return
    # RECIPE_PARAM:POSTHOG_API_KEY_ENV=POSTHOG_API_KEY
    api_key = os.environ.get("POSTHOG_API_KEY")
    # RECIPE_PARAM:POSTHOG_HOST_ENV=POSTHOG_HOST
    host = os.environ.get(
        "POSTHOG_HOST",
        "https://us.i.posthog.com",  # RECIPE_PARAM:POSTHOG_HOST_DEFAULT=https://us.i.posthog.com
    )
    if not api_key:
        import logging

        logging.getLogger("analytics").warning(
            "POSTHOG_API_KEY not set - track_event is a no-op."
        )
        return
    _posthog.api_key = api_key
    _posthog.host = host
    _initialized = True


def track_event(
    name: str,
    *,
    distinct_id: str,
    properties: Optional[dict[str, Any]] = None,
) -> None:
    """Emit a standardized analytics event server-side.

    Parameters
    ----------
    name:
        One of the standard AARRR taxonomy event names (see events template).
    distinct_id:
        Stable user identifier (the same id used by the web client shim).
    properties:
        Event-specific properties.

    Auto-attaches: mission_id, feature_id, variant, segment, business_model.
    For business_model "b2b" it additionally attaches account_id.

    Z9 T5A — when SEGMENT_PREDICATE is set, the event is dropped for users
    whose segment does not match (cohort gate).
    """
    business_model = _context.get("business_model", "b2c")
    enriched: dict[str, Any] = dict(properties or {})
    enriched["mission_id"] = _context.get("mission_id", "unknown")
    enriched["feature_id"] = _context.get("feature_id", "unknown")
    enriched["business_model"] = business_model
    if "variant" in _context:
        enriched["variant"] = _context["variant"]
    if "segment" in _context:
        enriched["segment"] = _context["segment"]
    # Z9 T5B — B2B events carry the account/org id alongside distinct_id.
    if business_model == "b2b" and "account_id" in _context:
        enriched["account_id"] = _context["account_id"]

    if not _initialized or _posthog is None:
        import logging

        logging.getLogger("analytics").warning(
            "track_event(%s) dropped - PostHog not initialized.", name
        )
        return
    # Z9 T5A — cohort gate: skip emission for non-matching segments.
    if not _segment_matches():
        return
    _posthog.capture(distinct_id=distinct_id, event=name, properties=enriched)
