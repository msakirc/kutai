"""Z8 T3D — rule-based severity classifier for inbound webhook alerts.

``classify(integration_id, event_type, payload) -> str``
  Returns one of: ``"critical" | "high" | "medium" | "low" | "uncertain"``.

* Rules are pure functions of ``payload``; missing keys never raise.
* When no rules match the integration/event_type pair → ``"uncertain"``
  so the caller can fall back to LLM grading.
* When rules exist but no predicate fires → ``"low"`` (catch-all digest).

Severity bands (v1, mirrors plan):
  * critical — sentry event_count>100 in ≤5min, stripe live-mode payment
               failure, betterstack monitor down, github advisory critical.
  * high     — sentry affected_users>5, stripe dispute, betterstack
               degraded, github advisory high.
  * medium   — sentry regression, stripe recoverable invoice failure.
  * low      — everything else.
"""
from __future__ import annotations

from typing import Callable

Severity = str  # "critical" | "high" | "medium" | "low" | "uncertain"

_Rule = tuple[Severity, Callable[[dict], bool]]


def classify(integration_id: str, event_type: str, payload: dict) -> Severity:
    rules = _RULES.get(integration_id, {}).get(event_type)
    if not rules:
        return "uncertain"
    for severity, predicate in rules:
        try:
            if predicate(payload or {}):
                return severity
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return "low"


# ---------------------------------------------------------------------------
# Rule table
# ---------------------------------------------------------------------------


def _get(d: dict, *path, default=None):
    """Nested .get() that tolerates non-dict intermediates."""
    cur: object = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


_RULES: dict[str, dict[str, list[_Rule]]] = {
    "sentry": {
        "issue_alert": [
            (
                "critical",
                lambda p: int(p.get("event_count", 0)) > 100
                and int(p.get("timeframe_minutes", 60)) <= 5,
            ),
            ("high", lambda p: int(p.get("affected_users", 0)) > 5),
            ("medium", lambda p: bool(p.get("is_regression", False))),
        ],
    },
    "stripe": {
        "payment_intent.payment_failed": [
            (
                "critical",
                lambda p: bool(_get(p, "data", "object", "livemode", default=False)),
            ),
        ],
        "charge.dispute.created": [("high", lambda p: True)],
        "invoice.payment_failed": [("medium", lambda p: True)],
    },
    "betterstack": {
        "incident": [
            (
                "critical",
                lambda p: _get(p, "monitor", "status") == "down",
            ),
            (
                "high",
                lambda p: _get(p, "monitor", "status") == "degraded",
            ),
        ],
    },
    "github": {
        "repository_advisory": [
            (
                "critical",
                lambda p: _get(p, "advisory", "severity") == "critical",
            ),
            (
                "high",
                lambda p: _get(p, "advisory", "severity") == "high",
            ),
        ],
    },
}
