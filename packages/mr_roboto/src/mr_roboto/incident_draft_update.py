"""Z7 T3D — B3: incident/draft_update mr_roboto verb.

Draft a customer-friendly status update from internal alert details.

CRITICAL: Redacts internal hostnames, stack traces, and customer PII
before the draft is returned.  The draft is returned as text for
founder review — it is NOT published automatically.
"""
from __future__ import annotations

import json as _json
import re
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.incident_draft_update")

# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------

# Patterns that signal internal / non-customer-facing content.
_HOSTNAME_RE = re.compile(
    r'\b(?:'
    r'[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?'          # first label
    r'(?:\.[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?){1,}'  # subsequent labels
    r')'
    r'(?::(?:6553[0-5]|655[0-2]\d|65[0-4]\d{2}|6[0-4]\d{3}|[1-5]\d{4}|[1-9]\d{0,3}))?',
    re.IGNORECASE,
)

# Common internal FQDN suffixes worth flagging explicitly.
_INTERNAL_SUFFIX_RE = re.compile(
    r'\b[a-z0-9][\w\-]*\.'
    r'(?:internal|local|lan|corp|intra|cluster\.local|svc\.cluster|k8s)\b',
    re.IGNORECASE,
)

# Stack-trace indicators (Python, JS, Java).
_STACK_TRACE_RE = re.compile(
    r'(?:'
    r'Traceback \(most recent call last\):|'   # Python
    r'  File "[^"]+", line \d+|'               # Python frame
    r'at [A-Za-z_$][A-Za-z0-9_$]*\(?.*\)?\s+\([^)]+\)|'  # JS/Java
    r'\tat [a-z][a-zA-Z0-9_.]+\([^)]*\)|'     # Java
    r'Exception in thread|'
    r'Caused by:|'
    r'^\s+(?:raise|throw) '                    # Python raise / Java throw line
    r')',
    re.IGNORECASE | re.MULTILINE,
)

# IPv4 addresses that shouldn't reach customers.
_IP_RE = re.compile(
    r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    r'|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}'
    r'|192\.168\.\d{1,3}\.\d{1,3})\b'
)


def _contains_stack_trace(text: str) -> bool:
    return bool(_STACK_TRACE_RE.search(text))


def redact_internal(text: str) -> str:
    """Remove stack traces, private IPs, and internal hostnames from *text*.

    Returns a redacted copy safe for external publication.  Stack-trace blocks
    are replaced wholesale; hostnames / IPs are replaced token-by-token.
    """
    if not text:
        return text

    # 1. Wipe stack traces (multi-line blocks)
    if _contains_stack_trace(text):
        # Replace the entire traceback block with a placeholder.
        text = _STACK_TRACE_RE.sub("[internal error detail redacted]", text)

    # 2. Replace private-network IPs
    text = _IP_RE.sub("[internal-ip]", text)

    # 3. Replace obvious internal hostnames
    text = _INTERNAL_SUFFIX_RE.sub("[internal-host]", text)

    return text


def _redact_alert(alert_details: dict) -> dict:
    """Return a copy of *alert_details* with sensitive fields redacted."""
    safe: dict[str, Any] = {}
    for key, val in alert_details.items():
        if isinstance(val, str):
            # Strip PII and internal infrastructure from every string field.
            from src.security.sensitivity import redact_user_pii, redact_secrets
            cleaned = redact_internal(val)
            cleaned = redact_secrets(cleaned)
            cleaned = redact_user_pii(cleaned)
            safe[key] = cleaned
        elif isinstance(val, dict):
            safe[key] = _redact_alert(val)
        elif isinstance(val, list):
            safe[key] = [
                _redact_alert(item) if isinstance(item, dict) else item
                for item in val
            ]
        else:
            safe[key] = val
    return safe


# ---------------------------------------------------------------------------
# Shared helpers — used by run() and the CPS sink
# ---------------------------------------------------------------------------

def fallback_draft(status_kind: str, affected_components: list) -> str:
    """Deterministic customer-facing draft when the LLM produced nothing."""
    return (
        f"We are currently {status_kind} an issue affecting "
        f"{', '.join(affected_components) or 'some services'}. "
        "We will provide an update as soon as possible."
    )


def finalize_redaction(text: str) -> str:
    """Final safety pass over any draft before it reaches a customer-facing card."""
    from src.security.sensitivity import redact_user_pii, redact_secrets
    text = redact_internal(text)
    text = redact_secrets(text)
    text = redact_user_pii(text)
    return text


# ---------------------------------------------------------------------------
# Public run() — called by mr_roboto._run_dispatch
# ---------------------------------------------------------------------------

async def run(payload: dict) -> dict:
    """Execute incident/draft_update.

    Expected payload keys:
      - incident_id  (int, required)
      - product_id   (str, required)
      - alert_details (dict, required) — raw internal alert from Z8 oncall
      - status_kind  ('investigating'|'identified'|'monitoring'|'resolved')

    Returns:
      {"status": "ok", "draft": str, "redaction_applied": bool}
    """
    incident_id = payload.get("incident_id")
    product_id = payload.get("product_id") or ""
    alert_details: dict = payload.get("alert_details") or {}
    status_kind = payload.get("status_kind") or "investigating"

    if not incident_id:
        return {"status": "error", "error": "incident_id is required"}
    if not product_id:
        return {"status": "error", "error": "product_id is required"}

    # Fetch incident for context.
    severity = "minor"
    affected_components: list[str] = []
    existing_summary = ""
    try:
        from dabidabi import get_db
        db = await get_db()
        async with db.execute(
            "SELECT severity, affected_components_json, customer_impact_summary "
            "FROM incidents WHERE incident_id = ? AND product_id = ?",
            (incident_id, product_id),
        ) as cur:
            row = await cur.fetchone()
        if row:
            severity = row[0] or "minor"
            try:
                affected_components = _json.loads(row[1] or "[]")
            except Exception:
                affected_components = []
            existing_summary = row[2] or ""
    except Exception as exc:
        logger.warning("incident_draft_update: could not fetch incident", error=str(exc))

    # Redact internal details before passing to LLM.
    safe_alert_details = _redact_alert(alert_details)
    redaction_applied = safe_alert_details != alert_details

    from src.comms.producers import enqueue_incident_update
    tid = await enqueue_incident_update(
        incident_id=incident_id, product_id=product_id, status_kind=status_kind,
        severity=severity, affected_components=affected_components,
        safe_alert_details=safe_alert_details, existing_summary=existing_summary,
    )
    logger.info("incident_draft_update: producer enqueued", incident_id=incident_id,
                product_id=product_id, status_kind=status_kind,
                redaction_applied=redaction_applied)
    return {
        "status": "ok", "producer_task_id": tid, "deferred": True,
        "redaction_applied": redaction_applied, "incident_id": incident_id,
        "product_id": product_id, "status_kind": status_kind,
    }
