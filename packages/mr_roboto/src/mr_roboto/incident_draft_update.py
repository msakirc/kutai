"""Z7 T3D — B3: incident/draft_update mr_roboto verb.

Draft a customer-friendly status update from internal alert details.

CRITICAL: Redacts internal hostnames, stack traces, and customer PII
before the draft is returned.  The draft is returned as text for
founder review — it is NOT published automatically.
"""
from __future__ import annotations

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
# LLM draft helper (testable via monkeypatch)
# ---------------------------------------------------------------------------

async def _call_llm_draft(
    severity: str,
    affected_components: list[str],
    safe_alert_details: dict,
    existing_summary: str,
    status_kind: str,
) -> str:
    """Call LLM (OVERHEAD lane via beckman.enqueue) to draft the update.

    Returns the raw draft text.  Caller redacts again before returning.
    """
    from general_beckman import enqueue
    from general_beckman.lanes import LANE_OVERHEAD
    import asyncio
    import json

    components_str = ", ".join(affected_components) if affected_components else "the service"
    safe_details_str = json.dumps(safe_alert_details, ensure_ascii=False)[:800]

    prompt = (
        f"You are drafting a public-facing status page update for customers.\n"
        f"Incident severity: {severity}\n"
        f"Affected components: {components_str}\n"
        f"Status kind: {status_kind} "
        f"(investigating|identified|monitoring|resolved)\n"
        f"Current summary: {existing_summary or 'none'}\n"
        f"Internal alert details (already redacted, for context only):\n"
        f"{safe_details_str}\n\n"
        f"Write 2-4 clear, calm sentences suitable for customers.\n"
        f"Rules:\n"
        f"- Do NOT mention internal hostnames, IPs, stack traces, or team names.\n"
        f"- Do NOT include customer PII.\n"
        f"- Use plain language — no jargon.\n"
        f"- Acknowledge the impact, state what you know, give next-update ETA.\n"
        f"Draft only — no sign-off or signature needed."
    )

    result_holder: list[str] = []
    done_event = asyncio.Event()

    async def _on_finish(task_result: dict) -> None:
        result_holder.append(task_result.get("output") or task_result.get("result") or "")
        done_event.set()

    await enqueue(
        {
            "title": "incident_draft_update:llm",
            "description": "Draft customer-facing status update.",
            "agent_type": "assistant",
            "kind": "overhead",
            "context": {
                "prompt": prompt,
                "_callback": _on_finish,
            },
        },
        lane=LANE_OVERHEAD,
    )

    # Await with a 30s timeout (OVERHEAD tasks should be fast).
    try:
        await asyncio.wait_for(done_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("incident_draft_update: LLM task timed out; returning empty draft")
        return ""

    return result_holder[0] if result_holder else ""


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
        from src.infra.db import get_db
        db = await get_db()
        async with db.execute(
            "SELECT severity, affected_components_json, customer_impact_summary "
            "FROM incidents WHERE incident_id = ? AND product_id = ?",
            (incident_id, product_id),
        ) as cur:
            row = await cur.fetchone()
        if row:
            import json as _json
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

    # Get LLM draft.
    draft_raw = ""
    try:
        draft_raw = await _call_llm_draft(
            severity=severity,
            affected_components=affected_components,
            safe_alert_details=safe_alert_details,
            existing_summary=existing_summary,
            status_kind=status_kind,
        )
    except Exception as exc:
        logger.warning("incident_draft_update: LLM draft failed", error=str(exc))
        draft_raw = (
            f"We are currently {status_kind} an issue affecting "
            f"{', '.join(affected_components) or 'some services'}. "
            "We will provide an update as soon as possible."
        )

    # Final redaction pass over the LLM output itself.
    from src.security.sensitivity import redact_user_pii, redact_secrets
    draft_clean = redact_internal(draft_raw)
    draft_clean = redact_secrets(draft_clean)
    draft_clean = redact_user_pii(draft_clean)

    logger.info(
        "incident_draft_update: draft ready",
        incident_id=incident_id,
        product_id=product_id,
        status_kind=status_kind,
        redaction_applied=redaction_applied,
    )

    return {
        "status": "ok",
        "draft": draft_clean,
        "redaction_applied": redaction_applied,
        "incident_id": incident_id,
        "product_id": product_id,
        "status_kind": status_kind,
    }
