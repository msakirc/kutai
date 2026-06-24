"""Z7 T3D — B3: incident/draft_postmortem mr_roboto verb.

Auto-drafts a postmortem template when an incident is resolved.
The template is written as a workspace artifact (markdown) and a
founder_action is emitted requesting review + publication within 7 days.

The draft includes:
  - Incident timeline (from status_updates rows)
  - Affected components
  - Placeholder sections for: root cause, contributing factors,
    timeline of events, action items, lessons learned
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from yazbunu import get_logger

logger = get_logger("mr_roboto.incident_draft_postmortem")

_POSTMORTEM_TEMPLATE = """\
# Postmortem: {title}

**Incident ID:** {incident_id}
**Severity:** {severity}
**Opened:** {opened_at}
**Resolved:** {resolved_at}
**Affected Components:** {components}

---

## Summary

{impact_summary}

---

## Timeline

{timeline_md}

---

## Root Cause

<!-- TODO: Describe the root cause. What went wrong? -->

_To be filled in by the team within 7 days of resolution._

---

## Contributing Factors

<!-- TODO: List systemic or environmental factors that contributed. -->

---

## Action Items

<!-- TODO: List concrete follow-up tasks with owners and due dates. -->

| Action | Owner | Due |
|--------|-------|-----|
| | | |

---

## Lessons Learned

<!-- TODO: What did we learn? What would we do differently? -->

---

## Customer Communication

Updates sent to customers during this incident:

{customer_updates_md}

---

*Postmortem auto-drafted by KutAI on {drafted_at}. Review and publish within 7 days.*
"""


async def _fetch_incident(incident_id: int, product_id: str) -> dict | None:
    try:
        from dabidabi import get_db
        db = await get_db()
        async with db.execute(
            "SELECT incident_id, product_id, opened_at, resolved_at, severity, "
            "affected_components_json, customer_impact_summary, current_status_md "
            "FROM incidents WHERE incident_id = ? AND product_id = ?",
            (incident_id, product_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "incident_id": row[0],
            "product_id": row[1],
            "opened_at": row[2] or "",
            "resolved_at": row[3] or "",
            "severity": row[4] or "minor",
            "affected_components": json.loads(row[5] or "[]"),
            "customer_impact_summary": row[6] or "",
            "current_status_md": row[7] or "",
        }
    except Exception as exc:
        logger.warning("incident_draft_postmortem: fetch_incident failed", error=str(exc))
        return None


async def _fetch_updates(incident_id: int, product_id: str) -> list[dict]:
    try:
        from dabidabi import get_db
        db = await get_db()
        async with db.execute(
            "SELECT posted_at, status_kind, body_md "
            "FROM status_updates "
            "WHERE incident_id = ? AND product_id = ? "
            "ORDER BY posted_at ASC",
            (incident_id, product_id),
        ) as cur:
            rows = await cur.fetchall()
        return [{"posted_at": r[0], "status_kind": r[1], "body_md": r[2]} for r in rows]
    except Exception as exc:
        logger.warning("incident_draft_postmortem: fetch_updates failed", error=str(exc))
        return []


def _format_timeline(updates: list[dict]) -> str:
    if not updates:
        return "_(No status updates recorded)_"
    lines = []
    for u in updates:
        ts = u.get("posted_at") or "?"
        kind = u.get("status_kind") or ""
        body = (u.get("body_md") or "").strip().replace("\n", " ")[:200]
        lines.append(f"- **{ts}** [{kind}] {body}")
    return "\n".join(lines)


def _format_customer_updates(updates: list[dict]) -> str:
    if not updates:
        return "_(No customer updates sent)_"
    parts = []
    for u in updates:
        ts = u.get("posted_at") or "?"
        body = (u.get("body_md") or "").strip()
        parts.append(f"### {ts}\n\n{body}")
    return "\n\n---\n\n".join(parts)


async def _emit_founder_action(
    *,
    incident_id: int,
    product_id: str,
    mission_id: int,
    artifact_path: str,
) -> object:
    try:
        from src.founder_actions import create as fa_create
        title = (
            f"Postmortem draft ready for incident #{incident_id} ({product_id}) — "
            "please review and publish within 7 days."
        )
        why = (
            f"Incident #{incident_id} has been resolved. "
            "A postmortem template has been drafted automatically. "
            "Please fill in root cause, contributing factors, and action items, "
            "then publish the postmortem publicly within 7 days to maintain trust."
        )
        instructions = [
            f"Open the draft: {artifact_path}",
            "Fill in: root cause, contributing factors, action items, lessons learned.",
            "Review customer communication section.",
            "Publish to your status page or docs within 7 days of resolution.",
            "Update incidents.postmortem_url after publishing.",
        ]
        return await fa_create(
            mission_id=mission_id,
            kind="generic",
            title=title,
            why=why,
            instructions=instructions,
            expected_output_kind="ack_only",
            notify_telegram=True,
        )
    except Exception as exc:
        logger.warning(
            "incident_draft_postmortem: _emit_founder_action failed", error=str(exc)
        )
        return None


async def run(payload: dict) -> dict:
    """Execute incident/draft_postmortem.

    Expected payload keys:
      - incident_id   (int, required)
      - product_id    (str, required)
      - workspace_path (str, optional) — where to write the .md artifact
      - mission_id    (int, optional) — for founder_action context

    Returns:
      {"status": "ok", "artifact_path": str, "founder_action_id": int|None}
    """
    from dabidabi.times import db_now

    incident_id = payload.get("incident_id")
    product_id = payload.get("product_id") or ""
    workspace_path = payload.get("workspace_path") or ""
    mission_id = int(payload.get("mission_id") or 0)

    if not incident_id:
        return {"status": "error", "error": "incident_id is required"}
    if not product_id:
        return {"status": "error", "error": "product_id is required"}

    incident = await _fetch_incident(int(incident_id), product_id)
    if incident is None:
        return {
            "status": "error",
            "error": f"incident {incident_id} not found for product {product_id!r}",
        }

    updates = await _fetch_updates(int(incident_id), product_id)
    drafted_at = db_now()

    components = incident.get("affected_components") or []
    components_str = ", ".join(components) if components else "_(unknown)_"

    body = _POSTMORTEM_TEMPLATE.format(
        title=f"Incident #{incident_id} — {incident.get('severity', 'unknown').capitalize()} severity",
        incident_id=incident_id,
        severity=incident.get("severity", "minor"),
        opened_at=incident.get("opened_at") or "unknown",
        resolved_at=incident.get("resolved_at") or "unknown",
        components=components_str,
        impact_summary=incident.get("customer_impact_summary") or "_(No impact summary recorded)_",
        timeline_md=_format_timeline(updates),
        customer_updates_md=_format_customer_updates(updates),
        drafted_at=drafted_at,
    )

    # Determine artifact path.
    if workspace_path:
        artifact_dir = workspace_path
    else:
        artifact_dir = os.path.join("data", "incidents", product_id)

    os.makedirs(artifact_dir, exist_ok=True)
    artifact_filename = f"postmortem_incident_{incident_id}.md"
    artifact_path = os.path.join(artifact_dir, artifact_filename)

    with open(artifact_path, "w", encoding="utf-8") as fh:
        fh.write(body)

    logger.info(
        "incident_draft_postmortem: artifact written",
        artifact_path=artifact_path,
        incident_id=incident_id,
        product_id=product_id,
    )

    # Emit founder_action.
    fa = await _emit_founder_action(
        incident_id=int(incident_id),
        product_id=product_id,
        mission_id=mission_id,
        artifact_path=artifact_path,
    )
    fa_id = getattr(fa, "id", None) if fa else None

    return {
        "status": "ok",
        "artifact_path": artifact_path,
        "founder_action_id": fa_id,
        "incident_id": incident_id,
        "product_id": product_id,
    }
