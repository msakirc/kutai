"""Z8 T3A — FastAPI webhook listener (embedded in orchestrator process).

Single port (default ``WEBHOOK_PORT`` env=9882; nerd_herd already binds 9881).
One generic route ``/webhook/{integration_id}`` covers known vendors
(sentry, stripe, github, betterstack, twilio) and any future entries
that gain a verifier + event-id extractor.

Flow per delivery:
  1. Read raw body.
  2. ``verify_signature(integration_id, raw, headers)`` — 401 on failure.
  3. Extract ``event_id`` from payload (per-vendor extractor).
  4. ``already_seen`` → return 200 ``duplicate`` (no work).
  5. Compute payload_hash + look up routing mission via
     ``integration_mappings`` (T3E table — falls back to NULL).
  6. ``mark_seen`` + enqueue mechanical ``alert_triage`` task on the
     ongoing lane.
  7. Return 200 ``accepted``.

Health endpoint ``/webhook/__health`` is reserved for Yaşar Usta wiring
in T5H — for now returns ``{"status": "ok"}``.
"""
from __future__ import annotations

import hashlib
import json
import logging

from fastapi import FastAPI, HTTPException, Request

from src.app.webhook_dedup import already_seen, mark_seen
from src.app.webhook_signing import verify_signature
from src.infra.db import get_db

logger = logging.getLogger("kutai.webhook")
app = FastAPI()

# Z9 T3A — providers whose webhooks carry *growth signals* (support tickets,
# error reports, analytics events) rather than ops alerts. These are
# normalized, PII-redacted, and stored to ``growth_events`` as ``raw_signal``
# instead of enqueuing an ``alert_triage`` task. sentry stays on the Z8
# ops/alert path — it is intentionally NOT in this set.
_SIGNAL_PROVIDERS = frozenset({"intercom", "zendesk", "posthog"})


@app.get("/webhook/__health")
async def webhook_health() -> dict:
    """T5H reserved hook — minimal liveness probe."""
    return {"status": "ok"}


# ── Z7 T2A — Email provider webhook routes ─────────────────────────────────
# Routes: POST /webhook/email/{provider}/{product_id}
# Covers: open, click, bounce, unsub, complaint, delivery for
#         brevo and resend (free-tier adapters).
# Signature verification is intentionally skipped here (the generic
# /webhook/{integration_id} route does sig verification via webhook_signing.py;
# email webhooks are typically unsigned or use a shared secret that can be
# added to webhook_signing.py when wired to a real domain).


@app.post("/webhook/email/{provider}/{product_id}")
async def email_webhook_inbound(
    provider: str, product_id: str, request: Request
) -> dict:
    """Receive and process email provider webhook events.

    Accepts Brevo / Resend (and future Postmark / SES) delivery events.
    Normalises via the provider adapter, persists to email_events, and
    adds to email_suppression on bounce / complaint / unsub.

    Dedup is NOT applied here (email webhooks rarely re-deliver the same
    event unlike ops webhooks); if needed, wrap in already_seen/mark_seen.
    """
    try:
        raw = await request.body()
        payload = __import__("json").loads(raw) if raw else {}
    except (ValueError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"bad json: {exc}") from exc

    try:
        from src.integrations.email.service import handle_webhook_event

        await handle_webhook_event(
            product_id=product_id,
            provider=provider,
            raw_payload=payload,
        )
    except Exception as exc:
        logger.error(
            "email webhook handler error",
            provider=provider,
            product_id=product_id,
            exc=str(exc),
        )
        raise HTTPException(status_code=500, detail="handler error") from exc

    return {"status": "accepted", "provider": provider, "product_id": product_id}


@app.post("/webhook/{integration_id}")
async def webhook_inbound(integration_id: str, request: Request) -> dict:
    raw = await request.body()
    headers = {k.lower(): v for k, v in request.headers.items()}

    if not await verify_signature(integration_id, raw, headers):
        raise HTTPException(status_code=401, detail="bad signature")

    try:
        payload = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"bad json: {exc}") from exc

    event_id = _extract_event_id(integration_id, payload)
    if not event_id:
        raise HTTPException(status_code=400, detail="missing event_id")

    if await already_seen(integration_id, event_id):
        return {"status": "duplicate", "event_id": event_id}

    payload_hash = hashlib.sha256(raw).hexdigest()
    mission_id = await _route_to_mission(integration_id, payload)
    await mark_seen(integration_id, event_id, payload_hash, mission_id)

    # Z9 T3A — growth signal intake. intercom / zendesk / posthog deliveries
    # are normalized + PII-redacted + stored as a ``raw_signal`` growth_event.
    # No LLM, no triage task — later tiers (T3B classifier) read the sink.
    if integration_id in _SIGNAL_PROVIDERS:
        await _store_raw_signal(integration_id, event_id, payload, mission_id)
        return {"status": "accepted", "event_id": event_id, "kind": "raw_signal"}

    # Enqueue the alert_triage task on the ongoing lane.
    #
    # Two conventions need to coexist:
    #   * Mechanical execution path: agent_type="mechanical" → runner derives
    #     to "mechanical" → mr_roboto.run dispatches on payload["action"].
    #     payload is stored under ``context["payload"]``; Beckman's apply.py
    #     `_mechanical_context` produces the same shape.
    #   * Lane routing: the ``lane`` kwarg pinned to LANE_ONGOING wins over
    #     ``pick_lane(agent_type)`` so mechanical alert_triage still lands
    #     on the ongoing pool. The legacy mapping
    #     ``pick_lane("alert_triage") == ongoing`` remains intact for any
    #     direct LLM-agent caller, but the webhook path is mechanical.
    from general_beckman import enqueue
    from general_beckman.lanes import LANE_ONGOING

    await enqueue(
        {
            "title": f"alert_triage:{integration_id}:{event_id}",
            "description": (
                f"Triage inbound webhook from {integration_id} "
                f"(event_id={event_id})."
            ),
            "mission_id": mission_id,
            "agent_type": "mechanical",
            "context": {
                "executor": "mechanical",
                "payload": {
                    "action": "alert_triage",
                    "integration_id": integration_id,
                    "event_id": event_id,
                    "payload": payload,
                },
            },
            "kind": "main_work",
        },
        lane=LANE_ONGOING,
    )
    return {"status": "accepted", "event_id": event_id}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_event_id(integration_id: str, payload: dict) -> str | None:
    """Vendor-specific event id extractor. Falls back to generic id/event_id."""
    extractors = {
        "sentry": lambda p: p.get("event_id") or p.get("id"),
        "stripe": lambda p: p.get("id"),
        "github": lambda p: p.get("delivery") or p.get("id"),
        "betterstack": lambda p: p.get("id") or p.get("uuid"),
        "twilio": lambda p: p.get("MessageSid") or p.get("CallSid") or p.get("id"),
        # Z9 T3A — growth signal providers.
        "intercom": lambda p: (
            p.get("id")
            or (p.get("data", {}).get("item", {}) or {}).get("id")
        ),
        "zendesk": lambda p: (
            p.get("id")
            or p.get("ticket_id")
            or (p.get("ticket", {}) or {}).get("id")
        ),
        "posthog": lambda p: (
            p.get("uuid") or p.get("event_id") or p.get("id")
        ),
    }
    fn = extractors.get(integration_id, lambda p: p.get("event_id") or p.get("id"))
    val = fn(payload)
    return str(val) if val is not None else None


async def _route_to_mission(integration_id: str, payload: dict) -> int | None:
    """Map an inbound webhook to an ongoing mission via integration_mappings.

    T3A: the table may not exist yet (created in T3E migration). Treat any
    SQLite OperationalError (no such table) as "no mapping" and return None
    so the listener still accepts the event. Once T3E runs init_db, the
    real lookup kicks in.

    Preference order:
      1. Exact (integration_id, product_id) match.
      2. Generic (integration_id, NULL product_id) catch-all.
    """
    product_id = payload.get("product_id")
    try:
        conn = await get_db()
        async with conn.execute(
            "SELECT mission_id FROM integration_mappings "
            "WHERE integration_id = ? "
            "  AND (product_id = ? OR product_id IS NULL) "
            "ORDER BY (product_id IS NULL) ASC LIMIT 1",
            (integration_id, product_id),
        ) as cur:
            row = await cur.fetchone()
    except Exception as exc:  # OperationalError: no such table
        logger.debug("integration_mappings lookup skipped: %s", exc)
        return None
    if not row:
        return None
    return int(row[0])


# ---------------------------------------------------------------------------
# Z9 T3A — growth signal normalization + sink
# ---------------------------------------------------------------------------


def _normalize_signal(
    provider: str, event_id: str, payload: dict
) -> dict:
    """Parse a provider webhook payload into a normalized signal dict.

    Shape (stable contract — T3B classifier consumes this):
        {provider, signal_type, content, external_id, occurred_at, raw_meta}

    ``content`` is the human-readable free text (ticket body, error message,
    event name). ``raw_meta`` carries provider-specific structured fields the
    classifier may use. PII redaction is applied by the caller, not here.
    """
    signal_type = "unknown"
    content = ""
    occurred_at = None
    raw_meta: dict = {}

    if provider == "intercom":
        # Conversation / ticket events: payload.data.item holds the resource.
        topic = payload.get("topic") or payload.get("type") or ""
        item = (payload.get("data", {}) or {}).get("item", {}) or {}
        signal_type = "support_ticket"
        source = item.get("source", {}) or {}
        content = (
            source.get("body")
            or item.get("body")
            or (item.get("conversation_message", {}) or {}).get("body")
            or ""
        )
        occurred_at = item.get("created_at") or payload.get("created_at")
        raw_meta = {
            "topic": topic,
            "conversation_id": item.get("id"),
            "state": item.get("state"),
            "priority": item.get("priority"),
        }
    elif provider == "zendesk":
        ticket = payload.get("ticket", {}) or payload
        signal_type = "support_ticket"
        content = (
            ticket.get("description")
            or ticket.get("latest_comment", "")
            or payload.get("description")
            or payload.get("comment")
            or ""
        )
        occurred_at = ticket.get("created_at") or payload.get("created_at")
        raw_meta = {
            "ticket_id": ticket.get("id") or payload.get("ticket_id"),
            "subject": ticket.get("subject") or payload.get("subject"),
            "status": ticket.get("status"),
            "priority": ticket.get("priority"),
            "tags": ticket.get("tags"),
        }
    elif provider == "posthog":
        signal_type = "analytics_event"
        event_name = (
            payload.get("event")
            or payload.get("name")
            or (payload.get("data", {}) or {}).get("event")
            or ""
        )
        content = event_name
        properties = payload.get("properties", {}) or {}
        occurred_at = payload.get("timestamp") or payload.get("sent_at")
        raw_meta = {
            "event": event_name,
            "distinct_id": payload.get("distinct_id"),
            "properties": properties,
        }

    return {
        "provider": provider,
        "signal_type": signal_type,
        "content": content,
        "external_id": event_id,
        "occurred_at": occurred_at,
        "raw_meta": raw_meta,
    }


async def _store_raw_signal(
    provider: str,
    event_id: str,
    payload: dict,
    mission_id: int | None,
) -> None:
    """Normalize, PII-redact, and persist a growth signal as ``raw_signal``.

    Runs ``redact_user_pii`` over ``content`` and ``raw_meta`` (free-text and
    structured) before the row hits ``growth_events``. T3A does NOT classify
    or score — that is T3B/T3C.
    """
    from src.infra.db import insert_growth_event
    from src.security.sensitivity import redact_user_pii

    signal = _normalize_signal(provider, event_id, payload)
    # Redact user PII from every free-text / structured field before storage.
    signal["content"] = redact_user_pii(signal.get("content") or "")
    signal["raw_meta"] = redact_user_pii(signal.get("raw_meta") or {})

    try:
        await insert_growth_event(
            mission_id=mission_id,
            kind="raw_signal",
            properties=signal,
            segment=None,
        )
    except Exception as exc:
        # Don't fail the webhook response on a sink hiccup — the delivery is
        # already deduped (mark_seen ran); log and move on.
        logger.warning(
            "growth_events insert failed for %s/%s: %s",
            provider, event_id, exc,
        )
