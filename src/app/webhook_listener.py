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


@app.get("/webhook/__health")
async def webhook_health() -> dict:
    """T5H reserved hook — minimal liveness probe."""
    return {"status": "ok"}


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
