"""Z8 T5G — Twilio SMS send executor.

Routed via ``mr_roboto.run`` when ``payload["action"] == "sms_send"``.

Wraps the Z6 ``vendor_call`` executor — Twilio's
``POST /2010-04-01/Accounts/{account_sid}/Messages.json`` endpoint. Uses
HTTP Basic auth via ``TWILIO_ACCOUNT_SID`` + ``TWILIO_AUTH_TOKEN`` env vars
through ``src/integrations/configs/twilio.json``.

Payload::

    {
        "to": "+15551234567",
        "from": "+15557654321",   # optional; falls back to TWILIO_FROM env
        "body": "Alert: ..."
    }

Returns ``{"ok": bool, "service": "twilio", "sid": str|None, ...}``.

When ``TWILIO_DAILY_CAP_USD`` env is set, the executor refuses with
``capped=true`` once the day's outbound count × $0.0079 (Twilio US base)
exceeds the cap. Counting lives in the ``registry_events`` ``sms_send``
scope — best-effort, fail-open if the table is unavailable.
"""
from __future__ import annotations

import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.sms_send")

# Twilio US base price per outbound SMS (USD). Approximate, used only for
# the daily cap heuristic — actual billing is via the dispute_check ledger.
_TWILIO_USD_PER_SMS: float = 0.0079


async def _today_sms_count() -> int:
    """Best-effort count of sms_send events recorded today. Fail-open → 0."""
    try:
        from dabidabi import get_db
        db = await get_db()
        async with db.execute(
            "SELECT COUNT(*) FROM registry_events "
            "WHERE scope='sms_send' AND date(timestamp) = date('now')"
        ) as cur:
            row = await cur.fetchone()
        return int((row or [0])[0])
    except Exception:
        return 0


async def _record_sms_send(mission_id: int | None, payload: dict, status: str) -> None:
    """Best-effort write to registry_events for cap tracking + ops_log."""
    try:
        import json as _json
        from dabidabi import get_db
        db = await get_db()
        await db.execute(
            "INSERT INTO registry_events "
            "(scope, target, event, verb, mission_id, payload_json) "
            "VALUES ('sms_send', ?, 'send', 'send', ?, ?)",
            (
                str(payload.get("to") or "unknown"),
                mission_id,
                _json.dumps({"payload": payload, "status": status}),
            ),
        )
        await db.commit()
    except Exception as e:  # noqa: BLE001
        logger.debug("sms_send registry write skipped: %s", e)


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    to = payload.get("to") or ""
    body = payload.get("body") or ""
    from_ = payload.get("from") or os.environ.get("TWILIO_FROM") or ""
    mission_id = task.get("mission_id")

    if not to or not body:
        return {"ok": False, "service": "twilio", "reason": "missing 'to' or 'body'"}
    if not from_:
        return {
            "ok": False, "service": "twilio",
            "reason": "missing 'from' (set TWILIO_FROM env or payload.from)",
        }

    # Daily cap heuristic.
    cap_env = os.environ.get("TWILIO_DAILY_CAP_USD")
    if cap_env:
        try:
            cap_usd = float(cap_env)
            sent_today = await _today_sms_count()
            projected = (sent_today + 1) * _TWILIO_USD_PER_SMS
            if projected > cap_usd:
                logger.warning(
                    "sms_send capped — projected $%.4f > cap $%.2f",
                    projected, cap_usd,
                )
                await _record_sms_send(mission_id, payload, "capped")
                return {
                    "ok": False, "service": "twilio",
                    "capped": True,
                    "reason": (
                        f"TWILIO_DAILY_CAP_USD=${cap_usd:.2f} "
                        f"would be exceeded ({sent_today} sent today)"
                    ),
                }
        except (TypeError, ValueError):
            pass

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID") or ""
    if not account_sid:
        return {
            "ok": False, "service": "twilio",
            "reason": "missing TWILIO_ACCOUNT_SID env",
        }

    # Delegate the HTTP call through vendor_call so retries + audit ride on
    # the standard adapter path.
    from .vendor_call import run as vendor_call_run

    sub_task = {
        "mission_id": mission_id,
        "context": {
            "post_hook": {
                "service": "twilio",
                "action": "send_sms",
                "params": {
                    "account_sid": account_sid,
                    "To": to,
                    "From": from_,
                    "Body": body,
                },
            }
        },
    }
    result = await vendor_call_run(sub_task)
    status = "ok" if result.get("ok") else "failed"
    await _record_sms_send(mission_id, payload, status)

    sid = None
    res_body = result.get("result")
    if isinstance(res_body, dict):
        sid = res_body.get("sid")
    return {
        "ok": bool(result.get("ok")),
        "service": "twilio",
        "to": to,
        "sid": sid,
        "vendor_result": result,
    }
