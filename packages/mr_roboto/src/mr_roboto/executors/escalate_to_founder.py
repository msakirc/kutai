"""Z8 T5G — escalate_to_founder executor.

Routed via ``mr_roboto.run`` when ``payload["action"] == "escalate_to_founder"``.
Also invoked by the on-call agent's verb-of-the-same-name (T4A whitelist).

Pipeline:

1. Look up the mission's :mod:`src.ops.escalation_policy` (defaults applied
   when no row exists).
2. Resolve the dispatch channel via severity + quiet hours — one of
   ``"telegram"`` / ``"telegram_log_only"`` / ``"sms"`` / ``"email"`` / ``"log"``.
3. Dispatch:
   - ``sms`` → call the :mod:`mr_roboto.executors.sms_send` executor.
   - ``telegram`` / ``telegram_log_only`` → create a founder_action
     (kind=``escalation``); the founder_actions notifier surfaces it on
     Telegram (urgent flag set when channel is just ``"telegram"`` — that
     pushes DM-to-admin; the log-only variant lands on the mission thread).
   - other → log + create a founder_action so nothing is silently dropped.

Payload::

    {
        "severity":   "critical",     # one of low/medium/high/critical
        "title":      "DB disk full",
        "summary":    "p99 disk free <2% for 5min",
        "to":         "+15551234567", # optional SMS target
        "instructions": [...],         # optional list of next steps
    }

Returns ``{"ok": bool, "channel": str, "tier": int, "founder_action_id": int|None,
"sms_sid": str|None, "quiet": bool}``.
"""
from __future__ import annotations

import os
from typing import Any

from yazbunu import get_logger
from src.ops.escalation_policy import resolve_channel

logger = get_logger("mr_roboto.escalate_to_founder")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    mission_id = task.get("mission_id")
    severity = str(payload.get("severity") or "medium").lower()
    title = str(payload.get("title") or "Escalation")
    summary = str(payload.get("summary") or "")
    instructions = payload.get("instructions") or [summary or title]

    if not isinstance(instructions, list):
        instructions = [str(instructions)]

    resolution = await resolve_channel(
        int(mission_id) if mission_id is not None else 0,
        severity,
    )
    channel = resolution.get("channel") or "telegram"
    tier = int(resolution.get("tier") or 1)
    quiet = bool(resolution.get("quiet"))

    out: dict[str, Any] = {
        "ok": False,
        "channel": channel,
        "tier": tier,
        "quiet": quiet,
        "founder_action_id": None,
        "sms_sid": None,
    }

    # ── SMS path ────────────────────────────────────────────────────────────
    if channel == "sms":
        to = payload.get("to") or os.environ.get("ONCALL_PHONE") or ""
        if not to:
            logger.warning(
                "escalate_to_founder: sms channel requested but no recipient "
                "(payload.to / ONCALL_PHONE env both empty) — falling back "
                "to founder_action"
            )
        else:
            from .sms_send import run as sms_run

            sms_res = await sms_run({
                "mission_id": mission_id,
                "payload": {
                    "to": to,
                    "body": f"[{severity.upper()}] {title}\n{summary}"[:1500],
                },
            })
            out["sms_sid"] = sms_res.get("sid")
            out["ok"] = bool(sms_res.get("ok"))
            out["sms_result"] = sms_res
            # Always also record a founder_action so the incident is
            # tracked in the action queue — SMS is just the page.

    # ── founder_action path (telegram, telegram_log_only, fallback) ─────────
    try:
        from src.founder_actions import create as fa_create

        action = await fa_create(
            mission_id=int(mission_id) if mission_id is not None else 0,
            kind="escalation",
            title=title,
            why=summary or f"severity={severity}",
            instructions=[str(s) for s in instructions],
            urgent=(channel == "telegram" or channel == "sms" or tier >= 3),
            notify_telegram=(channel != "log"),
        )
        out["founder_action_id"] = int(getattr(action, "id", 0) or 0)
        if channel != "sms":
            out["ok"] = True
    except Exception as e:  # noqa: BLE001
        logger.error("escalate_to_founder: founder_action create failed: %s", e)
        if channel != "sms":
            out["ok"] = False
            out["reason"] = str(e)

    return out
