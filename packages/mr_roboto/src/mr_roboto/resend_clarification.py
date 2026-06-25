"""Mechanical re-send of a pending clarification's ORIGINAL interactive
message (question + content + buttons).

Sweep escalation reminders (4h / 24h / 48h) used to be bare pointers —
"Task #N needs your input" + the step title — so the founder had to scroll
the entire chat history to find the actual question and its options/buttons
(e.g. 0.6a non_goals_confirm: the draft + OK/Regenerate/Edit keyboard).

This executor re-runs the clarify executor against the SOURCE task, so the
reminder re-sends EXACTLY the interactive message that was first sent —
whatever the gate kind (artifact_confirm / variant_choice / surface_choice /
plain question). The founder acts straight from the reminder.

It re-sends against the source task id, so the re-sent keyboard's callbacks
carry the right task_id; the resend row itself just completes.
"""
from __future__ import annotations

import json as _json

from yazbunu import get_logger
from dabidabi import get_task
from src.app.telegram_bot import get_telegram
from mr_roboto.clarify import clarify

logger = get_logger("mr_roboto.resend_clarification")


async def resend_clarification(task: dict) -> dict:
    payload = (task.get("payload")
               or (task.get("context") or {}).get("payload")
               or {})
    source_task_id = payload.get("source_task_id")
    if source_task_id is None:
        raise ValueError(
            "resend_clarification payload requires 'source_task_id'"
        )

    src = await get_task(int(source_task_id))
    if not src:
        logger.info(
            "resend_clarification: source task %s not found — skip",
            source_task_id,
        )
        return {"resent": False, "reason": "source_not_found"}
    src = dict(src)

    # Only re-send while the founder still owes an answer. If they already
    # replied (status moved off waiting_human between the sweep tick and
    # this dispatch), a re-send would be a spurious duplicate.
    if src.get("status") != "waiting_human":
        logger.info(
            "resend_clarification: source task %s no longer waiting "
            "(status=%s) — skip", source_task_id, src.get("status"),
        )
        return {"resent": False, "reason": "not_waiting"}

    ctx = src.get("context")
    if isinstance(ctx, str):
        try:
            ctx = _json.loads(ctx)
        except (ValueError, TypeError):
            ctx = {}
    if not isinstance(ctx, dict):
        ctx = {}
    src["context"] = ctx

    # Path A — mechanical clarify gate (workflow step). The source task's
    # context carries the ORIGINAL clarify payload (kind / question /
    # attach_file_paths / options / regenerate_step_id). Re-run clarify
    # verbatim against the source row.
    clar_payload = ctx.get("payload")
    if isinstance(clar_payload, dict) and clar_payload.get("action") == "clarify":
        rp = dict(clar_payload)
        # Escalation re-send must always reach Telegram — the question was
        # already asked; we are nudging, not asking fresh. Bypass the
        # attention-budget gate so it never gets deferred to a file instead.
        rp["attention_skip"] = True
        src_task = dict(src)
        src_task["payload"] = rp
        res = await clarify(src_task)
        return {
            "resent": True,
            "via": "clarify",
            "kind": clar_payload.get("kind"),
            "result": res,
        }

    # Path C — parked reviewer halt (escalated review). No clarify payload and
    # no _clarification_question; re-render the founder-halt KEYBOARD via the
    # telegram interface so the reminder carries actionable Regenerate/Accept
    # buttons instead of dead-ending as a no-op text nudge.
    if src.get("agent_type") == "reviewer" or ctx.get("_review_halt"):
        tg = get_telegram()
        if tg is None:
            logger.info(
                "resend_clarification: telegram unavailable for review halt "
                "task %s", source_task_id,
            )
            return {"resent": False, "reason": "telegram_unavailable"}
        ok = await tg.resurface_review_halt(src)
        return {"resent": bool(ok), "via": "review_halt"}

    # Path B — LLM-agent plain clarification. No clarify payload; the question
    # lives in _clarification_question (or the current item of a numbered Q&A
    # queue). Re-send via the same request_clarification path used on the
    # original send and on restart-restore.
    question = ctx.get("_clarification_question") or ""
    if not question:
        queue = ctx.get("_clarification_queue")
        if isinstance(queue, dict):
            questions = queue.get("questions") or []
            current = queue.get("current", 0)
            if isinstance(current, int) and 0 <= current < len(questions):
                question = questions[current]
    if question:
        tg = get_telegram()
        if tg is None:
            logger.info(
                "resend_clarification: telegram unavailable for task %s",
                source_task_id,
            )
            return {"resent": False, "reason": "telegram_unavailable"}
        await tg.request_clarification(
            int(source_task_id), src.get("title", ""), question,
        )
        return {"resent": True, "via": "request_clarification"}

    logger.info(
        "resend_clarification: source task %s has no re-sendable "
        "clarification payload — skip", source_task_id,
    )
    return {"resent": False, "reason": "no_payload"}
