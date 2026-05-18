"""Z7 T4 A10 — Follow-up reminder job.

Runs daily at 09:00 (registered in beckman cron_seed as
``_executor='follow_up_reminder'``).

Scans ``interactions WHERE follow_up_at <= today+7 AND done=0``
and produces a plain-text digest pushed to the founder via notify_user
(best-effort; no failure if Telegram is unreachable).

Public API
----------
- ``build_digest(product_id)`` — build the reminder text for one product
  (or all if product_id is None).  Called by tests to verify digest content.
- ``run_follow_up_reminder()`` — called by mr_roboto for the
  ``follow_up_reminder`` executor.  Returns {"ok": True} on success.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("app.jobs.follow_up_reminder")


async def build_digest(product_id: str | None = None, within_days: int = 7) -> str:
    """Build the follow-up reminder digest text.

    Returns a markdown-ish string summarising pending follow-ups.
    If there are no pending items, returns an "all clear" message.
    """
    from src.app.crm import get_pending_follow_ups
    items = await get_pending_follow_ups(product_id=product_id, within_days=within_days)

    if not items:
        return f"Follow-up reminder: no pending follow-ups in the next {within_days} days. All clear."

    lines = [f"Follow-up reminder — {len(items)} pending in the next {within_days} days:\n"]
    for item in items:
        handle = item.get("handle") or f"contact#{item['contact_id']}"
        display = item.get("display_name") or handle
        summary = item.get("summary") or "(no summary)"
        fu_at = item.get("follow_up_at") or "?"
        kind = item.get("kind") or "other"
        lines.append(
            f"• {display} ({handle}) — [{kind}] {summary}\n"
            f"  Due: {fu_at}"
        )

    return "\n".join(lines)


async def run_follow_up_reminder() -> dict:
    """Main entry point called by mr_roboto ``follow_up_reminder`` executor.

    Builds the digest across all products and notifies the founder.
    Returns {"ok": True} on success, {"ok": False, "reason": ...} on failure.
    """
    try:
        digest = await build_digest(product_id=None, within_days=7)

        # Best-effort Telegram notify — failure here is non-fatal
        try:
            from src.app.telegram_bot import get_telegram
            tg = get_telegram()
            if tg is not None:
                # Use the notify_user mechanical path if available,
                # otherwise send a plain message to the admin chat.
                import os
                admin_chat = os.getenv("TELEGRAM_ADMIN_CHAT_ID")
                if admin_chat:
                    await tg.app.bot.send_message(
                        chat_id=int(admin_chat),
                        text=digest,
                    )
        except Exception as notify_exc:
            logger.warning(
                "follow_up_reminder: telegram notify failed (non-fatal)",
                error=str(notify_exc),
            )

        logger.info("follow_up_reminder: digest built and sent")
        return {"ok": True, "digest": digest}

    except Exception as exc:
        logger.error("follow_up_reminder: failed", error=str(exc))
        return {"ok": False, "reason": str(exc)}
