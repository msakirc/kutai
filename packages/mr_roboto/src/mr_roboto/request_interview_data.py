"""Mechanical request_interview_data — Z1 Tier 2 (A4).

Surfaces the generated ``interview_script.md`` path to the founder via
Telegram and waits for a DONE/SKIP reply.

Behaviour:

* Sends a clarify-shape message with the script path and a short prompt:
  "Run interviews and reply DONE with N>=3 per persona, or SKIP at your
  own risk."
* Returns ``status="needs_clarification"`` with ``keyboard_sent=True``
  so general_beckman.result_router keeps the row ``waiting_human``
  until the founder replies.
* On DONE the founder is expected to drop interview transcripts into
  ``mission_{mission_id}/.intake/interviews/`` (created here as an
  empty directory if missing).
* On SKIP the caller (downstream classifier or a follow-up step) is
  expected to set ``missions.interview_skip_reason``; this mechanical
  exposes a ``record_skip`` sub-mode for that.

Payload:

* ``script_path`` (str, required): relative or absolute path to the
  interview script.
* ``mode`` (str, optional): ``"prompt"`` (default) or ``"record_skip"``.
* ``skip_reason`` (str, optional): used only when ``mode=="record_skip"``.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


async def _ensure_interviews_dir(mission_id: int, workspace_path: str | None) -> str:
    """Create ``mission_<id>/.intake/interviews/`` if missing. Returns path."""
    if workspace_path:
        base = workspace_path
    else:
        base = os.environ.get("WORKSPACE_DIR") or os.path.join(
            os.getcwd(), "workspace"
        )
    target = os.path.join(base, f"mission_{mission_id}", ".intake", "interviews")
    try:
        os.makedirs(target, exist_ok=True)
    except OSError as exc:
        logger.warning("request_interview_data: mkdir %s failed: %s", target, exc)
    return target


async def _record_skip(mission_id: int, reason: str) -> dict[str, Any]:
    """Persist ``missions.interview_skip_reason`` for a skipped mission."""
    try:
        from src.infra.db import get_db
    except Exception as exc:
        logger.warning("request_interview_data: db import failed: %s", exc)
        return {"recorded": False, "error": str(exc)}
    try:
        db = await get_db()
        await db.execute(
            "UPDATE missions SET interview_skip_reason = ? WHERE id = ?",
            (reason or "founder_skipped", int(mission_id)),
        )
        await db.commit()
        return {"recorded": True, "mission_id": mission_id, "reason": reason}
    except Exception as exc:
        logger.exception("request_interview_data: skip persist failed")
        return {"recorded": False, "error": str(exc)}


async def request_interview_data(task: dict) -> dict[str, Any]:
    """Send the interview-script prompt to the founder via Telegram."""
    payload = task.get("payload") or {}
    mode = (payload.get("mode") or "prompt").lower()
    mission_id = task.get("mission_id") or payload.get("mission_id")
    if mission_id is None:
        return {"status": "failed", "error": "missing mission_id"}
    mission_id = int(mission_id)

    if mode == "record_skip":
        reason = str(payload.get("skip_reason") or "founder_skipped")
        out = await _record_skip(mission_id, reason)
        return {
            "status": "completed",
            "mode": "record_skip",
            **out,
        }

    script_path = payload.get("script_path") or (
        f"mission_{mission_id}/.intake/interview_script.md"
    )
    target_dir = await _ensure_interviews_dir(
        mission_id, payload.get("workspace_path")
    )

    prompt = (
        f"Interview script is ready at {script_path}.\n"
        f"Run interviews and reply DONE with N>=3 per persona "
        f"(drop transcripts into {target_dir}), or SKIP at your own risk."
    )

    chat_id = payload.get("chat_id")
    if chat_id is None:
        try:
            from src.collaboration.blackboard import read_blackboard
            artifacts = await read_blackboard(mission_id, "artifacts")
            if isinstance(artifacts, dict):
                chat_id = artifacts.get("chat_id")
        except Exception as exc:
            logger.debug(
                "request_interview_data: blackboard chat_id lookup failed: %s",
                exc,
            )

    sent = False
    try:
        from src.app.telegram_bot import get_telegram
        tg = get_telegram()
        if tg is not None:
            if chat_id is not None:
                await tg.app.bot.send_message(chat_id=int(chat_id), text=prompt)
            else:
                await tg.send_notification(prompt)
            sent = True
    except Exception as exc:
        logger.warning("request_interview_data: telegram send failed: %s", exc)

    try:
        if sent:
            from src.infra.db import update_task
            await update_task(task["id"], status="waiting_human")
    except Exception as exc:
        logger.debug("request_interview_data: update_task waiting_human failed: %s", exc)

    return {
        "status": "needs_clarification" if sent else "completed",
        "kind": "interview_request",
        "prompt": prompt,
        "script_path": script_path,
        "interviews_dir": target_dir,
        "keyboard_sent": sent,
    }
