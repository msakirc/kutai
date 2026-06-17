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

# Telegram hard-caps a message at 4096 chars; leave headroom for the
# wrapper text + buttons.
_MAX_SCRIPT_CHARS = 3000


def _workspace_base(workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    return os.environ.get("WORKSPACE_DIR") or os.path.join(os.getcwd(), "workspace")


async def _ensure_interviews_dir(mission_id: int, workspace_path: str | None) -> str:
    """Create ``mission_<id>/.intake/interviews/`` if missing. Returns path."""
    target = os.path.join(
        _workspace_base(workspace_path),
        f"mission_{mission_id}", ".intake", "interviews",
    )
    try:
        os.makedirs(target, exist_ok=True)
    except OSError as exc:
        logger.warning("request_interview_data: mkdir %s failed: %s", target, exc)
    return target


def _read_script_body(
    mission_id: int, script_path: str, workspace_path: str | None
) -> str:
    """Best-effort read of the interview script so the founder receives the
    actual questions inline (not just a server-side path). Returns '' on any
    failure — the caller still surfaces the path as a fallback."""
    if os.path.isabs(script_path):
        full = script_path
    else:
        full = os.path.join(_workspace_base(workspace_path), script_path)
    try:
        with open(full, "r", encoding="utf-8") as fh:
            body = fh.read().strip()
    except OSError as exc:
        logger.warning(
            "request_interview_data: could not read script %s: %s", full, exc
        )
        return ""
    if len(body) > _MAX_SCRIPT_CHARS:
        body = body[:_MAX_SCRIPT_CHARS].rstrip() + "\n…(truncated — full script on disk)"
    return body


def _build_ack_keyboard(task_id: Any):
    """DONE/SKIP inline buttons. callback_data resolves against the live DB
    in handle_callback, so the gate survives bot restarts (the Telegram
    message persists in the founder's chat)."""
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    except Exception:
        return None
    if task_id is None:
        return None
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ DONE", callback_data=f"ivack:done:{task_id}"),
        InlineKeyboardButton("⏭️ SKIP", callback_data=f"ivack:skip:{task_id}"),
    ]])


async def _record_skip(mission_id: int, reason: str) -> dict[str, Any]:
    """Persist ``missions.interview_skip_reason`` for a skipped mission."""
    try:
        from general_beckman import update_mission_fields as _umf
        await _umf(int(mission_id), interview_skip_reason=(reason or "founder_skipped"))
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
    workspace_path = payload.get("workspace_path")
    target_dir = await _ensure_interviews_dir(mission_id, workspace_path)
    script_body = _read_script_body(mission_id, script_path, workspace_path)

    header = (
        "📋 *Interview script ready.* Run interviews and tap **DONE** with "
        f"N≥3 per persona (drop transcripts into `{target_dir}`), or **SKIP** "
        "at your own risk.\n"
    )
    if script_body:
        prompt = f"{header}\n———\n{script_body}"
    else:
        # Fallback: file unreadable — at least surface the path.
        prompt = f"{header}\nScript: {script_path}"

    # chat_id resolution: blackboard → task row → missions.context (the
    # load-bearing source — see workflows/engine/runner.py). Reuse the
    # canonical resolver so this gate matches clarify/confirm.
    chat_id = payload.get("chat_id")
    if chat_id is None:
        try:
            from mr_roboto.clarify import _resolve_chat_id
            chat_id = await _resolve_chat_id(task, mission_id)
        except Exception as exc:
            logger.debug(
                "request_interview_data: chat_id resolve failed: %s", exc
            )

    keyboard = _build_ack_keyboard(task.get("id"))

    sent = False
    try:
        from src.app.telegram_bot import get_telegram
        tg = get_telegram()
        if tg is not None:
            if chat_id is not None:
                try:
                    await tg.app.bot.send_message(
                        chat_id=int(chat_id), text=prompt,
                        parse_mode="Markdown", reply_markup=keyboard,
                    )
                    sent = True
                except Exception as exc:
                    # Bad chat_id / markdown — fall back to the admin-chat
                    # helper (retries + no-markdown fallback) so the gate is
                    # never silently skipped.
                    logger.warning(
                        "request_interview_data: direct send failed (%s); "
                        "falling back to send_notification", exc,
                    )
                    await tg.send_notification(prompt, reply_markup=keyboard)
                    sent = True
            else:
                await tg.send_notification(prompt, reply_markup=keyboard)
                sent = True
    except Exception as exc:
        logger.warning("request_interview_data: telegram send failed: %s", exc)

    try:
        if sent:
            from general_beckman import update_task
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
