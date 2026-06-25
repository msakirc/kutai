"""Z10 T4A — ``mission_deliverable_bundle`` mechanical verb.

Final-step bundler: posts the mission's deliverable artifacts to the
mission's Telegram thread (T2B) or flat-fallback. Includes:

* ``demo.mp4`` as a video attachment (when present).
* Final commit hash from the mission workspace's git HEAD.
* Top-N provenance summary (files with most ``artifact_provenance`` rows
  + the model_id of the last writer).
* Cost summary (T2A's ``format_mission_cost``).

Reversibility: ``irreversible`` — sends a visible Telegram message.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.mission_deliverable_bundle")


def _project_root() -> str:
    here = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))


def _default_demo_path(mission_id: int) -> str:
    return os.path.join(
        _project_root(), "data", "missions", f"{int(mission_id)}", "demo.mp4"
    )


def _final_commit_sha(repo_path: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return (out or b"").decode("utf-8", "replace").strip() or None
    except Exception:
        return None


async def _top_provenance(mission_id: int, limit: int = 5) -> list[dict[str, Any]]:
    """Return top-N most-written paths + their last writer model_id."""
    try:
        from dabidabi import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT path, COUNT(*) AS write_count, "
            "       (SELECT model_id FROM artifact_provenance ap2 "
            "          WHERE ap2.path = ap.path "
            "          ORDER BY ap2.written_at DESC LIMIT 1) AS last_model_id "
            "FROM artifact_provenance ap "
            "WHERE mission_id = ? "
            "GROUP BY path "
            "ORDER BY write_count DESC, MAX(written_at) DESC "
            "LIMIT ?",
            (int(mission_id), int(limit)),
        )
        rows = await cur.fetchall()
        return [
            {
                "path": r[0],
                "write_count": int(r[1]),
                "last_model_id": r[2],
            }
            for r in (rows or [])
        ]
    except Exception as e:
        logger.warning(
            "top_provenance lookup failed",
            mission_id=mission_id,
            error=str(e),
        )
        return []


def _format_bundle_text(
    mission_id: int,
    commit_sha: str | None,
    provenance: list[dict[str, Any]],
    cost_text: str,
) -> str:
    lines = [f"📦 [deliverable] Mission {mission_id} bundle"]
    if commit_sha:
        lines.append(f"Commit: {commit_sha[:12]}")
    if provenance:
        lines.append("Top files:")
        for row in provenance:
            mid = row.get("last_model_id") or "?"
            lines.append(
                f" • {row['path']}  ({row['write_count']} writes, last: {mid})"
            )
    lines.append("")
    lines.append(cost_text)
    return "\n".join(lines)


async def run(
    bot: Any,
    mission_id: int,
    video_path: str | None = None,
    repo_path: str | None = None,
    chat_id: int | None = None,
) -> dict[str, Any]:
    """Post the deliverable bundle to the mission's Telegram thread.

    Returns dict with ``ok``, ``commit_sha``, ``video_sent``, ``provenance``,
    ``cost_text``, and the underlying sent message ids when available.
    """
    from src.app.telegram_topics import post_to_mission_thread
    from kuleden_donen_var import format_mission_cost

    if repo_path is None:
        from src.tools.workspace import get_mission_workspace
        repo_path = get_mission_workspace(int(mission_id))

    if video_path is None:
        video_path = _default_demo_path(int(mission_id))

    commit_sha = _final_commit_sha(repo_path)
    provenance = await _top_provenance(int(mission_id), limit=5)
    try:
        cost_text = await format_mission_cost(int(mission_id))
    except Exception as e:
        logger.warning("format_mission_cost failed", error=str(e))
        cost_text = f"Mission {mission_id} — cost unavailable"

    text = _format_bundle_text(mission_id, commit_sha, provenance, cost_text)

    # 1. Send the video first when it exists — Telegram caption is short,
    # so we follow with a text post for the full provenance/cost block.
    video_sent_id: int | None = None
    if os.path.exists(video_path):
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            # Try bot.send_video with thread routing manually — we can't
            # use post_to_mission_thread for binary uploads.
            from dabidabi import get_mission
            mission = await get_mission(int(mission_id))
            thread_id = (mission or {}).get("telegram_thread_id")
            caption = (
                f"[Mission {mission_id}] demo.mp4"
                if not thread_id
                else "demo.mp4"
            )
            if chat_id is None:
                from src.app.config import TELEGRAM_ADMIN_CHAT_ID
                chat_id_val = int(TELEGRAM_ADMIN_CHAT_ID) if TELEGRAM_ADMIN_CHAT_ID else None
            else:
                chat_id_val = int(chat_id)
            if chat_id_val is None:
                logger.warning("deliverable_bundle: no chat_id; skipping video send")
            else:
                kwargs: dict[str, Any] = {
                    "chat_id": chat_id_val,
                    "video": video_bytes,
                    "caption": caption,
                }
                if thread_id:
                    kwargs["message_thread_id"] = int(thread_id)
                msg = await bot.send_video(**kwargs)
                video_sent_id = getattr(msg, "message_id", None)
                if video_sent_id is None and isinstance(msg, dict):
                    video_sent_id = msg.get("message_id")
        except Exception as e:
            logger.warning(
                "deliverable_bundle video send failed",
                mission_id=mission_id,
                error=str(e),
            )

    # 2. Send the text bundle (provenance + cost + commit).
    text_sent_id: int | None = None
    try:
        msg = await post_to_mission_thread(
            bot, int(mission_id), text, chat_id=chat_id,
        )
        text_sent_id = getattr(msg, "message_id", None)
        if text_sent_id is None and isinstance(msg, dict):
            text_sent_id = msg.get("message_id")
    except Exception as e:
        logger.warning(
            "deliverable_bundle text send failed",
            mission_id=mission_id,
            error=str(e),
        )

    return {
        "ok": True,
        "mission_id": int(mission_id),
        "commit_sha": commit_sha,
        "video_path": video_path,
        "video_sent": video_sent_id is not None,
        "video_message_id": video_sent_id,
        "text_message_id": text_sent_id,
        "provenance": provenance,
        "cost_text": cost_text,
        "text": text,
    }
