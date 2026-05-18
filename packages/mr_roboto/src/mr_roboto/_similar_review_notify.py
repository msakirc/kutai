"""Z1 T6A — similar-missions review Telegram notification.

When ``find_similar_missions`` returns ``ok=False``, fire a Telegram
notification with three inline buttons per match:

* ``simrev:c:<mission_id>`` — Continue (proceed; index this idea)
* ``simrev:b:<from>:<mission_id>`` — Branch from prior #from
* ``simrev:a:<mission_id>`` — Abort current mission

Callback data is capped at 64 bytes by Telegram; long combinations are
trimmed via short slug for safety.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mr_roboto.similar_review")


async def enqueue_similar_review_notice(
    *,
    mission_id: int,
    matches: list[dict[str, Any]],
    report_path: str | None,
) -> None:
    if not matches:
        return
    top = matches[0]
    lines = [
        f"🪞 *Idea similarity review — mission #{mission_id}*",
        "",
        f"Top match: prior mission #{top.get('mission_id')} "
        f"(similarity {top.get('similarity', 0):.2f})",
    ]
    if top.get("title"):
        lines.append(f"_{top['title']}_")
    if report_path:
        lines.append("")
        lines.append(f"Full report: `{report_path}`")
    message = "\n".join(lines)

    buttons: list[dict[str, str]] = [
        {"label": "▶️ Continue", "callback_data": f"simrev:c:{mission_id}"},
    ]
    # First couple of matches each get a Branch button.
    for m in matches[:2]:
        from_mid = m.get("mission_id")
        if from_mid is None:
            continue
        buttons.append({
            "label": f"🌿 Branch #{from_mid}",
            "callback_data": f"simrev:b:{from_mid}:{mission_id}",
        })
    buttons.append({
        "label": "🛑 Abort",
        "callback_data": f"simrev:a:{mission_id}",
    })

    try:
        import general_beckman  # type: ignore
        import json as _json

        await general_beckman.enqueue({
            "title": f"notify_user:similar_review:m{mission_id}",
            "agent_type": "mechanical",
            "mission_id": mission_id or None,
            "context": _json.dumps({
                "executor": "mechanical",
                "payload": {
                    "action": "notify_user",
                    "message": message,
                    "inline_buttons": buttons,
                },
            }),
        })
    except Exception as exc:
        logger.warning("similar-review enqueue failed: %s", exc)
