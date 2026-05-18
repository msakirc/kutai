"""Artifact-emit notifications with inline action buttons (Z1 T4B).

When a mechanical action produces an artifact (today: ``regen_artifact``),
enqueue a ``notify_user`` task that includes inline buttons so the founder
can iterate without typing commands.

Buttons (callback_data is capped at 64 bytes by Telegram — paths longer
than that are silently skipped by ``notify_user._build_reply_markup``):

* ``regen:<mid>:<artifact_path>``     — re-emit with a different change
* ``propagate:<mid>:<artifact_path>`` — walk produces/consumes graph,
  propose downstream patches
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("mr_roboto.artifact_notify")


async def enqueue_artifact_emit_notice(
    *,
    mission_id: int,
    artifact_path: str,
    change_description: str,
    new_version: str,
) -> None:
    """Enqueue a notify_user task for an artifact-emit event.

    Best-effort: any failure (Beckman missing, DB write failure) is logged
    and swallowed so callers can chain the call after a successful action
    without risking the action's status.
    """
    if not artifact_path:
        return
    label = os.path.basename(artifact_path) or artifact_path
    text_lines = [
        f"📦 *Artifact regen:* `{artifact_path}`",
        f"_Change:_ {change_description}",
    ]
    if new_version:
        text_lines.append(f"_New version:_ `{os.path.basename(new_version)}`")
    message = "\n".join(text_lines)

    buttons = [
        {
            "label": "🔄 Regen",
            "callback_data": f"regen:{mission_id}:{artifact_path}",
        },
        {
            "label": "🎯 Propagate",
            "callback_data": f"propagate:{mission_id}:{artifact_path}",
        },
    ]

    try:
        import general_beckman  # type: ignore

        await general_beckman.enqueue(
            {
                "title": f"notify_user:artifact_emit:{label}",
                "context": _mechanical_context(
                    "notify_user",
                    message=message,
                    inline_buttons=buttons,
                ),
                "agent_type": "mechanical",
                "mission_id": mission_id or None,
            }
        )
    except Exception as exc:
        logger.warning("artifact-emit notify enqueue failed: %s", exc)


def _mechanical_context(action: str, **payload) -> str:
    """Serialize a mechanical task context exactly like Beckman's sweep helper.

    Kept local so the helper has zero coupling to internal Beckman APIs.
    """
    import json as _json

    return _json.dumps(
        {
            "executor": "mechanical",
            "payload": {"action": action, **payload},
        }
    )
