"""Z7 T5 B2 — changelog_freshness posthook handler.

Fires monthly. If any ``goal:public_release`` mission has shipped without
a corresponding changelog_entries row (published=1 OR published=0) that
references its mission_id, surfaces a founder_action.

Handler contract
----------------
``handle(task, result) -> dict``

Returns one of:

- ``{"status": "ok", "reason": "entry_found"}``   — entry exists for mission
- ``{"status": "flagged", "mission_id": N, ...}``  — no entry → founder_action emitted
- ``{"status": "skip", "reason": str}``            — no product_id / no mission
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.changelog_freshness")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _has_changelog_entry_for_mission(mission_id: int) -> bool:
    """Return True if any changelog_entries row references this mission_id."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT entry_id FROM changelog_entries "
            "WHERE related_mission_ids_json LIKE ?",
            (f"%{mission_id}%",),
        )
        row = await cur.fetchone()
        return row is not None
    except Exception as exc:
        logger.warning(
            "changelog_freshness: _has_changelog_entry_for_mission failed",
            error=str(exc),
        )
        # Conservative: assume entry exists to avoid false positives
        return True


async def _get_mission_info(mission_id: int) -> dict | None:
    """Return mission row fields needed by the handler.

    The `goal` field is stored in the missions.context JSON (not a dedicated
    column) — we read title and status from the DB row; is_public_release is
    passed via the task context by the caller (workflow or posthook wiring).
    """
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT id, title, status FROM missions WHERE id=?",
            (mission_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "status": row[2],
        }
    except Exception as exc:
        logger.warning(
            "changelog_freshness: _get_mission_info failed",
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Founder-action emitter
# ---------------------------------------------------------------------------

async def _emit_founder_action(
    *,
    mission_id: int,
    mission_title: str,
    product_id: str,
) -> Any:
    """Emit a founder_action surfacing the missing changelog entry."""
    try:
        from src.founder_actions import create as fa_create
        return await fa_create(
            mission_id=mission_id,
            kind="generic",
            title=(
                f"Mission '{mission_title}' shipped without a changelog entry — publish one?"
            ),
            why=(
                f"Mission {mission_id} (goal=public_release) completed without a "
                f"corresponding changelog_entries row. Publish a changelog entry to "
                "keep your public changelog current."
            ),
            instructions=[
                "Run changelog/draft to auto-generate an entry from recent commits.",
                "Review and edit the draft entry.",
                "Run changelog/publish when satisfied.",
            ],
            expected_output_kind="ack_only",
            notify_telegram=False,
        )
    except Exception as exc:
        logger.warning(
            "changelog_freshness: _emit_founder_action failed", error=str(exc)
        )
        return None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

async def handle(task: dict, result: dict) -> dict[str, Any]:
    """changelog_freshness posthook handler."""
    task_id = task.get("id")
    mission_id: int | None = task.get("mission_id")

    # Parse task context
    ctx_raw = task.get("context", "{}")
    if isinstance(ctx_raw, str):
        try:
            ctx: dict = json.loads(ctx_raw)
        except Exception:
            ctx = {}
    elif isinstance(ctx_raw, dict):
        ctx = dict(ctx_raw)
    else:
        ctx = {}

    product_id: str = str(ctx.get("product_id") or "").strip()

    if not product_id:
        logger.debug(
            "changelog_freshness: no product_id in task context — skip",
            task_id=task_id,
        )
        return {"status": "skip", "reason": "no product_id in task context"}

    if not mission_id:
        logger.debug(
            "changelog_freshness: no mission_id on task — skip",
            task_id=task_id,
        )
        return {"status": "skip", "reason": "no mission_id on task"}

    # Check mission info
    mission = await _get_mission_info(mission_id)
    if mission is None:
        return {"status": "skip", "reason": f"mission {mission_id} not found"}

    # Only care about public_release missions that are completed.
    # is_public_release is passed via task context (set by workflow wiring
    # or the caller) since `goal` is not a dedicated column on missions.
    is_public_release = bool(ctx.get("is_public_release", False))
    if not is_public_release:
        return {
            "status": "ok",
            "reason": "not a public_release mission",
            "mission_id": mission_id,
        }
    if (mission.get("status") or "").lower() != "completed":
        return {
            "status": "ok",
            "reason": "mission not completed yet",
            "mission_id": mission_id,
        }

    # Check if a changelog entry already references this mission
    entry_exists = await _has_changelog_entry_for_mission(mission_id)
    if entry_exists:
        logger.info(
            "changelog_freshness: entry found for mission",
            mission_id=mission_id,
        )
        return {
            "status": "ok",
            "reason": "entry_found",
            "mission_id": mission_id,
        }

    # No entry → surface founder_action
    mission_title = mission.get("title") or f"Mission {mission_id}"
    logger.warning(
        "changelog_freshness: no changelog entry for completed public_release mission",
        mission_id=mission_id,
        product_id=product_id,
    )

    fa = await _emit_founder_action(
        mission_id=mission_id,
        mission_title=mission_title,
        product_id=product_id,
    )

    return {
        "status": "flagged",
        "mission_id": mission_id,
        "product_id": product_id,
        "mission_title": mission_title,
        "founder_action_id": getattr(fa, "id", None) if fa else None,
    }
