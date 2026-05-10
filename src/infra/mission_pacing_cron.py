"""Z10 T3A — pacing-driven tradeoff prompt cron logic.

Runs every 30 minutes (registered in beckman cron_seed). For each
running mission with a time budget, checks ``compute_mission_pacing``;
if ``tradeoff_due`` and no row in ``mission_tradeoff_prompts`` for
``(mission_id, today)``, posts a single ``[asking]`` event suggesting
a ~30%-cut of remaining tasks ranked by ``created_at``.

NOTE: Z9 (growth zone) owns proper founder-priority ordering. Here we
fall back to ``created_at`` ASC — older tasks first — and tag the
ranking inline in the [asking] payload as a TODO.
"""
from __future__ import annotations

from typing import Any

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.infra.pacing import compute_mission_pacing

logger = get_logger("infra.mission_pacing_cron")

# Cut roughly this fraction of the remaining estimated cost.
TARGET_CUT_FRACTION = 0.30


async def _running_missions_with_budget() -> list[dict]:
    """Missions whose status is non-terminal AND ``time_budget_hours`` set."""
    db = await get_db()
    cur = await db.execute(
        "SELECT id, title, time_budget_hours "
        "FROM missions "
        "WHERE time_budget_hours IS NOT NULL "
        "  AND COALESCE(status, '') NOT IN "
        "    ('completed', 'failed', 'cancelled', 'archived')"
    )
    return [dict(r) for r in await cur.fetchall()]


async def _has_open_tradeoff_today(mission_id: int) -> bool:
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM mission_tradeoff_prompts "
        "WHERE mission_id = ? AND DATE(posted_at) = DATE('now')",
        (int(mission_id),),
    )
    return await cur.fetchone() is not None


async def _suggest_cut(mission_id: int) -> dict:
    """Pick a slate of remaining tasks summing to ~30% of remaining cost.

    Returns ``{"tasks": [{id, title, est}], "total_est": float}``.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT id, title, "
        "       COALESCE(estimated_cost_usd, 0) AS est, "
        "       created_at "
        "FROM tasks "
        "WHERE mission_id = ? "
        "  AND COALESCE(status, '') NOT IN "
        "    ('completed', 'failed', 'cancelled', 'skipped') "
        # TODO: Z9 owns proper founder-priority ordering. For now, the
        # loose proxy is created_at ASC (older = lower priority).
        "ORDER BY created_at ASC",
        (int(mission_id),),
    )
    rows = [dict(r) for r in await cur.fetchall()]
    if not rows:
        return {"tasks": [], "total_est": 0.0}

    total_remaining = sum(float(r["est"] or 0.0) for r in rows)
    if total_remaining <= 0:
        # Fall back to ~30% by task count.
        cut_n = max(1, int(round(len(rows) * TARGET_CUT_FRACTION)))
        chosen = rows[:cut_n]
        return {
            "tasks": [
                {"id": r["id"], "title": r["title"], "est": 0.0}
                for r in chosen
            ],
            "total_est": 0.0,
        }

    target_cut = total_remaining * TARGET_CUT_FRACTION
    chosen: list[dict] = []
    acc = 0.0
    for r in rows:
        chosen.append({
            "id": r["id"],
            "title": r["title"],
            "est": float(r["est"] or 0.0),
        })
        acc += float(r["est"] or 0.0)
        if acc >= target_cut:
            break
    return {"tasks": chosen, "total_est": acc}


async def _record_prompt(
    mission_id: int, mission_event_id: int | None,
) -> None:
    """Insert (or no-op if today already has one) into the prompt log."""
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO mission_tradeoff_prompts "
            "(mission_id, mission_event_id) VALUES (?, ?)",
            (int(mission_id), mission_event_id),
        )
        await db.commit()
    except Exception as e:
        # UNIQUE violation = another tick won the race; treat as idempotent.
        logger.debug(f"mission_tradeoff_prompts insert skipped: {e}")


async def _post_asking(
    mission_id: int, payload: dict,
) -> int | None:
    """Post via mission_events.post_event if telegram is available.

    Falls back to logging when the bot is not initialised (unit tests).
    Returns the mission_events row id or None.
    """
    bot: Any = None
    try:
        # Late, optional import: TelegramInterface owns the bot handle.
        from src.app.telegram_bot import TelegramInterface  # type: ignore
        bot = getattr(TelegramInterface, "_singleton_bot", None)
    except Exception:
        bot = None

    try:
        from src.app.mission_events import post_event
        # post_event handles a None bot gracefully (post fails, row kept).
        return await post_event(bot, mission_id, "asking", payload)
    except Exception as e:
        logger.warning(
            "tradeoff post_event failed (no event row created)",
            mission_id=mission_id, error=str(e),
        )
        return None


async def check_and_post_tradeoff_prompts() -> int:
    """Main entry point. Returns the number of NEW prompts posted."""
    posted = 0
    missions = await _running_missions_with_budget()
    for m in missions:
        mid = int(m["id"])
        if await _has_open_tradeoff_today(mid):
            continue
        try:
            pacing = await compute_mission_pacing(mid)
        except Exception as e:
            logger.warning(
                "pacing compute failed",
                mission_id=mid, error=str(e),
            )
            continue
        if not pacing.get("tradeoff_due"):
            continue

        suggestion = await _suggest_cut(mid)
        title = (m.get("title") or f"Mission {mid}")[:50]
        burn_pct = int(round(float(pacing["percent_burn"]) * 100))
        scope_pct = int(round(float(pacing["scope_remaining_pct"]) * 100))
        cut_lines = []
        for t in suggestion["tasks"]:
            est_part = (
                f" — ${t['est']:.2f}" if t["est"] > 0 else ""
            )
            cut_lines.append(f"#{t['id']} {t['title'][:60]}{est_part}")
        cut_text = (
            "\n".join(f"  • {line}" for line in cut_lines)
            if cut_lines else "  (no remaining tasks to cut)"
        )
        question = (
            f"Mission '{title}' is at {burn_pct}% burn with "
            f"{scope_pct}% scope remaining. Suggested ~30% cut "
            f"(ranked by created_at; Z9 priority order TODO):\n"
            f"{cut_text}\n\n"
            f"Approve the cut, modify, or push the deadline?"
        )
        payload = {
            "question": question,
            "options": ["approve_cut", "modify", "extend_deadline"],
            "mission_id": mid,
            "burn_pct": burn_pct,
            "scope_remaining_pct": scope_pct,
            "suggested_cut": suggestion["tasks"],
            "kind_subtype": "tradeoff_prompt",
        }
        event_id = await _post_asking(mid, payload)
        await _record_prompt(mid, event_id)
        posted += 1
    return posted
