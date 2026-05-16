"""Z7 A0 — Daily founder briefing job.

Runs at 09:00 founder-tz (registered in beckman cron_seed as
``_executor='daily_briefing'``).

Aggregates:
  - In-flight missions (status NOT IN terminal states)
  - Pending founder_actions
  - Cost burn since last 24h
Writes one ``mission_briefings`` row (kind='daily', product_id='__daily__').

Idempotent: if a row with kind='daily' already exists for today (DATE(prepared_at)
= DATE('now')), returns early to avoid duplicate rows.

Public API
----------
- ``run_daily_briefing()`` — called by mr_roboto for the ``daily_briefing`` executor
- ``sum_founder_minutes_saved(period_days)`` — sum of mission_events.founder_minutes_saved
  over the last N days; used by /founder_hours_saved
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("app.jobs.daily_briefing")

# product_id sentinel for daily briefings (not tied to a single mission)
_DAILY_PRODUCT_ID = "__daily__"


async def _in_flight_missions() -> list[dict]:
    """Return non-terminal missions."""
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT id, title, COALESCE(status, 'unknown') AS status "
        "FROM missions "
        "WHERE COALESCE(status, '') NOT IN "
        "  ('completed', 'failed', 'cancelled', 'archived') "
        "ORDER BY id DESC LIMIT 20"
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in await cur.fetchall()]


async def _pending_founder_actions() -> tuple[list[dict], list[dict]]:
    """Return (active, resurfaced) founder_actions split by defer_until.

    - ``active``: pending/in_progress cards that are not deferred past now.
    - ``resurfaced``: cards that *were* deferred but whose review window
      has now arrived (``defer_until <= now``). These are surfaced as a
      distinct section so deferred attention cards re-enter the A0
      briefing when their window comes due — wiring up the
      ``attention_budget.next_review_window`` deferral cycle.
    """
    from src.infra.db import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT id, mission_id, kind, title, status, defer_until "
            "FROM founder_actions "
            "WHERE status IN ('pending', 'in_progress') "
            "ORDER BY id DESC LIMIT 30"
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in await cur.fetchall()]
    except Exception:
        return [], []

    import datetime as _dt

    now = _dt.datetime.utcnow()
    active: list[dict] = []
    resurfaced: list[dict] = []
    for r in rows:
        defer_until = r.get("defer_until")
        if defer_until:
            try:
                dt = _dt.datetime.strptime(defer_until, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                dt = None
            if dt is not None and dt > now:
                # Still deferred — its review window has not arrived; skip.
                continue
            if dt is not None and dt <= now:
                # Review window reached — re-surface this card.
                resurfaced.append(r)
                continue
        active.append(r)

    return active[:10], resurfaced[:10]


async def _cost_burn_last_24h() -> float:
    """Sum model_call_tokens.cost_usd for the last 24 hours."""
    from src.infra.db import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) "
            "FROM model_call_tokens "
            "WHERE called_at >= datetime('now', '-1 day')"
        )
        row = await cur.fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0


async def _already_ran_today() -> bool:
    """True if a daily briefing row exists for today."""
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM mission_briefings "
        "WHERE kind='daily' AND DATE(prepared_at)=DATE('now') "
        "LIMIT 1"
    )
    return await cur.fetchone() is not None


def _compose_daily_body(
    *,
    missions: list[dict],
    actions: list[dict],
    resurfaced: list[dict],
    cost_usd: float,
) -> str:
    sections: list[str] = []

    # In-flight missions
    if missions:
        lines = ["## In-Flight Missions"]
        for m in missions:
            lines.append(f"- **#{m['id']} {m['title']}** — {m['status']}")
        sections.append("\n".join(lines))
    else:
        sections.append("## In-Flight Missions\n_(none running)_")

    # Resurfaced (deferred cards whose review window has arrived)
    if resurfaced:
        lines = ["## Resurfaced — Deferred Review Window Reached"]
        for a in resurfaced:
            lines.append(
                f"- [{a['kind']}] #{a['id']} — {a['title']} "
                f"(was deferred to {a.get('defer_until') or '?'})"
            )
        sections.append("\n".join(lines))

    # Pending founder actions
    if actions:
        lines = ["## Pending Founder Actions"]
        for a in actions:
            lines.append(f"- [{a['kind']}] #{a['id']} — {a['title']} ({a['status']})")
        sections.append("\n".join(lines))
    else:
        sections.append("## Pending Founder Actions\n_(queue is clear)_")

    # Cost burn
    cost_line = f"${cost_usd:.4f}" if cost_usd else "$0.0000"
    sections.append(f"## Cost Burn (last 24h)\n{cost_line}")

    return "\n\n".join(sections)


async def run_daily_briefing() -> dict:
    """Main entry point. Returns {"ok": True} on success, {"ok": False, "reason": ...} on failure."""
    try:
        if await _already_ran_today():
            logger.info("daily_briefing: already ran today, skipping")
            return {"ok": True, "skipped": True, "reason": "already_ran_today"}

        missions = await _in_flight_missions()
        actions, resurfaced = await _pending_founder_actions()
        cost_usd = await _cost_burn_last_24h()

        body_md = _compose_daily_body(
            missions=missions,
            actions=actions,
            resurfaced=resurfaced,
            cost_usd=cost_usd,
        )

        from src.infra.db import get_db
        db = await get_db()
        await db.execute(
            "INSERT INTO mission_briefings "
            "(product_id, mission_id, kind, body_md, prepared_at) "
            "VALUES (?, NULL, 'daily', ?, datetime('now'))",
            (_DAILY_PRODUCT_ID, body_md),
        )
        await db.commit()

        logger.info(
            "daily_briefing: wrote daily briefing",
            missions_count=len(missions),
            actions_count=len(actions),
            resurfaced_count=len(resurfaced),
            cost_usd=cost_usd,
        )
        return {
            "ok": True,
            "missions": len(missions),
            "actions": len(actions),
            "resurfaced": len(resurfaced),
        }

    except Exception as exc:
        logger.error("daily_briefing: failed", error=str(exc))
        return {"ok": False, "reason": str(exc)}


async def sum_founder_minutes_saved(period_days: int = 7) -> int:
    """Sum mission_events.founder_minutes_saved over the last period_days days.

    Returns the total as an integer (minutes). Returns 0 on any error or when
    no rows match.
    """
    from src.infra.db import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT COALESCE(SUM(founder_minutes_saved), 0) "
            "FROM mission_events "
            "WHERE founder_minutes_saved IS NOT NULL "
            "  AND posted_at >= datetime('now', ?)",
            (f"-{int(period_days)} days",),
        )
        row = await cur.fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:
        logger.warning("sum_founder_minutes_saved: failed", error=str(exc))
        return 0
