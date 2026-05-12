"""Z8 T4B — per-verb action cooldowns for the on-call agent.

Mr. Roboto's ``oncall_action`` executor calls :func:`check` **before**
delegating to the verb sub-handler and :func:`record` **after** the
sub-handler returns. Cooldowns are scoped to ``(mission_id, verb)``
because the same kind of incident on the same mission is the runaway
case we are guarding against (rollback-loop, restart-storm, drain-storm).

Default policy (in calls per window)
------------------------------------
- rollback_to_last_green — 2/hr, 8/day
- restart_service        — 5/hr
- scale_up               — 3/hr
- scale_down             — 3/hr
- drain_traffic          — 3/hr
- rotate_failed_key      — 1/day
- archive_flake_test     — 10/hr

Unknown verbs default to ``max_per_hour=999`` so additions to the on-call
whitelist do not silently lock out the agent — they need an explicit
policy row to get gated.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("ops.action_cooldowns")

DEFAULT_POLICY: dict[str, dict[str, int]] = {
    "rollback_to_last_green": {"max_per_hour": 2, "max_per_day": 8},
    "restart_service": {"max_per_hour": 5},
    "scale_up": {"max_per_hour": 3},
    "scale_down": {"max_per_hour": 3},
    "drain_traffic": {"max_per_hour": 3},
    "rotate_failed_key": {"max_per_day": 1},
    "archive_flake_test": {"max_per_hour": 10},
}


async def check(mission_id: int, verb: str) -> bool:
    """Return ``True`` when ``verb`` may execute now for ``mission_id``.

    Returns ``False`` when an applicable rate-limit window has been hit.
    """
    policy = DEFAULT_POLICY.get(verb, {"max_per_hour": 999})
    from src.infra.db import get_db

    db = await get_db()
    if "max_per_hour" in policy:
        async with db.execute(
            "SELECT COUNT(*) FROM action_cooldowns "
            "WHERE mission_id = ? AND verb = ? "
            "AND invoked_at >= datetime('now', '-1 hour')",
            (mission_id, verb),
        ) as cur:
            row = await cur.fetchone()
        n = int(row[0]) if row else 0
        if n >= int(policy["max_per_hour"]):
            logger.info(
                "cooldown blocked (hourly)",
                mission_id=mission_id,
                verb=verb,
                count=n,
                limit=policy["max_per_hour"],
            )
            return False
    if "max_per_day" in policy:
        async with db.execute(
            "SELECT COUNT(*) FROM action_cooldowns "
            "WHERE mission_id = ? AND verb = ? "
            "AND invoked_at >= datetime('now', '-1 day')",
            (mission_id, verb),
        ) as cur:
            row = await cur.fetchone()
        n = int(row[0]) if row else 0
        if n >= int(policy["max_per_day"]):
            logger.info(
                "cooldown blocked (daily)",
                mission_id=mission_id,
                verb=verb,
                count=n,
                limit=policy["max_per_day"],
            )
            return False
    return True


async def record(mission_id: int, verb: str, outcome: str) -> None:
    """Record a verb invocation against the (mission, verb) cooldown ledger."""
    from src.infra.db import get_db

    db = await get_db()
    await db.execute(
        "INSERT INTO action_cooldowns (mission_id, verb, invoked_at, outcome) "
        "VALUES (?, ?, datetime('now'), ?)",
        (mission_id, verb, outcome),
    )
    await db.commit()
