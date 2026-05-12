"""Z8 T4D — per-mission tier→channel routing for escalations.

Schema lives in :mod:`src.infra.db` (migration ``2026-05-12-escalation-policy``).

Tiers (mapping intentionally loose so playbooks can pass either form):

- ``low`` / ``medium`` → tier1
- ``high`` → tier2
- ``critical`` (or ``sec_critical``) → tier3

Quiet-hours
-----------
``quiet_hours_start``/``quiet_hours_end`` are 24h ``HH:MM`` strings in the
policy's timezone. During quiet hours, **non-critical** alerts collapse to
``telegram_log_only`` (write to the mission thread but no push notification);
critical alerts always page on the tier3 channel regardless of quiet hours.

Default policy (when no row exists for the mission): tier1=tier2=telegram,
tier3=sms; no quiet hours; UTC.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("ops.escalation_policy")


@dataclass
class Policy:
    mission_id: int
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None
    tier1_channel: str = "telegram"
    tier2_channel: str = "telegram"
    tier3_channel: str = "sms"
    tz: str = "UTC"


DEFAULTS = Policy(mission_id=0)


def tier_of(severity: str) -> int:
    """Map severity string to tier (1/2/3)."""
    s = (severity or "").lower()
    if s in ("critical", "sec_critical", "tier3"):
        return 3
    if s in ("high", "tier2"):
        return 2
    return 1


async def load_policy(mission_id: int) -> Policy:
    """Return the escalation policy for a mission, defaults if no row exists."""
    from src.infra.db import get_db

    db = await get_db()
    async with db.execute(
        "SELECT mission_id, quiet_hours_start, quiet_hours_end, "
        "tier1_channel, tier2_channel, tier3_channel, tz "
        "FROM escalation_policy WHERE mission_id = ?",
        (int(mission_id),),
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return Policy(mission_id=int(mission_id))
    return Policy(
        mission_id=int(row[0]),
        quiet_hours_start=row[1],
        quiet_hours_end=row[2],
        tier1_channel=row[3] or "telegram",
        tier2_channel=row[4] or "telegram",
        tier3_channel=row[5] or "sms",
        tz=row[6] or "UTC",
    )


async def set_policy(policy: Policy) -> None:
    """Upsert the policy for a mission."""
    from src.infra.db import get_db

    db = await get_db()
    await db.execute(
        "INSERT INTO escalation_policy "
        "(mission_id, quiet_hours_start, quiet_hours_end, "
        " tier1_channel, tier2_channel, tier3_channel, tz) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(mission_id) DO UPDATE SET "
        " quiet_hours_start=excluded.quiet_hours_start, "
        " quiet_hours_end=excluded.quiet_hours_end, "
        " tier1_channel=excluded.tier1_channel, "
        " tier2_channel=excluded.tier2_channel, "
        " tier3_channel=excluded.tier3_channel, "
        " tz=excluded.tz",
        (
            policy.mission_id,
            policy.quiet_hours_start,
            policy.quiet_hours_end,
            policy.tier1_channel,
            policy.tier2_channel,
            policy.tier3_channel,
            policy.tz,
        ),
    )
    await db.commit()


def _parse_hhmm(s: str | None) -> tuple[int, int] | None:
    if not s or ":" not in s:
        return None
    try:
        h, m = s.split(":", 1)
        return int(h), int(m)
    except (ValueError, TypeError):
        return None


def in_quiet_hours(policy: Policy, now: datetime | None = None) -> bool:
    """True when ``now`` (UTC by default) falls inside the policy's quiet window.

    Supports overnight windows where ``end < start`` (e.g. 22:00→06:00).
    Missing/unparseable values mean "no quiet hours" — returns False.
    """
    start = _parse_hhmm(policy.quiet_hours_start)
    end = _parse_hhmm(policy.quiet_hours_end)
    if start is None or end is None:
        return False
    now = now or datetime.utcnow()
    cur = (now.hour, now.minute)
    if start <= end:
        return start <= cur < end
    # overnight wrap-around
    return cur >= start or cur < end


def channel_for(policy: Policy, severity: str, now: datetime | None = None) -> str:
    """Resolve the dispatch channel for ``severity`` given quiet hours."""
    tier = tier_of(severity)
    base = {
        1: policy.tier1_channel,
        2: policy.tier2_channel,
        3: policy.tier3_channel,
    }[tier]
    # Quiet hours muffle everything except tier-3 critical.
    if tier < 3 and in_quiet_hours(policy, now):
        return "telegram_log_only"
    return base


async def resolve_channel(
    mission_id: int,
    severity: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One-call helper: load policy + return ``{channel, tier, quiet}``."""
    policy = await load_policy(int(mission_id))
    tier = tier_of(severity)
    quiet = in_quiet_hours(policy, now)
    return {
        "channel": channel_for(policy, severity, now),
        "tier": tier,
        "quiet": quiet,
        "policy": {
            "tier1_channel": policy.tier1_channel,
            "tier2_channel": policy.tier2_channel,
            "tier3_channel": policy.tier3_channel,
        },
    }
