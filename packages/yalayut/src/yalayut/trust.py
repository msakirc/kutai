"""Source/owner trust → tier-cap mapping.

Per spec Tier classifier section:
  trust_cap = max(source_max, owner_max)   # owner elevates source
The lower the integer, the more trusted (T0 best). Owner promotion is always
manual (Telegram /yalayut owner promote) — this module only reads stored trust.
"""
from __future__ import annotations

import aiosqlite

# source-level automated trust → ceiling tier
SOURCE_MAX = {"trusted": 0, "review": 1, "untrusted": 2}

# owner trust_score thresholds → ceiling tier. Higher score = lower (better)
# tier. An owner not present in yalayut_owners offers no elevation (T3).
OWNER_MAX = [
    (0.8, 0),   # trust_score >= 0.8 -> T0
    (0.5, 1),   # >= 0.5 -> T1
    (0.0, 2),   # any recorded score -> T2
]
OWNER_MAX_FLOOR = 3  # unknown / very-low-trust owner


async def source_max_tier(db: aiosqlite.Connection, source_id: str) -> int:
    """Tier ceiling contributed by the source. Unknown source = untrusted."""
    cur = await db.execute(
        "SELECT trusted FROM yalayut_sources WHERE source_id = ?",
        (source_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return SOURCE_MAX["untrusted"]
    trusted = row["trusted"] if isinstance(row, aiosqlite.Row) else row[0]
    if trusted:
        return SOURCE_MAX["trusted"]
    return SOURCE_MAX["review"]


async def owner_max_tier(db: aiosqlite.Connection, owner: str | None) -> int:
    """Tier ceiling contributed by the owner allowlist."""
    if not owner:
        return OWNER_MAX_FLOOR
    cur = await db.execute(
        "SELECT trust_score FROM yalayut_owners WHERE owner_id = ?",
        (owner,),
    )
    row = await cur.fetchone()
    if row is None:
        return OWNER_MAX_FLOOR
    score = row["trust_score"] if isinstance(row, aiosqlite.Row) else row[0]
    score = score if score is not None else 0.0
    for threshold, tier in OWNER_MAX:
        if score >= threshold:
            return tier
    return OWNER_MAX_FLOOR
