"""Yalayut Phase 4 — demand-signal subsystem.

Seven signal types feed one queue with confidence stacking, dedupe by
``source_step_pattern`` and a per-(pattern, type) cooldown window.

Signal types (spec — 4 proactive + 3 reactive):
  proactive: planning_miss, step_entry_miss, tool_call, founder
  reactive:  hint_miss, dlq, repeat_pattern
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta

from dabidabi import get_db
from yazbunu import get_logger
from dabidabi.times import utc_now, to_db

logger = get_logger("yalayut.demand")

SIGNAL_TYPES: frozenset[str] = frozenset({
    "planning_miss", "step_entry_miss", "tool_call", "founder",
    "hint_miss", "dlq", "repeat_pattern",
})

#: Per-(pattern, type) cooldown — spec "start at 7d per source_step_pattern".
COOLDOWN_SECONDS: int = 7 * 24 * 3600

#: Stacked-confidence threshold a pattern must cross before an autonomous
#: on-demand discovery run is triggered for it. Shared with source_scout's
#: demand web-search scan so both autonomy paths agree on "high confidence".
DEMAND_DISCOVERY_THRESHOLD: float = 0.5


@dataclass
class DemandSignal:
    """One fired demand signal. ``confidence`` is the per-signal weight;
    the stacked confidence across signals is computed by ``stack_confidence``."""
    source_step_pattern: str
    intent_keywords: list[str]
    signal_type: str
    confidence: float = 0.3
    fired_at: str = field(default_factory=lambda: to_db(utc_now()))

    def __post_init__(self) -> None:
        if self.signal_type not in SIGNAL_TYPES:
            raise ValueError(f"unknown signal_type: {self.signal_type!r}")
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


async def _within_cooldown(pattern: str, signal_type: str) -> bool:
    """True when a same-(pattern, type) signal fired inside the cooldown."""
    db = await get_db()
    cutoff = to_db(utc_now() - timedelta(seconds=COOLDOWN_SECONDS))
    cur = await db.execute(
        "SELECT 1 FROM yalayut_demand_signals "
        "WHERE source_step_pattern = ? AND signal_type = ? AND fired_at >= ? "
        "LIMIT 1",
        (pattern, signal_type, cutoff),
    )
    row = await cur.fetchone()
    await cur.close()
    return row is not None


async def record_signal(sig: DemandSignal) -> int:
    """Insert a demand signal. Returns the new row id, or ``-1`` when the
    signal is deduped (same pattern + type already within cooldown)."""
    if await _within_cooldown(sig.source_step_pattern, sig.signal_type):
        logger.debug(
            "demand signal deduped (cooldown)",
            pattern=sig.source_step_pattern, type=sig.signal_type,
        )
        return -1
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO yalayut_demand_signals "
        "(source_step_pattern, intent_keywords_json, signal_type, "
        " confidence, fired_at, resulted_in_discovery) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (sig.source_step_pattern, json.dumps(sig.intent_keywords),
         sig.signal_type, sig.confidence, sig.fired_at),
    )
    await db.commit()
    logger.info("demand signal recorded", pattern=sig.source_step_pattern,
                type=sig.signal_type, confidence=sig.confidence)
    return int(cur.lastrowid)


async def stack_confidence(pattern: str) -> float:
    """Combine all un-discovered signals for one pattern into a single
    confidence via independent-probability stacking:
    ``1 - Π(1 - c_i)``. Bounded in [0, 1]."""
    db = await get_db()
    cur = await db.execute(
        "SELECT confidence FROM yalayut_demand_signals "
        "WHERE source_step_pattern = ? AND resulted_in_discovery = 0",
        (pattern,),
    )
    rows = await cur.fetchall()
    await cur.close()
    miss = 1.0
    for (c,) in rows:
        miss *= (1.0 - max(0.0, min(1.0, float(c or 0.0))))
    return round(1.0 - miss, 6)


async def pending_signals(limit: int = 20) -> list[dict]:
    """Return distinct un-discovered patterns with their stacked confidence
    and merged intent keywords, ordered by stacked confidence descending."""
    db = await get_db()
    cur = await db.execute(
        "SELECT DISTINCT source_step_pattern FROM yalayut_demand_signals "
        "WHERE resulted_in_discovery = 0",
    )
    patterns = [r[0] for r in await cur.fetchall()]
    await cur.close()
    out: list[dict] = []
    for pat in patterns:
        stacked = await stack_confidence(pat)
        kw_cur = await db.execute(
            "SELECT intent_keywords_json FROM yalayut_demand_signals "
            "WHERE source_step_pattern = ? AND resulted_in_discovery = 0",
            (pat,),
        )
        merged: set[str] = set()
        for (kj,) in await kw_cur.fetchall():
            try:
                for k in json.loads(kj or "[]"):
                    merged.add(str(k))
            except (json.JSONDecodeError, TypeError):
                continue
        await kw_cur.close()
        out.append({
            "source_step_pattern": pat,
            "stacked_confidence": stacked,
            "intent_keywords": sorted(merged),
        })
    out.sort(key=lambda d: d["stacked_confidence"], reverse=True)
    return out[:limit]


async def mark_discovered(pattern: str) -> None:
    """Flip ``resulted_in_discovery`` for every signal on a pattern once an
    on-demand discovery run consumed it."""
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_demand_signals SET resulted_in_discovery = 1 "
        "WHERE source_step_pattern = ?",
        (pattern,),
    )
    await db.commit()


async def record(
    *,
    source_step_pattern: str,
    intent_keywords: list[str],
    signal_type: str,
    confidence: float = 0.3,
) -> int:
    """Kwargs convenience over ``record_signal`` — build a ``DemandSignal``
    and insert it. Returns the new row id, or ``-1`` when deduped. Lets a
    firing site fire one signal without importing ``DemandSignal`` itself."""
    return await record_signal(DemandSignal(
        source_step_pattern=source_step_pattern,
        intent_keywords=list(intent_keywords),
        signal_type=signal_type,
        confidence=confidence,
    ))
