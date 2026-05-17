"""Yalayut Phase 4 — autonomous demand-signal drain.

``run_demand_drain()`` is the autonomous trigger the demand subsystem was
missing: it derives the ``repeat_pattern`` signal, then for every pattern
whose stacked confidence crosses ``DEMAND_DISCOVERY_THRESHOLD`` it runs an
on-demand discovery pass. ``on_demand_discovery`` marks the pattern
discovered, so a drained pattern drops out of the next sweep.

Folded into the ``yalayut_discovery`` daily mechanical executor — no new
orchestrator method, no new cron cadence row.
"""
from __future__ import annotations

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from yalayut.discovery import demand as _demand

logger = get_logger("yalayut.demand_drain")

#: A pattern with at least this many distinct un-discovered signal types is a
#: recurrence — worth one amplifying ``repeat_pattern`` signal.
REPEAT_PATTERN_MIN_TYPES: int = 3


async def _scan_repeat_patterns() -> int:
    """Reactive ``repeat_pattern`` derivation. For each un-discovered pattern
    with >= REPEAT_PATTERN_MIN_TYPES distinct *other* signal types, record one
    ``repeat_pattern`` signal. Returns the count recorded (deduped by the
    7-day cooldown). Best-effort — never raises into the drain."""
    added = 0
    try:
        db = await get_db()
        cur = await db.execute(
            "SELECT source_step_pattern, COUNT(DISTINCT signal_type) "
            "FROM yalayut_demand_signals "
            "WHERE resulted_in_discovery = 0 AND signal_type != 'repeat_pattern' "
            "GROUP BY source_step_pattern")
        rows = await cur.fetchall()
        await cur.close()
        for pattern, type_count in rows:
            if int(type_count) < REPEAT_PATTERN_MIN_TYPES:
                continue
            kw_cur = await db.execute(
                "SELECT intent_keywords_json FROM yalayut_demand_signals "
                "WHERE source_step_pattern = ? AND resulted_in_discovery = 0 "
                "LIMIT 1", (pattern,))
            kw_row = await kw_cur.fetchone()
            await kw_cur.close()
            import json
            try:
                keywords = json.loads(kw_row[0]) if kw_row and kw_row[0] else []
            except (json.JSONDecodeError, TypeError):
                keywords = []
            row_id = await _demand.record(
                source_step_pattern=pattern,
                intent_keywords=list(keywords),
                signal_type="repeat_pattern",
                confidence=0.3,
            )
            if row_id > 0:
                added += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("repeat_pattern scan failed: %s", exc)
    return added


async def run_demand_drain() -> dict:
    """Derive repeat_pattern, then drain every pattern at/above the
    discovery threshold through on_demand_discovery. Returns a summary."""
    summary = {
        "repeat_patterns_added": 0,
        "patterns_considered": 0,
        "patterns_discovered": 0,
        "errors": [],
    }
    summary["repeat_patterns_added"] = await _scan_repeat_patterns()

    pending = await _demand.pending_signals(limit=20)
    summary["patterns_considered"] = len(pending)
    for sig in pending:
        if sig["stacked_confidence"] < _demand.DEMAND_DISCOVERY_THRESHOLD:
            continue
        try:
            # Imported here (not at module top) so tests can monkeypatch
            # yalayut.discovery.on_demand.on_demand_discovery.
            from yalayut.discovery import on_demand as _on_demand
            await _on_demand.on_demand_discovery({
                "source_step_pattern": sig["source_step_pattern"],
                "intent_keywords": sig["intent_keywords"],
            })
            summary["patterns_discovered"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("on-demand drain failed for %s: %s",
                           sig["source_step_pattern"], exc)
            summary["errors"].append(f"{sig['source_step_pattern']}: {exc}")
    logger.info("run_demand_drain complete",
                repeat_patterns_added=summary["repeat_patterns_added"],
                patterns_considered=summary["patterns_considered"],
                patterns_discovered=summary["patterns_discovered"])
    return summary
