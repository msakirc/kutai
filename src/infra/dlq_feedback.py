"""DLQ feedback hook — Z9 Growth T3D.

Mines recurring failure patterns out of the dead-letter queue and surfaces
them as ``growth_events`` rows (kind=``dlq_pattern``). The weekly analytics
digest (Z9 T2) reads ``growth_events`` so DLQ failure clusters land in the
founder digest alongside product signals — closing the loop on what was a
manual ``/dead`` dead-end.

This is a deterministic, no-LLM pattern counter:

  group recent unresolved DLQ rows by (error_category, original_agent)
  -> any group with >= MIN_OCCURRENCES rows becomes a dlq_pattern event.

Idempotency
-----------
Re-running within the review window must not duplicate rows. Each pattern
carries a stable ``pattern_key``; before writing we scan recent
``dlq_pattern`` growth_events and skip keys already emitted inside
``DEDUP_WINDOW_SECONDS``.

Wired weekly via the ``dlq_signal_review`` internal cadence (see
``general_beckman.cron_seed``); fired as a mechanical ``mine_dlq_patterns``
executor.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("infra.dlq_feedback")

# A group must have at least this many DLQ rows to count as a pattern.
MIN_OCCURRENCES = 3

# Sliding window for which DLQ rows are mined (matches the weekly cadence).
WINDOW_DAYS = 7

# A pattern_key already emitted inside this window is not re-emitted.
DEDUP_WINDOW_SECONDS = 604800  # 7 days

GROWTH_KIND = "dlq_pattern"


def _pattern_key(error_category: str, agent: str, task_type: str) -> str:
    """Stable identity for a failure cluster; used for idempotent dedup."""
    return f"{error_category}|{agent}|{task_type}"


async def _recent_pattern_keys() -> set[str]:
    """pattern_keys already emitted as dlq_pattern events inside the window."""
    from src.infra.db import get_growth_events

    events = await get_growth_events(kind=GROWTH_KIND)
    keys: set[str] = set()
    # get_growth_events returns most-recent-first; the dedup window is a week
    # and the cron also runs weekly, so an unbounded scan of one kind is cheap.
    # Filter by occurred_at against the window to stay correct if the digest
    # cadence ever tightens.
    import datetime as _dt

    cutoff = _dt.datetime.utcnow() - _dt.timedelta(seconds=DEDUP_WINDOW_SECONDS)
    for ev in events:
        occurred = ev.get("occurred_at")
        if occurred:
            try:
                ts = _dt.datetime.strptime(str(occurred), "%Y-%m-%d %H:%M:%S")
                if ts < cutoff:
                    continue
            except (ValueError, TypeError):
                pass  # unparseable timestamp — keep it (conservative dedup)
        props = ev.get("properties_json")
        if isinstance(props, dict):
            key = props.get("pattern_key")
            if key:
                keys.add(str(key))
    return keys


async def mine_dlq_patterns() -> int:
    """Scan recent DLQ rows, emit dlq_pattern growth_events for clusters.

    Groups unresolved dead_letter_tasks rows from the last ``WINDOW_DAYS`` by
    (error_category, original_agent). Each group with at least
    ``MIN_OCCURRENCES`` rows is written as one ``growth_events`` row with
    ``kind="dlq_pattern"`` — unless the same ``pattern_key`` was already
    emitted inside the dedup window (idempotency).

    ``mission_id`` is left NULL: DLQ patterns are a global signal, not bound
    to one mission (``growth_events.mission_id`` is nullable).

    Returns the number of new dlq_pattern rows written.
    """
    from src.infra.db import get_db
    from general_beckman import record_growth_event

    # Make sure the DLQ table exists even on a fresh DB / test bootstrap.
    try:
        from src.infra.dead_letter import _ensure_dlq_table

        await _ensure_dlq_table()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("dlq_feedback: _ensure_dlq_table skipped: %s", exc)

    db = await get_db()
    cursor = await db.execute(
        """
        SELECT d.task_id, d.mission_id, d.error, d.error_category,
               d.original_agent, d.quarantined_at,
               COALESCE(t.agent_type, '') AS task_type
        FROM dead_letter_tasks d
        LEFT JOIN tasks t ON t.id = d.task_id
        WHERE d.resolved_at IS NULL
          AND d.quarantined_at >= datetime('now', ?)
        ORDER BY d.quarantined_at ASC
        """,
        (f"-{WINDOW_DAYS} days",),
    )
    rows = await cursor.fetchall()

    if not rows:
        logger.info("dlq_feedback: no unresolved DLQ rows in window")
        return 0

    # Group by (error_category, original_agent). task_type is recorded as the
    # dominant agent_type within the group (informational; not a group key —
    # keeps clusters from fragmenting when a category spans task types).
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        d = dict(row)
        category = d.get("error_category") or "unknown"
        agent = d.get("original_agent") or "executor"
        groups[(category, agent)].append(d)

    already_emitted = await _recent_pattern_keys()
    written = 0

    for (error_category, agent), entries in groups.items():
        if len(entries) < MIN_OCCURRENCES:
            continue

        # Dominant task_type within the cluster.
        type_counts: dict[str, int] = defaultdict(int)
        for e in entries:
            type_counts[e.get("task_type") or "unknown"] += 1
        task_type = max(type_counts.items(), key=lambda kv: kv[1])[0]

        key = _pattern_key(error_category, agent, task_type)
        if key in already_emitted:
            logger.debug("dlq_feedback: pattern %s already emitted; skipping", key)
            continue

        sample_task_ids = [e["task_id"] for e in entries][:20]
        # entries are ordered ASC by quarantined_at.
        first_seen = entries[0].get("quarantined_at")
        last_seen = entries[-1].get("quarantined_at")

        properties = {
            "pattern_key": key,
            "error_category": error_category,
            "agent": agent,
            "task_type": task_type,
            "occurrence_count": len(entries),
            "sample_task_ids": sample_task_ids,
            "first_seen": first_seen,
            "last_seen": last_seen,
        }

        try:
            await record_growth_event(
                mission_id=None,  # global signal — not mission-scoped
                kind=GROWTH_KIND,
                properties=properties,
            )
            already_emitted.add(key)
            written += 1
            logger.info(
                "dlq_feedback: emitted dlq_pattern key=%s count=%d",
                key,
                len(entries),
            )
        except Exception as exc:
            logger.warning(
                "dlq_feedback: failed to write growth_event for %s: %s",
                key,
                exc,
            )

    logger.info("dlq_feedback: mine_dlq_patterns wrote %d new pattern(s)", written)
    return written


if __name__ == "__main__":  # pragma: no cover - manual CLI
    import asyncio as _asyncio

    async def _main() -> None:
        import fatih_hoca  # noqa: F401 — registers the 5 registry tables on a fresh DB
        from src.infra.db import init_db

        await init_db()
        n = await mine_dlq_patterns()
        print(f"Wrote {n} dlq_pattern growth_event(s).")

    _asyncio.run(_main())
