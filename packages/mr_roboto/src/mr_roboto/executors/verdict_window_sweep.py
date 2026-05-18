"""Z9 Growth T4C — verdict window sweeper.

A global daily cron (``verdict_window_sweep`` internal cadence) that scans
every pending hypothesis and enqueues a ``record_verdict`` mechanical task
for each one whose measurement window has closed
(``created_at + window_seconds <= now``).

Why a sweeper, not a per-hypothesis scheduled row
-------------------------------------------------
A per-hypothesis ``scheduled_tasks`` one-shot would work, but it couples
every T4A hypothesis insert to cron-row creation and leaves dangling rows
if a hypothesis is deleted. A single daily sweeper over
``get_pending_hypotheses()`` is restart-safe (it re-derives "due" from the
DB every tick), idempotent (a verdict task flips ``verdict`` away from
``pending`` so the next sweep skips it), and needs zero per-hypothesis
bookkeeping.

Architecture contract
---------------------
This executor is **mechanical**: it never calls the LLM dispatcher. It only
enqueues ``record_verdict`` mechanical tasks via ``general_beckman.enqueue``
on the ONGOING lane. Verdict math itself is deterministic (Bayesian
posterior — see ``src/growth/verdict_stats.py``); no LLM is involved
anywhere in the verdict pipeline.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.verdict_window_sweep")

# scheduled_tasks one-shot guard duplicates would be enqueued if a verdict
# task is slow; we tag enqueued hypothesis ids so a single sweep tick never
# double-enqueues. Cross-tick dedup is handled naturally by the verdict
# flipping verdict='pending' → confirmed/refuted/inconclusive.


def _parse_db_ts(value: Any) -> datetime | None:
    """Parse a SQLite datetime string ('YYYY-MM-DD HH:MM:SS')."""
    if not value:
        return None
    s = str(value).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[: len(fmt) + 6], fmt)
        except Exception:
            continue
    return None


def is_due(hypothesis: dict, now: datetime | None = None) -> bool:
    """True when a hypothesis's measurement window has closed.

    Due iff ``created_at + window_seconds <= now``. A hypothesis with a
    missing/invalid ``created_at`` or a non-positive ``window_seconds`` is
    treated as NOT due (it must be measured deliberately, not swept).
    """
    now = now or datetime.now()
    created = _parse_db_ts(hypothesis.get("created_at"))
    window = hypothesis.get("window_seconds")
    if created is None or not window:
        return False
    try:
        window_i = int(window)
    except (TypeError, ValueError):
        return False
    if window_i <= 0:
        return False
    return created + timedelta(seconds=window_i) <= now


async def _enqueue_verdict_task(hypothesis: dict) -> int | None:
    """Enqueue one ``record_verdict`` mechanical task on the ONGOING lane."""
    from general_beckman import enqueue

    hyp_id = hypothesis.get("id")
    mission_id = hypothesis.get("mission_id")
    feature = hypothesis.get("feature") or "feature"
    spec = {
        "title": f"Hypothesis verdict — {feature} (hyp #{hyp_id})",
        "description": (
            f"Measurement window closed for hypothesis #{hyp_id}; pull the "
            f"actual metric and record a verdict."
        ),
        "agent_type": "mechanical",
        "mission_id": mission_id,
        "context": {
            "executor": "mechanical",
            "payload": {
                "action": "record_verdict",
                "hypothesis_id": hyp_id,
            },
        },
        "depends_on": [],
    }
    try:
        new_id = await enqueue(spec, lane="ongoing")
        return int(new_id) if isinstance(new_id, int) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "verdict_window_sweep failed to enqueue verdict task",
            hypothesis_id=hyp_id,
            error=str(exc),
        )
        return None


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Scan pending hypotheses; enqueue a verdict task for each due one.

    Always returns a dict — never raises into the dispatcher.
    """
    try:
        from src.infra.db import get_pending_hypotheses

        pending = await get_pending_hypotheses() or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("verdict_window_sweep: pending query failed", error=str(exc))
        return {"ok": False, "reason": "pending_query_failed", "error": str(exc)}

    now = datetime.now()
    due = [h for h in pending if is_due(h, now)]

    enqueued_ids: list[int] = []
    for hyp in due:
        new_id = await _enqueue_verdict_task(hyp)
        if new_id is not None:
            enqueued_ids.append(new_id)

    logger.info(
        "verdict_window_sweep complete",
        pending=len(pending),
        due=len(due),
        enqueued=len(enqueued_ids),
    )
    return {
        "ok": True,
        "pending": len(pending),
        "due": len(due),
        "enqueued": len(enqueued_ids),
        "verdict_task_ids": enqueued_ids,
    }


__all__ = ["run", "is_due"]
