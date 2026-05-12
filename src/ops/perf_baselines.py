"""Z8 T5F — perf_baselines storage + regression diff.

Public API
----------
``record_baseline(mission_id, release_tag, metric, p50, p95, p99)``
    Append one row to ``perf_baselines``.

``latest_green_baseline(mission_id, metric)``
    Return the most recent baseline row for ``(mission_id, metric)``. The
    "green" filter is implicit — callers persist only confirmed-green runs.
    Returns ``None`` if no prior row exists.

``regression_pct(baseline, current, statistic='p95')``
    Pure helper: percent change relative to baseline. Positive = slower.

``has_regression(baseline, current, threshold_pct=10.0)``
    True when p50/p95/p99 of ``current`` is more than ``threshold_pct`` slower
    than ``baseline``. A missing statistic on either side never trips.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("ops.perf_baselines")


@dataclass
class Baseline:
    mission_id: int
    release_tag: str
    metric: str
    p50: float | None
    p95: float | None
    p99: float | None
    recorded_at: str | None = None


async def record_baseline(
    mission_id: int,
    release_tag: str,
    metric: str,
    *,
    p50: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
) -> int:
    """Append one row to ``perf_baselines``. Returns the new row id."""
    from src.infra.db import get_db

    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO perf_baselines "
        "(mission_id, release_tag, metric, p50, p95, p99) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (int(mission_id), str(release_tag), str(metric), p50, p95, p99),
    )
    await db.commit()
    return int(cursor.lastrowid or 0)


async def latest_green_baseline(
    mission_id: int, metric: str,
) -> Baseline | None:
    """Return the most recent baseline for ``(mission_id, metric)`` or None."""
    from src.infra.db import get_db

    db = await get_db()
    async with db.execute(
        "SELECT mission_id, release_tag, metric, p50, p95, p99, recorded_at "
        "FROM perf_baselines WHERE mission_id = ? AND metric = ? "
        "ORDER BY id DESC LIMIT 1",
        (int(mission_id), str(metric)),
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return None
    return Baseline(
        mission_id=int(row[0]),
        release_tag=str(row[1]),
        metric=str(row[2]),
        p50=row[3],
        p95=row[4],
        p99=row[5],
        recorded_at=row[6],
    )


def regression_pct(
    baseline: float | None,
    current: float | None,
) -> float | None:
    """Percent change of current vs baseline. Positive = slower (worse).

    Returns ``None`` when either value is missing or baseline is zero/None.
    """
    if baseline is None or current is None:
        return None
    if baseline == 0:
        return None
    return ((float(current) - float(baseline)) / float(baseline)) * 100.0


def has_regression(
    baseline: Baseline | dict | None,
    current: dict[str, Any],
    threshold_pct: float = 10.0,
) -> bool:
    """Return True when any of p50/p95/p99 regresses past ``threshold_pct``.

    ``current`` is a dict like ``{"p50": ..., "p95": ..., "p99": ...}``.
    A missing statistic on either side is skipped (not a regression).
    """
    if baseline is None:
        return False
    if isinstance(baseline, Baseline):
        b = {"p50": baseline.p50, "p95": baseline.p95, "p99": baseline.p99}
    else:
        b = baseline
    for stat in ("p50", "p95", "p99"):
        delta = regression_pct(b.get(stat), current.get(stat))
        if delta is not None and delta > threshold_pct:
            return True
    return False
