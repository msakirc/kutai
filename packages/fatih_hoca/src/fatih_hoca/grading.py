"""Grading-derived perf_score from model_stats.

Used by ranking.py to replace the flat `perf_score=50` fallback. Reads the
main KutAI sqlite DB (DB_PATH env) in a tight, synchronous connection — this
is selection-path code, must not block on async machinery.
"""
from __future__ import annotations

import logging
import math
import os
import sqlite3
from datetime import datetime
from typing import Optional

GRADING_MIN_SAMPLES: int = 20
GRADING_PERF_FLOOR: float = 20.0
GRADING_PERF_CEIL: float = 95.0

# Z9 T4E reinforce loop — confirmed hypothesis verdicts write
# model_pick_log rows tagged call_category='reinforce' carrying a +0.05
# bonus in the ``reinforce`` column. We fold a time-decayed SUM of those
# bonuses into perf_score: each nudge halves every 30 days so old wins
# fade. REINFORCE_PERF_SCALE converts a [0..1] bonus sum into perf_score
# points; a single fresh win is a gentle ~+1pt nudge, never a takeover.
REINFORCE_HALF_LIFE_DAYS: float = 30.0
REINFORCE_PERF_SCALE: float = 20.0
REINFORCE_PERF_CAP: float = 8.0  # never more than +8 perf points total

logger = logging.getLogger(__name__)


def _db_path() -> str | None:
    return os.environ.get("DB_PATH")


def grading_perf_score(model_name: str) -> Optional[float]:
    """Aggregate model_stats rows for `model_name` into a 0-100 perf score.

    Returns None when total sample count across agent_types is below
    GRADING_MIN_SAMPLES. Otherwise maps weighted success_rate in [0, 1] to
    [GRADING_PERF_FLOOR, GRADING_PERF_CEIL] linearly.
    """
    path = _db_path()
    if not path or not os.path.exists(path):
        return None
    try:
        from dabidabi import connect_aux_sync
        conn = connect_aux_sync(path)
        try:
            cur = conn.execute(
                "SELECT total_calls, success_rate FROM model_stats WHERE model = ?",
                (model_name,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        logger.debug("grading_perf_score read failed for %s: %s", model_name, e)
        return None

    total = sum(r[0] or 0 for r in rows)
    if total < GRADING_MIN_SAMPLES:
        return None

    weighted_success = sum((r[0] or 0) * (r[1] or 0.0) for r in rows) / total
    weighted_success = max(0.0, min(1.0, weighted_success))
    base = GRADING_PERF_FLOOR + weighted_success * (
        GRADING_PERF_CEIL - GRADING_PERF_FLOOR
    )
    # Fold in the Z9 reinforce nudge — gentle, decaying, capped.
    return min(GRADING_PERF_CEIL, base + reinforce_bonus(model_name))


def reinforce_bonus(model_name: str) -> float:
    """Time-decayed perf-score bonus from confirmed hypothesis verdicts.

    Reads ``model_pick_log`` rows for ``model_name`` with
    ``call_category='reinforce'`` and sums each row's ``reinforce`` value
    weighted by ``0.5 ** (age_days / 30)`` — a 30-day half-life so wins
    fade. The decayed sum is scaled into perf-score points and capped at
    REINFORCE_PERF_CAP so the loop nudges, never dominates.

    Returns 0.0 on any error or when the column/rows are absent — the
    selection path must never break on telemetry.
    """
    path = _db_path()
    if not path or not os.path.exists(path):
        return 0.0
    try:
        from dabidabi import connect_aux_sync
        conn = connect_aux_sync(path)
        try:
            cur = conn.execute(
                "SELECT reinforce, timestamp FROM model_pick_log "
                "WHERE picked_model = ? AND call_category = 'reinforce' "
                "AND reinforce IS NOT NULL",
                (model_name,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        logger.debug("reinforce_bonus read failed for %s: %s", model_name, e)
        return 0.0

    if not rows:
        return 0.0

    now = datetime.now()
    decayed_sum = 0.0
    for amount, ts in rows:
        if not amount:
            continue
        age_days = 0.0
        if ts:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d %H:%M:%S.%f"):
                try:
                    age_days = max(
                        0.0,
                        (now - datetime.strptime(str(ts)[:26], fmt)).days,
                    )
                    break
                except Exception:
                    continue
        decay = 0.5 ** (age_days / REINFORCE_HALF_LIFE_DAYS)
        decayed_sum += float(amount) * decay

    return min(REINFORCE_PERF_CAP, decayed_sum * REINFORCE_PERF_SCALE)
