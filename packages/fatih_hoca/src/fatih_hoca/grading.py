"""Grading-derived perf_score from model_stats.

Used by ranking.py to replace the flat `perf_score=50` fallback. Reads the
main KutAI sqlite DB (DB_PATH env) in a tight, synchronous connection — this
is selection-path code, must not block on async machinery.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Optional

GRADING_MIN_SAMPLES: int = 20
GRADING_PERF_FLOOR: float = 20.0
GRADING_PERF_CEIL: float = 95.0

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
        from src.infra.db import connect_aux_sync
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
    return GRADING_PERF_FLOOR + weighted_success * (GRADING_PERF_CEIL - GRADING_PERF_FLOOR)
