"""B-table rollup: model_call_tokens → step_token_stats.

Hourly cron registered in Beckman scheduled_jobs (marker 'btable_rollup').
14-day rolling window. Computes p50/p90/p99 in Python (SQLite lacks percentile_disc).
"""
from __future__ import annotations

import os

import aiosqlite

from general_beckman.btable_cache import set_btable


WINDOW_DAYS = 14


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


async def run_rollup(db_path: str | None = None) -> int:
    """Aggregate model_call_tokens (last 14 days, non-streaming) into step_token_stats.

    Refreshes the in-memory btable_cache. Returns count of rows written.
    """
    db_path = db_path or os.environ.get("DB_PATH", "kutai.db")
    from src.infra.db import connect_aux
    async with connect_aux(db_path) as db:
        async with db.execute(
            f"""SELECT agent_type, workflow_step_id, workflow_phase,
                       prompt_tokens, completion_tokens, iteration_n
                FROM model_call_tokens
                WHERE timestamp > datetime('now', '-{WINDOW_DAYS} days')
                  AND is_streaming = 0
                  AND agent_type IS NOT NULL
                  AND workflow_step_id IS NOT NULL
                  AND workflow_phase IS NOT NULL"""
        ) as cur:
            rows = await cur.fetchall()

    # Group per key
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    iter_observations: dict[tuple[str, str, str], list[int]] = {}
    for agent_type, step_id, phase, in_tok, out_tok, iter_n in rows:
        key = (agent_type, step_id, phase)
        bucket = grouped.setdefault(key, {"in": [], "out": []})
        bucket["in"].append(float(in_tok or 0))
        bucket["out"].append(float(out_tok or 0))
        iter_observations.setdefault(key, []).append(int(iter_n or 1))

    rows_written = 0
    btable_dict: dict[tuple[str, str, str], dict] = {}
    from src.infra.db import connect_aux
    async with connect_aux(db_path) as db:
        for key, vals in grouped.items():
            in_sorted = sorted(vals["in"])
            out_sorted = sorted(vals["out"])
            samples_n = len(out_sorted)
            iter_sorted = sorted(float(v) for v in iter_observations.get(key, [1]))
            iters_p50 = _percentile(iter_sorted, 0.50)
            iters_p90 = _percentile(iter_sorted, 0.90)
            iters_p99 = _percentile(iter_sorted, 0.99)
            row = {
                "samples_n": samples_n,
                "in_p50": int(_percentile(in_sorted, 0.50)),
                "in_p90": int(_percentile(in_sorted, 0.90)),
                "in_p99": int(_percentile(in_sorted, 0.99)),
                "out_p50": int(_percentile(out_sorted, 0.50)),
                "out_p90": int(_percentile(out_sorted, 0.90)),
                "out_p99": int(_percentile(out_sorted, 0.99)),
                "iters_p50": iters_p50,
                "iters_p90": iters_p90,
                "iters_p99": iters_p99,
            }
            await db.execute(
                """INSERT INTO step_token_stats
                    (agent_type, workflow_step_id, workflow_phase,
                     samples_n, in_p50, in_p90, in_p99, out_p50, out_p90, out_p99,
                     iters_p50, iters_p90, iters_p99, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(agent_type, workflow_step_id, workflow_phase) DO UPDATE SET
                     samples_n=excluded.samples_n,
                     in_p50=excluded.in_p50, in_p90=excluded.in_p90, in_p99=excluded.in_p99,
                     out_p50=excluded.out_p50, out_p90=excluded.out_p90, out_p99=excluded.out_p99,
                     iters_p50=excluded.iters_p50, iters_p90=excluded.iters_p90,
                     iters_p99=excluded.iters_p99,
                     updated_at=datetime('now')""",
                (key[0], key[1], key[2], samples_n,
                 row["in_p50"], row["in_p90"], row["in_p99"],
                 row["out_p50"], row["out_p90"], row["out_p99"],
                 iters_p50, iters_p90, iters_p99),
            )
            rows_written += 1
            btable_dict[key] = row
        await db.commit()

    set_btable(btable_dict)  # refresh in-memory cache for admission gate
    return rows_written
