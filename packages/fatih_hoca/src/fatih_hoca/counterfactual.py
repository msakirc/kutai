"""Counterfactual scoring CLI — replays model_pick_log under candidate parameters.

Usage:
    python -m fatih_hoca.counterfactual [--k F] [--limit-days N] [--json PATH]

Reads DB_PATH from env. Joins model_pick_log against model_stats to compute
how often each pick aligned with the empirically-best model (highest
success_rate) at pick time. Does NOT re-run the full ranker — it rescales
stored candidate composites under the Phase 2d utilization equation and
re-ranks.

Sweeps K (UTILIZATION_K) instead of the old cap_gate_ratio. The gate was
retired in Phase 2d; the equation now handles capability fit via the
fit-excess dampener. Historical rows written before Phase 2d will still
have `urgency` and `cap_score` per candidate — interpreted as scarcity in
the new equation. Pre-Phase-2d rows lacking `fit_excess` fall back to
cap_score_100/10 treated as-is.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from typing import Any


def _load_rows(db_path: str, limit_days: int | None) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    q = ("SELECT picked_model, agent_type, difficulty, candidates_json "
         "FROM model_pick_log")
    if limit_days is not None:
        q += f" WHERE timestamp >= datetime('now','-{int(limit_days)} days')"
    try:
        cur = conn.execute(q)
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return rows


def _load_success_map(db_path: str) -> dict[str, float]:
    """model_name → weighted success rate across agent types."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT model, total_calls, success_rate FROM model_stats"
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    weighted: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for model, total, rate in rows:
        if total and total > 0 and rate is not None:
            weighted[model].append((total, rate))
    out: dict[str, float] = {}
    for m, pairs in weighted.items():
        tot = sum(t for t, _ in pairs)
        if tot > 0:
            out[m] = sum(t * r for t, r in pairs) / tot
    return out


def _rescore_utilization(
    original_score: float,
    cap_score_100: float,
    task_difficulty: int,
    scarcity: float,
    K: float = 1.0,
) -> float:
    """Re-score a historical candidate under the Phase 2d utilization equation.

    Applies:
        fit_excess     = (cap_score_100 - cap_needed_for_difficulty(d)) / 100
        if scarcity > 0:
            fit_dampener = max(0, 1 - abs(fit_excess))   # symmetric
        else:
            fit_dampener = 1 - max(0, fit_excess)        # over-qual only
        composite *= 1 + K * scarcity * fit_dampener
    """
    from fatih_hoca.capability_curve import cap_needed_for_difficulty
    cap_needed = cap_needed_for_difficulty(task_difficulty)
    fit_excess = (cap_score_100 - cap_needed) / 100.0
    if scarcity > 0:
        fit_dampener = max(0.0, 1.0 - abs(fit_excess))
    else:
        fit_dampener = 1.0 - max(0.0, fit_excess)
    adjustment = 1.0 + K * scarcity * fit_dampener
    return original_score * adjustment


def _rescore(
    candidates: list[dict[str, Any]],
    task_difficulty: int,
    K: float,
) -> list[dict[str, Any]]:
    """Apply Phase 2d utilization equation to stored candidate composites."""
    if not candidates:
        return candidates
    out = []
    for c in candidates:
        composite = float(c.get("composite", 0.0) or 0.0)
        # `urgency` in historical rows is the scarcity scalar in [-1, +1]
        # (Phase 2c stored it as [0, 1] non-negative urgency; those rows
        # get treated as positive scarcity, which matches the old semantic).
        scarcity = float(c.get("urgency", 0.0) or 0.0)
        cap = float(c.get("cap_score", 0.0) or 0.0)
        new_composite = _rescore_utilization(
            original_score=composite,
            cap_score_100=cap,
            task_difficulty=task_difficulty,
            scarcity=scarcity,
            K=K,
        )
        nc = dict(c)
        nc["composite_cf"] = new_composite
        out.append(nc)
    out.sort(key=lambda x: x["composite_cf"], reverse=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, default=1.0,
                        help="UTILIZATION_K magnitude (Phase 2d). Default 1.0.")
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    db_path = os.environ.get("DB_PATH")
    if not db_path or not os.path.exists(db_path):
        print(f"DB_PATH not set or missing: {db_path}", file=sys.stderr)
        return 1

    rows = _load_rows(db_path, args.limit_days)
    success_map = _load_success_map(db_path)

    agree_cf_hist = 0
    agree_cf_best = 0
    agree_hist_best = 0
    total = 0
    pool_counter: Counter[str] = Counter()

    for r in rows:
        try:
            candidates = json.loads(r["candidates_json"] or "[]")
        except Exception:
            continue
        if not candidates:
            continue
        total += 1
        difficulty = int(r.get("difficulty") or 5)
        rescored = _rescore(candidates, task_difficulty=difficulty, K=args.k)
        cf_pick = rescored[0].get("name") or rescored[0].get("model")
        hist_pick = r["picked_model"]
        best = None
        best_rate = -1.0
        for c in candidates:
            name = c.get("name") or c.get("model")
            rate = success_map.get(name)
            if rate is not None and rate > best_rate:
                best_rate = rate
                best = name
        if cf_pick == hist_pick:
            agree_cf_hist += 1
        if best is not None and cf_pick == best:
            agree_cf_best += 1
        if best is not None and hist_pick == best:
            agree_hist_best += 1
        pool_counter[rescored[0].get("pool", "?")] += 1

    summary = {
        "rows": total,
        "k": args.k,
        "agreement_cf_vs_historical": (agree_cf_hist / total) if total else 0.0,
        "agreement_cf_vs_best": (agree_cf_best / total) if total else 0.0,
        "agreement_historical_vs_best": (agree_hist_best / total) if total else 0.0,
        "pool_distribution_cf": dict(pool_counter),
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
