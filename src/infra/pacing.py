"""Z10 T3A — mission time awareness + pacing computation.

This module reads the time-budget fields on ``missions`` (target_launch,
time_budget_hours, phase_budget_json) and the per-task timing columns
(started_at, completed_at, step_started_at, phase_id) and renders a
single pacing dict suitable for the ``/mission <id>`` Telegram view and
for the ``mission_pacing_check`` cron job (D5).

Pure read-side except for ``take_pacing_snapshot``, which writes one
row to ``mission_pacing_snapshots`` and returns its id.

Tradeoff trigger (D5):
    tradeoff_due == True iff percent_burn > 0.75 AND
                            scope_remaining_pct > 0.25

Edge cases
----------
- ``time_budget_hours IS NULL`` → ``percent_burn`` / ``projected_finish_at``
  / ``remaining_budget_hours`` are all ``None``; ``tradeoff_due`` False.
- No tasks yet → ``elapsed_hours == 0.0``.
- Running tasks (status='processing') with ``step_started_at`` set
  contribute (now - step_started_at) to ``elapsed_hours``.
- Maintenance tasks (``agent_type='mechanical'`` and ``phase_id IS NULL``)
  count toward elapsed_hours like any other task (D6).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from src.infra.db import get_db
from src.infra.logging_config import get_logger

logger = get_logger("infra.pacing")


def _parse_ts(ts: Any) -> datetime | None:
    """Coerce SQLite TIMESTAMP value (str | datetime | None) → UTC datetime.

    SQLite stores `CURRENT_TIMESTAMP` as ``"YYYY-MM-DD HH:MM:SS"`` (space
    separator, no tz). isoformat values ``"YYYY-MM-DDTHH:MM:SS"`` may
    also slip in. Both are treated as UTC.
    """
    if ts is None or ts == "":
        return None
    if isinstance(ts, datetime):
        # Naive → assume UTC; aware → convert to UTC.
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    if not isinstance(ts, str):
        return None
    s = ts.strip()
    # Strip trailing 'Z'
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Try a couple of formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        # fromisoformat handles ±HH:MM offsets
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _hours_between(start: datetime | None, end: datetime | None) -> float:
    if start is None or end is None:
        return 0.0
    delta = end - start
    return max(delta.total_seconds() / 3600.0, 0.0)


async def compute_mission_pacing(mission_id: int) -> dict:
    """Compute the pacing snapshot for ``mission_id``. Pure read.

    See module docstring for shape. Returns a dict even when the mission
    has no tasks or no budget configured.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT id, created_at, target_launch, time_budget_hours, "
        "       phase_budget_json, completed_at, status "
        "FROM missions WHERE id = ?",
        (mission_id,),
    )
    mrow = await cur.fetchone()
    if mrow is None:
        return {
            "mission_id": mission_id,
            "started_at": None,
            "target_launch": None,
            "time_budget_hours": None,
            "elapsed_hours": 0.0,
            "remaining_budget_hours": None,
            "percent_burn": None,
            "scope_remaining_pct": 0.0,
            "projected_finish_at": None,
            "phase_breakdown": [],
            "tradeoff_due": False,
        }

    mrow = dict(mrow)
    started_at_raw = mrow.get("created_at")
    started_dt = _parse_ts(started_at_raw)
    target_launch = mrow.get("target_launch")
    time_budget = mrow.get("time_budget_hours")
    time_budget_f = float(time_budget) if time_budget is not None else None
    phase_budget_json = mrow.get("phase_budget_json")
    phase_budgets: dict[str, float] = {}
    if phase_budget_json:
        try:
            raw = json.loads(phase_budget_json)
            if isinstance(raw, dict):
                phase_budgets = {
                    str(k): float(v) for k, v in raw.items()
                    if v is not None
                }
        except Exception as e:
            logger.debug(f"phase_budget_json parse failed: {e}")

    # ── Tasks: elapsed + scope + per-phase breakdown ─────────────
    cur = await db.execute(
        "SELECT id, status, agent_type, started_at, completed_at, "
        "       step_started_at, phase_id "
        "FROM tasks WHERE mission_id = ?",
        (mission_id,),
    )
    trows = [dict(r) for r in await cur.fetchall()]

    now = _now_utc()
    elapsed = 0.0
    phase_elapsed: dict[str, float] = {}
    total_tasks = 0
    remaining_tasks = 0
    for t in trows:
        total_tasks += 1
        status = (t.get("status") or "").lower()
        # Scope: anything that hasn't reached terminal state.
        if status not in ("completed", "failed", "cancelled", "skipped"):
            remaining_tasks += 1
        s = _parse_ts(t.get("started_at")) or _parse_ts(t.get("step_started_at"))
        c = _parse_ts(t.get("completed_at"))
        if c is not None and s is not None:
            dur = _hours_between(s, c)
        elif s is not None and status in ("processing", "running"):
            dur = _hours_between(s, now)
        else:
            dur = 0.0
        elapsed += dur
        pid = t.get("phase_id")
        if pid is not None and pid != "":
            phase_elapsed[str(pid)] = phase_elapsed.get(str(pid), 0.0) + dur

    # ── Derived metrics ─────────────────────────────────────────
    scope_remaining_pct = (
        float(remaining_tasks) / float(total_tasks) if total_tasks > 0 else 0.0
    )

    percent_burn: float | None
    remaining_budget_hours: float | None
    projected_finish_at: str | None
    if time_budget_f is not None and time_budget_f > 0:
        percent_burn = elapsed / time_budget_f
        remaining_budget_hours = max(time_budget_f - elapsed, 0.0)
        # Extrapolate from current burn rate: if X% of work done in Y hours,
        # project total = Y * (1 / fraction_done). fraction_done = 1 -
        # scope_remaining_pct. Cap to avoid division by zero.
        fraction_done = max(1.0 - scope_remaining_pct, 1e-6)
        if elapsed > 0 and started_dt is not None:
            projected_total_hours = elapsed / fraction_done
            finish_dt = started_dt + timedelta(hours=projected_total_hours)
            projected_finish_at = finish_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            projected_finish_at = None
    else:
        percent_burn = None
        remaining_budget_hours = None
        projected_finish_at = None

    phase_breakdown = []
    seen_phases = set(phase_elapsed.keys()) | set(phase_budgets.keys())
    for pid in sorted(seen_phases):
        phase_breakdown.append({
            "phase_id": pid,
            "elapsed_h": round(phase_elapsed.get(pid, 0.0), 4),
            "budget_h": phase_budgets.get(pid),
        })

    tradeoff_due = bool(
        percent_burn is not None
        and percent_burn > 0.75
        and scope_remaining_pct > 0.25
    )

    return {
        "mission_id": mission_id,
        "started_at": (
            started_dt.strftime("%Y-%m-%d %H:%M:%S")
            if started_dt is not None else None
        ),
        "target_launch": str(target_launch) if target_launch else None,
        "time_budget_hours": time_budget_f,
        "elapsed_hours": round(elapsed, 4),
        "remaining_budget_hours": (
            round(remaining_budget_hours, 4)
            if remaining_budget_hours is not None else None
        ),
        "percent_burn": (
            round(percent_burn, 4) if percent_burn is not None else None
        ),
        "scope_remaining_pct": round(scope_remaining_pct, 4),
        "projected_finish_at": projected_finish_at,
        "phase_breakdown": phase_breakdown,
        "tradeoff_due": tradeoff_due,
    }


async def take_pacing_snapshot(mission_id: int) -> int:
    """Compute pacing and persist a row in ``mission_pacing_snapshots``.

    Returns the new row id (0 on failure / unknown mission).
    """
    p = await compute_mission_pacing(mission_id)
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO mission_pacing_snapshots "
        "(mission_id, elapsed_hours, remaining_budget_hours, "
        " projected_finish_at, percent_burn, scope_remaining_pct) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            int(mission_id),
            float(p["elapsed_hours"]),
            (
                float(p["remaining_budget_hours"])
                if p["remaining_budget_hours"] is not None else None
            ),
            p["projected_finish_at"],
            (
                float(p["percent_burn"])
                if p["percent_burn"] is not None else None
            ),
            float(p["scope_remaining_pct"]),
        ),
    )
    await db.commit()
    return cur.lastrowid or 0
