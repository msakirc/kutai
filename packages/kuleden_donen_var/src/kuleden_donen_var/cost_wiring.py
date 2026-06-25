"""Z10 T2A — cost transparency wiring (Telegram + decision helpers).

This module hosts the read-side formatters and the cost-at-decision gate
that wraps ``db.request_confirmation`` for multi-pass actions. T2B owns
the Telegram event posting; T2A only provides the formatter strings and
opens confirmation rows.
"""
from __future__ import annotations

from dabidabi import (
    check_confirmation,
    get_cost_by_iteration,
    get_mission_cost_breakdown,
    request_confirmation,
)


DEFAULT_COST_DECISION_THRESHOLD_USD = 1.0


async def format_mission_cost(mission_id: int) -> str:
    """Render the /mission_cost <id> body. Pure read; safe in any context.

    Reads ``get_mission_cost_breakdown`` + ``get_cost_by_iteration`` and
    formats a 5-line block. Pulls ``budget_ceiling_usd`` from the
    ``cost_budgets`` row scoped ``mission`` for the ratio.
    """
    breakdown = await get_mission_cost_breakdown(mission_id)
    rows = await get_cost_by_iteration(mission_id)
    first_pass = breakdown["first_pass_usd"]
    retry = breakdown["retry_usd"]
    vendor = breakdown["vendor_usd"]
    total = breakdown["total_usd"]

    fp_tokens = 0
    fp_calls = 0
    retry_tokens = 0
    retry_calls = 0
    for r in rows:
        if r["iteration_n"] == 0:
            fp_tokens += int(r["total_tokens"] or 0)
            fp_calls += int(r["calls"] or 0)
        else:
            retry_tokens += int(r["total_tokens"] or 0)
            retry_calls += int(r["calls"] or 0)

    # Pull ceiling from cost_budgets.
    from dabidabi import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT budget_ceiling_usd FROM cost_budgets "
        "WHERE scope = ? AND scope_id = ?",
        ("mission", str(mission_id)),
    )
    row = await cur.fetchone()
    ceiling = float(row[0]) if row and row[0] is not None else None

    # Estimate-vs-actual deviation (mean across completed tasks in mission).
    cur = await db.execute(
        "SELECT COALESCE(SUM(estimated_cost_usd), 0), "
        "       COALESCE(SUM(actual_cost_usd), 0) "
        "FROM tasks "
        "WHERE mission_id = ? AND actual_cost_usd IS NOT NULL",
        (mission_id,),
    )
    row = await cur.fetchone()
    sum_est = float(row[0] or 0.0) if row else 0.0
    sum_act = float(row[1] or 0.0) if row else 0.0

    lines = [f"Mission {mission_id} — cost"]
    lines.append(
        f"First-pass: ${first_pass:.2f} "
        f"({_fmt_tokens(fp_tokens)}, {fp_calls} call{'s' if fp_calls != 1 else ''})"
    )
    lines.append(
        f"Retries:    ${retry:.2f} "
        f"({_fmt_tokens(retry_tokens)}, {retry_calls} call{'s' if retry_calls != 1 else ''})"
    )
    lines.append(f"Vendor:     ${vendor:.2f}")
    if ceiling and ceiling > 0:
        pct = (total / ceiling) * 100.0
        lines.append(
            f"Total:      ${total:.2f} / ${ceiling:.2f} ceiling ({pct:.0f}%)"
        )
    else:
        lines.append(f"Total:      ${total:.2f}")

    if sum_est > 0:
        dev_pct = ((sum_act - sum_est) / sum_est) * 100.0
        sign = "+" if dev_pct >= 0 else ""
        lines.append(
            f"Est vs actual: ${sum_est:.2f} → ${sum_act:.2f} "
            f"({sign}{dev_pct:.0f}%)"
        )
    return "\n".join(lines)


def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.0f}k tokens"
    return f"{n} tokens"


async def open_cost_decision_confirmation(
    task_id: int,
    mission_id: int,
    estimated_usd: float,
    *,
    payload_summary: str | None = None,
) -> int | None:
    """Cost-at-decision gate for multi-pass actions.

    If ``estimated_usd`` exceeds the mission's
    ``cost_decision_threshold_usd`` (default $1.00), opens an
    ``action_confirmations`` row with ``verb='cost_decision'`` and
    returns the confirmation id. Otherwise returns None.

    T2B is responsible for posting ``[asking]`` to the mission thread and
    flipping the verdict via ``resolve_confirmation`` on founder reaction.
    """
    if estimated_usd <= 0:
        return None
    threshold = await _mission_cost_threshold(mission_id)
    if estimated_usd < threshold:
        return None
    summary = payload_summary or (
        f"this multi-pass action will cost ~${estimated_usd:.2f} (estimate)"
    )
    return await request_confirmation(
        task_id=task_id,
        verb="cost_decision",
        reversibility="partial",
        payload_summary=summary,
    )


async def await_cost_decision_verdict(
    confirmation_id: int,
    *,
    timeout_seconds: float = 0.0,
) -> str:
    """Return the verdict for an open cost_decision confirmation.

    With ``timeout_seconds == 0`` (default) this is one-shot: returns
    whatever the verdict is right now. Callers that want to actually
    block belong on the dispatcher side (T2B). Kept separate so unit
    tests don't have to wait.
    """
    res = await check_confirmation(confirmation_id)
    return res.get("verdict", "missing")


async def _mission_cost_threshold(mission_id: int) -> float:
    """Resolve the per-mission cost-decision floor, defaulting to $1."""
    from dabidabi import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT cost_decision_threshold_usd FROM missions WHERE id = ?",
            (mission_id,),
        )
        row = await cur.fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        pass
    return DEFAULT_COST_DECISION_THRESHOLD_USD


# Quality-mode adjustments (D7). Coulson + Fatih Hoca read these.
QUALITY_MODE_PROFILES: dict[str, dict] = {
    "quick": {
        "max_retries": 1,
        "reviewer_rounds": 0,
        "speed_weight_delta": +1.0,
        "benchmark_weight_delta": -0.5,
    },
    "balanced": {
        "max_retries": None,  # leave defaults alone
        "reviewer_rounds": None,
        "speed_weight_delta": 0.0,
        "benchmark_weight_delta": 0.0,
    },
    "thorough": {
        "max_retries": 5,
        "reviewer_rounds": 2,
        "speed_weight_delta": -1.0,
        "benchmark_weight_delta": +0.5,
    },
}


def quality_mode_profile(mode: str) -> dict:
    """Return the dial for ``mode`` (falls back to ``balanced``)."""
    return QUALITY_MODE_PROFILES.get(mode, QUALITY_MODE_PROFILES["balanced"])
