"""Z10 T3A — /mission <id> view formatter (with Pacing block).

Renders a mission summary suitable for the Telegram ``/mission <id>``
command. The Pacing block is appended when the mission has a
``time_budget_hours`` set (or has any tracked timing); fields that
require ``time_budget_hours`` (projection, percent-burn) are omitted
when it is None.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.infra.db import get_db
from src.infra.pacing import compute_mission_pacing, _parse_ts


def _fmt_hours(h: float | None) -> str:
    if h is None:
        return "—"
    total_minutes = int(round(h * 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    if hours == 0:
        return f"{minutes}m"
    if minutes == 0:
        return f"{hours}h"
    return f"{hours}h {minutes}m"


def _fmt_delta(d_seconds: float) -> str:
    """Render a duration in 'Xd Yh', 'Xh Ym', or 'Xm' depending on size."""
    s = int(abs(d_seconds))
    days = s // 86400
    hours = (s % 86400) // 3600
    minutes = (s % 3600) // 60
    if days >= 1:
        return f"{days}d {hours}h"
    if hours >= 1:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _render_pacing(pacing: dict) -> list[str]:
    """Return the lines of the Pacing block. Empty list ⇒ no block."""
    started = pacing.get("started_at")
    target = pacing.get("target_launch")
    budget = pacing.get("time_budget_hours")
    elapsed = float(pacing.get("elapsed_hours") or 0.0)
    burn = pacing.get("percent_burn")
    proj = pacing.get("projected_finish_at")
    scope = float(pacing.get("scope_remaining_pct") or 0.0)
    phase_breakdown = pacing.get("phase_breakdown") or []

    # If absolutely nothing is set, skip the block.
    if (
        started is None and target is None and budget is None
        and elapsed == 0.0 and not phase_breakdown
    ):
        return []

    lines: list[str] = ["", "*Pacing*"]
    now = datetime.now(timezone.utc)

    if started:
        sd = _parse_ts(started)
        ago = ""
        if sd is not None:
            ago = f" ({_fmt_delta((now - sd).total_seconds())} ago)"
        lines.append(f"Started:    {started} UTC{ago}")
    if target:
        try:
            td = datetime.strptime(str(target), "%Y-%m-%d").replace(
                tzinfo=timezone.utc,
            )
            remaining = td - now
            sign = "" if remaining.total_seconds() >= 0 else "-"
            rem_txt = (
                f"{sign}{_fmt_delta(remaining.total_seconds())} remaining"
                if remaining.total_seconds() >= 0
                else (
                    f"{_fmt_delta(remaining.total_seconds())} overdue"
                )
            )
            lines.append(f"Target:     {target} ({rem_txt})")
        except ValueError:
            lines.append(f"Target:     {target}")

    if budget is not None:
        pct_txt = f" ({int(round(burn * 100))}%)" if burn is not None else ""
        lines.append(
            f"Elapsed:    {_fmt_hours(elapsed)} / "
            f"{_fmt_hours(budget)} budget{pct_txt}"
        )
        if proj:
            # If target_launch set + projected past it, mark "late".
            late_note = ""
            if target:
                try:
                    td = datetime.strptime(
                        str(target), "%Y-%m-%d"
                    ).replace(tzinfo=timezone.utc)
                    pd = _parse_ts(proj)
                    if pd is not None and pd > td:
                        d = pd - td
                        late_note = f" ({_fmt_delta(d.total_seconds())} late)"
                except ValueError:
                    pass
            lines.append(f"Projected:  {proj} UTC{late_note}")
    else:
        lines.append(f"Elapsed:    {_fmt_hours(elapsed)} (no budget set)")

    lines.append(f"Scope:      {int(round(scope * 100))}% remaining")

    for entry in phase_breakdown:
        pid = entry.get("phase_id", "?")
        eh = float(entry.get("elapsed_h") or 0.0)
        bh = entry.get("budget_h")
        if bh is not None and float(bh) > 0:
            pct = int(round(eh / float(bh) * 100))
            lines.append(
                f"{pid}:    {_fmt_hours(eh)} / {_fmt_hours(float(bh))} "
                f"({pct}%)"
            )
        else:
            lines.append(f"{pid}:    {_fmt_hours(eh)}")

    return lines


async def format_mission_view(mission_id: int) -> str:
    """Render the /mission <id> output."""
    db = await get_db()
    cur = await db.execute(
        "SELECT id, title, description, status, priority, created_at, "
        "       completed_at, target_launch, time_budget_hours "
        "FROM missions WHERE id = ?",
        (mission_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return f"❌ Mission #{mission_id} not found."
    m = dict(row)

    # Task summary
    cur = await db.execute(
        "SELECT status, COUNT(*) AS c FROM tasks "
        "WHERE mission_id = ? GROUP BY status",
        (mission_id,),
    )
    status_counts: dict[str, int] = {}
    for r in await cur.fetchall():
        rd = dict(r)
        status_counts[rd["status"]] = int(rd["c"])
    total = sum(status_counts.values())

    title = (m.get("title") or "(untitled)")[:80]
    status = m.get("status") or "?"
    lines = [
        f"🎯 *Mission #{mission_id}* — {title}",
        f"Status:  {status}",
        f"Tasks:   {total} total — "
        + ", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))
        if total else "Tasks:   (none yet)",
    ]

    pacing = await compute_mission_pacing(mission_id)
    lines.extend(_render_pacing(pacing))

    # Z6 T7B — surface pending founder_actions inline so the founder
    # sees the wall while reading mission detail (instead of bouncing
    # through /actions). Cap titles at 5; append "...and N-5 more" tail
    # when over.
    try:
        import src.founder_actions as fa
        pending = await fa.list_by_mission(
            mission_id, status_filter=["pending", "in_progress"],
        )
    except Exception:
        pending = []
    if pending:
        lines.append("")
        lines.append(f"*Pending founder_actions: {len(pending)}*")
        for r in pending[:5]:
            lines.append(f"  ⚠ #{r.id} [{r.kind}] {r.title[:60]}")
        if len(pending) > 5:
            lines.append(f"  ...and {len(pending) - 5} more — /actions {mission_id}")

    return "\n".join(lines)
