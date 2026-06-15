"""Z7 A0 — briefing_compose posthook handler.

Fires on terminal mission completion. Composes a `mission_briefings` row
(kind='completion') from:
  - phase summaries extracted from result
  - changed-files list (result.changed_files or task_context.produces)
  - deferred items (result.deferred_items)
  - cost actuals (result.cost_actual_usd or DB query)
  - failed-then-recovered events from mission_lessons

founder_minutes_saved_estimate heuristic
-----------------------------------------
Each completed task in the mission represents work the founder would otherwise
have to direct manually: writing the spec, reviewing the output, fixing bugs, etc.
Conservative estimate: **3 minutes of founder attention per completed task step**.
This is intentionally low — it only captures the coordination overhead,
not the execution time.  Source: observed ~3-5 min/task in manual workflows;
using the lower bound to avoid over-claiming.

Constant: MINUTES_PER_STEP = 3
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.briefing_compose")

# ---------------------------------------------------------------------------
# Heuristic constant: founder minutes saved per completed agent step.
# Rationale: coordination overhead (reviewing spec, unblocking, checking
# output) is ~3-5 min/step in manual workflows. We use the conservative
# lower bound so ROI claims hold up under scrutiny.
# ---------------------------------------------------------------------------
MINUTES_PER_STEP: int = 3


async def _count_completed_steps(mission_id: int) -> int:
    """Count completed (non-mechanical) tasks for this mission."""
    from dabidabi import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT COUNT(*) FROM tasks "
        "WHERE mission_id = ? "
        "  AND COALESCE(status, '') = 'completed' "
        "  AND COALESCE(agent_type, '') != 'mechanical'",
        (int(mission_id),),
    )
    row = await cur.fetchone()
    return int(row[0]) if row else 0


async def _get_cost_actual(mission_id: int) -> float:
    """Sum actual cost from model_call_tokens for this mission."""
    from dabidabi import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) "
            "FROM model_call_tokens "
            "WHERE mission_id = ?",
            (int(mission_id),),
        )
        row = await cur.fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0


async def _get_recovered_lessons(mission_id: int) -> list[dict]:
    """Pull mission_lessons rows that reference this mission_id in source_ref.

    ``source_ref`` is a JSON TEXT column (no real ``mission_id`` column on
    ``mission_lessons``). We match with SQLite's ``json_extract`` so
    ``mission_id == 42`` does not also match ``421``/``4210`` the way a
    naive ``LIKE '%"mission_id": 42%'`` substring match would.
    """
    from dabidabi import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT pattern, fix, domain FROM mission_lessons "
            "WHERE json_extract(source_ref, '$.mission_id') = ? "
            "  AND suppressed = 0 "
            "ORDER BY occurrences DESC LIMIT 5",
            (int(mission_id),),
        )
        return [dict(zip([d[0] for d in cur.description], r)) for r in await cur.fetchall()]
    except Exception:
        return []


def _compose_body_md(
    *,
    phase_summaries: list[dict],
    changed_files: list[str],
    deferred_items: list[str],
    cost_usd: float,
    recovered_lessons: list[dict],
) -> str:
    """Compose the Markdown body for the briefing."""
    sections: list[str] = []

    # Phase summary
    if phase_summaries:
        lines = ["## Phase Summary"]
        for ps in phase_summaries:
            phase = ps.get("phase") or ps.get("id") or "Phase"
            outcome = ps.get("outcome") or ps.get("summary") or ""
            lines.append(f"- **{phase}**: {outcome}")
        sections.append("\n".join(lines))
    else:
        sections.append("## Phase Summary\n_(no phase data)_")

    # Changed files
    if changed_files:
        lines = ["## Changed Files"]
        for f in changed_files[:20]:  # cap at 20
            lines.append(f"- `{f}`")
        if len(changed_files) > 20:
            lines.append(f"- _(+{len(changed_files) - 20} more)_")
        sections.append("\n".join(lines))
    else:
        sections.append("## Changed Files\n_(none recorded)_")

    # Deferred items
    if deferred_items:
        lines = ["## Deferred Items"]
        for d in deferred_items:
            lines.append(f"- {d}")
        sections.append("\n".join(lines))
    else:
        sections.append("## Deferred Items\n_(none)_")

    # Cost
    cost_line = f"${cost_usd:.4f}" if cost_usd else "_(not tracked)_"
    sections.append(f"## Cost\n{cost_line}")

    # Recovered failures
    if recovered_lessons:
        lines = ["## Recovered Failures"]
        for lesson in recovered_lessons:
            domain = lesson.get("domain", "")
            pattern = lesson.get("pattern", "")
            fix = lesson.get("fix", "")
            entry = f"- [{domain}] {pattern}"
            if fix:
                entry += f" → _{fix}_"
            lines.append(entry)
        sections.append("\n".join(lines))
    else:
        sections.append("## Recovered Failures\n_(none)_")

    return "\n\n".join(sections)


async def handle(task: dict, result: dict) -> dict:
    """briefing_compose posthook handler.

    Composes a mission_briefings row (kind='completion') for the founder.

    Returns {"status": "ok"} on success.
    Returns {"status": "skip", "reason": ...} when mission_id is absent.
    """
    mission_id = task.get("mission_id")
    if not mission_id:
        logger.debug("briefing_compose: no mission_id — skipping")
        return {"status": "skip", "reason": "no mission_id"}

    try:
        mission_id = int(mission_id)
    except (TypeError, ValueError):
        return {"status": "skip", "reason": f"invalid mission_id: {mission_id}"}

    logger.debug(
        "briefing_compose: composing completion briefing",
        task_id=task.get("id"),
        mission_id=mission_id,
    )

    # ── Gather data ──────────────────────────────────────────────────────────
    phase_summaries: list[dict] = result.get("phase_summaries") or []
    changed_files: list[str] = result.get("changed_files") or []
    deferred_items: list[str] = result.get("deferred_items") or []
    cost_usd: float = float(result.get("cost_actual_usd") or 0.0)

    # Fall back to DB cost if not in result
    if cost_usd == 0.0:
        cost_usd = await _get_cost_actual(mission_id)

    recovered_lessons = await _get_recovered_lessons(mission_id)

    # ── Estimate founder minutes saved ───────────────────────────────────────
    step_count = await _count_completed_steps(mission_id)
    # If no tasks recorded yet (e.g. called from test before tasks inserted),
    # fall back to counting any explicitly provided phase_summaries.
    if step_count == 0 and phase_summaries:
        step_count = len(phase_summaries)
    minutes_saved = max(1, step_count * MINUTES_PER_STEP)

    # ── Compose body ─────────────────────────────────────────────────────────
    body_md = _compose_body_md(
        phase_summaries=phase_summaries,
        changed_files=changed_files,
        deferred_items=deferred_items,
        cost_usd=cost_usd,
        recovered_lessons=recovered_lessons,
    )

    # ── Write to DB ──────────────────────────────────────────────────────────
    from dabidabi import get_db
    db = await get_db()
    await db.execute(
        "INSERT INTO mission_briefings "
        "(product_id, mission_id, kind, body_md, founder_minutes_saved_estimate, prepared_at) "
        "VALUES (?, ?, 'completion', ?, ?, datetime('now'))",
        (str(mission_id), str(mission_id), body_md, minutes_saved),
    )
    await db.commit()

    logger.info(
        "briefing_compose: completion briefing written",
        mission_id=mission_id,
        minutes_saved=minutes_saved,
        step_count=step_count,
    )
    return {"status": "ok", "founder_minutes_saved": minutes_saved}
