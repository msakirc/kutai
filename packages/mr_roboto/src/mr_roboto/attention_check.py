"""Z1 Tier 5A (A5) — founder attention budget check.

Mechanical pre-hook for clarify-shape steps. Reads
``missions.founder_attention_budget_minutes`` minus
``SUM(founder_attention_log.minutes_debited)`` for the mission.

Returns:

- ``remaining``: int (minutes left; may be negative)
- ``ok``: True iff remaining >= reserve_minutes
- ``would_exceed``: True when ok is False AND budget is set
- ``budget_set``: bool — when False, attention is treated as unbounded
  (ok=True, remaining=None) so existing missions without z0 don't break.

When ``ok=False``, callers (orchestrator pre-hook) write the pending
clarify questions to ``mission_<id>/deferred_questions.md`` and skip the
step rather than firing on Telegram. See spec A5 in
``docs/i2p-evolution/01-pre-code-additions-claude.md``.
"""
from __future__ import annotations

import datetime
import os
from typing import Any


async def attention_check(
    mission_id: int,
    reserve_minutes: int = 5,
) -> dict[str, Any]:
    """Return current attention-budget status.

    DB columns:
      - missions.founder_attention_budget_minutes (INTEGER, NULL = unbounded)
      - founder_attention_log.minutes_debited (per-debit log)
    """
    from src.infra.db import get_db
    db = await get_db()

    # Budget
    cur = await db.execute(
        "SELECT founder_attention_budget_minutes FROM missions WHERE id = ?",
        (mission_id,),
    )
    row = await cur.fetchone()
    if not row:
        return {
            "ok": True,
            "remaining": None,
            "budget_set": False,
            "would_exceed": False,
            "reserve_minutes": reserve_minutes,
            "error": "mission not found",
        }
    budget = row[0]
    if budget is None:
        return {
            "ok": True,
            "remaining": None,
            "budget_set": False,
            "would_exceed": False,
            "reserve_minutes": reserve_minutes,
        }

    # Debits
    cur = await db.execute(
        "SELECT COALESCE(SUM(minutes_debited), 0) FROM founder_attention_log "
        "WHERE mission_id = ?",
        (mission_id,),
    )
    drow = await cur.fetchone()
    spent = int(drow[0] if drow and drow[0] is not None else 0)
    remaining = int(budget) - spent
    ok = remaining >= int(reserve_minutes)
    return {
        "ok": ok,
        "remaining": remaining,
        "budget_set": True,
        "would_exceed": not ok,
        "reserve_minutes": int(reserve_minutes),
        "spent": spent,
        "budget": int(budget),
    }


async def attention_debit(
    mission_id: int,
    step_id: str,
    action: str,
    minutes_debited: int,
) -> dict[str, Any]:
    """Record a debit row. Returns ``{"ok": True, "id": rowid}`` on success."""
    from src.infra.db import get_db
    db = await get_db()
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur = await db.execute(
        "INSERT INTO founder_attention_log "
        "(mission_id, step_id, action, minutes_debited, ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (mission_id, step_id, action, int(minutes_debited), ts),
    )
    await db.commit()
    return {"ok": True, "id": cur.lastrowid}


async def write_deferred_question(
    mission_id: int,
    step_id: str,
    question_text: str,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Append a deferred clarify question to ``deferred_questions.md``."""
    if workspace_path is None:
        from src.tools.workspace import get_mission_workspace
        workspace_path = get_mission_workspace(int(mission_id))
    os.makedirs(workspace_path, exist_ok=True)
    path = os.path.join(workspace_path, "deferred_questions.md")
    header_needed = not os.path.isfile(path)
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    with open(path, "a", encoding="utf-8") as fh:
        if header_needed:
            fh.write("# Deferred clarify questions\n\n")
            fh.write(
                "_These questions were skipped because the founder's "
                "attention budget was insufficient. Run `/budget set <minutes>` "
                "to top up, then `/dlq retry <task_id>` on the deferred step._\n\n"
            )
        fh.write(f"- **{step_id}** ({ts}): {question_text}\n")
    return {"ok": True, "path": path}
