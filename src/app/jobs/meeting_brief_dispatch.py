"""Z7 T4 B4 — Meeting brief dispatch job.

Runs every 5 minutes (registered in beckman cron_seed as
``_executor='meeting_brief_dispatch'``).

Two phases per tick:
  1. Brief phase: picks meetings where ``scheduled_for - now ∈ [25, 35] min``
     AND ``brief_generated_at IS NULL``.  For each, enqueues a
     ``meeting/brief`` MAIN_WORK task via general_beckman (LLM-bound).

  2. Outcome-prompt phase: picks meetings where ``scheduled_for`` is between
     20 and 60 minutes in the past AND ``outcome_logged_interaction_id IS NULL``.
     For each, fires ``meeting/outcome_prompt`` immediately (non-LLM mechanical).

Public API
----------
- ``_pick_meetings_for_brief()`` — query helper exposed for tests
- ``_pick_meetings_for_outcome_prompt()`` — query helper exposed for tests
- ``run_meeting_brief_dispatch()`` — called by mr_roboto for the
  ``meeting_brief_dispatch`` executor.  Returns {"ok": True} on success.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.infra.logging_config import get_logger

logger = get_logger("app.jobs.meeting_brief_dispatch")

# Window (minutes) before meeting to generate brief
_BRIEF_WINDOW_EARLY_MIN = 25
_BRIEF_WINDOW_LATE_MIN = 35

# Window (minutes) after meeting to prompt for outcome logging
_OUTCOME_PROMPT_AFTER_MIN = 20
_OUTCOME_PROMPT_BEFORE_MIN = 60


def _db_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


async def _pick_meetings_for_brief() -> list[dict]:
    """Return meetings due for brief generation.

    Criteria:
      - scheduled_for - now ∈ [25min, 35min]   (window open, not yet started)
      - brief_generated_at IS NULL              (not already generated)
    """
    from src.infra.db import get_db
    db = await get_db()
    now = datetime.now(timezone.utc)
    window_start = _db_str(now + timedelta(minutes=_BRIEF_WINDOW_EARLY_MIN))
    window_end = _db_str(now + timedelta(minutes=_BRIEF_WINDOW_LATE_MIN))
    cur = await db.execute(
        "SELECT meeting_id, product_id, contact_id, scheduled_for, purpose "
        "FROM meetings "
        "WHERE scheduled_for >= ? AND scheduled_for <= ? "
        "AND brief_generated_at IS NULL",
        (window_start, window_end),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def _pick_meetings_for_outcome_prompt() -> list[dict]:
    """Return meetings due for the outcome-log prompt.

    Criteria:
      - scheduled_for is between 20 and 60 minutes in the past
      - outcome_logged_interaction_id IS NULL (outcome not yet logged)
    """
    from src.infra.db import get_db
    db = await get_db()
    now = datetime.now(timezone.utc)
    past_start = _db_str(now - timedelta(minutes=_OUTCOME_PROMPT_BEFORE_MIN))
    past_end = _db_str(now - timedelta(minutes=_OUTCOME_PROMPT_AFTER_MIN))
    cur = await db.execute(
        "SELECT meeting_id, product_id, contact_id, scheduled_for, purpose "
        "FROM meetings "
        "WHERE scheduled_for >= ? AND scheduled_for <= ? "
        "AND outcome_logged_interaction_id IS NULL",
        (past_start, past_end),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def run_meeting_brief_dispatch() -> dict:
    """Main entry point called by mr_roboto ``meeting_brief_dispatch`` executor.

    Phase 1 — brief generation:
      Enqueues a MAIN_WORK beckman task for each meeting in the 25–35min window.
      The enqueued task carries action='meeting/brief' in its context.
      Marks brief_generated_at immediately to prevent double-dispatch.

    Phase 2 — outcome prompt:
      For each meeting 20-60min past its scheduled_for with no outcome:
      fires meeting/outcome_prompt directly (non-LLM, best-effort).

    Returns {"ok": True} on overall success (partial failures are logged,
    not propagated, so the cron cadence continues).
    """
    brief_count = 0
    outcome_count = 0
    errors: list[str] = []

    # ── Phase 1: brief generation ──────────────────────────────────────────
    try:
        brief_meetings = await _pick_meetings_for_brief()
        for m in brief_meetings:
            meeting_id = m["meeting_id"]
            product_id = m["product_id"]
            try:
                # Mark immediately to prevent double-dispatch on next tick
                from src.infra.db import get_db
                db = await get_db()
                await db.execute(
                    "UPDATE meetings SET brief_generated_at=? WHERE meeting_id=?",
                    (
                        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        meeting_id,
                    ),
                )
                await db.commit()

                # Enqueue MAIN_WORK LLM task via general_beckman
                import general_beckman
                task_id = await general_beckman.enqueue(
                    {
                        "agent_type": "mechanical",
                        "title": f"Meeting brief: {m.get('purpose') or 'meeting'} ({m['scheduled_for']})",
                        "context": {
                            "action": "meeting/brief",
                            "meeting_id": meeting_id,
                            "product_id": product_id,
                        },
                        "priority": 8,
                    }
                )
                logger.info(
                    "meeting_brief_dispatch: enqueued brief task",
                    meeting_id=meeting_id,
                    task_id=task_id,
                )
                brief_count += 1
            except Exception as exc:
                err = f"brief/{meeting_id}: {exc}"
                logger.error("meeting_brief_dispatch: brief enqueue failed", error=err)
                errors.append(err)
    except Exception as exc:
        err = f"brief_phase: {exc}"
        logger.error("meeting_brief_dispatch: brief phase failed", error=err)
        errors.append(err)

    # ── Phase 2: outcome prompt ────────────────────────────────────────────
    try:
        outcome_meetings = await _pick_meetings_for_outcome_prompt()
        for m in outcome_meetings:
            meeting_id = m["meeting_id"]
            product_id = m["product_id"]
            try:
                from src.app.meetings import emit_outcome_prompt
                res = await emit_outcome_prompt(
                    meeting_id=meeting_id,
                    product_id=product_id,
                )
                if res.get("ok"):
                    outcome_count += 1
                    logger.info(
                        "meeting_brief_dispatch: outcome prompt sent",
                        meeting_id=meeting_id,
                    )
                else:
                    errors.append(f"outcome/{meeting_id}: {res.get('reason')}")
            except Exception as exc:
                err = f"outcome/{meeting_id}: {exc}"
                logger.error("meeting_brief_dispatch: outcome prompt failed", error=err)
                errors.append(err)
    except Exception as exc:
        err = f"outcome_phase: {exc}"
        logger.error("meeting_brief_dispatch: outcome phase failed", error=err)
        errors.append(err)

    logger.info(
        "meeting_brief_dispatch: tick complete",
        briefs=brief_count,
        outcomes=outcome_count,
        errors=len(errors),
    )
    return {
        "ok": True,
        "briefs": brief_count,
        "outcomes": outcome_count,
        "errors": errors or None,
    }
