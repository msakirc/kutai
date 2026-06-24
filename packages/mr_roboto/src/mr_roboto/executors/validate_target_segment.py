"""Z9 T5A — validate_target_segment mechanical executor.

Cohort-awareness nag. ``mission.context['target_segment']`` is a free-form
string declaring WHO a Phase 8+ growth change targets:

    "paid_users" | "new_signups" | "week2_churners" | "any"

It is a plain context key — NO schema migration. The default is ``"any"``
(the whole user base). A change scoped to "everyone" is rarely the right
default for a Growth mission, so when a Phase 8+ mission reaches the
implementation backlog with no explicit ``target_segment`` this executor
emits a **WARNING** — never a block. The mission proceeds either way.

Mirrors ``inject_north_star`` (Z9 T4B):
  - reads ``missions.context`` JSON,
  - graceful when no mission / unreadable context,
  - idempotent — when ``target_segment`` is absent it back-fills the
    explicit ``"any"`` default so downstream readers never KeyError.

Pure mechanical — NO LLM.

Public API
----------
validate_target_segment(mission_id) -> dict
    ``{"ok": True, "target_segment": <str>, "explicit": bool,
       "warned": bool, "mission_id": int}``
"""
from __future__ import annotations

import json

from yazbunu import get_logger

logger = get_logger("mr_roboto.validate_target_segment")

# The recognised cohort vocabulary. Free-form is allowed, but these are the
# canonical values the i2p Phase 2 brief step prompts for.
KNOWN_SEGMENTS = frozenset(
    {"paid_users", "new_signups", "week2_churners", "any"}
)

DEFAULT_SEGMENT = "any"


async def validate_target_segment(mission_id: int) -> dict:
    """Warn (never block) when a Phase 8+ mission lacks a target_segment.

    Parameters
    ----------
    mission_id:
        Target mission row in the ``missions`` table.

    Returns
    -------
    dict
        ``{"ok": True, "target_segment": <str>, "explicit": bool,
           "warned": bool, "unknown_value": bool, "mission_id": int}``
    """
    try:
        from dabidabi import get_db

        db = await get_db()
        async with db.execute(
            "SELECT context FROM missions WHERE id = ?",
            (mission_id,),
        ) as cur:
            row = await cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"validate_target_segment: could not read context: {exc}"
        )
        return {
            "ok": True,
            "target_segment": DEFAULT_SEGMENT,
            "explicit": False,
            "warned": False,
            "unknown_value": False,
            "mission_id": mission_id,
        }

    if row is None:
        logger.warning(
            f"validate_target_segment: mission {mission_id} not found"
        )
        return {
            "ok": True,
            "target_segment": DEFAULT_SEGMENT,
            "explicit": False,
            "warned": False,
            "unknown_value": False,
            "mission_id": mission_id,
        }

    raw_ctx = row[0] or "{}"
    if isinstance(raw_ctx, str):
        try:
            ctx = json.loads(raw_ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    elif isinstance(raw_ctx, dict):
        ctx = raw_ctx
    else:
        ctx = {}

    raw_seg = ctx.get("target_segment")
    explicit = isinstance(raw_seg, str) and raw_seg.strip() != ""
    segment = raw_seg.strip() if explicit else DEFAULT_SEGMENT
    unknown_value = explicit and segment not in KNOWN_SEGMENTS

    warned = False
    if not explicit:
        # The cohort-awareness nag — WARN ONLY, mission still proceeds.
        logger.warning(
            "validate_target_segment: Phase 8+ mission %s has no "
            "target_segment — defaulting to 'any' (whole user base). "
            "A growth change scoped to everyone is rarely intentional; "
            "set mission.context['target_segment'] at i2p step 2.9.",
            mission_id,
        )
        warned = True
    elif unknown_value:
        logger.warning(
            "validate_target_segment: mission %s target_segment "
            "%r is not a recognised cohort (%s) — proceeding anyway.",
            mission_id,
            segment,
            sorted(KNOWN_SEGMENTS),
        )

    # Idempotent back-fill: persist the explicit default so downstream
    # readers (instrumentation context, scorer) never KeyError.
    if not explicit and ctx.get("target_segment") != DEFAULT_SEGMENT:
        ctx["target_segment"] = DEFAULT_SEGMENT
        try:
            from general_beckman import update_mission_fields as _umf
            await _umf(mission_id, context=json.dumps(ctx, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                f"validate_target_segment: default back-fill failed: {exc}"
            )

    return {
        "ok": True,
        "target_segment": segment,
        "explicit": explicit,
        "warned": warned,
        "unknown_value": unknown_value,
        "mission_id": mission_id,
    }


async def run(task: dict) -> dict:
    """Dispatcher entry point — mirrors the executors/ run(task) convention."""
    payload = task.get("payload") or {}
    mission_id = task.get("mission_id") or payload.get("mission_id")
    if mission_id is None:
        return {
            "ok": False,
            "error": "validate_target_segment requires mission_id",
        }
    return await validate_target_segment(int(mission_id))
