"""B10 — mission-level rework / regression metric.

Single source of truth for "phase rollback" events: a phase >=7 step that
has to drop back to a phase <=6 step (spec amend, schema fail, reviewer
reject, founder request). Increments ``missions.phase_7_rework_loops``
AND emits a structured ``phase_rollback`` yazbunu event.

Spec: docs/i2p-evolution/01-pre-code-master-synthesis.md §B10
Strategic context: project_z1_strategic_locks_20260509 — without this,
KutAI cannot prove its spec-first bet against Lovable's spec-skip bet.

Usage:
    from src.telemetry.rework import record_rollback

    await record_rollback(
        mission_id=task["mission_id"],
        from_phase="8.3",
        to_phase="4.16",
        reason="reviewer_reject",
        triggered_by="code_reviewer",
    )

The helper is async-safe and never raises — telemetry must not crash
the caller. Failures are logged at WARNING and swallowed.
"""
from __future__ import annotations

from typing import Literal

# Lazy import yazbunu via the standard re-export path. Direct ``from yazbunu``
# would break in environments where yazbunu's source dir is missing
# (worktrees, partial checkouts) — the re-export tolerates that.
from src.infra.logging_config import get_logger

logger = get_logger("telemetry.rework")

RollbackReason = Literal[
    "spec_drift",
    "schema_failure",
    "founder_request",
    "reviewer_reject",
    "other",
]

_VALID_REASONS = frozenset({
    "spec_drift", "schema_failure", "founder_request",
    "reviewer_reject", "other",
})


def _phase_num(phase: str) -> int | None:
    """Extract leading integer from "8.3" / "phase_8" / "8" → 8.

    Returns None if no integer can be parsed.
    """
    if not phase:
        return None
    s = str(phase).strip()
    if s.startswith("phase_"):
        s = s[len("phase_"):]
    # Take the part before the first '.'
    head = s.split(".", 1)[0]
    try:
        return int(head)
    except (ValueError, TypeError):
        return None


def is_phase_7_rework(from_phase: str, to_phase: str) -> bool:
    """True iff from_phase >= 7 AND to_phase <= 6 (per B10 definition)."""
    f = _phase_num(from_phase)
    t = _phase_num(to_phase)
    if f is None or t is None:
        return False
    return f >= 7 and t <= 6


async def record_rollback(
    mission_id: int | None,
    from_phase: str,
    to_phase: str,
    reason: str,
    triggered_by: str,
) -> None:
    """Record a phase rollback: bump the counter and emit a yazbunu event.

    Args:
        mission_id: Mission this rollback belongs to. None means the
            rollback is mission-less (rare, e.g. preflight) — we still
            emit the event but skip the DB increment.
        from_phase: e.g. "8.3" (step id) or "phase_8" — the phase that
            failed.
        to_phase: e.g. "4.16" — the phase being re-entered.
        reason: one of ``RollbackReason``. Unknown values are coerced
            to "other" with a warning so the caller never crashes.
        triggered_by: agent name ("code_reviewer"), mechanical
            executor ("mechanical:verify_artifacts"), or "founder".

    Only mutates ``missions.phase_7_rework_loops`` when
    ``is_phase_7_rework(from_phase, to_phase)`` is True — sub-phase
    retries inside the same band don't count toward the headline metric.
    The yazbunu event is ALWAYS emitted so that lower-tier rework is
    still observable in the log.
    """
    if reason not in _VALID_REASONS:
        logger.warning(
            "phase_rollback unknown reason coerced to 'other'",
            given_reason=reason,
        )
        reason = "other"

    is_p7 = is_phase_7_rework(from_phase, to_phase)
    new_count = 0
    if is_p7 and mission_id is not None:
        try:
            from src.infra.db import increment_mission_rework_loops
            new_count = await increment_mission_rework_loops(int(mission_id))
        except Exception as exc:
            # Swallow — telemetry must not crash the caller
            logger.warning(
                "phase_rollback counter bump failed",
                mission_id=mission_id,
                error=str(exc),
            )

    # Always emit the event (yazbunu phase_rollback schema)
    try:
        logger.info(
            "phase_rollback",
            event="phase_rollback",
            mission_id=mission_id,
            from_phase=str(from_phase),
            to_phase=str(to_phase),
            reason=reason,
            triggered_by=triggered_by,
            phase_7_rework=is_p7,
            new_count=new_count,
        )
    except Exception as exc:
        # Even the log emit can theoretically fail — never let it kill
        # the calling pipeline.
        logger.warning(
            "phase_rollback emit failed",
            error=str(exc),
        )
