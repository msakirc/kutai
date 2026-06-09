"""Autonomous routing of a reviewer 'fail' to the at-fault producer(s).

tag-map (deterministic) -> LLM fallback for unresolved -> re-pend each
producer's EXISTING task row with feedback. Escalate to the founder-halt only
when nothing is localisable or a producer has exhausted its normal attempts.
Per-producer attempt bounding is the existing retry rail (worker_attempts) —
there is no separate budget here.

NOTE: _repend_producer / _assign_unresolved / _escalate_to_founder are
implemented in a later task; their signatures are fixed by this module."""
from __future__ import annotations

import json as _json
from typing import Any

from src.infra.logging_config import get_logger
from src.workflows.engine.producer_index import build_producer_index
from coulson.posthooks.review_router import map_tagged_issues

logger = get_logger("beckman.review_routing")


def _feedback_text(issues: list[dict]) -> str:
    lines = [f"- [{i.get('severity')}] {i.get('problem')}" for i in issues]
    return "Reviewer rejected this artifact. Fix:\n" + "\n".join(lines)


async def _repend_producer(mission_id: int, step_id: str, feedback: str) -> bool:
    """Re-pend the producer step's EXISTING task row with reviewer feedback.

    Loads the producer's latest task row by (mission_id, workflow_step_id),
    increments ``worker_attempts``, stamps the feedback into the task context
    (same mechanism the grade / mechanical-check re-pend rails use, so the
    next worker attempt sees it), and flips the row back to ``pending``.

    Returns False (no re-pend) when:
      * no matching producer task row exists, or
      * the next attempt would exceed ``max_worker_attempts`` (exhausted) —
        the caller escalates to the founder-halt in that case.

    Mirrors the worker_attempts/feedback mechanics of the grounding-gate
    re-pend in ``apply.py`` (``_stamp_retry_feedback`` is the single place that
    guarantees per-attempt invariants for every quality re-pend).
    """
    from src.infra.db import get_task_by_workflow_step, update_task
    from general_beckman.apply import _stamp_retry_feedback

    row = await get_task_by_workflow_step(
        mission_id, step_id, statuses=("completed", "failed"),
    )
    if row is None:
        logger.info(
            "review-route: no producer task row to re-pend",
            mission_id=mission_id, step_id=step_id,
        )
        return False

    attempts = int(row.get("worker_attempts") or 0) + 1
    max_attempts = int(row.get("max_worker_attempts") or 15)
    if attempts > max_attempts:
        logger.info(
            "review-route: producer exhausted, cannot re-pend",
            mission_id=mission_id, step_id=step_id,
            attempts=attempts, max_attempts=max_attempts,
        )
        return False

    try:
        ctx = _json.loads(row.get("context") or "{}")
        if not isinstance(ctx, dict):
            ctx = {}
    except (ValueError, TypeError):
        ctx = {}

    # Feed the reviewer rejection back into the retry prompt via the same
    # _schema_error / _prev_output channel post_execute_workflow_step reads.
    ctx["_schema_error"] = feedback
    prev_output = row.get("result") or ""
    if isinstance(prev_output, str) and prev_output.strip():
        ctx["_prev_output"] = prev_output[:6000]
    _stamp_retry_feedback(ctx, attempts)

    await update_task(
        int(row["id"]),
        status="pending",
        worker_attempts=attempts,
        max_worker_attempts=max_attempts,
        error="reviewer rejected artifact — re-pended for fix",
        error_category="quality",
        next_retry_at=None,
        context=_json.dumps(ctx),
    )
    logger.info(
        "review-route: re-pended producer with reviewer feedback",
        mission_id=mission_id, step_id=step_id,
        task_id=row["id"], worker_attempts=attempts,
    )
    return True


async def _assign_unresolved(unresolved: list[dict], candidates: list[tuple[str, str]]) -> dict:
    # TODO(task 9b): LLM fallback — ask a model to attribute each unresolved
    # issue to one of `candidates` (producer step ids the reviewer consumed).
    # For now return {} so unresolved issues fall through to founder escalation
    # rather than being silently mis-attributed.
    return {}


async def _escalate_to_founder(**kwargs) -> None:
    """Safe minimal founder-halt stub. MUST NOT raise.

    Best-effort: park the reviewer task in ``waiting_human`` so the mission
    stops auto-advancing on an unlocalisable / exhausted reviewer failure, and
    log a warning. The full founder-halt card (Continue/Branch/Abort notice)
    is wired in a later task.
    """
    # TODO(task 10): send the founder-halt card (Continue/Branch/Abort notice).
    reason = kwargs.get("reason") or "unspecified"
    reviewer_task_id = kwargs.get("reviewer_task_id")
    try:
        if reviewer_task_id is not None:
            from src.infra.db import update_task
            await update_task(int(reviewer_task_id), status="waiting_human")
    except Exception as exc:  # noqa: BLE001 — escalation must never raise
        logger.warning(
            "review escalation: could not park reviewer task",
            reviewer_task_id=reviewer_task_id, error=str(exc),
        )
    logger.warning(
        "review escalated to founder: %s", reason,
        mission_id=kwargs.get("mission_id"),
        reviewer_id=kwargs.get("reviewer_id"),
        producer=kwargs.get("producer"),
    )


async def route_review_failure(
    *, mission_id: int, reviewer_id: str, review_result: dict, workflow: dict,
) -> dict[str, Any]:
    issues = review_result.get("issues") or []
    index = build_producer_index(workflow)
    grouped, unresolved = map_tagged_issues(issues, index)

    if unresolved:
        reviewer = next((s for s in workflow["steps"] if s["id"] == reviewer_id), {})
        candidates = [
            (pid, art)
            for art in (reviewer.get("input_artifacts") or [])
            for pid in index.get(art, [])
        ]
        assigned = await _assign_unresolved(unresolved, candidates)
        if isinstance(assigned, dict):
            for pid, pissues in assigned.items():
                items = pissues if isinstance(pissues, list) else [pissues]
                grouped.setdefault(pid, []).extend(items)

    if not grouped:
        await _escalate_to_founder(
            mission_id=mission_id, reviewer_id=reviewer_id,
            review_result=review_result, workflow=workflow, reason="no_localisable_target",
        )
        return {"routed": [], "escalated": True}

    routed: list[str] = []
    escalated = False
    for pid, pissues in grouped.items():
        ok = await _repend_producer(mission_id, pid, _feedback_text(pissues))
        if ok:
            routed.append(pid)
        else:
            escalated = True
            await _escalate_to_founder(
                mission_id=mission_id, reviewer_id=reviewer_id,
                review_result=review_result, workflow=workflow,
                reason="producer_exhausted", producer=pid,
            )
    return {"routed": routed, "escalated": escalated}
