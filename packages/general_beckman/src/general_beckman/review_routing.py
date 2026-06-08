"""Autonomous routing of a reviewer 'fail' to the at-fault producer(s).

tag-map (deterministic) -> LLM fallback for unresolved -> re-pend each
producer's EXISTING task row with feedback. Escalate to the founder-halt only
when nothing is localisable or a producer has exhausted its normal attempts.
Per-producer attempt bounding is the existing retry rail (worker_attempts) —
there is no separate budget here.

NOTE: _repend_producer / _assign_unresolved / _escalate_to_founder are
implemented in a later task; their signatures are fixed by this module."""
from __future__ import annotations

from typing import Any

from src.workflows.engine.producer_index import build_producer_index
from coulson.posthooks.review_router import map_tagged_issues


def _feedback_text(issues: list[dict]) -> str:
    lines = [f"- [{i.get('severity')}] {i.get('problem')}" for i in issues]
    return "Reviewer rejected this artifact. Fix:\n" + "\n".join(lines)


async def _repend_producer(mission_id: int, step_id: str, feedback: str) -> bool:
    raise NotImplementedError  # implemented in a later task


async def _assign_unresolved(unresolved: list[dict], candidates: list[tuple[str, str]]) -> dict:
    raise NotImplementedError  # implemented in a later task


async def _escalate_to_founder(**kwargs) -> None:
    raise NotImplementedError  # implemented in a later task


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
