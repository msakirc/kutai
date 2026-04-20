"""Salako executor: delegate mission advance to workflow_engine."""
from __future__ import annotations


async def run(task: dict) -> dict:
    import json
    raw_payload = task.get("payload") or {}
    if isinstance(raw_payload, str):
        try:
            payload = json.loads(raw_payload)
        except Exception:
            payload = {}
    else:
        payload = dict(raw_payload)

    mission_id = payload.get("mission_id")
    completed_task_id = payload.get("completed_task_id")
    previous_result = payload.get("previous_result") or {}

    if mission_id is None or completed_task_id is None:
        return {
            "status": "failed",
            "error": "workflow_advance payload missing mission_id/completed_task_id",
        }

    from workflow_engine import advance

    result = await advance(mission_id, completed_task_id, previous_result)

    if result.status == "needs_clarification":
        return {"status": "needs_clarification", "question": result.error}
    if result.status == "failed":
        return {"status": "failed", "error": result.error}
    if result.next_subtasks:
        return {"status": "needs_subtasks", "subtasks": result.next_subtasks}
    return {"status": "completed", "result": "advance complete"}
