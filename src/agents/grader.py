"""GraderAgent — post-hook wrapper over src.core.grading.grade_task.

No ReAct loop. Reads `source_task_id` from the task context, fetches
the source task row, invokes `grade_task`, and returns a result dict
whose `posthook_verdict` field the Beckman rewrite layer translates
into a PostHookVerdict action.
"""
from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.infra.logging_config import get_logger

logger = get_logger("agents.grader")


class GraderAgent(BaseAgent):
    name = "grader"
    allowed_tools: list[str] = []

    async def execute(self, task: dict) -> dict:
        from src.core.grading import grade_task
        from src.infra.db import get_task

        ctx_raw = task.get("context") or "{}"
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except (json.JSONDecodeError, ValueError):
            ctx = {}

        source_task_id = ctx.get("source_task_id")
        if source_task_id is None:
            return {"status": "failed", "error": "grader: source_task_id missing"}

        source = await get_task(source_task_id)
        if source is None:
            return {
                "status": "failed",
                "error": f"grader: source task {source_task_id} missing",
            }

        verdict = await grade_task(source)
        passed = bool(verdict.get("passed", False))
        return {
            "status": "completed",
            "result": json.dumps(verdict, default=str),
            "model": verdict.get("grader_model", "unknown"),
            "cost": float(verdict.get("cost", 0.0)),
            "iterations": 1,
            "posthook_verdict": {
                "kind": "grade",
                "source_task_id": source_task_id,
                "passed": passed,
                "raw": verdict,
            },
        }
