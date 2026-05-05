"""CodeReviewerAgent — post-hook wrapper over src.core.code_review.code_review_task.

Mirrors GraderAgent. Reads source_task_id from ctx, fetches the source,
runs an LLM code review, returns a result dict whose posthook_verdict
field the Beckman rewrite layer translates into a PostHookVerdict action.
"""
from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.infra.logging_config import get_logger

logger = get_logger("agents.code_reviewer")


class CodeReviewerAgent(BaseAgent):
    name = "code_reviewer"
    allowed_tools: list[str] = []

    async def execute(self, task: dict) -> dict:
        from src.core.code_review import code_review_task
        from src.infra.db import get_task

        ctx_raw = task.get("context") or "{}"
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except (json.JSONDecodeError, ValueError):
            ctx = {}

        source_task_id = ctx.get("source_task_id")
        if source_task_id is None:
            return {"status": "failed", "error": "code_reviewer: source_task_id missing"}

        source = await get_task(source_task_id)
        if source is None:
            return {
                "status": "failed",
                "error": f"code_reviewer: source task {source_task_id} missing",
            }

        try:
            result = await code_review_task(source)
        except Exception as e:
            logger.warning(
                f"code reviewer dispatch exception for source #{source_task_id}: {e!r}",
                task_id=task.get("id"),
            )
            return {
                "status": "failed",
                "error": f"code_reviewer: {type(e).__name__}: {str(e)[:200]}",
            }

        raw_dict = {
            "passed": bool(result.passed),
            "issues": list(result.issues),
            "raw": result.raw,
        }

        return {
            "status": "completed",
            "result": json.dumps(raw_dict, default=str),
            "iterations": 1,
            "posthook_verdict": {
                "kind": "code_review",
                "source_task_id": source_task_id,
                "passed": bool(result.passed),
                "raw": raw_dict,
            },
        }
