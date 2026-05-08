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

    def get_system_prompt(self, task: dict) -> str:
        # NOTE: Live grading does NOT use this prompt — see src/core/grading.py
        # which injects GRADING_SYSTEM + GRADING_PROMPT directly via raw_dispatch.
        # This method exists only as a fallback if grader is ever invoked via
        # the standard agent dispatch path.
        return (
            "You are a task output grader. You evaluate completed task results "
            "for relevance, completeness, coherence, and well-formedness.\n"
            "\n"
            "## Grading Criteria\n"
            "- **Relevant** — The output addresses the original task description.\n"
            "- **Complete** — All required deliverables are present.\n"
            "- **Well-formed** — Output structure matches the expected format.\n"
            "- **Coherent** — The output is internally consistent and logical.\n"
            "\n"
            "## Rules\n"
            "- Never pass an output that is empty or clearly off-topic.\n"
            "- Always base your verdict on the actual output content, not effort.\n"
            "- Do not penalize for style if the substance is correct.\n"
            "- You must return a structured verdict — passed or failed with reasons.\n"
            "\n"
            "## Output format\n"
            "\n"
            "Live grading (src/core/grading.py) expects plain-text key-value lines "
            "parsed by `_parse_yes_no` and `_parse_text_field`:\n"
            "\n"
            "```text\n"
            "RELEVANT: YES or NO\n"
            "COMPLETE: YES or NO\n"
            "VERDICT: PASS or FAIL\n"
            "WELL_FORMED: PASS or FAIL\n"
            "COHERENT: PASS or FAIL\n"
            "SITUATION: one line, what type of problem was solved\n"
            "STRATEGY: one line, what approach worked\n"
            "TOOLS: comma-separated list of tools used effectively\n"
            "PREFERENCE: one-line user preference signal observed, or NONE\n"
            "INSIGHT: one-line reusable learning from this task, or NONE\n"
            "```\n"
            "\n"
            "If invoked via standard agent dispatch (not raw_dispatch), use this "
            "`final_answer` instead:\n"
            "\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "PASS",\n'
            '  "memories": {"insight": "..."}\n'
            "}\n"
            "```\n"
        )

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

        try:
            verdict = await grade_task(source)
        except Exception as e:
            # grade_task now auto-fails internally on parse incapability, so
            # only infra errors (DB, dispatcher with no candidates) reach here.
            logger.warning(
                f"grader dispatch exception for source #{source_task_id}: {e!r}",
                task_id=task.get("id"),
            )
            return {
                "status": "failed",
                "error": f"grader: {type(e).__name__}: {str(e)[:200]}",
            }

        # grade_task returns dict in legacy/mocked paths or GradeResult dataclass
        # in production. Normalize to dict for serialization + posthook payload.
        if isinstance(verdict, dict):
            raw_dict = verdict
            passed = bool(verdict.get("passed", False))
        else:
            raw_dict = {
                "passed": bool(verdict.passed),
                "relevant": verdict.relevant,
                "complete": verdict.complete,
                "well_formed": verdict.well_formed,
                "coherent": verdict.coherent,
                "situation": verdict.situation,
                "strategy": verdict.strategy,
                "tools": verdict.tools,
                "preference": verdict.preference,
                "insight": verdict.insight,
                "raw": verdict.raw,
            }
            passed = bool(verdict.passed)

        return {
            "status": "completed",
            "result": json.dumps(raw_dict, default=str),
            "model": raw_dict.get("grader_model", "unknown"),
            "cost": float(raw_dict.get("cost", 0.0)),
            "iterations": 1,
            "posthook_verdict": {
                "kind": "grade",
                "source_task_id": source_task_id,
                "passed": passed,
                "raw": raw_dict,
            },
        }
