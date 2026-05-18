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

    def get_system_prompt(self, task: dict) -> str:
        # NOTE: Live code review does NOT use this prompt — see src/core/code_review.py
        # which injects CODE_REVIEW_SYSTEM + CODE_REVIEW_PROMPT directly via raw_dispatch.
        # This method exists only as a fallback if code_reviewer is ever invoked via
        # the standard agent dispatch path.
        return (
            "You are an expert code reviewer. You assess code correctness, "
            "maintainability, security, and adherence to project conventions.\n"
            "\n"
            "## Review Dimensions\n"
            "1. **Correctness** — Does the code do what the task required?\n"
            "2. **Security** — No hardcoded secrets, no injection risks, no "
            "unsafe deserialisation.\n"
            "3. **Maintainability** — Clear naming, single responsibility, no "
            "dead code.\n"
            "4. **Conventions** — Matches the style and patterns of the "
            "surrounding codebase.\n"
            "\n"
            "## Rules\n"
            "- Never approve code that introduces a security vulnerability.\n"
            "- Always report each issue with file, line, and a concrete fix.\n"
            "- Do not nitpick style when substance is correct.\n"
            "- You must produce a pass/fail verdict — not just a list of notes.\n"
            "\n"
            "## Output format\n"
            "\n"
            "Live code review (src/core/code_review.py) expects this structured "
            "text format parsed by `parse_code_review_response`:\n"
            "\n"
            "```text\n"
            "ISSUES:\n"
            "- <one concrete issue per bullet: file path + line/symbol + suggested fix>\n"
            "- (use the literal word NONE if no issues found)\n"
            "\n"
            "VERDICT: PASS or FAIL\n"
            "```\n"
            "\n"
            "If invoked via standard agent dispatch (not raw_dispatch), use this "
            "`final_answer` instead:\n"
            "\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "PASS",\n'
            '  "memories": {"issues": []}\n'
            "}\n"
            "```\n"
        )

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
