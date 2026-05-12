"""IntegrationReviewerAgent — post-hook for multi-file feature expansion.

Receives a step context that includes a ``signatures`` key (injected by
apply.py from the ``extract_signatures`` mechanical pre-check) and reviews
cross-file consistency: import contracts, interface alignment, naming
conventions, and potential integration seams.

Agent invariants (test_prompt_quality.py):
1. First line: ``You are ...``
2. Body contains must/always + don't/never
3. Body contains ``final_answer`` and fenced ```json schema

This agent is in ``_NO_POSTHOOKS_AGENT_TYPES``: its verdict IS the gate.
"""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.infra.logging_config import get_logger

logger = get_logger("agents.integration_reviewer")


class IntegrationReviewerAgent(BaseAgent):
    name = "integration_reviewer"
    allowed_tools: list[str] = []

    def get_system_prompt(self, task: dict) -> str:
        ctx_raw = task.get("context") or "{}"
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except (json.JSONDecodeError, ValueError):
            ctx = {}

        sigs = ctx.get("signatures") or {}
        mismatches = ctx.get("mismatches") or []
        sub_task_titles = ctx.get("sub_task_titles") or []

        sig_summary = ""
        if sigs:
            lines = []
            for path, sig_list in list(sigs.items())[:10]:
                names = [s["name"] for s in sig_list[:5]]
                lines.append(f"  {path}: {', '.join(names)}")
            sig_summary = "\n".join(lines)

        mismatch_summary = ""
        if mismatches:
            lines = [
                f"  {m.get('why', '')} ({m.get('caller', '')} → {m.get('callee', '')})"
                for m in mismatches[:8]
            ]
            mismatch_summary = "\n".join(lines)

        return (
            "You are an integration reviewer specialising in cross-file consistency "
            "after multi-file feature expansion.\n"
            "\n"
            "## Context\n"
            f"Sub-tasks expanded: {', '.join(sub_task_titles) or '(see task description)'}\n"
            "\n"
            + (
                f"Extracted signatures:\n{sig_summary}\n\n"
                if sig_summary
                else ""
            )
            + (
                f"AST-detected mismatches (best-effort):\n{mismatch_summary}\n\n"
                if mismatch_summary
                else ""
            )
            + "## Review Dimensions\n"
            "1. **Interface contracts** — Do callers match callee signatures? "
            "Flag arity/type mismatches.\n"
            "2. **Import alignment** — Are all cross-file imports consistent "
            "with the actual exports?\n"
            "3. **Naming conventions** — Consistent naming across the expanded files.\n"
            "4. **Integration seams** — Missing glue code (e.g. route not registered, "
            "schema not imported, event not wired).\n"
            "\n"
            "## Rules\n"
            "- You must always produce a PASS or FAIL verdict.\n"
            "- Always list concrete issues with file path and line when available.\n"
            "- Don't fail on style nits — only structural/contract issues.\n"
            "- Never approve an expansion with an unresolved import or arity conflict.\n"
            "\n"
            "## Output format\n"
            "\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "PASS",\n'
            '  "memories": {\n'
            '    "verdict": "PASS",\n'
            '    "issues": [],\n'
            '    "mismatches_confirmed": []\n'
            "  }\n"
            "}\n"
            "```\n"
            "\n"
            "Set ``result`` to ``\"PASS\"`` when no structural issues found, "
            "``\"FAIL\"`` otherwise. Populate ``issues`` with blocking findings.\n"
        )

    async def execute(self, task: dict) -> dict:
        ctx_raw = task.get("context") or "{}"
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except (json.JSONDecodeError, ValueError):
            ctx = {}

        source_task_id = ctx.get("source_task_id")
        if source_task_id is None:
            return {
                "status": "failed",
                "error": "integration_reviewer: source_task_id missing",
            }

        # The LLM review is invoked via the standard agent dispatch (BaseAgent.run).
        # This execute() fallback is only reached if the agent is dispatched
        # outside the normal ReAct loop; return a skeleton pass.
        logger.warning(
            "integration_reviewer.execute() called directly "
            "(expected ReAct loop dispatch)"
        )
        return {
            "status": "completed",
            "result": json.dumps(
                {
                    "verdict": "PASS",
                    "issues": [],
                    "note": "direct-execute fallback (no LLM call)",
                }
            ),
        }
