"""ArtifactSummarizerAgent — wraps _llm_summarize for the post-hook pipeline."""
from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.infra.logging_config import get_logger

logger = get_logger("agents.artifact_summarizer")


class ArtifactSummarizerAgent(BaseAgent):
    name = "artifact_summarizer"
    allowed_tools: list[str] = []

    async def execute(self, task: dict) -> dict:
        from src.workflows.engine.hooks import _llm_summarize
        from src.workflows.engine.artifacts import ArtifactStore

        ctx_raw = task.get("context") or "{}"
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except (json.JSONDecodeError, ValueError):
            ctx = {}

        source_task_id = ctx.get("source_task_id")
        artifact_name = ctx.get("artifact_name")
        if source_task_id is None or not artifact_name:
            return {
                "status": "failed",
                "error": "artifact_summarizer: source_task_id/artifact_name missing",
            }

        mission_id = task.get("mission_id")
        text = ""
        if mission_id is not None:
            val = await ArtifactStore().retrieve(mission_id, artifact_name)
            if isinstance(val, str):
                text = val
        if not text:
            return {
                "status": "failed",
                "error": f"artifact '{artifact_name}' empty or missing on blackboard",
            }

        summary = await _llm_summarize(text, artifact_name)
        passed = bool(summary) and isinstance(summary, str) and len(summary) >= 50

        return {
            "status": "completed",
            "result": summary or "",
            "model": "artifact_summarizer",
            "cost": 0.0,
            "iterations": 1,
            "posthook_verdict": {
                "kind": f"summary:{artifact_name}",
                "source_task_id": source_task_id,
                "passed": passed,
                "raw": {"summary": summary or "", "artifact_name": artifact_name},
            },
        }
