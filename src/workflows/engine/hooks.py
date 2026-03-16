"""Pre/post execution hooks for workflow steps in the orchestrator.

Handles artifact injection, output storage, conditional evaluation,
template expansion triggers, and CodingPipeline delegation detection.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from .artifacts import ArtifactStore, format_artifacts_for_prompt

logger = logging.getLogger(__name__)

# ── Module-level singleton ─────────────────────────────────────────────────

_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Return the module-level ArtifactStore singleton (lazy init)."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore(use_db=False)
    return _artifact_store


# ── Helper functions ───────────────────────────────────────────────────────


def is_workflow_step(context: dict) -> bool:
    """Check whether the task context marks this as a workflow step."""
    return bool(context.get("is_workflow_step"))


def extract_output_artifact_names(context: dict) -> list[str]:
    """Get output_artifacts list from context, defaulting to empty."""
    return context.get("output_artifacts", [])


def _parse_context(task: dict) -> dict:
    """Parse task context, handling both dict and JSON string forms."""
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    if not isinstance(ctx, dict):
        ctx = {}
    return ctx


def enrich_task_description(task: dict, artifact_contents: dict) -> str:
    """Build an enriched description with artifact context and done_when.

    Parameters
    ----------
    task:
        Task dict with "description" and optional "context".
    artifact_contents:
        Mapping of artifact name -> content (already fetched).

    Returns
    -------
    str
        The enriched description string.
    """
    instruction = task.get("description", "")
    ctx = _parse_context(task)

    context_strategy = ctx.get("context_strategy")
    done_when = ctx.get("done_when")

    parts: list[str] = [instruction]

    # Append formatted artifacts if any are available
    if artifact_contents:
        filtered = {k: v for k, v in artifact_contents.items() if v is not None}
        if filtered:
            formatted = format_artifacts_for_prompt(
                filtered, context_strategy=context_strategy
            )
            if formatted:
                parts.append(f"\n\n## Context Artifacts\n\n{formatted}")

    # Append done_when section if present
    if done_when:
        parts.append(f"\n\n## Done When\n{done_when}")

    return "".join(parts)


# ── Pre/Post hooks ─────────────────────────────────────────────────────────


async def pre_execute_workflow_step(task: dict) -> dict:
    """Pre-hook: inject artifact context into workflow step descriptions.

    If the task is not a workflow step, returns it unchanged.
    Otherwise fetches input artifacts from the store and enriches
    the task description.
    """
    ctx = _parse_context(task)
    if not is_workflow_step(ctx):
        return task

    goal_id = ctx.get("goal_id")
    input_artifact_names: list[str] = ctx.get("input_artifacts", [])

    # Fetch artifacts from store
    store = get_artifact_store()
    artifact_contents: dict[str, Optional[str]] = {}
    if goal_id is not None and input_artifact_names:
        artifact_contents = await store.collect(goal_id, input_artifact_names)

    # Enrich description
    task["description"] = enrich_task_description(task, artifact_contents)

    logger.info(
        f"[Workflow Hook] Pre-execute: injected {len(input_artifact_names)} "
        f"artifact(s) into task description"
    )

    return task


async def post_execute_workflow_step(task: dict, result: dict) -> None:
    """Post-hook: store output artifacts after successful workflow step execution.

    If the task is not a workflow step, returns immediately.
    For single output artifacts, stores the full result output.
    For multiple output artifacts, stores the full result under each name.
    """
    ctx = _parse_context(task)
    if not is_workflow_step(ctx):
        return

    goal_id = ctx.get("goal_id")
    output_names = extract_output_artifact_names(ctx)

    if not goal_id or not output_names:
        return

    store = get_artifact_store()
    output_value = result.get("output", "")

    for name in output_names:
        await store.store(goal_id, name, output_value)
        logger.info(
            f"[Workflow Hook] Post-execute: stored artifact '{name}' "
            f"for goal {goal_id} ({len(output_value)} chars)"
        )
