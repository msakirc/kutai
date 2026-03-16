"""Pre/post execution hooks for workflow steps in the orchestrator.

Handles artifact injection, output storage, conditional evaluation,
template expansion triggers, and CodingPipeline delegation detection.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from .artifacts import ArtifactStore, format_artifacts_for_prompt
from .conditions import evaluate_condition, resolve_group
from .policies import ReviewTracker

logger = logging.getLogger(__name__)

# ── Module-level singleton ─────────────────────────────────────────────────

_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Return the module-level ArtifactStore singleton (lazy init)."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore(use_db=True)
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


_review_tracker = ReviewTracker()


async def post_execute_workflow_step(task: dict, result: dict) -> None:
    """Post-hook: store output artifacts, evaluate conditional groups,
    trigger template expansion, and track review cycles.

    If the task is not a workflow step, returns immediately.
    """
    ctx = _parse_context(task)
    if not is_workflow_step(ctx):
        return

    goal_id = ctx.get("goal_id")
    output_names = extract_output_artifact_names(ctx)
    step_id = ctx.get("step_id", "")

    if not goal_id or not output_names:
        return

    store = get_artifact_store()
    output_value = result.get("result", "")

    for name in output_names:
        await store.store(goal_id, name, output_value)
        logger.info(
            f"[Workflow Hook] Post-execute: stored artifact '{name}' "
            f"for goal {goal_id} ({len(output_value)} chars)"
        )

    # ── Check conditional group triggers ──
    await _check_conditional_triggers(goal_id, output_names, store)

    # ── Check template expansion trigger ──
    if "implementation_backlog" in output_names:
        await _trigger_template_expansion(goal_id, output_value)

    # ── Track review status ──
    status = result.get("status", "completed")
    if status in ("needs_review", "failed"):
        action = _review_tracker.record_failure(step_id)
        if action == "escalate":
            logger.warning(
                f"[Workflow Hook] Step '{step_id}' exceeded max review "
                f"cycles — escalating to needs_clarification"
            )

    # ── Check phase completion for checkpoint/resume support ──
    workflow_phase = ctx.get("workflow_phase")
    if goal_id and workflow_phase:
        await _check_phase_completion(goal_id, workflow_phase)


async def _check_phase_completion(goal_id: int, phase_id: str) -> bool:
    """Detect when all tasks in a workflow phase are done and checkpoint it.

    Returns True if the phase is complete, False otherwise.
    """
    try:
        from ...infra.db import get_tasks_for_goal, get_workflow_checkpoint, upsert_workflow_checkpoint
    except ImportError as exc:
        logger.debug(f"[Workflow Hook] Phase completion check skipped (import): {exc}")
        return False

    try:
        tasks = await get_tasks_for_goal(goal_id)
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Could not fetch tasks for goal {goal_id}: {exc}")
        return False

    terminal_states = {"completed", "skipped", "cancelled"}
    phase_tasks = []
    for t in tasks:
        ctx = _parse_context(t)
        if ctx.get("workflow_phase") == phase_id:
            phase_tasks.append(t)

    if not phase_tasks:
        return False

    all_done = all(t.get("status") in terminal_states for t in phase_tasks)
    if not all_done:
        return False

    # Phase complete — update checkpoint
    try:
        checkpoint = await get_workflow_checkpoint(goal_id)
        completed = checkpoint["completed_phases"] if checkpoint else []
        workflow_name = checkpoint["workflow_name"] if checkpoint else ""

        if phase_id not in completed:
            completed.append(phase_id)

        await upsert_workflow_checkpoint(
            goal_id=goal_id,
            workflow_name=workflow_name,
            current_phase=phase_id,
            completed_phases=completed,
        )
        logger.info(
            f"[Workflow Hook] Phase '{phase_id}' complete for goal {goal_id} "
            f"({len(phase_tasks)} tasks). Checkpoint updated."
        )
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Could not update checkpoint: {exc}")

    return True


async def _check_conditional_triggers(
    goal_id: int, output_names: list[str], store: ArtifactStore
) -> None:
    """Evaluate conditional groups when their trigger artifact is produced."""
    try:
        from .loader import load_workflow

        wf = load_workflow("idea_to_product_v2")
    except Exception:
        logger.debug("[Workflow Hook] Could not load workflow for conditional eval")
        return

    for group in wf.conditional_groups:
        condition_artifact = group.get("condition_artifact", "")
        if condition_artifact not in output_names:
            continue

        artifact_value = await store.retrieve(goal_id, condition_artifact)
        if artifact_value is None:
            continue

        condition_check = group.get("condition_check", "")
        result_bool = evaluate_condition(condition_check, artifact_value)
        included, excluded = resolve_group(group, artifact_value)

        logger.info(
            f"[Workflow Hook] Conditional group '{group.get('group_id')}': "
            f"condition={result_bool}, include={len(included)}, "
            f"exclude={len(excluded)} steps"
        )

        # Update task statuses in DB for excluded steps
        if excluded:
            try:
                from ...infra.db import update_task_by_context_field

                for step in excluded:
                    await update_task_by_context_field(
                        goal_id=goal_id,
                        field="step_id",
                        value=step,
                        status="skipped",
                    )
            except (ImportError, Exception) as e:
                logger.debug(
                    f"[Workflow Hook] Could not skip excluded steps: {e}"
                )


async def _trigger_template_expansion(goal_id: int, backlog_text: str) -> None:
    """Expand feature_implementation_template for each feature in backlog."""
    import json as _json

    try:
        features = _json.loads(backlog_text)
        if not isinstance(features, list):
            logger.debug("[Workflow Hook] implementation_backlog is not a list")
            return
    except (ValueError, TypeError):
        logger.debug("[Workflow Hook] Could not parse implementation_backlog as JSON")
        return

    try:
        from .loader import load_workflow
        from .expander import expand_template, expand_steps_to_tasks
        from ...infra.db import insert_task

        wf = load_workflow("idea_to_product_v2")
        template = wf.get_template("feature_implementation_template")
        if not template:
            logger.warning("[Workflow Hook] feature_implementation_template not found")
            return

        for feature in features:
            if not isinstance(feature, dict):
                continue
            fid = feature.get("id", feature.get("feature_id", "unknown"))
            fname = feature.get("name", feature.get("feature_name", "Unnamed"))

            expanded = expand_template(
                template,
                params={"feature_id": fid, "feature_name": fname},
                prefix=f"8.{fid}.",
            )

            tasks = expand_steps_to_tasks(
                expanded, goal_id=goal_id, initial_context={}
            )

            for t in tasks:
                try:
                    await insert_task(**t)
                except Exception as e:
                    logger.debug(
                        f"[Workflow Hook] Could not insert expanded task: {e}"
                    )

            logger.info(
                f"[Workflow Hook] Expanded template for feature '{fid}' "
                f"({len(expanded)} steps → {len(tasks)} tasks)"
            )

    except (ImportError, Exception) as e:
        logger.debug(f"[Workflow Hook] Template expansion failed: {e}")
