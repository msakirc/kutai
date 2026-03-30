"""Pre/post execution hooks for workflow steps in the orchestrator.

Handles artifact injection, output storage, conditional evaluation,
template expansion triggers, and CodingPipeline delegation detection.
"""
from __future__ import annotations

import json
from typing import Optional

from src.infra.logging_config import get_logger
from .artifacts import ArtifactStore, format_artifacts_for_prompt, get_phase_summaries
from .conditions import evaluate_condition, resolve_group
from .policies import ReviewTracker
from .quality_gates import evaluate_gate, format_gate_result

logger = get_logger("workflows.engine.hooks")

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

    mission_id = ctx.get("mission_id") or task.get("mission_id")
    input_artifact_names: list[str] = ctx.get("input_artifacts", [])

    # Fetch artifacts from store
    store = get_artifact_store()
    artifact_contents: dict[str, Optional[str]] = {}
    if mission_id is not None and input_artifact_names:
        artifact_contents = await store.collect(mission_id, input_artifact_names)

    # Inject phase summaries from earlier phases
    workflow_phase = ctx.get("workflow_phase")
    if mission_id is not None and workflow_phase:
        phase_summaries = await get_phase_summaries(store, mission_id, workflow_phase)
        if phase_summaries:
            artifact_contents.update(phase_summaries)
            # Ensure phase summaries are included at reference tier
            context_strategy = ctx.get("context_strategy")
            if isinstance(context_strategy, dict):
                ref_list = context_strategy.setdefault("reference", [])
                for sname in phase_summaries:
                    if sname not in ref_list:
                        ref_list.append(sname)
                # Re-serialize updated strategy into context so enrich picks it up
                if isinstance(task.get("context"), str):
                    ctx["context_strategy"] = context_strategy
                    task["context"] = json.dumps(ctx)
                else:
                    task["context"]["context_strategy"] = context_strategy

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

    mission_id = ctx.get("mission_id") or task.get("mission_id")
    output_names = extract_output_artifact_names(ctx)
    step_id = ctx.get("workflow_step_id", "")

    if not mission_id or not output_names:
        return

    store = get_artifact_store()
    output_value = result.get("result", "")

    for name in output_names:
        await store.store(mission_id, name, output_value)
        logger.info(
            f"[Workflow Hook] Post-execute: stored artifact '{name}' "
            f"for mission {mission_id} ({len(output_value)} chars)"
        )

    # ── Check conditional group triggers ──
    await _check_conditional_triggers(mission_id, output_names, store)

    # ── Check template expansion trigger ──
    if "implementation_backlog" in output_names:
        await _trigger_template_expansion(mission_id, output_value)

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
    if mission_id and workflow_phase:
        await _check_phase_completion(mission_id, workflow_phase)


async def _check_phase_completion(mission_id: int, phase_id: str) -> bool:
    """Detect when all tasks in a workflow phase are done and checkpoint it.

    Returns True if the phase is complete, False otherwise.
    """
    try:
        from ...infra.db import get_tasks_for_mission, get_workflow_checkpoint, upsert_workflow_checkpoint
    except ImportError as exc:
        logger.debug(f"[Workflow Hook] Phase completion check skipped (import): {exc}")
        return False

    try:
        tasks = await get_tasks_for_mission(mission_id)
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Could not fetch tasks for mission {mission_id}: {exc}")
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
        checkpoint = await get_workflow_checkpoint(mission_id)
        completed = checkpoint["completed_phases"] if checkpoint else []
        workflow_name = checkpoint["workflow_name"] if checkpoint else ""

        if phase_id not in completed:
            completed.append(phase_id)

        await upsert_workflow_checkpoint(
            mission_id=mission_id,
            workflow_name=workflow_name,
            current_phase=phase_id,
            completed_phases=completed,
        )
        logger.info(
            f"[Workflow Hook] Phase '{phase_id}' complete for mission {mission_id} "
            f"({len(phase_tasks)} tasks). Checkpoint updated."
        )
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Could not update checkpoint: {exc}")

    # Generate a summary artifact for the completed phase
    await _generate_phase_summary(mission_id, phase_id, phase_tasks)

    # ── Evaluate quality gate ──
    await _evaluate_phase_gate(mission_id, phase_id)

    return True


async def _evaluate_phase_gate(mission_id: int, phase_id: str) -> None:
    """Evaluate the quality gate for a completed phase and store the result."""
    store = get_artifact_store()
    try:
        phase_num = phase_id.replace("phase_", "")
        passed, details = await evaluate_gate(mission_id, phase_id, store)

        # Store gate result as artifact
        result_text = format_gate_result(phase_id, passed, details)
        await store.store(mission_id, f"phase_{phase_num}_gate_result", result_text)

        if details:  # Only log if there was actually a gate
            if passed:
                logger.info(
                    f"[Workflow Hook] Quality gate for '{phase_id}' PASSED "
                    f"(mission {mission_id})"
                )
            else:
                logger.warning(
                    f"[Workflow Hook] Quality gate for '{phase_id}' FAILED "
                    f"(mission {mission_id}): {result_text}"
                )
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Quality gate evaluation failed: {exc}")


async def _generate_phase_summary(
    mission_id: int, phase_id: str, phase_tasks: list[dict]
) -> None:
    """Build a structured summary from a completed phase's output artifacts.

    The summary is stored as ``phase_{N}_summary`` in the artifact store so
    that subsequent phases can receive it as context.
    """
    from .status import PHASE_NAMES

    store = get_artifact_store()

    # Collect output artifact names from all phase tasks
    output_names: list[str] = []
    for t in phase_tasks:
        ctx = _parse_context(t)
        output_names.extend(ctx.get("output_artifacts", []))

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in output_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    # Fetch artifact contents
    artifact_contents = await store.collect(mission_id, unique_names)

    # Build summary text
    phase_name = PHASE_NAMES.get(phase_id, phase_id)
    # Extract phase number for the artifact key
    try:
        phase_num = phase_id.split("_", 1)[1]
    except IndexError:
        phase_num = phase_id

    names_with_content = [
        n for n in unique_names if artifact_contents.get(n)
    ]
    artifact_count = len(names_with_content)

    lines: list[str] = [
        f"## Phase {phase_num}: {phase_name} — Summary",
        f"**Key outputs:** {', '.join(names_with_content) if names_with_content else 'none'}",
        f"**Artifacts produced:** {artifact_count}",
        "",
    ]

    for name in names_with_content:
        content = artifact_contents[name] or ""
        excerpt = content[:200]
        if len(content) > 200:
            excerpt += "..."
        lines.append(f"### {name}\n{excerpt}")
        lines.append("")

    summary_text = "\n".join(lines).rstrip()

    summary_artifact_name = f"phase_{phase_num}_summary"
    await store.store(mission_id, summary_artifact_name, summary_text)
    logger.info(
        f"[Workflow Hook] Generated summary for '{phase_id}' "
        f"({artifact_count} artifacts) -> '{summary_artifact_name}'"
    )


async def _check_conditional_triggers(
    mission_id: int, output_names: list[str], store: ArtifactStore
) -> None:
    """Evaluate conditional groups when their trigger artifact is produced."""
    try:
        from .loader import load_workflow

        wf = load_workflow("i2p_v2")
    except Exception:
        logger.debug("[Workflow Hook] Could not load workflow for conditional eval")
        return

    for group in wf.conditional_groups:
        condition_artifact = group.get("condition_artifact", "")
        if condition_artifact not in output_names:
            continue

        artifact_value = await store.retrieve(mission_id, condition_artifact)
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
                from ...infra.db import update_task_by_context_field, propagate_skips

                for step in excluded:
                    await update_task_by_context_field(
                        mission_id=mission_id,
                        field="workflow_step_id",
                        value=step,
                        status="skipped",
                    )
                # Cascade skips to downstream dependents
                skipped_count = await propagate_skips(mission_id)
                if skipped_count:
                    logger.info(
                        f"[Workflow Hook] Cascaded skip to {skipped_count} dependent tasks"
                    )
            except (ImportError, Exception) as e:
                logger.debug(
                    f"[Workflow Hook] Could not skip excluded steps: {e}"
                )


async def _trigger_template_expansion(mission_id: int, backlog_text: str) -> None:
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
        from ...infra.db import add_task as insert_task, update_task

        wf = load_workflow("i2p_v2")
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
                expanded, mission_id=mission_id, initial_context={}
            )

            # Batch insert with rollback on failure
            inserted_ids = []
            try:
                for t in tasks:
                    t.pop("depends_on_steps", None)
                    task_id = await insert_task(**t)
                    inserted_ids.append(task_id)
            except Exception as insert_err:
                # Rollback: cancel partially inserted tasks
                for tid in inserted_ids:
                    try:
                        await update_task(tid, status="cancelled")
                    except Exception:
                        pass
                logger.error(
                    f"[Workflow Hook] Partial expansion rollback for '{fid}': {insert_err}"
                )
                continue  # Skip this feature, try next one

            logger.info(
                f"[Workflow Hook] Expanded template for feature '{fid}' "
                f"({len(expanded)} steps \u2192 {len(tasks)} tasks)"
            )

    except (ImportError, Exception) as e:
        logger.debug(f"[Workflow Hook] Template expansion failed: {e}")
