"""Workflow runner -- creates goals and tasks from workflow definitions.

Usage:
    runner = WorkflowRunner()
    goal_id = await runner.start("idea_to_product_v2", initial_input={"raw_idea": "Build a ..."})
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.workflows.engine.artifacts import ArtifactStore, format_artifacts_for_prompt
from src.workflows.engine.expander import (
    expand_steps_to_tasks,
    filter_steps_for_context,
)
from src.workflows.engine.loader import load_workflow

logger = logging.getLogger(__name__)

# ── Trigger-to-cron mapping ────────────────────────────────────────────────

_TRIGGER_CRON_MAP: dict[str, str] = {
    "daily": "0 9 * * *",
    "weekly": "0 9 * * 1",
    "continuous": "0 */4 * * *",
}


def _trigger_to_cron(trigger: str) -> Optional[str]:
    """Map a trigger description to a cron expression.

    Returns None for alert-triggered steps (manual only).
    """
    trigger_lower = trigger.lower()
    for keyword, cron in _TRIGGER_CRON_MAP.items():
        if keyword in trigger_lower:
            return cron
    # Alert-triggered or unrecognized -> no cron (manual only)
    return None


# ── Pure utility functions ─────────────────────────────────────────────────


def resolve_dependencies(
    step_dep_ids: list[str],
    step_to_task_map: dict[str, int],
) -> list[int]:
    """Convert step IDs to DB task IDs, skipping missing with a warning."""
    task_ids: list[int] = []
    for step_id in step_dep_ids:
        if step_id in step_to_task_map:
            task_ids.append(step_to_task_map[step_id])
        else:
            logger.warning(
                "Dependency step '%s' not found in step_to_task_map; skipping",
                step_id,
            )
    return task_ids


def build_step_description(
    instruction: str,
    input_artifacts: list[str],
    artifact_contents: dict[str, str],
    done_when: str = "",
) -> str:
    """Build a full task description from instruction, artifacts, and done_when.

    Combines the instruction text with formatted artifact contents (using
    ``format_artifacts_for_prompt``) and an optional done_when section.
    Notes any artifacts that are listed but not available.
    """
    parts: list[str] = [instruction]

    if input_artifacts:
        # Separate available from missing
        available: dict[str, str] = {}
        missing: list[str] = []
        for name in input_artifacts:
            if name in artifact_contents and artifact_contents[name] is not None:
                available[name] = artifact_contents[name]
            else:
                missing.append(name)

        if available:
            formatted = format_artifacts_for_prompt(available)
            parts.append(f"\n\n## Input Artifacts\n\n{formatted}")

        if missing:
            missing_str = ", ".join(missing)
            parts.append(
                f"\n\nNote: The following artifacts are not available yet: {missing_str}"
            )

    if done_when:
        parts.append(f"\n\n## Done when\n\n{done_when}")

    return "".join(parts)


# ── WorkflowRunner ─────────────────────────────────────────────────────────


class WorkflowRunner:
    """Orchestrates workflow execution: load, expand, and insert into DB."""

    def __init__(self) -> None:
        self.artifact_store = ArtifactStore(use_db=True)

    async def start(
        self,
        workflow_name: str,
        initial_input: Optional[dict] = None,
        title: Optional[str] = None,
        existing_codebase_path: Optional[str] = None,
    ) -> int:
        """Load a workflow, create a goal, expand steps, and insert tasks.

        Returns the goal_id.
        """
        # Lazy imports to avoid circular dependencies
        from src.infra.db import add_goal, add_task

        # 1. Load workflow definition
        wf = load_workflow(workflow_name)

        # 2. Create goal
        goal_title = title or f"Workflow: {wf.metadata.get('title', workflow_name)}"
        goal_description = wf.metadata.get("description", "")
        goal_context = {
            "workflow_name": workflow_name,
            "workflow_version": wf.version,
            "plan_id": wf.plan_id,
        }
        if initial_input:
            goal_context["initial_input"] = initial_input

        goal_id = await add_goal(
            title=goal_title,
            description=goal_description,
            priority=8,
            context=goal_context,
        )

        # 3. Store initial inputs as artifacts
        if initial_input:
            for key, value in initial_input.items():
                await self.artifact_store.store(
                    goal_id, key, value if isinstance(value, str) else json.dumps(value)
                )

        # 4. Store existing_codebase_path if provided
        if existing_codebase_path:
            await self.artifact_store.store(
                goal_id, "existing_codebase_path", existing_codebase_path
            )

        # 5. Filter steps based on context
        has_existing = existing_codebase_path is not None
        filtered_steps = filter_steps_for_context(wf.steps, has_existing_codebase=has_existing)

        # 6. Separate recurring vs non-recurring steps
        recurring_steps = [s for s in filtered_steps if s.get("type") == "recurring"]
        non_recurring_steps = [s for s in filtered_steps if s.get("type") != "recurring"]

        # 7. Expand non-recurring steps to tasks
        task_dicts = expand_steps_to_tasks(
            non_recurring_steps,
            goal_id=goal_id,
            initial_context=initial_input,
        )

        # 8. Insert tasks into DB, resolving dependencies
        step_to_task: dict[str, int] = {}

        for task_dict in task_dicts:
            step_id = task_dict["context"].get("workflow_step_id", "")

            # Resolve step-level dependencies to DB task IDs
            depends_on_steps = task_dict.pop("depends_on_steps", [])
            depends_on = resolve_dependencies(depends_on_steps, step_to_task)

            task_id = await add_task(
                title=task_dict["title"],
                description=task_dict["description"],
                goal_id=task_dict["goal_id"],
                agent_type=task_dict["agent_type"],
                tier=task_dict["tier"],
                priority=task_dict["priority"],
                depends_on=depends_on if depends_on else None,
                context=task_dict["context"],
            )

            step_to_task[step_id] = task_id

        # 9. Register recurring steps as scheduled tasks
        if recurring_steps:
            await self._register_recurring_steps(
                recurring_steps, goal_id, wf.plan_id, step_to_task
            )

        # 10. Store workflow metadata artifact
        template_ids = [t.get("template_id", "") for t in wf.templates]
        cg_data = [
            {"group_id": cg.get("group_id", ""), "condition": cg.get("condition", "")}
            for cg in wf.conditional_groups
        ]
        metadata = {
            "conditional_groups": cg_data,
            "template_ids": template_ids,
            "step_to_task": step_to_task,
        }
        await self.artifact_store.store(
            goal_id, "_workflow_metadata", json.dumps(metadata)
        )

        logger.info(
            "Workflow '%s' started: goal_id=%d, %d tasks created, %d recurring steps",
            workflow_name,
            goal_id,
            len(task_dicts),
            len(recurring_steps),
        )

        return goal_id

    async def _register_recurring_steps(
        self,
        recurring_steps: list[dict],
        goal_id: int,
        workflow_id: str,
        step_to_task: dict[str, int],
    ) -> None:
        """Register Phase 15 recurring steps as scheduled tasks."""
        from src.infra.db import add_scheduled_task

        for step in recurring_steps:
            step_id = step["id"]
            trigger = step.get("trigger", "")
            cron = _trigger_to_cron(trigger)

            if cron is None:
                logger.info(
                    "Recurring step '%s' has alert-based trigger; skipping cron registration",
                    step_id,
                )
                continue

            context = {
                "workflow_id": workflow_id,
                "goal_id": goal_id,
                "workflow_step_id": step_id,
                "step_to_task": step_to_task,
            }

            await add_scheduled_task(
                title=f"[{step_id}] {step.get('name', step_id)}",
                description=step.get("instruction", ""),
                cron_expression=cron,
                agent_type=step.get("agent", "executor"),
                tier="cheap",
                context=context,
            )

            logger.info(
                "Registered recurring step '%s' with cron '%s'",
                step_id,
                cron,
            )
