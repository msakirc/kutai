"""Workflow runner -- creates missions and tasks from workflow definitions.

Usage:
    runner = WorkflowRunner()
    mission_id = await runner.start("i2p_v3", initial_input={"raw_idea": "Build a ..."})
"""

from __future__ import annotations

import json
from typing import Optional

from src.infra.logging_config import get_logger
from src.workflows.engine.artifacts import ArtifactStore, format_artifacts_for_prompt
from src.workflows.engine.expander import (
    expand_steps_to_tasks,
    filter_steps_for_context,
)
from src.workflows.engine.loader import load_workflow, validate_dependencies

logger = get_logger("workflows.engine.runner")

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

    async def find_resumable(self, workflow_name: str) -> Optional[int]:
        """Find an active mission with a matching workflow checkpoint."""
        from src.infra.db import get_active_missions, get_workflow_checkpoint

        missions = await get_active_missions()
        for mission in missions:
            ctx = mission.get("context", "{}")
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except (json.JSONDecodeError, TypeError):
                    ctx = {}
            if not isinstance(ctx, dict):
                ctx = {}
            if ctx.get("workflow_name") != workflow_name:
                continue

            checkpoint = await get_workflow_checkpoint(mission["id"])
            if checkpoint and checkpoint.get("workflow_name") == workflow_name:
                return mission["id"]
        return None

    async def resume(self, mission_id: int) -> int:
        """Resume a workflow from its last checkpoint.

        Resets failed/stuck tasks to pending, identifies steps not yet
        created as tasks, and inserts them. Returns mission_id.
        Raises ValueError if checkpoint not found.
        """
        from src.infra.db import (
            get_tasks_for_mission, get_workflow_checkpoint,
            update_task, add_task,
        )

        checkpoint = await get_workflow_checkpoint(mission_id)
        if checkpoint is None:
            raise ValueError(
                f"Mission {mission_id} has no workflow checkpoint — cannot resume"
            )

        wf_name = checkpoint["workflow_name"]
        wf = load_workflow(wf_name)

        # Warm artifact cache
        await self.artifact_store.warm_cache(mission_id)

        tasks = await get_tasks_for_mission(mission_id)

        # Identify completed step IDs and reset failed tasks
        completed_step_ids: set[str] = set()
        existing_step_ids: set[str] = set()
        for t in tasks:
            ctx = t.get("context", "{}")
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except (json.JSONDecodeError, TypeError):
                    ctx = {}
            sid = ctx.get("workflow_step_id", "")
            if sid:
                existing_step_ids.add(sid)
            if t.get("status") in ("completed", "skipped"):
                completed_step_ids.add(sid)
            elif t.get("status") in ("failed", "needs_clarification"):
                await update_task(t["id"], status="pending", retry_count=0, error=None)

        # Find and insert missing steps
        missing_steps = [s for s in wf.steps if s["id"] not in existing_step_ids]
        if missing_steps:
            task_dicts = expand_steps_to_tasks(missing_steps, mission_id=mission_id, initial_context={})
            step_to_task: dict[str, int] = {}
            for t in tasks:
                ctx = t.get("context", "{}")
                if isinstance(ctx, str):
                    try:
                        ctx = json.loads(ctx)
                    except (json.JSONDecodeError, TypeError):
                        ctx = {}
                sid = ctx.get("workflow_step_id", "")
                if sid:
                    step_to_task[sid] = t["id"]

            for task_dict in task_dicts:
                step_id = task_dict["context"].get("workflow_step_id", "")
                depends_on_steps = task_dict.pop("depends_on_steps", [])
                depends_on = resolve_dependencies(depends_on_steps, step_to_task)
                task_id = await add_task(
                    title=task_dict["title"],
                    description=task_dict["description"],
                    mission_id=task_dict["mission_id"],
                    agent_type=task_dict["agent_type"],
                    tier=task_dict["tier"],
                    priority=task_dict["priority"],
                    depends_on=depends_on if depends_on else None,
                    context=task_dict["context"],
                )
                step_to_task[step_id] = task_id

        logger.info(
            "Workflow '%s' resumed: mission_id=%d, %d completed, %d new tasks inserted",
            wf_name, mission_id, len(completed_step_ids), len(missing_steps),
        )
        return mission_id

    async def preview(
        self,
        workflow_name: str,
        initial_input: Optional[dict] = None,
        existing_codebase_path: Optional[str] = None,
    ) -> dict:
        """Preview what a workflow would create without actually starting it.

        Returns a dict with:
        - total_steps: int
        - phases: list of {phase_id, phase_name, step_count, agents}
        - estimated_cost: float (rough estimate based on step count and agent types)
        - recurring_steps: int
        - conditional_groups: int
        - templates: int
        """
        wf = load_workflow(workflow_name)

        # Validate DAG integrity before preview
        dag_errors = validate_dependencies(wf)
        if dag_errors:
            logger.warning(
                "Workflow '%s' has DAG issues: %s", workflow_name, dag_errors
            )

        has_existing = existing_codebase_path is not None
        filtered_steps = filter_steps_for_context(wf.steps, has_existing_codebase=has_existing)

        recurring = [s for s in filtered_steps if s.get("type") == "recurring"]
        non_recurring = [s for s in filtered_steps if s.get("type") != "recurring"]

        # Group by phase
        phases: dict[str, dict] = {}
        for step in non_recurring:
            phase = step.get("phase", "unknown")
            if phase not in phases:
                phases[phase] = {"phase_id": phase, "steps": [], "agents": set()}
            phases[phase]["steps"].append(step)
            phases[phase]["agents"].add(step.get("agent", "executor"))

        # Count template expansion estimate
        template_step_count = 0
        for tmpl in wf.templates:
            steps_in_template = len(tmpl.get("steps", []))
            # Estimate 3 features average if we don't have backlog yet
            estimated_features = 3
            template_step_count += steps_in_template * estimated_features

        # Cost estimate: rough heuristic
        # cheap tier ~ $0.001/step, auto ~ $0.01/step, expensive ~ $0.05/step
        COST_PER_STEP = {"cheap": 0.001, "auto": 0.01, "expensive": 0.05}
        estimated_cost = 0.0
        for step in non_recurring:
            tier = step.get("tier", "auto")
            estimated_cost += COST_PER_STEP.get(tier, 0.01)
        estimated_cost += template_step_count * COST_PER_STEP["auto"]

        # Build phase summaries
        from .status import PHASE_NAMES

        phase_list = []
        for phase_id in sorted(phases.keys()):
            p = phases[phase_id]
            phase_name = PHASE_NAMES.get(phase_id, phase_id)
            phase_list.append({
                "phase_id": phase_id,
                "phase_name": phase_name,
                "step_count": len(p["steps"]),
                "agents": sorted(p["agents"]),
            })

        return {
            "workflow_name": workflow_name,
            "title": wf.metadata.get("title", workflow_name),
            "total_steps": len(non_recurring) + template_step_count,
            "direct_steps": len(non_recurring),
            "template_estimated_steps": template_step_count,
            "phases": phase_list,
            "recurring_steps": len(recurring),
            "conditional_groups": len(wf.conditional_groups),
            "templates": len(wf.templates),
            "estimated_cost": round(estimated_cost, 2),
            "dag_warnings": dag_errors,
        }

    async def start(
        self,
        workflow_name: str,
        initial_input: Optional[dict] = None,
        title: Optional[str] = None,
        existing_codebase_path: Optional[str] = None,
    ) -> int:
        """Load a workflow, create a mission, expand steps, and insert tasks.

        Returns the mission_id.
        """
        # Lazy imports to avoid circular dependencies
        from src.infra.db import add_mission, add_task

        # 1. Load workflow definition
        wf = load_workflow(workflow_name)

        # 1b. Validate DAG integrity — block on cycles/unknown refs, warn on orphans
        dag_errors = validate_dependencies(wf)
        critical_errors = [
            e for e in dag_errors
            if "cycle" in e.lower() or "unknown step" in e.lower()
        ]
        if critical_errors:
            raise ValueError(
                f"Workflow '{workflow_name}' has critical DAG errors and "
                f"cannot be started:\n" + "\n".join(critical_errors)
            )
        if dag_errors:
            logger.warning(
                "Workflow '%s' has non-critical DAG warnings: %s",
                workflow_name, dag_errors,
            )

        # 2. Create mission
        mission_title = title or f"Workflow: {wf.metadata.get('title', workflow_name)}"
        mission_description = wf.metadata.get("description", "")
        # Workflow-level timeout: default 72 hours for full workflows
        timeout_hours = wf.metadata.get("timeout_hours", 72)

        mission_context = {
            "workflow_name": workflow_name,
            "workflow_version": wf.version,
            "plan_id": wf.plan_id,
            "workflow_timeout_hours": timeout_hours,
        }
        if initial_input:
            mission_context["initial_input"] = initial_input

        mission_id = await add_mission(
            title=mission_title,
            description=mission_description,
            priority=8,
            context=mission_context,
        )

        # 3. Store initial inputs as artifacts
        if initial_input:
            for key, value in initial_input.items():
                await self.artifact_store.store(
                    mission_id, key, value if isinstance(value, str) else json.dumps(value)
                )

        # 4. Store existing_codebase_path if provided
        if existing_codebase_path:
            await self.artifact_store.store(
                mission_id, "existing_codebase_path", existing_codebase_path
            )

        # 5. Filter steps based on context
        has_existing = existing_codebase_path is not None
        filtered_steps = filter_steps_for_context(wf.steps, has_existing_codebase=has_existing)

        # 5b. Evaluate skip_when conditions
        from .expander import filter_skipped_steps
        active_conditions: set[str] = set()
        if initial_input:
            # User-provided skip conditions
            active_conditions.update(initial_input.get("skip_conditions", []))

        non_skipped, skipped_steps = filter_skipped_steps(filtered_steps, active_conditions)
        if skipped_steps:
            logger.info(
                "Skipping %d steps due to conditions: %s",
                len(skipped_steps),
                active_conditions,
            )
            filtered_steps = non_skipped

        # 6. Separate recurring vs non-recurring steps
        recurring_steps = [s for s in filtered_steps if s.get("type") == "recurring"]
        non_recurring_steps = [s for s in filtered_steps if s.get("type") != "recurring"]

        # 7. Expand non-recurring steps to tasks
        task_dicts = expand_steps_to_tasks(
            non_recurring_steps,
            mission_id=mission_id,
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
                mission_id=task_dict["mission_id"],
                agent_type=task_dict["agent_type"],
                tier=task_dict["tier"],
                priority=task_dict["priority"],
                depends_on=depends_on if depends_on else None,
                context=task_dict["context"],
            )

            step_to_task[step_id] = task_id

        # 8b. Log skipped steps (add_task does not support status param)
        if skipped_steps:
            skipped_ids = [s.get("id", "?") for s in skipped_steps]
            logger.info(
                "Skipped %d steps due to skip_when conditions: %s",
                len(skipped_steps),
                skipped_ids,
            )

        # 9. Register recurring steps as scheduled tasks
        if recurring_steps:
            await self._register_recurring_steps(
                recurring_steps, mission_id, wf.plan_id, step_to_task
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
            mission_id, "_workflow_metadata", json.dumps(metadata)
        )

        logger.info(
            "Workflow '%s' started: mission_id=%d, %d tasks created, %d recurring steps",
            workflow_name,
            mission_id,
            len(task_dicts),
            len(recurring_steps),
        )

        return mission_id

    async def _register_recurring_steps(
        self,
        recurring_steps: list[dict],
        mission_id: int,
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
                "mission_id": mission_id,
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
