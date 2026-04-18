"""Expand workflow definitions into concrete tasks for the orchestrator.

Handles v2 features: Phase -1 conditional inclusion, recurring step types,
template expansion with context_strategy, and agent name mapping.
"""

from __future__ import annotations

from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("workflows.engine.expander")

DIFFICULTY_MAP: dict[str, int] = {
    "easy": 3,
    "medium": 6,
    "hard": 8,
}

# Maps workflow agent names to system agent types.
# Most map 1:1; only special case is router -> executor.
AGENT_MAP: dict[str, str] = {
    "router": "executor",
}


def map_agent_type(agent_name: str) -> str:
    """Map a workflow agent name to the system agent type.

    Uses :data:`AGENT_MAP` for known overrides; unmapped names pass through
    unchanged.
    """
    return AGENT_MAP.get(agent_name, agent_name)


def filter_steps_for_context(
    steps: list[dict],
    has_existing_codebase: bool = False,
) -> list[dict]:
    """Filter steps based on project context.

    If *has_existing_codebase* is ``False``, steps belonging to
    ``phase_-1`` are excluded (they only apply when onboarding an
    existing project).  Otherwise all steps are returned.
    """
    if has_existing_codebase:
        return list(steps)
    return [s for s in steps if s.get("phase") != "phase_-1"]


def _phase_to_priority(phase: str) -> int:
    """Derive a numeric priority from a phase ID.

    Phase -1 and 0 receive priority 10 (highest).  Phase 15 receives
    priority 1 (lowest).  Intermediate phases are linearly interpolated.
    """
    # Extract the numeric part from "phase_N" or "phase_-1"
    try:
        phase_num = int(phase.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 5  # default for unparseable phases

    if phase_num <= 0:
        return 10
    if phase_num >= 15:
        return 1
    # Linear interpolation: phase 1 -> 9, phase 14 -> 2
    return max(1, 10 - phase_num)


def expand_steps_to_tasks(
    steps: list[dict],
    mission_id: str,
    initial_context: Optional[dict] = None,
) -> list[dict]:
    """Convert workflow step dicts into task dicts for DB insertion.

    Parameters
    ----------
    steps:
        List of step dicts from a :class:`WorkflowDefinition`.
    mission_id:
        The mission ID to associate each task with.
    initial_context:
        Optional dict of initial context (e.g. user idea) to propagate
        into each task's ``workflow_context``.

    Returns
    -------
    list[dict]
        Task dicts ready for DB insertion.  ``depends_on_steps`` contains
        step ID strings; the runner resolves these to actual DB task IDs.
    """
    tasks: list[dict] = []

    for step in steps:
        step_id = step["id"]
        phase = step.get("phase", "phase_0")

        context: dict = {
            "workflow_step_id": step_id,
            "step_name": step.get("name", ""),
            "workflow_phase": phase,
            "input_artifacts": step.get("input_artifacts", []),
            "output_artifacts": step.get("output_artifacts", []),
            "may_need_clarification": step.get("may_need_clarification", False),
            "is_workflow_step": True,
        }

        # Optional fields — only include if present on the step
        if "condition" in step:
            context["condition"] = step["condition"]
        if "type" in step:
            context["step_type"] = step["type"]
        if "trigger" in step:
            context["trigger"] = step["trigger"]
        if "done_when" in step:
            context["done_when"] = step["done_when"]
        if initial_context is not None:
            context["workflow_context"] = initial_context

        # v3 fields — difficulty, tools_hint, artifact_schema, skip_when
        difficulty = step.get("difficulty")
        if difficulty and difficulty in DIFFICULTY_MAP:
            context["difficulty"] = DIFFICULTY_MAP[difficulty]
            if difficulty == "hard":
                context["needs_thinking"] = True
                context["prefer_quality"] = True

        tools_hint = step.get("tools_hint")
        if tools_hint and isinstance(tools_hint, list):
            context["tools_hint"] = tools_hint

        api_hints = step.get("api_hints")
        if api_hints and isinstance(api_hints, list):
            context["api_hints"] = api_hints

        artifact_schema = step.get("artifact_schema")
        if artifact_schema and isinstance(artifact_schema, dict):
            context["artifact_schema"] = artifact_schema

        skip_when = step.get("skip_when")
        if skip_when and isinstance(skip_when, list):
            context["skip_when"] = skip_when

        if step.get("triggers_clarification"):
            context["triggers_clarification"] = True

        # Mechanical-executor steps (salako): propagate executor tag + payload
        # into context so the orchestrator can route them without an LLM call.
        agent_name = step.get("agent", "executor")
        if step.get("executor") == "mechanical" or agent_name == "mechanical":
            context["executor"] = "mechanical"
            if "payload" in step:
                context["payload"] = step["payload"]

        task = {
            "title": f"[{step_id}] {step['name']}",
            "description": step.get("instruction", ""),
            "agent_type": map_agent_type(agent_name),
            "mission_id": mission_id,
            "depends_on_steps": list(step.get("depends_on", [])),
            "context": context,
            "priority": _phase_to_priority(phase),
            "tier": "auto",
        }

        tasks.append(task)

    return tasks


def expand_template(
    template: dict,
    params: dict,
    prefix: str = "",
) -> list[dict]:
    """Expand a template into concrete step dicts.

    Parameters
    ----------
    template:
        A template dict containing ``steps``, ``context_artifacts``,
        and optionally ``context_strategy``.
    params:
        Parameter values to substitute into step instructions.
        Placeholders like ``{feature_name}`` are replaced.
    prefix:
        Prefix for generated step IDs.  If non-empty, IDs become
        ``"{prefix}.{template_step_id}"``.

    Returns
    -------
    list[dict]
        Concrete step dicts with substituted instructions and propagated
        context strategy.
    """
    context_artifacts = template.get("context_artifacts", [])
    context_strategy = template.get("context_strategy")
    expanded: list[dict] = []

    for tpl_step in template.get("steps", []):
        tpl_step_id = tpl_step["template_step_id"]

        # Build the step ID
        if prefix:
            step_id = f"{prefix}.{tpl_step_id}"
        else:
            step_id = tpl_step_id

        # Parameter substitution in instruction
        instruction = tpl_step.get("instruction", "")
        for param_name, param_value in params.items():
            instruction = instruction.replace(f"{{{param_name}}}", str(param_value))

        # Prefix artifact names with feature_id to avoid collisions
        # across features.  e.g. "backend_service_files" becomes
        # "auth__backend_service_files" for feature_id="auth".
        feature_id = params.get("feature_id", "")
        art_prefix = f"{feature_id}__" if feature_id else ""

        output_arts = [
            f"{art_prefix}{a}" for a in tpl_step.get("output_artifacts", [])
        ]
        # Input artifacts: prefix template-local refs, keep global refs as-is
        template_output_names = set()
        for ts in template.get("steps", []):
            template_output_names.update(ts.get("output_artifacts", []))

        raw_inputs = tpl_step.get("input_artifacts", context_artifacts)
        input_arts = [
            f"{art_prefix}{a}" if a in template_output_names else a
            for a in raw_inputs
        ]

        step: dict = {
            "id": step_id,
            "name": tpl_step.get("name", ""),
            "agent": tpl_step.get("agent", "executor"),
            "instruction": instruction,
            "output_artifacts": output_arts,
            "input_artifacts": input_arts,
        }

        # Propagate condition if present
        if "condition" in tpl_step:
            step["condition"] = tpl_step["condition"]

        # Propagate context_strategy from template
        if context_strategy is not None:
            step["context_strategy"] = dict(context_strategy)

        # Propagate v3 fields from template steps if present
        if "difficulty" in tpl_step:
            step["difficulty"] = tpl_step["difficulty"]
        if "tools_hint" in tpl_step:
            step["tools_hint"] = list(tpl_step["tools_hint"])
        if "artifact_schema" in tpl_step:
            step["artifact_schema"] = dict(tpl_step["artifact_schema"])

        expanded.append(step)

    return expanded


def filter_skipped_steps(
    steps: list[dict],
    active_skip_conditions: set[str],
) -> tuple[list[dict], list[dict]]:
    """Split steps into (active, skipped) based on skip_when conditions.

    A step is skipped if ANY of its skip_when conditions are in active_skip_conditions.
    """
    active = []
    skipped = []
    for step in steps:
        skip_when = step.get("skip_when", [])
        if skip_when and active_skip_conditions.intersection(skip_when):
            skipped.append(step)
        else:
            active.append(step)
    return active, skipped
