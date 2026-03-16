"""Bridge between workflow template steps and CodingPipeline.

When a feature_implementation_template step is tagged for pipeline delegation,
this module packages the step's context into a CodingPipeline-compatible task.
"""

import re

# Template step IDs (feat.N) that should be delegated to CodingPipeline.
PIPELINE_DELEGATE_STEPS: set[str] = {
    "feat.3", "feat.4", "feat.5", "feat.6", "feat.7",
    "feat.10", "feat.11", "feat.12", "feat.13", "feat.14",
    "feat.15", "feat.16", "feat.17", "feat.18",
}

# Agent types that qualify for pipeline delegation.
PIPELINE_DELEGATE_AGENTS: set[str] = {"implementer", "coder"}

# Pattern to extract the feat.N suffix from a full step ID like "8.F-001.feat.5".
_FEAT_PATTERN = re.compile(r"feat\.(\d+)$")

# Pattern to extract feature_id from step IDs like "8.<feature_id>.feat.<N>".
_STEP_ID_PATTERN = re.compile(r"^[^.]+\.(.+)\.feat\.\d+$")


def should_delegate_to_pipeline(template_step_id: str, agent_type: str) -> bool:
    """Decide whether a workflow step should be delegated to CodingPipeline.

    Args:
        template_step_id: Full step ID, e.g. "8.F-001.feat.5".
        agent_type: The agent assigned to this step, e.g. "implementer".

    Returns:
        True if the feat.N portion is in PIPELINE_DELEGATE_STEPS
        AND the agent_type is in PIPELINE_DELEGATE_AGENTS.
    """
    match = _FEAT_PATTERN.search(template_step_id)
    if not match:
        return False
    feat_key = f"feat.{match.group(1)}"
    return feat_key in PIPELINE_DELEGATE_STEPS and agent_type in PIPELINE_DELEGATE_AGENTS


def extract_feature_context(step_context: dict) -> tuple[str, str]:
    """Extract feature_id and feature_name from a step context dict.

    Parses the step ID pattern ``<goal>.<feature_id>.feat.<N>`` to pull out
    the feature_id segment.  The feature_name is read from
    ``step_context["workflow_context"]["feature_name"]`` when available,
    falling back to feature_id.

    Args:
        step_context: Dict containing at least ``step_id`` and optionally
            ``workflow_context.feature_name``.

    Returns:
        A ``(feature_id, feature_name)`` tuple.
    """
    step_id: str = step_context["step_id"]
    match = _STEP_ID_PATTERN.match(step_id)
    feature_id = match.group(1) if match else step_id

    workflow_ctx = step_context.get("workflow_context", {})
    feature_name = workflow_ctx.get("feature_name", feature_id)

    return feature_id, feature_name


def build_pipeline_task(
    step_title: str,
    step_instruction: str,
    goal_id: str,
    feature_name: str,
    artifact_context: str = "",
) -> dict:
    """Build a task dict compatible with CodingPipeline.

    Args:
        step_title: Human-readable title for the task.
        step_instruction: The instruction text from the template step.
        goal_id: The parent goal identifier.
        feature_name: Name of the feature being implemented.
        artifact_context: Optional additional context from prior artifacts.

    Returns:
        A dict with keys ``title``, ``description``, ``goal_id``, and ``context``.
    """
    description = f"Feature: {feature_name}\n\n{step_instruction}\n\n## Context\n{artifact_context}"

    return {
        "title": step_title,
        "description": description,
        "goal_id": goal_id,
        "context": {
            "pipeline_mode": "feature",
            "prefer_quality": True,
        },
    }
