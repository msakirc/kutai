"""Workflow definition loader for v2-aware workflow JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

WORKFLOW_DIR = Path(__file__).parent.parent  # src/workflows/


@dataclass
class WorkflowDefinition:
    """Parsed and indexed representation of a workflow JSON definition."""

    plan_id: str
    version: str
    metadata: dict
    phases: list[dict]
    steps: list[dict]
    templates: list[dict] = field(default_factory=list)
    conditional_groups: list[dict] = field(default_factory=list)

    # --- internal indexes, built on __post_init__ ---
    _step_index: dict[str, dict] = field(default_factory=dict, repr=False)
    _template_index: dict[str, dict] = field(default_factory=dict, repr=False)
    _cg_index: dict[str, dict] = field(default_factory=dict, repr=False)
    _phase_index: dict[str, dict] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._step_index = {s["id"]: s for s in self.steps}
        self._template_index = {t["template_id"]: t for t in self.templates}
        self._cg_index = {cg["group_id"]: cg for cg in self.conditional_groups}
        self._phase_index = {p["id"]: p for p in self.phases}

    # --- helper methods ---

    def get_step(self, step_id: str) -> Optional[dict]:
        """Return a step by its id, or None."""
        return self._step_index.get(step_id)

    def get_phase_steps(self, phase_id: str) -> list[dict]:
        """Return all steps belonging to *phase_id*."""
        return [s for s in self.steps if s.get("phase") == phase_id]

    def get_template(self, template_id: str) -> Optional[dict]:
        """Return a template by its template_id, or None."""
        return self._template_index.get(template_id)

    def get_conditional_group(self, group_id: str) -> Optional[dict]:
        """Return a conditional group by its group_id, or None."""
        return self._cg_index.get(group_id)

    def get_recurring_steps(self) -> list[dict]:
        """Return all steps whose type is 'recurring'."""
        return [s for s in self.steps if s.get("type") == "recurring"]

    def get_phase(self, phase_id: str) -> Optional[dict]:
        """Return a phase dict by its id, or None."""
        return self._phase_index.get(phase_id)


def _resolve_workflow_path(workflow_name: str) -> Path:
    """Resolve *workflow_name* to a JSON file path.

    Name normalization: strip ``_v1`` / ``_v2`` suffixes for the directory
    lookup but use the exact name for the file match.
    """
    # Directory name strips version suffixes
    dir_name = workflow_name
    for suffix in ("_v1", "_v2", "_v3"):
        if dir_name.endswith(suffix):
            dir_name = dir_name[: -len(suffix)]
            break

    workflow_dir = WORKFLOW_DIR / dir_name
    if not workflow_dir.is_dir():
        raise FileNotFoundError(
            f"Workflow directory not found: {workflow_dir}"
        )

    json_file = workflow_dir / f"{workflow_name}.json"
    if not json_file.is_file():
        raise FileNotFoundError(
            f"Workflow definition file not found: {json_file}"
        )

    return json_file


def load_workflow(workflow_name: str) -> WorkflowDefinition:
    """Load a workflow JSON and return a :class:`WorkflowDefinition`.

    Parameters
    ----------
    workflow_name:
        Logical name such as ``idea_to_product_v2``.  The directory is
        derived by stripping the version suffix (``idea_to_product``).
    """
    path = _resolve_workflow_path(workflow_name)
    data = json.loads(path.read_text(encoding="utf-8"))

    conditional_groups = data.get("metadata", {}).get("conditional_groups", [])

    return WorkflowDefinition(
        plan_id=data["plan_id"],
        version=data["version"],
        metadata=data.get("metadata", {}),
        phases=data.get("phases", []),
        steps=data.get("steps", []),
        templates=data.get("templates", []),
        conditional_groups=conditional_groups,
    )


def validate_dependencies(wf: WorkflowDefinition) -> list[str]:
    """Check that every ``depends_on`` reference in *wf* resolves.

    Also validates fallback_steps inside conditional groups.

    Returns a list of human-readable error strings (empty == valid).
    """
    # Collect all known step IDs (main steps + fallback steps from CGs)
    known_ids: set[str] = {s["id"] for s in wf.steps}
    for cg in wf.conditional_groups:
        for fb in cg.get("fallback_steps", []):
            known_ids.add(fb["id"])

    errors: list[str] = []

    # Validate main steps
    for step in wf.steps:
        for dep in step.get("depends_on", []):
            if dep not in known_ids:
                errors.append(
                    f"Step '{step['id']}' depends on unknown step '{dep}'"
                )

    # Validate fallback steps inside conditional groups
    for cg in wf.conditional_groups:
        for fb in cg.get("fallback_steps", []):
            for dep in fb.get("depends_on", []):
                if dep not in known_ids:
                    errors.append(
                        f"Fallback step '{fb['id']}' (group '{cg['group_id']}') "
                        f"depends on unknown step '{dep}'"
                    )

    return errors
