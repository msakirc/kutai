"""Workflow definition loader for v2-aware workflow JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("workflows.engine.loader")

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

    Supports two calling conventions:

    1. Slash-separated: ``"shopping/quick_search"`` →
       ``WORKFLOW_DIR/shopping/quick_search.json``
    2. Plain name: ``"idea_to_product_v2"`` → strips version suffix, looks for
       ``WORKFLOW_DIR/<dir_name>/<workflow_name>.json`` (original behaviour).
       If that directory does not exist the loader scans every immediate
       sub-directory of WORKFLOW_DIR for ``<workflow_name>.json`` so that
       sub-workflows stored under a parent directory (e.g.
       ``shopping/quick_search.json``) are found automatically.
    """
    # ── 1. Explicit slash-separated path ────────────────────────────────
    if "/" in workflow_name:
        json_file = WORKFLOW_DIR / f"{workflow_name}.json"
        if not json_file.is_file():
            raise FileNotFoundError(
                f"Workflow definition file not found: {json_file}"
            )
        return json_file

    # ── 2. Plain name — strip version suffix for the directory lookup ───
    dir_name = workflow_name
    for suffix in ("_v1", "_v2", "_v3"):
        if dir_name.endswith(suffix):
            dir_name = dir_name[: -len(suffix)]
            break

    workflow_dir = WORKFLOW_DIR / dir_name
    if workflow_dir.is_dir():
        json_file = workflow_dir / f"{workflow_name}.json"
        if not json_file.is_file():
            raise FileNotFoundError(
                f"Workflow definition file not found: {json_file}"
            )
        return json_file

    # ── 3. Fallback: scan sub-directories for the file ──────────────────
    for candidate in WORKFLOW_DIR.iterdir():
        if candidate.is_dir():
            json_file = candidate / f"{workflow_name}.json"
            if json_file.is_file():
                logger.debug(
                    "Resolved '%s' via fallback scan in '%s'",
                    workflow_name,
                    candidate.name,
                )
                return json_file

    raise FileNotFoundError(
        f"Workflow directory not found: {workflow_dir} "
        f"(also scanned all sub-directories of {WORKFLOW_DIR})"
    )


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
    """Check that every ``depends_on`` reference in *wf* resolves,
    detect dependency cycles, and flag orphan steps.

    Also validates fallback_steps inside conditional groups.

    Returns a list of human-readable error strings (empty == valid).
    """
    # Collect all known step IDs (main steps + fallback steps from CGs)
    known_ids: set[str] = {s["id"] for s in wf.steps}
    for cg in wf.conditional_groups:
        for fb in cg.get("fallback_steps", []):
            known_ids.add(fb["id"])

    errors: list[str] = []

    # ── 1. Unknown-reference check ──────────────────────────────────────
    # Build adjacency list at the same time for cycle detection later
    adjacency: dict[str, list[str]] = {sid: [] for sid in known_ids}

    for step in wf.steps:
        for dep in step.get("depends_on", []):
            if dep not in known_ids:
                errors.append(
                    f"Step '{step['id']}' depends on unknown step '{dep}'"
                )
            else:
                adjacency[step["id"]].append(dep)

    for cg in wf.conditional_groups:
        for fb in cg.get("fallback_steps", []):
            for dep in fb.get("depends_on", []):
                if dep not in known_ids:
                    errors.append(
                        f"Fallback step '{fb['id']}' (group '{cg['group_id']}') "
                        f"depends on unknown step '{dep}'"
                    )
                else:
                    adjacency[fb["id"]].append(dep)

    # ── 2. Cycle detection (DFS-based) ──────────────────────────────────
    # We reverse the adjacency (depends_on → "is depended on by") and
    # look for back-edges with a standard 3-colour DFS.
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {sid: WHITE for sid in known_ids}
    cycle_participants: list[str] = []

    def _dfs_cycle(node: str, path: list[str]) -> bool:
        """Return True if a cycle is detected reachable from *node*."""
        colour[node] = GREY
        path.append(node)
        for neighbour in adjacency.get(node, []):
            if colour[neighbour] == GREY:
                # Back-edge → cycle
                cycle_start = path.index(neighbour)
                cycle_participants.extend(path[cycle_start:])
                return True
            if colour[neighbour] == WHITE:
                if _dfs_cycle(neighbour, path):
                    return True
        path.pop()
        colour[node] = BLACK
        return False

    for sid in known_ids:
        if colour[sid] == WHITE:
            if _dfs_cycle(sid, []):
                break  # one cycle is enough to report

    if cycle_participants:
        cycle_str = " → ".join(cycle_participants)
        errors.append(f"Dependency cycle detected: {cycle_str}")

    # ── 3. Orphan step detection ────────────────────────────────────────
    # Steps that have no dependencies AND no other step depends on them
    # (excluding phase-1 root steps which are legitimate roots).
    depended_on: set[str] = set()
    for deps in adjacency.values():
        depended_on.update(deps)

    for step in wf.steps:
        sid = step["id"]
        has_deps = bool(step.get("depends_on"))
        is_depended_on = sid in depended_on
        phase = step.get("phase", "")
        # Phase 1 steps are roots — they're expected to have no deps
        if not has_deps and not is_depended_on and phase != "phase_1":
            errors.append(
                f"Orphan step '{sid}' (phase {phase}): not connected "
                f"to any other step via depends_on"
            )

    return errors
