"""Multi-file feature expansion for workflow templates.

When a workflow step targets a single logical feature but a stack requires
multiple files (e.g. model + schema + service + tests), this module expands
the parent step into ordered sub-task specs — one per file role.

Convention for target_file inference
-------------------------------------
The parent step is expected to have a ``produces`` list whose first entry
contains the primary artifact path (may contain ``{{feature}}`` tokens).
Sub-tasks inherit that path pattern, substituting the role name as a suffix
or using FILE_ROLE_TO_PATH for the canonical path template.  If the parent
produces list is empty or missing, target_file falls back to
``"{{feature}}/<role>"`` as a best-effort placeholder.

LLM fallback for missing (template_id, stack) combos
------------------------------------------------------
When ``expand_template`` finds no rule for the given ``(template_id, stack)``
pair it returns ``None``.  The caller (expander integration, wired in T2C)
should treat ``None`` as "ask the LLM to enumerate sub-tasks at runtime".
That fallback path is a T1-followup or T2C task — not implemented here.

No imports from general_beckman or other packages to avoid circular dep risk.
This module is pure data + pure functions only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SubTaskSpec:
    """Spec for a single file-role sub-task derived from a parent step."""

    step_id: str
    """Child step ID in the form ``<parent_step_id>.<role>``."""

    template_id: str
    """The originating template (e.g. ``"backend_service"``)."""

    target_file: str
    """Best-effort path template for the file this sub-task should produce.
    May contain ``{{feature}}`` tokens resolved later by the expander."""

    produces: list[str]
    """Artifact slot names this sub-task is expected to emit."""

    inherited_post_hooks: list[str]
    """Copy of the parent step's ``post_hooks`` list (not a reference)."""

    inherited_from: str
    """The parent step_id this sub-task was derived from."""


# ---------------------------------------------------------------------------
# Multi-file rules
# ---------------------------------------------------------------------------

# Keyed on (template_id, stack_slug).
# Value: ordered list of file-role names for that template × stack combo.
MULTI_FILE_RULES: dict[tuple[str, str], list[str]] = {
    ("backend_service", "fastapi+nextjs"): [
        "model",
        "schema",
        "service",
        "repository",
        "error_mapper",
        "fixtures",
        "tests",
    ],
    ("frontend_component", "fastapi+nextjs"): [
        "component",
        "hook",
        "story",
        "test",
    ],
}


# ---------------------------------------------------------------------------
# Role-to-path templates
# ---------------------------------------------------------------------------

# FILE_ROLE_TO_PATH[(template_id, stack_slug)][role] → path template string.
# Path templates may contain ``{{feature}}`` which is resolved at expansion time.
FILE_ROLE_TO_PATH: dict[tuple[str, str], dict[str, str]] = {
    ("backend_service", "fastapi+nextjs"): {
        "model":        "src/models/{{feature}}.py",
        "schema":       "src/schemas/{{feature}}.py",
        "service":      "src/services/{{feature}}_service.py",
        "repository":   "src/repositories/{{feature}}_repository.py",
        "error_mapper": "src/errors/{{feature}}_errors.py",
        "fixtures":     "tests/fixtures/{{feature}}_fixtures.py",
        "tests":        "tests/test_{{feature}}_service.py",
    },
    ("frontend_component", "fastapi+nextjs"): {
        "component": "src/components/{{feature}}/{{feature}}.tsx",
        "hook":      "src/hooks/use{{feature}}.ts",
        "story":     "src/components/{{feature}}/{{feature}}.stories.tsx",
        "test":      "src/components/{{feature}}/{{feature}}.test.tsx",
    },
}


# ---------------------------------------------------------------------------
# Expansion logic
# ---------------------------------------------------------------------------

def _infer_target_file(
    template_id: str,
    stack: str,
    role: str,
    parent_step: dict,
) -> str:
    """Return the canonical path template for a file role.

    Lookup order:
    1. ``FILE_ROLE_TO_PATH[(template_id, stack)][role]`` — preferred.
    2. First entry in ``parent_step["produces"]`` with role appended.
    3. Fallback: ``"{{feature}}/<role>"``.
    """
    role_map = FILE_ROLE_TO_PATH.get((template_id, stack), {})
    if role in role_map:
        return role_map[role]

    produces = parent_step.get("produces") or []
    if produces and isinstance(produces[0], str):
        base = produces[0]
        # Strip known extension, append role as suffix before re-adding it.
        if "." in base:
            stem, ext = base.rsplit(".", 1)
            return f"{stem}_{role}.{ext}"
        return f"{base}_{role}"

    return f"{{{{feature}}}}/{role}"


def expand_template(
    template_id: str,
    stack: str,
    parent_step: dict,
    artifacts: dict,  # noqa: ARG001  (reserved for future context injection)
) -> list[SubTaskSpec] | None:
    """Expand a workflow step into ordered per-file sub-task specs.

    Parameters
    ----------
    template_id:
        The template to expand (e.g. ``"backend_service"``).
    stack:
        The project stack slug (e.g. ``"fastapi+nextjs"``).
    parent_step:
        The workflow step dict.  Expected keys (all optional but used when
        present): ``step_id``, ``produces``, ``post_hooks``.
    artifacts:
        Current artifact registry snapshot (reserved; unused in this scaffold
        — T2C will inject feature-name resolution here).

    Returns
    -------
    list[SubTaskSpec]
        Ordered sub-task specs, one per file role.
    None
        When no rule exists for ``(template_id, stack)``; caller should fall
        back to LLM enumeration (T1-followup / T2C).
    """
    roles = MULTI_FILE_RULES.get((template_id, stack))
    if roles is None:
        return None

    parent_id = parent_step.get("step_id", "unknown")
    # Deep-copy post_hooks so child lists are independent of the parent dict.
    parent_hooks: list[str] = copy.deepcopy(parent_step.get("post_hooks") or [])

    specs: list[SubTaskSpec] = []
    for role in roles:
        target = _infer_target_file(template_id, stack, role, parent_step)
        spec = SubTaskSpec(
            step_id=f"{parent_id}.{role}",
            template_id=template_id,
            target_file=target,
            produces=[target],
            inherited_post_hooks=copy.deepcopy(parent_hooks),
            inherited_from=parent_id,
        )
        specs.append(spec)

    return specs
