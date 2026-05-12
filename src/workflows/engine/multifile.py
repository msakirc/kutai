"""Multi-file feature expansion — T1B.

When a mission dial enables ``multi_file_expansion``, the expander can
decompose a single step into N per-file sub-task steps rather than asking
one agent to write all files in a single loop.

**Key types**

``SubTaskSpec``
    A single concrete file assignment within an expanded template.

``MULTI_FILE_RULES``
    Registry mapping ``(template_id, stack_slug)`` → list of ``SubTaskSpec``.
    Rules are kept minimal; callers add domain-specific rules at init time.

``FILE_ROLE_TO_PATH``
    Lookup from logical role (e.g. "model", "router", "schema") to a
    workspace-relative path template. Stack-specific; used when the rule
    itself does not supply a concrete ``produces`` path.

``expand_template``
    Entry point: given a ``template_id`` + ``stack`` + parent step + artifacts,
    return a list of ``SubTaskSpec`` (or ``None`` when no rule matches).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SubTaskSpec:
    """Specification for one file-level sub-task within a multi-file expansion.

    Attributes
    ----------
    step_id_suffix:
        Appended to the parent step's ID to form the child step ID,
        e.g. ``"model"`` → parent ``"3.4"`` becomes ``"3.4.model"``.
    name:
        Human-readable name for the sub-task (used as task title).
    instruction:
        Agent instruction / description.  Placeholders ``{feature_name}``,
        ``{stack}``, etc. are substituted by ``expand_template``.
    produces:
        List of workspace-relative paths this sub-task is expected to write.
        May contain ``{feature_name}`` placeholders.
    agent:
        Agent type for this sub-task (default: ``"coder"``).
    tools_hint:
        Optional list of tool names to hint to the agent.
    phase:
        Workflow phase string (inherits from parent if ``None``).
    extra_context:
        Arbitrary dict merged into the sub-task's ``workflow_context``.
    """

    step_id_suffix: str
    name: str
    instruction: str
    produces: list[str] = field(default_factory=list)
    agent: str = "coder"
    tools_hint: list[str] = field(default_factory=list)
    phase: str | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# File-role → path mappings (stack-specific)
# ---------------------------------------------------------------------------

FILE_ROLE_TO_PATH: dict[str, dict[str, str]] = {
    # FastAPI (Python) stack
    "fastapi": {
        "model":   "app/models/{feature_slug}.py",
        "schema":  "app/schemas/{feature_slug}.py",
        "router":  "app/routers/{feature_slug}.py",
        "service": "app/services/{feature_slug}.py",
        "tests":   "tests/test_{feature_slug}.py",
    },
    # Next.js (TypeScript) stack
    "nextjs": {
        "page":       "pages/{feature_slug}/index.tsx",
        "component":  "components/{feature_slug}/{feature_slug}.tsx",
        "api_route":  "pages/api/{feature_slug}.ts",
        "types":      "types/{feature_slug}.ts",
        "hook":       "hooks/use{FeatureSlug}.ts",
        "tests":      "__tests__/{feature_slug}.test.tsx",
    },
    # Django (Python) stack
    "django": {
        "model":   "{feature_slug}/models.py",
        "views":   "{feature_slug}/views.py",
        "urls":    "{feature_slug}/urls.py",
        "serializer": "{feature_slug}/serializers.py",
        "tests":   "{feature_slug}/tests.py",
    },
    # Generic fallback
    "generic": {
        "implementation": "src/{feature_slug}.py",
        "tests":          "tests/test_{feature_slug}.py",
    },
}


# ---------------------------------------------------------------------------
# Multi-file rule registry
# ---------------------------------------------------------------------------

#: Registry mapping (template_id, stack_slug) → list of SubTaskSpec.
#: Populated below and extensible at runtime via register_rule().
MULTI_FILE_RULES: dict[tuple[str, str], list[SubTaskSpec]] = {}


def register_rule(
    template_id: str,
    stack_slug: str,
    specs: list[SubTaskSpec],
) -> None:
    """Register a multi-file expansion rule.

    Idempotent: calling twice with the same key overwrites the previous rule.
    """
    MULTI_FILE_RULES[(template_id, stack_slug)] = specs


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------

register_rule(
    "crud_feature",
    "fastapi",
    [
        SubTaskSpec(
            step_id_suffix="model",
            name="Write SQLAlchemy model",
            instruction=(
                "Write the SQLAlchemy ORM model for the {feature_name} feature. "
                "Place it at app/models/{feature_slug}.py. "
                "Follow project conventions (Base class, __tablename__, typed columns)."
            ),
            produces=["app/models/{feature_slug}.py"],
            tools_hint=["write_file", "read_file"],
        ),
        SubTaskSpec(
            step_id_suffix="schema",
            name="Write Pydantic schemas",
            instruction=(
                "Write Pydantic request/response schemas for the {feature_name} feature "
                "at app/schemas/{feature_slug}.py. "
                "Include Create, Update, and Read variants."
            ),
            produces=["app/schemas/{feature_slug}.py"],
            tools_hint=["write_file", "read_file"],
        ),
        SubTaskSpec(
            step_id_suffix="router",
            name="Write FastAPI router",
            instruction=(
                "Write the FastAPI router for the {feature_name} feature "
                "at app/routers/{feature_slug}.py. "
                "Wire CRUD endpoints; import from schemas and models."
            ),
            produces=["app/routers/{feature_slug}.py"],
            tools_hint=["write_file", "read_file"],
        ),
        SubTaskSpec(
            step_id_suffix="tests",
            name="Write integration tests",
            instruction=(
                "Write pytest integration tests for the {feature_name} feature "
                "at tests/test_{feature_slug}.py. "
                "Cover create, read, update, delete endpoints."
            ),
            produces=["tests/test_{feature_slug}.py"],
            agent="test_generator",
            tools_hint=["write_file", "read_file"],
        ),
    ],
)

register_rule(
    "crud_feature",
    "nextjs",
    [
        SubTaskSpec(
            step_id_suffix="types",
            name="Write TypeScript types",
            instruction=(
                "Write TypeScript type definitions for the {feature_name} feature "
                "at types/{feature_slug}.ts."
            ),
            produces=["types/{feature_slug}.ts"],
            tools_hint=["write_file"],
        ),
        SubTaskSpec(
            step_id_suffix="api",
            name="Write API route handler",
            instruction=(
                "Write the Next.js API route handler for the {feature_name} feature "
                "at pages/api/{feature_slug}.ts. "
                "Import types from types/{feature_slug}.ts."
            ),
            produces=["pages/api/{feature_slug}.ts"],
            tools_hint=["write_file", "read_file"],
        ),
        SubTaskSpec(
            step_id_suffix="component",
            name="Write React component",
            instruction=(
                "Write the React component for the {feature_name} feature "
                "at components/{feature_slug}/{feature_slug}.tsx. "
                "Use the types from types/{feature_slug}.ts."
            ),
            produces=["components/{feature_slug}/{feature_slug}.tsx"],
            tools_hint=["write_file", "read_file"],
        ),
        SubTaskSpec(
            step_id_suffix="tests",
            name="Write component tests",
            instruction=(
                "Write Jest/RTL tests for the {feature_name} React component "
                "at __tests__/{feature_slug}.test.tsx."
            ),
            produces=["__tests__/{feature_slug}.test.tsx"],
            agent="test_generator",
            tools_hint=["write_file", "read_file"],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Convert a feature name to a filesystem slug."""
    import re
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "feature"


def _camelize(slug: str) -> str:
    """Convert snake_slug to CamelCase."""
    return "".join(part.capitalize() for part in slug.split("_"))


def _substitute(text: str, params: dict[str, str]) -> str:
    """Substitute {placeholder} patterns in text."""
    for k, v in params.items():
        text = text.replace("{" + k + "}", v)
    return text


def expand_template(
    template_id: str,
    stack: str,
    parent_step: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> list[SubTaskSpec] | None:
    """Expand a multi-file template into concrete SubTaskSpec list.

    Returns ``None`` when no rule is registered for ``(template_id, stack)``,
    signalling to the caller to fall back to single-step (LLM) behaviour.

    Parameters
    ----------
    template_id:
        Logical template name (e.g. ``"crud_feature"``).
    stack:
        Stack slug (e.g. ``"fastapi"``, ``"nextjs"``).
    parent_step:
        Original parent step dict — used to extract ``feature_name`` and
        ``feature_slug`` substitution params.
    artifacts:
        Mission artifact dict — may supply ``tech_stack_detected`` or
        ``feature_name`` when not present in the step.

    Returns
    -------
    list[SubTaskSpec] | None
        Concretised sub-task specs with placeholders resolved, or ``None``
        when no rule exists for the ``(template_id, stack)`` combo.
    """
    key = (template_id, stack)
    rule = MULTI_FILE_RULES.get(key)
    if rule is None:
        return None

    # Gather substitution parameters
    ctx = artifacts or {}
    step_ctx = (parent_step or {}).get("context") or {}
    if isinstance(step_ctx, str):
        import json as _json
        try:
            step_ctx = _json.loads(step_ctx)
        except Exception:
            step_ctx = {}

    feature_name: str = (
        str(step_ctx.get("feature_name") or ctx.get("feature_name") or "feature")
    )
    feature_slug = _slugify(feature_name)
    feature_camel = _camelize(feature_slug)

    params = {
        "feature_name": feature_name,
        "feature_slug": feature_slug,
        "FeatureSlug": feature_camel,
        "stack": stack,
        "template_id": template_id,
    }

    result: list[SubTaskSpec] = []
    for spec in rule:
        result.append(
            SubTaskSpec(
                step_id_suffix=spec.step_id_suffix,
                name=_substitute(spec.name, params),
                instruction=_substitute(spec.instruction, params),
                produces=[_substitute(p, params) for p in spec.produces],
                agent=spec.agent,
                tools_hint=list(spec.tools_hint),
                phase=spec.phase,
                extra_context=dict(spec.extra_context),
            )
        )
    return result
