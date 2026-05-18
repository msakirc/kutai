"""Render the task['skills'] envelope into agent-prompt text.

Rendering ownership (spec): SkillApplication is structured data;
coulson — the agent-prompt builder — renders it. intersect never knows
prompt conventions. This module reads the plain-dict envelope intersect
attaches and produces the markdown block coulson appends to the user
context.

Phase 2 consumes the ``applies_to == "execution"`` slice only — grading
exposure is v1.1. ``inject``-class entries render to prose; ``tool``-class
entries feed the per-execution allowed_tools list (see
``tool_names_from_envelope``); ``preempt`` never rides the envelope.
"""
from __future__ import annotations


def _render_one(app: dict) -> str:
    """Render a single inject-class SkillApplication to a markdown block."""
    name = app.get("name", "unknown")
    payload = app.get("payload") or {}
    body = (payload.get("body") or "").strip()
    if app.get("render") == "prebind":
        bound = payload.get("bound_args") or {}
        arg_str = ", ".join(f"{k}: {v}" for k, v in bound.items())
        lines = [
            f"### Skill: {name} (ready-to-run)",
            f"`{name}({arg_str})`",
        ]
        if body:
            lines.append(body)
        return "\n".join(lines)
    # prose render
    lines = [f"### Skill: {name}"]
    if body:
        lines.append(body)
    return "\n".join(lines)


def render_skill_envelope(envelope: list[dict]) -> str:
    """Render the inject/execution slice of the envelope to prompt text.

    Returns an empty string when nothing applies. The heading matches
    the legacy ``format_skills_for_prompt`` heading so downstream
    expectations don't shift.
    """
    if not envelope:
        return ""
    inject = [
        a for a in envelope
        if a.get("exposure_class") == "inject"
        and a.get("applies_to") == "execution"
    ]
    if not inject:
        return ""
    blocks = ["## Relevant Skills from Library", ""]
    for app in inject:
        blocks.append(_render_one(app))
        blocks.append("")
    return "\n".join(blocks)


def tool_names_from_envelope(envelope: list[dict]) -> list[str]:
    """Return tool-class artifact names for per-execution allowed_tools."""
    return [
        a.get("name")
        for a in envelope
        if a.get("exposure_class") == "tool"
        and a.get("applies_to") == "execution"
        and a.get("name")
    ]
