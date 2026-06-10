"""Profile — duck-typed runtime surface, ex-agent config as data.

Mirrors the attribute surface the runtime consumes (was src/agents/base.py
BaseAgent). Static profiles return a fixed seed string from get_system_prompt;
carve-outs (oncall_agent, writer) subclass and override it.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

# Mirrors src/app/config.MAX_AGENT_ITERATIONS default; profiles override.
DEFAULT_MAX_ITERATIONS = 10


@dataclass
class Profile:
    name: str
    description: str = ""
    system_prompt: str = ""          # the seed (frozen reference; DB override wins)
    default_tier: str = "cheap"
    min_tier: str = "cheap"
    allowed_tools: Optional[list[str]] = None   # None = all; [] = none
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    can_create_subtasks: bool = False
    execution_pattern: str = "react_loop"        # or "single_shot"
    enable_self_reflection: bool = False
    min_confidence: int = 0
    confidence_gate: str = "fail_closed"
    markdown_prompt: str = ""        # writer carve-out alt branch (else "")

    # ── runtime-mutable per-execution attrs (NOT profile data) ──
    # Set/restored by coulson.execute(); declared here so the duck-type holds.
    _prompt_version_override: Optional[str] = field(default=None, repr=False)
    _suppress_clarification: bool = field(default=False, repr=False)
    progress_callback: Optional[Callable] = field(default=None, repr=False)
    _original_allowed_tools: Optional[list[str]] = field(default=None, repr=False)
    # Sentinel: True only when a setup step has snapshotted allowed_tools into
    # _original_allowed_tools and must restore it in execute()'s finally. A plain
    # `is not None` guard is insufficient — auto-strip legitimately snapshots None
    # (original allowed_tools is None) and must restore None. For a data Profile,
    # _original_allowed_tools is a declared field (always present), so hasattr is
    # always True; without this flag the finally would wipe allowed_tools to None.
    _tools_overridden: bool = field(default=False, repr=False)

    def get_system_prompt(self, task: dict) -> str:
        return self.system_prompt

    # NOTE: NO execute() and NO _build_context() here. Those were BaseAgent
    # methods; they leave the profile contract entirely (Task 5.5). The worker
    # (coulson) drives execution: coulson.execute(profile, task). Keeping them
    # off the Profile is what lets the leaf stay pure (they delegate to
    # src.runtime/coulson, which the leaf may not import).


# ---------------------------------------------------------------------------
# Writer carve-out helpers
# ---------------------------------------------------------------------------

def _detect_markdown_schema(task: dict) -> bool:
    """Return True if the task's artifact_schema declares a markdown output.

    Returns False when the step also declares a non-empty ``produces`` list.
    In that case the grounding guard (coulson.guards.check_grounding_sub_iter)
    requires an actual write_file call against the produces path — the inline
    prompt would tell the agent "DO NOT call write_file" and the two paths
    fight each other, exhausting iteration budgets (production 2026-05-14
    mission 69 step 0.0z: 5 attempts × 3 model swaps × max_iterations_reached).
    """
    ctx = task.get("context") or {}
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
            if isinstance(ctx, str):
                ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False
    if not isinstance(ctx, dict):
        return False
    produces = ctx.get("produces")
    if isinstance(produces, list) and produces:
        return False
    schema = ctx.get("artifact_schema")
    if not isinstance(schema, dict):
        return False
    for v in schema.values():
        if isinstance(v, dict) and v.get("type") == "markdown":
            return True
    return False


class WriterProfile(Profile):
    """Writer carve-out: switches system prompt based on artifact_schema type.

    No new fields — markdown_prompt is already declared on the base Profile
    dataclass. Overrides get_system_prompt only.
    """

    def get_system_prompt(self, task: dict) -> str:
        if _detect_markdown_schema(task):
            return self.markdown_prompt or self.system_prompt
        return self.system_prompt
