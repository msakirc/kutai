"""Profile — duck-typed runtime surface, ex-agent config as data.

Mirrors the attribute surface the runtime consumes (was src/agents/base.py
BaseAgent). Static profiles return a fixed seed string from get_system_prompt;
carve-outs (oncall_agent, writer) subclass and override it.
"""
from __future__ import annotations
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
    _original_allowed_tools: object = field(default=None, repr=False)

    def get_system_prompt(self, task: dict) -> str:
        return self.system_prompt

    # NOTE: NO execute() and NO _build_context() here. Those were BaseAgent
    # methods; they leave the profile contract entirely (Task 5.5). The worker
    # (coulson) drives execution: coulson.execute(profile, task). Keeping them
    # off the Profile is what lets the leaf stay pure (they delegate to
    # src.runtime/coulson, which the leaf may not import).
