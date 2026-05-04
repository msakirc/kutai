"""Sub-iteration guards and task-type heuristics.

Pure functions. No side effects, no instance state.

Guards fire *inside* a single outer iteration (sub-iteration guards):
  - blocked_clarification: agent tried to clarify but it's suppressed
  - hallucination: agent answered without using any tools on an action task
  - search_required: task needs web search but agent answered without it

Public API
----------
GuardCorrection   — dataclass returned when a guard fires
MAX_SUB_CORRECTIONS  — budget for guard + format corrections within one iter
MAX_FORMAT_CORRECTIONS — budget for JSON format corrections within one iter
DATA_FETCH_TOOLS  — frozenset of tools that satisfy the data-fetch requirement
is_action_task(task)   — heuristic: does the task need real tool execution?
get_search_depth(task) — extract classification.search_depth from task.context
check_sub_iter_guards(...)  — run all three guards, return first match or None
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..tools import TOOL_REGISTRY

if TYPE_CHECKING:
    pass  # profile is duck-typed; no import needed


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

# Max sub-iteration corrections (guards + format) within a single outer iteration.
MAX_SUB_CORRECTIONS: int = 3

# Max JSON format corrections (sub-iteration) before falling through to final_answer.
MAX_FORMAT_CORRECTIONS: int = 2

# Tools whose results satisfy the data-fetch requirement (hallucination guard).
DATA_FETCH_TOOLS: frozenset[str] = frozenset({
    "web_search", "api_call", "api_lookup", "http_request",
    "shopping_search", "read_file", "read_blackboard",
})


# ────────────────────────────────────────────────────────────────────────────
# Dataclass
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardCorrection:
    """Result from a sub-iteration guard check."""
    guard_name: str
    message: str


# ────────────────────────────────────────────────────────────────────────────
# Pure functions
# ────────────────────────────────────────────────────────────────────────────

def is_action_task(task: dict) -> bool:
    """Heuristically detect whether a task requires real execution (tools)
    vs. pure text generation (answering questions, writing prose).

    Used by the hallucination guard to catch models that claim to have
    performed actions without actually calling any tools.
    """
    text = (
        f"{task.get('title', '')} {task.get('description', '')}"
    ).lower().strip()

    # ── Questions are almost never action tasks ──
    question_starts = [
        "what ", "who ", "why ", "when ", "where ",
        "how does ", "how is ", "how do ",
        "explain ", "describe ", "summarize ",
        "what's ", "what is ", "do you ", "can you tell",
        "is there ", "are there ", "which ",
    ]
    if any(text.startswith(q) for q in question_starts):
        return False

    # ── Strong action verbs: almost always need tools ──
    strong_verbs = [
        "fetch", "download", "install", "deploy", "execute",
        "run ", "run:", "clone", "pull ", "push ", "start ",
        "stop ", "restart", "compile", "test ", "debug",
        "setup", "set up", "configure", "scan", "scrape",
        "crawl", "ping", "ssh ", "curl ", "grep ", "find ",
        "launch", "migrate", "import ", "export ", "research",
    ]
    if any(v in text for v in strong_verbs):
        return True

    # ── Contextual verbs that need tools ONLY with technical targets ──
    context_verbs = [
        "list", "create", "build", "write", "read",
        "check", "update", "delete", "remove", "add ",
        "modify", "edit", "open", "search", "look up",
        "analyze", "monitor", "show",
    ]
    tech_targets = [
        "file", "folder", "directory", "repo", "repos",
        "repository", "repositories", "server", "database",
        "api", "endpoint", "package", "container", "docker",
        "service", "script", "code", "project", "workspace",
        "branch", "commit", "log ", "logs", "port", "process",
        "module", "dependency", "dependencies", "config",
        # Shopping / research targets
        "product", "price", "review", "shop", "store",
        "compare", "alternative", "spec",
    ]

    has_verb = any(v in text for v in context_verbs)
    has_target = any(t in text for t in tech_targets)

    return has_verb and has_target


def get_search_depth(task: dict) -> str:
    """Extract search_depth from task classification context.

    Returns "none" if not classified or missing.
    """
    ctx = task.get("context") or {}
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    cls = ctx.get("classification", {})
    return cls.get("search_depth", "none") or "none"


def check_sub_iter_guards(
    parsed: dict,
    *,
    profile,
    iteration: int,
    tools_used: bool,
    tools_used_names: set[str],
    task: dict,
    search_depth: str,
    suppress_guards: bool,
) -> GuardCorrection | None:
    """Check Category-A guards that should not burn an outer iteration.

    ``profile`` is duck-typed: needs ``.name``, ``.allowed_tools``,
    ``.can_create_subtasks``, and ``._suppress_clarification``.

    Returns a ``GuardCorrection`` if a guard fires, or ``None`` if all pass.
    """
    if suppress_guards:
        return None

    action_type = parsed.get("action", "final_answer")

    # 1. Blocked clarification guard
    if action_type == "clarify" and profile._suppress_clarification:
        return GuardCorrection(
            guard_name="blocked_clarification",
            message=(
                "You cannot ask for clarification on this task. "
                "Work with the information you have and provide "
                "your best answer using final_answer."
            ),
        )

    # 2. Hallucination guard (action tasks)
    # Skip when: agent returned subtasks (planner's job), or the task
    # has retry context with previous output (no need to re-do tools).
    has_tools = (
        profile.allowed_tools is None or len(profile.allowed_tools) > 0
    )
    _task_ctx = task.get("context") or {}
    if isinstance(_task_ctx, str):
        try:
            _task_ctx = json.loads(_task_ctx)
        except (json.JSONDecodeError, TypeError):
            _task_ctx = {}
    # Subtask plans are valid for non-workflow planners only
    is_wf_step = bool(_task_ctx.get("is_workflow_step"))
    has_subtasks = (
        bool(parsed.get("subtasks"))
        and profile.can_create_subtasks
        and not is_wf_step
    )
    has_retry_context = bool(_task_ctx.get("_prev_output") or _task_ctx.get("_schema_error"))
    if (
        action_type == "final_answer"
        and not tools_used
        and not has_subtasks
        and not has_retry_context
        and has_tools
        and is_action_task(task)
    ):
        available = (
            list(TOOL_REGISTRY.keys())[:6]
            if profile.allowed_tools is None
            else profile.allowed_tools[:6]
        )
        tool_list = ", ".join(available)
        task_title = task.get("title", "")
        return GuardCorrection(
            guard_name="hallucination",
            message=(
                "STOP. You did NOT actually perform this task. "
                "You just described what you would do, but nothing "
                "was executed.\n\n"
                f"Your task: {task_title}\n\n"
                "You MUST call a tool to take real action. "
                f"Available tools: {tool_list}\n\n"
                "Example — to run a shell command:\n"
                "```json\n"
                '{"action": "tool_call", "tool": "shell", '
                '"args": {"command": "ls -la"}}\n'
                "```\n\n"
                "Respond with ONLY the JSON block. No explanation."
            ),
        )

    # 3. Search-required guard
    _WEB_TOOLS = {"web_search", "smart_search", "api_call", "http_request"}
    _has_web_search = (
        profile.allowed_tools is None
        or "web_search" in (profile.allowed_tools or [])
    )
    _data_fetched = bool(tools_used_names & DATA_FETCH_TOOLS)
    _web_searched = bool(tools_used_names & _WEB_TOOLS)
    # Researcher agents must use actual web tools, not just read_file.
    # Other agents: any data fetch tool satisfies the guard.
    _search_satisfied = _web_searched if profile.name == "researcher" else _data_fetched
    if (
        action_type == "final_answer"
        and _has_web_search
        and (search_depth in ("quick", "standard", "deep") or profile.name == "researcher")
        and not _search_satisfied
    ):
        task_title = task.get("title", "")
        return GuardCorrection(
            guard_name="search_required",
            message=(
                "STOP. This task requires a web search but you "
                "answered without searching. Your answer may contain "
                "fabricated information.\n\n"
                f"Task: {task_title}\n\n"
                "You MUST call web_search or api_call first to get "
                "real, up-to-date information. Example:\n"
                "```json\n"
                '{"action": "tool_call", "tool": "web_search", '
                '"args": {"query": "your search query here"}}\n'
                "```\n\n"
                "Respond with ONLY the JSON block. No explanation."
            ),
        )

    return None
