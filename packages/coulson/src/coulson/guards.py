"""Sub-iteration guards and task-type heuristics.

Pure functions. No side effects, no instance state.

Guards fire *inside* a single outer iteration (sub-iteration guards):
  - blocked_clarification: agent tried to clarify but it's suppressed
  - hallucination: agent answered without using any tools on an action task
  - search_required: task needs web search but agent answered without it
  - grounding: agent answered without writing files it declared in
                workflow ``produces``
  - self_critique: agent's own one-shot review of its diff (after grounding,
                   before format guard); uses a SEPARATE pass budget so it
                   does NOT consume MAX_SUB_CORRECTIONS slots

Public API
----------
GuardCorrection   — dataclass returned when a guard fires
MAX_SUB_CORRECTIONS  — budget for guard + format corrections within one iter
MAX_FORMAT_CORRECTIONS — budget for JSON format corrections within one iter
DATA_FETCH_TOOLS  — frozenset of tools that satisfy the data-fetch requirement
is_action_task(task)   — heuristic: does the task need real tool execution?
get_search_depth(task) — extract classification.search_depth from task.context
check_sub_iter_guards(...)  — run all five guards, return first match or None
check_grounding_sub_iter(...) — declarative produces vs tool_calls check
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.tools import TOOL_REGISTRY

from .grounding import (
    WRITE_TOOLS,
    build_grounding_message,
    extract_written_paths,
    unmatched_produces,
)

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
    tool_calls: list[dict] | None = None,
    self_critique_passes: int = 0,
) -> GuardCorrection | None:
    """Check Category-A guards that should not burn an outer iteration.

    ``profile`` is duck-typed: needs ``.name``, ``.allowed_tools``,
    ``.can_create_subtasks``, and ``._suppress_clarification``.

    ``self_critique_passes`` tracks how many self-critique sub-iterations
    have already fired for this task.  It uses a SEPARATE budget from
    ``MAX_SUB_CORRECTIONS`` — the caller increments it independently when
    a ``self_critique`` GuardCorrection is consumed.

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

    # 2b. Grounding guard (workflow steps with declared produces).
    # Fires when the agent emits final_answer but never called write_file
    # for any of the paths it was supposed to produce. Path-aware
    # complement to the heuristic hallucination guard above. Skipped if
    # the runtime didn't pass a tool_calls log (legacy callers).
    if tool_calls is not None:
        grounding_correction = check_grounding_sub_iter(
            parsed=parsed,
            task=task,
            tool_calls=tool_calls,
            suppress_guards=False,  # already gated by caller
            allowed_tools=profile.allowed_tools,
        )
        if grounding_correction is not None:
            return grounding_correction

    # 2c. Self-critique guard (fires AFTER grounding, BEFORE search/format).
    # Uses its own dedicated pass counter — does NOT consume MAX_SUB_CORRECTIONS.
    # Opt-out roles bypass entirely via SELF_CRITIQUE_OPT_OUT_AGENT_TYPES.
    from .self_critique import check_self_critique_sub_iter  # lazy import avoids circular
    sc_correction = check_self_critique_sub_iter(
        parsed=parsed,
        task=task,
        agent_type=profile.name,
        self_critique_passes=self_critique_passes,
        tool_calls=tool_calls,
    )
    if sc_correction is not None:
        return sc_correction

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


# ────────────────────────────────────────────────────────────────────────────
# Grounding guard (declarative; uses tool_calls + workflow produces)
# ────────────────────────────────────────────────────────────────────────────

def _produces_from_task(task: dict) -> list:
    """Pull workflow ``produces`` out of task.context, decoding json string
    form if needed. Returns [] when missing or malformed."""
    ctx = task.get("context") or {}
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            return []
    produces = ctx.get("produces")
    return produces if isinstance(produces, list) else []


def check_grounding_sub_iter(
    parsed: dict,
    *,
    task: dict,
    tool_calls: list[dict],
    suppress_guards: bool = False,
    allowed_tools: list[str] | None = None,
) -> GuardCorrection | None:
    """Path-aware grounding guard.

    Fires when ALL of:
      - guards not suppressed
      - parsed action is final_answer
      - workflow step declares a non-empty ``produces`` list in context
      - the agent actually HAS a write tool available (see below)
      - none of the produces entries match a successful write-tool call
        in ``tool_calls``

    Returns a ``GuardCorrection`` with a path-specific retry message that
    spells out which paths are missing and gives a concrete write_file
    example. Caller honors MAX_SUB_CORRECTIONS budget like the existing
    sub-iter guards.

    Mechanical agents and zero-produces tasks are no-ops by virtue of
    the produces empty-check.

    Write-capability gate (task #525001): a step that declares BOTH an
    ``artifact_schema`` AND a ``produces`` path has its write tools
    auto-stripped by ``_apply_auto_strip`` — the workflow engine persists
    the ``result`` to the produces path itself, so write tools are
    redundant. The agent is then physically unable to call write_file.
    Demanding a write it cannot perform loops the step to max_iterations
    and DLQs it (the file IS written, by the engine). So: when
    ``allowed_tools`` is supplied and contains none of ``WRITE_TOOLS``,
    grounding is moot — skip it. ``allowed_tools=None`` means "all tools"
    (write available) → guard proceeds, preserving prior behaviour.
    """
    if suppress_guards:
        return None
    if parsed.get("action", "final_answer") != "final_answer":
        return None
    produces = _produces_from_task(task)
    if not produces:
        return None
    if allowed_tools is not None and not (set(allowed_tools) & WRITE_TOOLS):
        # No write tool available — the engine materializes the result to
        # disk. The agent cannot satisfy a write demand; do not fire.
        return None
    written = extract_written_paths(tool_calls or [])
    missing = unmatched_produces(produces, written)
    if not missing:
        return None
    return GuardCorrection(
        guard_name="grounding",
        message=build_grounding_message(
            missing=missing,
            written=written,
            task_title=task.get("title", ""),
        ),
    )
