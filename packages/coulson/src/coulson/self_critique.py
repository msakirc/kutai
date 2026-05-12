"""Self-critique sub-iteration guard.

Pure functions. No LLM calls — those happen via the ReAct outer loop's
next sub-iteration when this guard fires and injects a critic-shaped prompt.

The guard fires ONCE per task execution (MAX_SELF_CRITIQUE_PASSES = 1)
on agents that emit files (have ``produces`` in context).  Non-emitting
and already-reviewer roles are opted out via SELF_CRITIQUE_OPT_OUT_AGENT_TYPES.

Public API
----------
MAX_SELF_CRITIQUE_PASSES         — int constant (1 pass)
SELF_CRITIQUE_OPT_OUT_AGENT_TYPES — frozenset of agent names that skip
build_self_critique_message()    — critic-shaped prompt for the same agent
check_self_critique_sub_iter()   — guard entry-point; returns GuardCorrection
                                   or None
"""
from __future__ import annotations

import json

from .guards import GuardCorrection


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

MAX_SELF_CRITIQUE_PASSES: int = 1

# Roles that opt out of self-critique:
#  - code_reviewer / integration_reviewer: already a reviewer — looping back
#    creates infinite review chains
#  - researcher: synthesises prose, no file-emit target
#  - decider: pure reasoning, no produces
#  - planner: produces subtasks not files
#  - interviewer: conversation-only role
SELF_CRITIQUE_OPT_OUT_AGENT_TYPES: frozenset[str] = frozenset({
    "code_reviewer",
    "researcher",
    "decider",
    "planner",
    "interviewer",
    "integration_reviewer",
})


# ────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ────────────────────────────────────────────────────────────────────────────

def build_self_critique_message(
    diff_summary: str,
    produces: list[str],
    agent_type: str,
) -> str:
    """Build a critic-shaped one-shot re-review prompt.

    The agent (same role, not a separate reviewer) is asked to verify its
    own output against the declared produces paths.  It must respond with
    a structured JSON verdict so the caller can decide whether to accept or
    re-do.

    Parameters
    ----------
    diff_summary:
        Short summary of what was written (e.g. extracted from tool_calls
        or the task description).  Passed verbatim into the prompt so the
        model has context without a full file re-read.
    produces:
        List of paths / glob patterns the step declared it would produce.
        Each appears in the prompt so the model can check completeness.
    agent_type:
        Name of the current agent (for personalised framing).
    """
    if not isinstance(produces, list):
        produces = []

    paths_block = "\n".join(f"  - {p}" for p in produces) or "  (none declared)"

    return (
        f"You are acting as your own critic ({agent_type} self-review).\n\n"
        "Review the work you just completed for the following declared output "
        "paths:\n"
        f"{paths_block}\n\n"
        f"Work summary:\n{diff_summary or '(no summary provided)'}\n\n"
        "Check for:\n"
        "  1. Missing or empty files that were declared in the path list above\n"
        "  2. Obvious correctness errors visible from the summary alone\n"
        "  3. Incomplete implementations (stubs, TODOs, placeholder content)\n\n"
        "Respond with ONLY a JSON block in this exact schema:\n"
        "```json\n"
        '{"verdict": "clean"|"issues", "findings": ['
        '{"severity": "error"|"warning", "file": "<path>", "why": "<reason>"}]}\n'
        "```\n\n"
        'Use "clean" when all declared paths look correct. Use "issues" and '
        "populate findings when there are real problems that need fixing. "
        "Return ONLY the JSON — no prose before or after it."
    )


# ────────────────────────────────────────────────────────────────────────────
# Guard entry-point
# ────────────────────────────────────────────────────────────────────────────

def _produces_from_task(task: dict) -> list[str]:
    """Extract produces list from task.context.  Returns [] when missing."""
    ctx = task.get("context") or {}
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            return []
    produces = ctx.get("produces")
    return list(produces) if isinstance(produces, list) else []


def _diff_summary_from_tool_calls(tool_calls: list[dict] | None) -> str:
    """Build a short summary of what was written from the tool-calls audit log."""
    if not tool_calls:
        return ""
    written = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        if not call.get("ok"):
            continue
        name = call.get("name", "")
        args = call.get("args") or {}
        path_val = None
        for k in ("filepath", "path", "file", "target", "dest"):
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                path_val = v
                break
        if path_val:
            written.append(f"{name}({path_val})")
    if written:
        return "Files written: " + ", ".join(written)
    return "No file writes detected in tool audit log."


def check_self_critique_sub_iter(
    parsed: dict,
    *,
    task: dict,
    agent_type: str,
    self_critique_passes: int,
    tool_calls: list[dict] | None = None,
) -> GuardCorrection | None:
    """Self-critique sub-iteration guard.

    Returns a ``GuardCorrection`` when all of:
      - agent_type not in SELF_CRITIQUE_OPT_OUT_AGENT_TYPES
      - parsed action is final_answer
      - task context declares a non-empty produces list
      - self_critique_passes < MAX_SELF_CRITIQUE_PASSES

    Returns None otherwise (guard doesn't apply or budget exhausted).

    Counter management: the caller is responsible for incrementing
    ``self_critique_passes`` after consuming the returned GuardCorrection,
    and for passing the updated value on the next call.
    """
    # Early exits — cheapest checks first
    if agent_type in SELF_CRITIQUE_OPT_OUT_AGENT_TYPES:
        return None

    if parsed.get("action", "final_answer") != "final_answer":
        return None

    produces = _produces_from_task(task)
    if not produces:
        return None

    if self_critique_passes >= MAX_SELF_CRITIQUE_PASSES:
        return None

    diff_summary = _diff_summary_from_tool_calls(tool_calls)
    message = build_self_critique_message(
        diff_summary=diff_summary,
        produces=produces,
        agent_type=agent_type,
    )
    return GuardCorrection(
        guard_name="self_critique",
        message=message,
    )
