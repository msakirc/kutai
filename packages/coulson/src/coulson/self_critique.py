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
from dataclasses import dataclass

from .guards import GuardCorrection
from .grounding import WRITE_TOOLS


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

    # Prompt TEXT lives in the Foundry rubric (rubrics/self_critique.yaml); this
    # builder owns only the dynamic field resolution (paths_block formatting +
    # diff_summary fallback). The rubric has an empty system, so the one-shot
    # critic string is the user message content. (Phase 3 Task 12 Batch H.)
    from finch import build_messages
    msgs = build_messages(
        "self_critique",
        {
            "agent_type": agent_type,
            "paths_block": paths_block,
            "diff_summary": diff_summary or "(no summary provided)",
        },
    )
    return msgs[1]["content"]


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
    allowed_tools: list[str] | None = None,
) -> GuardCorrection | None:
    """Self-critique sub-iteration guard.

    Returns a ``GuardCorrection`` when all of:
      - agent_type not in SELF_CRITIQUE_OPT_OUT_AGENT_TYPES
      - parsed action is final_answer
      - task context declares a non-empty produces list
      - write tools are available (not auto-stripped — see below)
      - self_critique_passes < MAX_SELF_CRITIQUE_PASSES

    Returns None otherwise (guard doesn't apply or budget exhausted).

    ``allowed_tools`` is the agent's live tool set. When it is supplied and
    contains NO write tool, the step is write-stripped (``_apply_auto_strip``
    removes write tools for structured-output schemas) — the artifact IS the
    final_answer and the engine materializes it. The critic's "did you write
    the declared files?" premise is then moot, and its "Call write_file"
    re-emit instruction is physically impossible → the agent loops to
    max_iterations (task 567381 [1.0a] prior_art_query_plan). Mirror the
    grounding guard and skip. ``allowed_tools=None`` means "all tools" (write
    present) — guard applies as before.

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

    # Write-stripped step → final_answer is the artifact; nothing to re-write.
    if allowed_tools is not None and not (set(allowed_tools) & WRITE_TOOLS):
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


# ────────────────────────────────────────────────────────────────────────────
# Reply resolution — what to do with the agent's answer to the critique prompt
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SelfCritiqueResolution:
    """Decision derived from the agent's reply to a self-critique prompt.

    ``is_verdict``    — the reply WAS a recognizable verdict envelope (had a
                        ``verdict`` key). When False the agent ignored the
                        critique schema and re-sent real content — the react
                        loop must KEEP that reply as the result, NOT restore.
    ``is_clean``      — no actionable issue (keep / restore the artifact).
                        When ``is_verdict`` and ``is_clean`` are both True the
                        loop restores the pre-critique artifact (the verdict
                        envelope must never become the task result).
    ``findings``      — parsed actionable findings (only when not clean).
    ``fix_message``   — when not clean, the re-emit prompt to feed back so the
                        agent re-produces the COMPLETE corrected artifact.
                        None when clean.
    """
    is_clean: bool
    findings: list
    fix_message: str | None = None
    is_verdict: bool = False


def _extract_balanced_json(text: str) -> dict | None:
    """Pull the first balanced ``{...}`` object out of ``text`` and json-load it.

    Tolerant of prose / ```json fences around the object. Returns None when
    no parseable object is present (e.g. the agent re-sent raw markdown).
    """
    if not isinstance(text, str):
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def build_self_critique_fix_message(findings: list, produces: list[str]) -> str:
    """Build the re-emit prompt fed back when the critic found real issues.

    Critical contract: the agent must re-emit the COMPLETE corrected artifact
    (and overwrite the declared path on disk), NOT a verdict envelope — the
    react loop takes this next reply as the task's final result.
    """
    if not isinstance(produces, list):
        produces = []
    paths_block = "\n".join(f"  - {p}" for p in produces) or "  (none declared)"
    lines = []
    for f in findings or []:
        if not isinstance(f, dict):
            continue
        sev = f.get("severity", "issue")
        fpath = f.get("file", "")
        why = f.get("why") or f.get("reason") or f.get("issue") or ""
        lines.append(f"  - [{sev}] {fpath}: {why}".rstrip())
    findings_block = "\n".join(lines) or "  (no specific findings provided)"
    return (
        "Your self-review found problems with the work you just produced:\n"
        f"{findings_block}\n\n"
        "Fix every issue above, then RE-EMIT THE COMPLETE corrected artifact. "
        "Call write_file to overwrite the declared output path(s):\n"
        f"{paths_block}\n\n"
        "Respond with your normal final_answer containing the FULL corrected "
        "content — do NOT respond with a verdict / review summary."
    )


def parse_self_critique_verdict(
    reply_text: str | None,
    *,
    produces: list[str] | None = None,
) -> SelfCritiqueResolution:
    """Interpret the agent's reply to the self-critique prompt.

    FAIL OPEN: a malformed / non-verdict / empty reply resolves to *clean* so
    a good artifact is never destroyed because the critic reply didn't parse.
    Only an explicit ``verdict == "issues"`` WITH at least one actionable
    finding triggers a re-emit.
    """
    produces = produces or []
    obj = _extract_balanced_json(reply_text or "")
    # Not a verdict envelope → the agent re-sent real content; keep it as-is.
    if obj is None or "verdict" not in obj:
        return SelfCritiqueResolution(
            is_clean=True, findings=[], fix_message=None, is_verdict=False,
        )
    verdict = obj.get("verdict")
    findings = obj.get("findings")
    if not isinstance(findings, list):
        findings = []
    actionable = [f for f in findings if isinstance(f, dict) and (
        f.get("why") or f.get("reason") or f.get("issue") or f.get("file")
    )]
    if isinstance(verdict, str) and verdict.strip().lower() == "issues" and actionable:
        return SelfCritiqueResolution(
            is_clean=False,
            findings=actionable,
            fix_message=build_self_critique_fix_message(actionable, produces),
            is_verdict=True,
        )
    # Verdict envelope with no actionable issue → restore the pre-critique
    # artifact (the envelope itself must not survive as the result).
    return SelfCritiqueResolution(
        is_clean=True, findings=[], fix_message=None, is_verdict=True,
    )
