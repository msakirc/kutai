"""Tool VM helpers — pure functions for tool schema lookup, permission checks,
and tool-call partitioning.

This module holds the stateless helpers extracted from ``src/agents/base.py``
during Phase A.4 of the runtime extraction (2026-05-04).

The heavy execute_tool_call / execute_tool_calls_batch wrappers live in
Phase A.8 (src/runtime/react.py) once the inline blocks inside
_execute_react_loop are moved.

Public API
----------
SIDE_EFFECT_TOOLS              frozenset[str]
CACHEABLE_READ_TOOLS           frozenset[str]
TOOL_FAILURE_ESCALATION_THRESHOLD  int
TOOL_SCHEMAS_BY_NAME           dict[str, dict]

partition_tool_calls(tools)    → (parallel_read_only, sequential_side_effect)
check_tool_permission(agent_name, tool_name) → bool
build_litellm_tools(allowed_tools, exclude)  → list[dict] | None
"""
from __future__ import annotations

from ..infra.logging_config import get_logger
from ..tools import TOOL_SCHEMAS

logger = get_logger("runtime.tools")


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

# Tools whose execution has side effects and should not be re-run on retry.
# Read-only tools (file_tree, read_file, git_log, etc.) are always re-executed.
SIDE_EFFECT_TOOLS: frozenset[str] = frozenset({
    "shell", "shell_stdin", "shell_sequential",
    "write_file", "edit_file", "patch_file", "apply_diff", "lint",
    "verify_deps", "run_code",
    "git_init", "git_commit", "git_branch", "git_rollback",
})

# Phase 5.6: Read-only tools whose results can be cached within a single
# agent execution. Cache is invalidated when any SIDE_EFFECT_TOOL runs.
CACHEABLE_READ_TOOLS: frozenset[str] = frozenset({
    "read_file", "file_tree", "git_status", "git_log", "git_diff",
    "web_search", "smart_search", "extract_url", "read_pdf", "read_docx",
    "read_spreadsheet", "extract_text",
})

# Model escalation: after this many consecutive tool failures,
# escalate to the next tier up.
TOOL_FAILURE_ESCALATION_THRESHOLD: int = 3


# ────────────────────────────────────────────────────────────────────────────
# Pre-built schema lookup
# ────────────────────────────────────────────────────────────────────────────

# Pre-build tool schema lookup by name for O(1) access during arg validation.
TOOL_SCHEMAS_BY_NAME: dict[str, dict] = {}
for _ts in TOOL_SCHEMAS:
    _fn = _ts.get("function", {})
    _ts_name = _fn.get("name")
    if _ts_name:
        TOOL_SCHEMAS_BY_NAME[_ts_name] = _fn.get("parameters", {})
del _ts, _fn, _ts_name


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def partition_tool_calls(tools: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split tool calls into parallel (read-only) and sequential (side-effect).

    Unknown tools are treated as side-effect (safe default).
    """
    parallel, sequential = [], []
    for tc in tools:
        if tc.get("tool", "") in CACHEABLE_READ_TOOLS:
            parallel.append(tc)
        else:
            sequential.append(tc)
    return parallel, sequential


def check_tool_permission(agent_name: str, tool_name: str) -> bool:
    """Check if agent_name is permitted to use tool_name (Phase 8.1).

    Fail-closed on runtime errors (returns False).
    ImportError → returns True (module not installed yet — allow).
    """
    try:
        from ..security.permissions import check_permission
        return check_permission(agent_name, tool_name)
    except ImportError:
        return True  # Module not installed yet — allow
    except Exception as exc:
        logger.warning(f"Permission check failed for {agent_name}/{tool_name}: {exc}")
        return False  # Fail-closed on runtime errors


def build_litellm_tools(
    allowed_tools: list[str] | None,
    exclude: set[str] | None = None,
) -> list[dict] | None:
    """Build filtered tool schemas for LiteLLM function calling.

    allowed_tools:
        None  → all schemas (minus exclude)
        []    → no tools → returns None
        [...]  → only those tools ∪ {final_answer, clarify} minus exclude

    exclude: tool names to strip from the schema list (e.g. ``read_file``
    once the agent has already read a file this task — discourages the
    LLM from re-reading content already present in the blackboard).

    final_answer and clarify are pseudo-tools that terminate the ReAct
    loop. They are NEVER excludable — without them the LLM has no way
    to signal "done" and the task wedges to iteration cap. Any caller
    that puts them in the exclude set has those entries silently dropped.
    """
    exclude = exclude or set()
    # Pseudo-tools never excludable — protects loop termination.
    exclude = {t for t in exclude if t not in ("final_answer", "clarify")}

    if allowed_tools is not None and not allowed_tools:
        return None  # explicitly no tools

    if allowed_tools is not None:
        allowed = set(allowed_tools) | {"final_answer", "clarify"}
        return [
            s for s in TOOL_SCHEMAS
            if s["function"]["name"] in allowed
            and s["function"]["name"] not in exclude
        ]
    return [
        s for s in TOOL_SCHEMAS
        if s["function"]["name"] not in exclude
    ]
